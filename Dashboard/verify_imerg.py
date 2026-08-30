"""IMERG verification loop for the V10 predictions.

Grades every mature prediction CSV against observed IMERG rain, with the exact
label definition the models were trained on (Thailand_Rain_V3): label = IMERG
cell-max hourly accumulation (`precipitation_max_mm`, complete hours only)
>= threshold in ANY of the next h hours, all h hours present.

Per run it:
 1. finds prediction stamps old enough for IMERG Late Run to exist and not yet
    fully scored;
 2. tops up "IMERG_THAILAND_DATA" for the needed dates via the existing Earth
    Engine fetcher (idempotent - complete chunks are skipped);
 3. scores each (issue, target) that has complete labels: Brier, ROC-AUC, and
    the flag confusion (POD / FAR / CSI), appended to v10_verification_log.csv;
 4. extends the labeled V9 health metric where the prediction CSV carries the
    per-cell hw_cold235_env_lag1 column: p(cold-top | observed rain next hour),
    appended to v10_health_labeled_live.csv.

Both logs live next to the prediction CSVs. Re-running is safe: already-scored
(issue, target) pairs are skipped, and issues whose labels are still missing
stay pending for the next run.

Usage:
    python Dashboard/verify_imerg.py [--min-age-hours 7] [--no-fetch] [--max-issues N]
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
PRED_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
FETCH_SCRIPT = PROJ / "IMERG_Script" / "fetch_imerg_thailand_data_gee.py"
VERIF_LOG = PRED_DIR / "v10_verification_log.csv"
HEALTH_LABELED_LOG = PRED_DIR / "v10_health_labeled_live.csv"

IMERG_TABLE = '"IMERG_THAILAND_DATA"'
N_CELLS = 833
MIN_COVERAGE = 0.98      # score a target only when >=98% of cells have full labels
MAX_LEAD = 6

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def prediction_issues():
    """[(stamp, issue_ts, csv_path)] for every prediction on disk, oldest first."""
    out = []
    for p in sorted(PRED_DIR.glob("v10_predictions_*.csv")):
        stamp = p.stem.replace("v10_predictions_", "")
        try:
            out.append((stamp, datetime.strptime(stamp, "%Y%m%d_%H%M"), p))
        except ValueError:
            log(f"skipping unparseable stamp {stamp}")
    return out


def targets_from_columns(cols):
    """[('h1_0.1mm', 1, 0.1), ...] parsed from p_* prediction columns."""
    out = []
    for c in cols:
        if c.startswith("p_h"):
            name = c[2:]
            h, thr = name.split("_")
            out.append((name, int(h[1:]), float(thr.replace("mm", ""))))
    return out


def scored_pairs():
    if not VERIF_LOG.exists():
        return set()
    df = pd.read_csv(VERIF_LOG, usecols=["stamp", "target"], dtype=str)
    return set(zip(df["stamp"], df["target"]))


def run_fetch(start_date, end_date):
    """Top up IMERG for [start_date, end_date] (UTC dates); complete chunks are skipped."""
    cmd = [sys.executable, str(FETCH_SCRIPT),
           "--start-date", start_date.strftime("%Y-%m-%d"),
           "--end-date", end_date.strftime("%Y-%m-%d")]
    log("fetching IMERG " + " ".join(cmd[2:]))
    r = subprocess.run(cmd, cwd=str(PROJ), capture_output=True, text=True, timeout=30 * 60)
    tail = (r.stdout + r.stderr).strip().splitlines()
    for line in tail[-4:]:
        log(f"  fetch| {line}")
    if r.returncode != 0:
        raise RuntimeError(f"IMERG fetch exited {r.returncode}")


def load_labels(conn, lo, hi):
    """precipitation_max_mm pivot (grid x local hour) over [lo, hi], complete hours only."""
    df = pd.read_sql(
        f"SELECT grid_number, local_observation_time, precipitation_max_mm, run_type "
        f"FROM {IMERG_TABLE} WHERE is_complete_hour "
        f"AND local_observation_time BETWEEN %s AND %s",
        conn, params=(lo, hi))
    if df.empty:
        return None, None
    pmax = df.pivot(index="grid_number", columns="local_observation_time",
                    values="precipitation_max_mm")
    rt = df.pivot(index="grid_number", columns="local_observation_time", values="run_type")
    return pmax, rt


def score_issue(stamp, issue, csv_path, pmax, rt, done):
    """Score every unscored target of one issue; returns (log_rows, health_row_or_None)."""
    pred = pd.read_csv(csv_path).set_index("grid_number")
    lead_hours = [issue + timedelta(hours=k) for k in range(1, MAX_LEAD + 1)]
    have = [h for h in lead_hours if pmax is not None and h in pmax.columns]
    rows, health = [], None

    # lead matrix aligned to prediction cells (NaN where a label hour is missing)
    lead = pd.DataFrame(index=pred.index)
    for h in lead_hours:
        lead[h] = pmax[h].reindex(pred.index) if h in have else np.nan

    for name, horizon, thr in targets_from_columns(pred.columns):
        if (stamp, name) in done:
            continue
        window = lead[lead_hours[:horizon]]
        ok = window.notna().all(axis=1)
        n = int(ok.sum())
        if n < MIN_COVERAGE * N_CELLS:
            continue  # labels not fully ingested yet - stays pending
        y = (window[ok] >= thr).any(axis=1).astype(int).to_numpy()
        p = pred.loc[ok, f"p_{name}"].to_numpy()
        f = pred.loc[ok, f"flag_{name}"].to_numpy().astype(int)
        tp = int(((f == 1) & (y == 1)).sum())
        fp = int(((f == 1) & (y == 0)).sum())
        fn = int(((f == 0) & (y == 1)).sum())
        tn = int(((f == 0) & (y == 0)).sum())
        roc = np.nan
        if 0 < y.mean() < 1:
            from sklearn.metrics import roc_auc_score
            roc = float(roc_auc_score(y, p))
        prov = np.nan
        if rt is not None:
            used = rt.reindex(index=pred.index[ok], columns=lead_hours[:horizon])
            vals = used.to_numpy().ravel()
            vals = vals[pd.notna(vals)]
            if len(vals):
                prov = float((vals == "provisional").mean())
        rows.append({
            "stamp": stamp, "issue_local": issue, "target": name,
            "n": n, "coverage": round(n / N_CELLS, 4),
            "base_rate": round(float(y.mean()), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "roc_auc": round(roc, 4) if np.isfinite(roc) else np.nan,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pod": round(tp / (tp + fn), 4) if tp + fn else np.nan,
            "far": round(fp / (tp + fp), 4) if tp + fp else np.nan,
            "csi": round(tp / (tp + fp + fn), 4) if tp + fp + fn else np.nan,
            "provisional_share": round(prov, 4) if np.isfinite(prov) else np.nan,
            "scored_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

        # labeled V9 health from the h1_1.0mm target, where the CSV has per-cell IR
        if name == "h1_1.0mm" and "hw_cold235_env_lag1" in pred.columns:
            cold = pred.loc[ok, "hw_cold235_env_lag1"].to_numpy()
            valid = (pred.loc[ok, "hw_valid"].to_numpy() == 1) & np.isfinite(cold)
            wet = (y == 1) & valid
            if wet.sum() > 0:
                health = {
                    "stamp": stamp, "issue_local": issue,
                    "wet_cells": int(wet.sum()),
                    "cold235_given_rain": round(float((cold[wet] > 0).mean()), 4),
                }
    return rows, health


def append_csv(path, new_rows, key_cols):
    if not new_rows:
        return 0
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        new["issue_local"] = new["issue_local"].astype(str)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
    else:
        merged = new
    merged = merged.sort_values(key_cols)
    tmp = path.with_suffix(".csv.tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-age-hours", type=float, default=7.0,
                    help="only score issues whose full 6h window ended at least this "
                         "long ago (IMERG Late Run latency)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="score against whatever IMERG is already in the table")
    ap.add_argument("--max-issues", type=int, default=None)
    args = ap.parse_args()

    done = scored_pairs()
    now = datetime.now()
    mature = [(s, t, p) for s, t, p in prediction_issues()
              if t + timedelta(hours=MAX_LEAD + args.min_age_hours) <= now]
    pending = [(s, t, p) for s, t, p in mature
               if any((s, name) not in done for name, _, _ in
                      targets_from_columns(pd.read_csv(p, nrows=0).columns))]
    if args.max_issues:
        pending = pending[: args.max_issues]
    log(f"{len(mature)} mature issues, {len(pending)} with unscored targets")
    if not pending:
        print("VERIFY_SUMMARY " + json.dumps({"scored": 0, "pending": 0}))
        return

    lo = min(t for _, t, _ in pending) + timedelta(hours=1)
    hi = max(t for _, t, _ in pending) + timedelta(hours=MAX_LEAD)
    if not args.no_fetch:
        # local (ICT, UTC+7) label hours span these UTC dates; fetcher takes UTC dates
        run_fetch((lo - timedelta(hours=7)).date(), (hi - timedelta(hours=7)).date())

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        pmax, rt = load_labels(conn, lo, hi)
    finally:
        conn.close()
    if pmax is None:
        log("no complete IMERG labels in the window yet - nothing scored")
        print("VERIFY_SUMMARY " + json.dumps({"scored": 0, "pending": len(pending)}))
        return

    all_rows, health_rows = [], []
    for stamp, issue, path in pending:
        rows, health = score_issue(stamp, issue, path, pmax, rt, done)
        all_rows.extend(rows)
        if health:
            health_rows.append(health)
        if rows:
            log(f"{stamp}: scored {len(rows)} targets"
                + (f", health wet_cells={health['wet_cells']}" if health else ""))

    n1 = append_csv(VERIF_LOG, all_rows, ["stamp", "target"])
    n2 = append_csv(HEALTH_LABELED_LOG, health_rows, ["stamp"])
    log(f"appended {n1} verification rows, {n2} labeled-health rows")
    print("VERIFY_SUMMARY " + json.dumps(
        {"scored": len({r['stamp'] for r in all_rows}), "rows": n1,
         "pending": len(pending) - len({r["stamp"] for r in all_rows})}))


if __name__ == "__main__":
    main()
