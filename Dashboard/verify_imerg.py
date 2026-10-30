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
    appended to v10_health_labeled_live.csv;
 5. scores two reference forecasts on exactly the same (issue, target, cells) -
    persistence and climatology (see BASELINES below) - appended to
    v10_baseline_log.csv, so the model's live skill can be read as a lift.

All logs live next to the prediction CSVs. Re-running is safe: already-scored
(issue, target[, baseline]) rows are skipped, and issues whose labels are still
missing stay pending for the next run. Baselines for issues scored before the
baseline log existed are backfilled on the next run.

Usage:
    python Dashboard/verify_imerg.py [--min-age-hours 7] [--no-fetch] [--max-issues N]
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

# Prediction stamps are Bangkok local time, whatever zone the machine clock is in.
BKK = timezone(timedelta(hours=7))


def bkk_now():
    return datetime.now(BKK).replace(tzinfo=None)
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parent
PRED_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
FETCH_SCRIPT = PROJ / "IMERG_Script" / "fetch_imerg_thailand_data_gee.py"
# Fallback when Earth Engine's IMERG copy lags NASA's archive: fills only the hours the table
# is still missing, straight from GES DISC (needs the Earthdata login in ~/_netrc).
NASA_FETCH_SCRIPT = PROJ / "IMERG_Script" / "fetch_imerg_thailand_data_nasa.py"
# Label set version (2026-10-27): window [issue, issue+h) on precipitation_pixelmax_mm, the
# wettest pixel's hourly total. The logs without a suffix hold the earlier scoring, which used
# [issue+1, issue+1+h) on precipitation_max_mm (an upper bound on the pixel total); they are
# kept as history and never rewritten. Both sets grade the same prediction files.
LABEL_VERSION = "labels_v2"
IMERG_LABEL_COLUMN = "precipitation_pixelmax_mm"
VERIF_LOG = PRED_DIR / f"v10_verification_log_{LABEL_VERSION}.csv"
HEALTH_LABELED_LOG = PRED_DIR / f"v10_health_labeled_live_{LABEL_VERSION}.csv"
BASELINE_LOG = PRED_DIR / f"v10_baseline_log_{LABEL_VERSION}.csv"
CLIM_PATH = PRED_DIR / f"v10_climatology_{LABEL_VERSION}.csv.gz"   # Dashboard/build_climatology.py
DEPLOY_DIR = PRED_DIR / "deploy"

IMERG_TABLE = '"IMERG_THAILAND_DATA"'
N_CELLS = 833
MIN_COVERAGE = 0.98      # score a target only when >=98% of cells have full labels
MAX_LEAD = 6

# Reference forecasts scored on the same (issue, target, cells) as the model:
#   persistence  - the last `horizon` observed hours ending AT the issue hour, [issue-h,
#                  issue); the score is their IMERG pixel-max (mm) and the flag is
#                  "score >= rain threshold", i.e. "it just rained, so it will rain". Uses
#                  observed IMERG up to issue time, which the live model never has (4h+
#                  latency) - this is a skill reference, not a deployable forecast.
#   persistence_raw - the same forecast, flags and Brier; only its ROC differs. `persistence`
#                  ranks the score after clipping at the threshold, which ties every wet
#                  cell, so its ROC reads low (2026-10-29: 0.005-0.025 on average). This row
#                  ranks the raw score. `persistence` is still logged unchanged so earlier
#                  numbers reproduce.
#   persistence_lag6h - persistence a live system could actually issue: the same raw-score
#                  forecast from the `horizon` hours ending AVAILABLE_LAG_H before issue,
#                  [issue-6-h, issue-6). 6 h is the protocol's deployable IMERG lag
#                  (BKK_Rain_V3.IMERG_DEPLOY_LAG_HOURS), about the Early Run's latency; the
#                  Late Run this table holds lands later, so this is generous to persistence.
#   climatology  - per-cell, per-month, per-hour-of-day observed label frequency from
#                  2024-07..2026-07 (build_climatology.py); flagged with the same
#                  probability threshold as the deployed model for that target.
#   om_only      - the no-IR shadow model's om_p_* / om_flag_* columns (Thailand_Rain_V10_deploy
#                  `shadow`), scored only on prediction files that carry them (2026-10-30 on).
BASELINES = ("persistence", "persistence_raw", "persistence_lag6h", "climatology", "om_only")
AVAILABLE_LAG_H = 6
RECENT_DAYS = 30   # baseline gaps on older issues are filled only by an explicit --backfill run

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}


def log(msg):
    print(f"[{bkk_now():%H:%M:%S}] {msg}", flush=True)


def prediction_issues():
    """[(stamp, issue_ts, csv_path)] for every prediction on disk, oldest first.
    Includes archive/*.csv.gz (rainwatch_server.prune_old) so a relabel can rescore them."""
    out = []
    for p in [*PRED_DIR.glob("v10_predictions_*.csv"),
              *(PRED_DIR / "archive").glob("v10_predictions_*.csv.gz")]:
        stamp = p.name.split(".")[0].replace("v10_predictions_", "")
        try:
            out.append((stamp, datetime.strptime(stamp, "%Y%m%d_%H%M"), p))
        except ValueError:
            log(f"skipping unparseable stamp {stamp}")
    return sorted(out)


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


def scored_baselines():
    """{(stamp, target, baseline)} already in the baseline log."""
    if not BASELINE_LOG.exists():
        return set()
    df = pd.read_csv(BASELINE_LOG, usecols=["stamp", "target", "baseline"], dtype=str)
    return set(zip(df["stamp"], df["target"], df["baseline"]))


def deploy_thresholds():
    """{target: probability threshold} of the models currently in deploy/."""
    import warnings
    import joblib
    out = {}
    for p in sorted(DEPLOY_DIR.glob("th_rain_*.joblib")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = joblib.load(p)
        out[d["target"]] = float(d["threshold"])
    return out


def load_climatology():
    """(grid, month, hour)-indexed climatology, or None if it has not been built."""
    if not CLIM_PATH.exists():
        log(f"no climatology at {CLIM_PATH.name} - run Dashboard/build_climatology.py")
        return None
    df = pd.read_csv(CLIM_PATH)
    return df.set_index(["month", "hour", "grid_number"]).sort_index()


def metrics(y, p, flag):
    """Brier, ROC-AUC and the flag confusion for one (issue, target, forecaster)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    f = np.asarray(flag, dtype=int)
    tp = int(((f == 1) & (y == 1)).sum())
    fp = int(((f == 1) & (y == 0)).sum())
    fn = int(((f == 0) & (y == 1)).sum())
    tn = int(((f == 0) & (y == 0)).sum())
    roc = np.nan
    if 0 < y.mean() < 1:
        from sklearn.metrics import roc_auc_score
        roc = float(roc_auc_score(y, p))
    return {
        "n": int(len(y)), "coverage": round(len(y) / N_CELLS, 4),
        "base_rate": round(float(y.mean()), 4),
        "brier": round(float(np.mean((np.clip(p, 0, 1) - y) ** 2)), 4),
        "roc_auc": round(roc, 4) if np.isfinite(roc) else np.nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pod": round(tp / (tp + fn), 4) if tp + fn else np.nan,
        "far": round(fp / (tp + fp), 4) if tp + fp else np.nan,
        "csi": round(tp / (tp + fp + fn), 4) if tp + fp + fn else np.nan,
    }


def run_fetch(start_date, end_date):
    """Top up IMERG for [start_date, end_date] (UTC dates): Earth Engine first (complete chunks
    are skipped), then NASA's archive for any hour Earth Engine has not delivered yet."""
    dates = ["--start-date", start_date.strftime("%Y-%m-%d"),
             "--end-date", end_date.strftime("%Y-%m-%d")]
    for script, fatal in ((FETCH_SCRIPT, True), (NASA_FETCH_SCRIPT, False)):
        log(f"fetching IMERG via {script.stem} " + " ".join(dates))
        # creationflags: harmless here (sys.executable is pythonw under the scheduled
        # task) but keeps the no-console rule uniform across the project's spawns.
        r = subprocess.run([sys.executable, str(script)] + dates, cwd=str(PROJ),
                           capture_output=True, text=True, timeout=30 * 60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tail = (r.stdout + r.stderr).strip().splitlines()
        for line in tail[-4:]:
            log(f"  fetch| {line}")
        if r.returncode != 0:
            if fatal:
                raise RuntimeError(f"IMERG fetch exited {r.returncode}")
            log(f"  {script.stem} exited {r.returncode}; scoring with what Earth Engine delivered")


def lead_hours(issue):
    """Label hours of the 6 h window starting AT the issue hour: [issue, issue+6)."""
    return [issue + timedelta(hours=k) for k in range(MAX_LEAD)]


def hist_hours(issue, lag=0):
    """The 6 observed hours ending `lag` hours before the issue hour, oldest first:
    [issue-lag-6, issue-lag)."""
    return [issue - timedelta(hours=lag + k) for k in range(MAX_LEAD, 0, -1)]


def load_labels(conn, lo, hi):
    """IMERG_LABEL_COLUMN pivot (grid x local hour) over [lo, hi], complete hours only."""
    df = pd.read_sql(
        f"SELECT grid_number, local_observation_time, {IMERG_LABEL_COLUMN} AS label_mm, run_type "
        f"FROM {IMERG_TABLE} WHERE is_complete_hour "
        f"AND local_observation_time BETWEEN %s AND %s",
        conn, params=(lo, hi))
    if df.empty:
        return None, None
    pmax = df.pivot(index="grid_number", columns="local_observation_time", values="label_mm")
    rt = df.pivot(index="grid_number", columns="local_observation_time", values="run_type")
    return pmax, rt


def score_issue(stamp, issue, csv_path, pmax, rt, done, done_base, clim, thr_map):
    """Score every unscored target (model + baselines) of one issue.

    Returns (model_rows, baseline_rows, health_row_or_None).
    """
    pred = pd.read_csv(csv_path).set_index("grid_number")
    leads, hist_hrs = lead_hours(issue), hist_hours(issue)
    rows, base_rows, health = [], [], None
    now_str = bkk_now().strftime("%Y-%m-%d %H:%M")

    # lead / history matrices aligned to prediction cells (NaN where an hour is missing)
    def aligned(hours):
        m = pd.DataFrame(index=pred.index)
        for h in hours:
            m[h] = pmax[h].reindex(pred.index) if (pmax is not None and h in pmax.columns) \
                else np.nan
        return m
    lag_hrs = hist_hours(issue, AVAILABLE_LAG_H)
    lead, hist, hist_lag = aligned(leads), aligned(hist_hrs), aligned(lag_hrs)

    for name, horizon, thr in targets_from_columns(pred.columns):
        need_model = (stamp, name) not in done
        need_base = {b: (stamp, name, b) not in done_base for b in BASELINES}
        if clim is None:
            need_base["climatology"] = False
        if f"om_p_{name}" not in pred.columns:
            need_base["om_only"] = False
        if not need_model and not any(need_base.values()):
            continue
        window = lead[leads[:horizon]]
        ok = window.notna().all(axis=1)
        n = int(ok.sum())
        if n < MIN_COVERAGE * N_CELLS:
            continue  # labels not fully ingested yet - stays pending
        y = (window[ok] >= thr).any(axis=1).astype(int).to_numpy()
        common = {"stamp": stamp, "issue_local": issue, "target": name,
                  # fit that produced the file; "" for files written before 2026-10-28
                  "model_build": str(pred["model_build"].iloc[0])
                  if "model_build" in pred.columns and pd.notna(pred["model_build"].iloc[0]) else ""}

        if need_model:
            p = pred.loc[ok, f"p_{name}"].to_numpy()
            f = pred.loc[ok, f"flag_{name}"].to_numpy().astype(int)
            prov = np.nan
            if rt is not None:
                used = rt.reindex(index=pred.index[ok], columns=leads[:horizon])
                vals = used.to_numpy().ravel()
                vals = vals[pd.notna(vals)]
                if len(vals):
                    prov = float((vals == "provisional").mean())
            rows.append({**common, **metrics(y, p, f),
                         "provisional_share": round(prov, 4) if np.isfinite(prov) else np.nan,
                         "scored_at": now_str})

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

        if need_base["persistence"] or need_base["persistence_raw"]:
            # the last `horizon` observed hours ending at issue time, same cells as ok
            recent = hist[hist_hrs[-horizon:]]
            ok_p = ok & recent.notna().all(axis=1)
            if int(ok_p.sum()) >= MIN_COVERAGE * N_CELLS:
                y_p = (window[ok_p] >= thr).any(axis=1).astype(int).to_numpy()
                score = recent[ok_p].max(axis=1).to_numpy()
                if need_base["persistence"]:
                    # Brier needs a probability: rank the score into [0,1] by thr saturation
                    p_p = np.clip(score / thr, 0, 1)
                    base_rows.append({**common, "baseline": "persistence",
                                      **metrics(y_p, p_p, score >= thr), "scored_at": now_str})
                if need_base["persistence_raw"]:
                    # metrics() clips for Brier itself, so only the ROC ranking changes
                    base_rows.append({**common, "baseline": "persistence_raw",
                                      **metrics(y_p, score / thr, score >= thr), "scored_at": now_str})

        if need_base["persistence_lag6h"]:
            stale = hist_lag[lag_hrs[-horizon:]]
            ok_l = ok & stale.notna().all(axis=1)
            if int(ok_l.sum()) >= MIN_COVERAGE * N_CELLS:
                score = stale[ok_l].max(axis=1).to_numpy()
                base_rows.append({**common, "baseline": "persistence_lag6h",
                                  **metrics((window[ok_l] >= thr).any(axis=1).astype(int).to_numpy(),
                                            score / thr, score >= thr), "scored_at": now_str})

        if need_base["om_only"]:
            base_rows.append({**common, "baseline": "om_only",
                              **metrics(y, pred.loc[ok, f"om_p_{name}"].to_numpy(),
                                        pred.loc[ok, f"om_flag_{name}"].to_numpy()),
                              "scored_at": now_str})

        if need_base["climatology"]:
            try:
                c = clim.loc[(issue.month, issue.hour)]
            except KeyError:
                c = None
            if c is not None:
                p_c = c[f"p_{name}"].reindex(pred.index[ok]).to_numpy(dtype=float)
                good = np.isfinite(p_c)
                if int(good.sum()) >= MIN_COVERAGE * N_CELLS:
                    flag = p_c[good] >= thr_map.get(name, 0.5)
                    base_rows.append({**common, "baseline": "climatology",
                                      **metrics(y[good], p_c[good], flag),
                                      "scored_at": now_str})
    return rows, base_rows, health


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
    ap.add_argument("--backfill", action="store_true",
                    help=f"also fill missing baselines on issues older than {RECENT_DAYS} days "
                         "(the archive); pair with --no-fetch, the labels are already stored")
    args = ap.parse_args()

    done = scored_pairs()
    done_base = scored_baselines()
    clim = load_climatology()
    want_base = [b for b in BASELINES if b != "climatology" or clim is not None]
    now = bkk_now()
    mature = [(s, t, p) for s, t, p in prediction_issues()
              if t + timedelta(hours=MAX_LEAD + args.min_age_hours) <= now]

    def unscored(s, t, p):
        cols = pd.read_csv(p, nrows=0).columns
        names = [n for n, _, _ in targets_from_columns(cols)]
        if any((s, n) not in done for n in names):
            return True
        if t < now - timedelta(days=RECENT_DAYS) and not args.backfill:
            # keeps a routine run's label window (and its Earth Engine top-up) to recent days
            return False
        wanted = [b for b in want_base if b != "om_only" or any(c.startswith("om_p_") for c in cols)]
        return any((s, n, b) not in done_base for n in names for b in wanted)
    pending = [(s, t, p) for s, t, p in mature if unscored(s, t, p)]
    if args.max_issues:
        pending = pending[: args.max_issues]
    log(f"{len(mature)} mature issues, {len(pending)} with unscored targets/baselines")
    if not pending:
        print("VERIFY_SUMMARY " + json.dumps({"scored": 0, "pending": 0}))
        return

    # labels for the lead window plus the hours before issue the persistence baselines read
    lo = min(t for _, t, _ in pending) - timedelta(hours=MAX_LEAD + AVAILABLE_LAG_H)
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

    thr_map = deploy_thresholds() if clim is not None else {}
    all_rows, all_base, health_rows = [], [], []
    for stamp, issue, path in pending:
        rows, base_rows, health = score_issue(stamp, issue, path, pmax, rt,
                                              done, done_base, clim, thr_map)
        all_rows.extend(rows)
        all_base.extend(base_rows)
        if health:
            health_rows.append(health)
        if rows or base_rows:
            log(f"{stamp}: scored {len(rows)} targets, {len(base_rows)} baseline rows"
                + (f", health wet_cells={health['wet_cells']}" if health else ""))

    n1 = append_csv(VERIF_LOG, all_rows, ["stamp", "target"])
    n2 = append_csv(HEALTH_LABELED_LOG, health_rows, ["stamp"])
    n3 = append_csv(BASELINE_LOG, all_base, ["stamp", "target", "baseline"])
    log(f"appended {n1} verification rows, {n2} labeled-health rows, {n3} baseline rows")
    touched = {r["stamp"] for r in all_rows} | {r["stamp"] for r in all_base}
    print("VERIFY_SUMMARY " + json.dumps(
        {"scored": len({r['stamp'] for r in all_rows}), "rows": n1, "baseline_rows": n3,
         "pending": len(pending) - len(touched)}))


if __name__ == "__main__":
    main()
