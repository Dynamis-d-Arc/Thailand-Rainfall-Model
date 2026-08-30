"""Rainwatch dashboard server.

Serves Dashboard/index.html on localhost and keeps the prediction data fresh:
once per hour (at minute >= RUN_MINUTE, when no CSV exists for the current
hour yet) it runs `Thailand_Rain_V10_deploy.py predict`, converts every
v10_predictions_*.csv into Dashboard/data/pred_<stamp>.json, and updates
data/index.json + data/status.json. The page polls those files.

Usage:
    python Dashboard/rainwatch_server.py [--port 8901] [--no-predict]

--no-predict serves whatever CSVs already exist without ever calling the
live APIs (useful offline or when rate-limited).
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent           # Dashboard/
PROJ = ROOT.parent
PRED_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
V9_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v9_2026_decay"
DATA = ROOT / "data"
RUN_MINUTE = 10        # earliest minute past the hour to launch a predict run
PREDICT_TIMEOUT = 25 * 60
VERIFY_INTERVAL = 6 * 3600      # how often the IMERG verification loop wakes up
VERIFY_TIMEOUT = 45 * 60
VERIFY_MIN_AGE_H = 7            # IMERG Late Run latency margin passed to the verifier
RETENTION_DAYS = 30             # raw hourly prediction CSVs older than this are pruned
                                # (the verification/health logs keep the distilled record)

P_COLS = ["p_h1_0.1mm", "p_h3_0.1mm", "p_h6_0.1mm",
          "p_h1_1.0mm", "p_h3_1.0mm", "p_h6_1.0mm"]
F_COLS = [c.replace("p_", "flag_") for c in P_COLS]

_status_lock = threading.Lock()
STATUS = {"last_attempt": None, "last_success": None, "last_error": None,
          "running": False, "run_minute": RUN_MINUTE, "predict_enabled": True,
          "verify_last_attempt": None, "verify_last_success": None,
          "verify_last_error": None, "verify_enabled": True}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_status():
    with _status_lock:
        atomic_write(DATA / "status.json", json.dumps(STATUS))


def csv_to_json(csv_path: Path, out_path: Path):
    df = pd.read_csv(csv_path)
    cells = []
    for row in df.itertuples(index=False):
        d = dict(zip(df.columns, row))
        flags = sum(int(d[F_COLS[i]]) << i for i in range(6))
        cells.append([int(d["grid_number"]),
                      round(float(d["longitude"]), 4),
                      round(float(d["latitude"]), 4)]
                     + [round(float(d[c]), 3) for c in P_COLS] + [flags])
    payload = {"issue": str(df["issue_local"].iloc[0])[:16], "cells": cells}
    atomic_write(out_path, json.dumps(payload, separators=(",", ":")))


def prune_old():
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    removed = 0
    for p in PRED_DIR.glob("v10_predictions_*.csv"):
        stamp = p.stem.replace("v10_predictions_", "")
        try:
            ts = datetime.strptime(stamp, "%Y%m%d_%H%M")
        except ValueError:
            continue
        if ts < cutoff:
            p.unlink()
            (DATA / f"pred_{stamp}.json").unlink(missing_ok=True)
            removed += 1
    if removed:
        log(f"pruned {removed} prediction snapshot(s) older than {RETENTION_DAYS} d")


def convert_all():
    """Convert any new/updated prediction CSVs and rewrite the index."""
    DATA.mkdir(exist_ok=True)
    prune_old()
    stamps = []
    for csv_path in sorted(PRED_DIR.glob("v10_predictions_*.csv")):
        stamp = csv_path.stem.replace("v10_predictions_", "")
        out = DATA / f"pred_{stamp}.json"
        try:
            if not out.exists() or out.stat().st_mtime < csv_path.stat().st_mtime:
                csv_to_json(csv_path, out)
                log(f"converted {csv_path.name}")
            stamps.append(stamp)
        except Exception as exc:
            log(f"convert failed for {csv_path.name}: {exc}")
    index = {"issues": stamps, "latest": stamps[-1] if stamps else None,
             "generated": datetime.now().isoformat(timespec="seconds")}
    atomic_write(DATA / "index.json", json.dumps(index))
    build_health()
    build_verification(stamps)
    return stamps


def build_health():
    """data/health.json — the V9 regime metric.

    `labeled` is the canonical monthly p(cold-top <=235 K near cell | IMERG rain) from the
    V9 analysis; it only extends when IMERG is backfilled. `live` is the per-run proxy the
    V10 predict phase appends to v10_health_log.csv (cold-top share among all cells and
    among alert cells) — unlabeled, but it moves hourly.
    """
    payload = {"labeled": [], "baseline": None, "live": []}
    rain_type = V9_DIR / "v9_rain_type_by_month.csv"
    if rain_type.exists():
        df = pd.read_csv(rain_type, dtype={"month": str})
        payload["labeled"] = [
            {"month": r.month, "p": round(float(r.p_cold235_given_rain), 4),
             "wet_rows": int(r.wet_rows)}
            for r in df.itertuples()]
        base = df[df["month"].astype(int) < 202600]["p_cold235_given_rain"]
        if len(base):
            payload["baseline"] = {"mean": round(float(base.mean()), 4),
                                   "min": round(float(base.min()), 4),
                                   "max": round(float(base.max()), 4)}
    health_log = PRED_DIR / "v10_health_log.csv"
    if health_log.exists():
        try:
            lg = pd.read_csv(health_log).sort_values("issue_local").tail(240)
            payload["live"] = json.loads(lg.to_json(orient="records"))
        except Exception as exc:
            log(f"health log read failed: {exc}")
    # labeled-live: the verification loop's monthly p(cold-top | observed rain),
    # computed from per-cell IR in the prediction CSVs joined against IMERG labels
    lab_live = PRED_DIR / "v10_health_labeled_live.csv"
    if lab_live.exists():
        try:
            df = pd.read_csv(lab_live, parse_dates=["issue_local"])
            df["month"] = df["issue_local"].dt.strftime("%Y%m")
            agg = df.groupby("month").apply(
                lambda g: pd.Series({
                    "p": (g["cold235_given_rain"] * g["wet_cells"]).sum()
                         / g["wet_cells"].sum(),
                    "wet_rows": g["wet_cells"].sum(),
                }), include_groups=False)
            payload["labeled_live"] = [
                {"month": m, "p": round(float(r["p"]), 4), "wet_rows": int(r["wet_rows"])}
                for m, r in agg.iterrows() if r["wet_rows"] >= 50]
        except Exception as exc:
            log(f"labeled-live health read failed: {exc}")
    atomic_write(DATA / "health.json", json.dumps(payload))


def build_verification(stamps):
    """data/verification.json — rolling scores of past predictions vs observed IMERG."""
    payload = {"targets": {}, "window_days": 14, "pending": 0,
               "updated": datetime.now().isoformat(timespec="seconds")}
    vlog = PRED_DIR / "v10_verification_log.csv"
    scored_stamps = set()
    if vlog.exists():
        try:
            df = pd.read_csv(vlog, parse_dates=["issue_local"])
            scored_stamps = {s for s, g in df.groupby("stamp") if len(g) >= 6}
            recent = df[df["issue_local"] >= datetime.now() - timedelta(days=14)]
            use = recent if len(recent) else df
            for tgt, g in use.groupby("target"):
                tp, fp, fn = int(g["tp"].sum()), int(g["fp"].sum()), int(g["fn"].sum())
                roc = g["roc_auc"].dropna()
                payload["targets"][tgt] = {
                    "issues": int(len(g)),
                    "brier": round(float((g["brier"] * g["n"]).sum() / g["n"].sum()), 4),
                    "roc": round(float(roc.mean()), 4) if len(roc) else None,
                    "pod": round(tp / (tp + fn), 3) if tp + fn else None,
                    "far": round(fp / (tp + fp), 3) if tp + fp else None,
                    "csi": round(tp / (tp + fp + fn), 3) if tp + fp + fn else None,
                    "base_rate": round(float((g["base_rate"] * g["n"]).sum()
                                             / g["n"].sum()), 4),
                    "last_issue": str(g["issue_local"].max())[:16],
                    "provisional_share": round(float(g["provisional_share"].mean()), 3)
                                         if g["provisional_share"].notna().any() else None,
                }
        except Exception as exc:
            log(f"verification log read failed: {exc}")
    mature_cut = datetime.now() - timedelta(hours=6 + VERIFY_MIN_AGE_H)
    for s in stamps:
        try:
            if datetime.strptime(s, "%Y%m%d_%H%M") <= mature_cut and s not in scored_stamps:
                payload["pending"] += 1
        except ValueError:
            pass
    atomic_write(DATA / "verification.json", json.dumps(payload))


def run_verify():
    with _status_lock:
        STATUS["verify_last_attempt"] = datetime.now().isoformat(timespec="seconds")
    save_status()
    log("launching IMERG verification")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "verify_imerg.py"),
             "--min-age-hours", str(VERIFY_MIN_AGE_H)],
            cwd=str(PROJ), capture_output=True, text=True, timeout=VERIFY_TIMEOUT)
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip()[-500:]
            raise RuntimeError(f"verify exited {result.returncode}: {tail}")
        summary = next((ln for ln in result.stdout.splitlines()
                        if ln.startswith("VERIFY_SUMMARY")), "VERIFY_SUMMARY {}")
        log(f"verification done: {summary.split(' ', 1)[1]}")
        with _status_lock:
            STATUS["verify_last_success"] = datetime.now().isoformat(timespec="seconds")
            STATUS["verify_last_error"] = None
    except Exception as exc:
        with _status_lock:
            STATUS["verify_last_error"] = f"{datetime.now():%Y-%m-%d %H:%M} {exc}"
        log(f"verification FAILED: {exc}")
    finally:
        convert_all()
        save_status()


def verify_loop():
    time.sleep(180)   # let the first predict/convert settle before hitting GEE
    while True:
        try:
            run_verify()
        except Exception as exc:
            log(f"verify loop error: {exc}")
        time.sleep(VERIFY_INTERVAL)


def run_predict():
    with _status_lock:
        STATUS["last_attempt"] = datetime.now().isoformat(timespec="seconds")
        STATUS["running"] = True
    save_status()
    log("launching V10 predict")
    try:
        result = subprocess.run(
            [sys.executable, str(PROJ / "Thailand_Rain_V10_deploy.py"), "predict"],
            cwd=str(PROJ), capture_output=True, text=True, timeout=PREDICT_TIMEOUT)
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip()[-500:]
            raise RuntimeError(f"predict exited {result.returncode}: {tail}")
        with _status_lock:
            STATUS["last_success"] = datetime.now().isoformat(timespec="seconds")
            STATUS["last_error"] = None
        log("predict finished ok")
    except Exception as exc:
        with _status_lock:
            STATUS["last_error"] = f"{datetime.now():%Y-%m-%d %H:%M} {exc}"
        log(f"predict FAILED: {exc}")
    finally:
        with _status_lock:
            STATUS["running"] = False
        convert_all()
        save_status()


def predict_loop():
    while True:
        try:
            now = datetime.now()
            want = now.strftime("%Y%m%d_%H00")
            have = (PRED_DIR / f"v10_predictions_{want}.csv").exists()
            if now.minute >= RUN_MINUTE and not have:
                run_predict()
        except Exception as exc:
            log(f"predict loop error: {exc}")
        time.sleep(60)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        # data files must never be cached — the page polls them
        if "/data/" in self.path or self.path.endswith(".json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # keep the console readable; predict progress is what matters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--no-predict", action="store_true",
                    help="serve existing predictions only, never call live APIs")
    ap.add_argument("--no-verify", action="store_true",
                    help="disable the IMERG verification loop")
    ap.add_argument("--log-file", default=None,
                    help="append output here instead of the console (required when "
                         "run windowless via pythonw / Task Scheduler)")
    args = ap.parse_args()

    if args.log_file:
        lf = Path(args.log_file)
        lf.parent.mkdir(parents=True, exist_ok=True)
        if lf.exists() and lf.stat().st_size > 5_000_000:
            lf.replace(lf.with_name(lf.name + ".1"))   # single rotation, ~5 MB
        stream = open(lf, "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = stream
        log("=== server start ===")

    stamps = convert_all()
    STATUS["predict_enabled"] = not args.no_predict
    STATUS["verify_enabled"] = not (args.no_predict or args.no_verify)
    save_status()
    log(f"{len(stamps)} prediction snapshot(s) available"
        + (f", latest {stamps[-1]}" if stamps else ""))

    if args.no_predict:
        log("predict loop disabled (--no-predict)")
    else:
        threading.Thread(target=predict_loop, daemon=True).start()
        log(f"predict loop armed: runs when no CSV exists for the current "
            f"hour and minute >= {RUN_MINUTE:02d}")
    if STATUS["verify_enabled"]:
        threading.Thread(target=verify_loop, daemon=True).start()
        log(f"verify loop armed: IMERG scoring every {VERIFY_INTERVAL // 3600} h")
    else:
        log("verify loop disabled")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    log(f"Rainwatch dashboard: http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopped")


if __name__ == "__main__":
    main()
