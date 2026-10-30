"""Rainwatch dashboard server.

Serves Dashboard/index.html on localhost and keeps the prediction data fresh:
once per hour (at minute >= RUN_MINUTE, when no CSV exists for the current
hour yet) it runs `Thailand_Rain_V10_deploy.py predict`, converts every
v10_predictions_*.csv into Dashboard/data/pred_<stamp>.json, and updates
data/index.json + data/status.json. The page polls those files.

Usage:
    python Dashboard/rainwatch_server.py [--port 8901] [--no-predict]

It also pulls snapshots the GitHub Actions job published to the 'predictions'
branch, which is what covers the hours this machine spent asleep.

--no-predict serves whatever CSVs already exist without ever calling the
live APIs (useful offline or when rate-limited); combined with the sync loop
it turns the dashboard into a pure viewer of cloud-produced predictions.
--no-sync disables the pull instead.
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

# Prediction stamps are Bangkok local time, whatever zone the machine clock is in.
BKK = timezone(timedelta(hours=7))


def bkk_now():
    return datetime.now(BKK).replace(tzinfo=None)
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent           # Dashboard/
PROJ = ROOT.parent
PRED_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
V9_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v9_2026_decay_labels_v2"  # 28 Oct 2026 rerun on the corrected labels
DATA = ROOT / "data"
RUN_MINUTE = 10        # earliest minute past the hour to launch a predict run
MAX_LAUNCHES_PER_HOUR = 2   # the :10 run plus one retry after the cool-down; the cloud run covers the rest
QUOTA_RESET_HOUR_BKK = 7    # Open-Meteo's free-tier day resets at 00:00 UTC
PREDICT_TIMEOUT = 25 * 60
VERIFY_INTERVAL = 6 * 3600      # how often the IMERG verification loop wakes up
VERIFY_TIMEOUT = 45 * 60
VERIFY_MIN_AGE_H = 7            # IMERG Late Run latency margin passed to the verifier
RETENTION_DAYS = 30             # raw hourly prediction CSVs older than this leave the live set
                                # for PRED_DIR/archive/ as .csv.gz (kept so they can be rescored)
# Windows: a windowless parent (pythonw, as the scheduled task runs us) spawning a
# CONSOLE program gets a fresh console allocated for it - a black window that flashes
# on the user's desktop. git.exe is a console program and the sync loop calls it every
# 10 minutes, so this is not cosmetic. watchdog.py already did this for its toast.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PRED_BRANCH = "predictions"     # data branch the GitHub Actions predict job publishes to
SYNC_INTERVAL = 10 * 60         # how often to look for cloud-produced snapshots
SYNC_TIMEOUT = 120

P_COLS = ["p_h1_0.1mm", "p_h3_0.1mm", "p_h6_0.1mm",
          "p_h1_1.0mm", "p_h3_1.0mm", "p_h6_1.0mm"]
F_COLS = [c.replace("p_", "flag_") for c in P_COLS]

_status_lock = threading.Lock()
STATUS = {"last_attempt": None, "last_success": None, "last_error": None,
          "running": False, "run_minute": RUN_MINUTE, "predict_enabled": True,
          "verify_last_attempt": None, "verify_last_success": None,
          "verify_last_error": None, "verify_enabled": True,
          "sync_last_attempt": None, "sync_last_success": None,
          "sync_last_error": None, "sync_enabled": True, "sync_pulled_total": 0,
          "started_at": bkk_now().isoformat(timespec="seconds")}


def log(msg):
    print(f"[{bkk_now():%H:%M:%S}] {msg}", flush=True)


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
    """Move snapshots older than RETENTION_DAYS to archive/ as .csv.gz, never delete them:
    6-27 Sep 2026 (231 local-only hours) were lost to the old unlink and could not be rescored."""
    cutoff = bkk_now() - timedelta(days=RETENTION_DAYS)
    archive = PRED_DIR / "archive"
    moved = 0
    for p in PRED_DIR.glob("v10_predictions_*.csv"):
        stamp = p.stem.replace("v10_predictions_", "")
        try:
            ts = datetime.strptime(stamp, "%Y%m%d_%H%M")
        except ValueError:
            continue
        if ts < cutoff:
            archive.mkdir(exist_ok=True)
            dest = archive / (p.name + ".gz")
            tmp = archive / (p.name + ".gz.tmp")
            tmp.write_bytes(gzip.compress(p.read_bytes(), mtime=0))   # mtime=0: same bytes on every re-run
            os.replace(tmp, dest)
            p.unlink()
            (DATA / f"pred_{stamp}.json").unlink(missing_ok=True)
            moved += 1
    if moved:
        log(f"archived {moved} prediction snapshot(s) older than {RETENTION_DAYS} d")


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
             "generated": bkk_now().isoformat(timespec="seconds")}
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
    lab_live = PRED_DIR / "v10_health_labeled_live_labels_v2.csv"
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
               "updated": bkk_now().isoformat(timespec="seconds")}
    # labels_v2 logs: window [t, t+h) on the wettest-pixel label (verify_imerg.py)
    vlog = PRED_DIR / "v10_verification_log_labels_v2.csv"
    blog = PRED_DIR / "v10_baseline_log_labels_v2.csv"
    scored_stamps = set()
    window_start = bkk_now() - timedelta(days=14)

    def skill(g):
        """Pooled confusion + mean per-issue ROC + n-weighted Brier over a log slice."""
        tp, fp, fn = int(g["tp"].sum()), int(g["fp"].sum()), int(g["fn"].sum())
        roc = g["roc_auc"].dropna()
        return {
            "issues": int(len(g)),
            "brier": round(float((g["brier"] * g["n"]).sum() / g["n"].sum()), 4),
            "roc": round(float(roc.mean()), 4) if len(roc) else None,
            "pod": round(tp / (tp + fn), 3) if tp + fn else None,
            "far": round(fp / (tp + fp), 3) if tp + fp else None,
            "csi": round(tp / (tp + fp + fn), 3) if tp + fp + fn else None,
        }

    if vlog.exists():
        try:
            df = pd.read_csv(vlog, parse_dates=["issue_local"])
            scored_stamps = {s for s, g in df.groupby("stamp") if len(g) >= 6}
            recent = df[df["issue_local"] >= window_start]
            use = recent if len(recent) else df
            for tgt, g in use.groupby("target"):
                payload["targets"][tgt] = {
                    **skill(g),
                    "base_rate": round(float((g["base_rate"] * g["n"]).sum()
                                             / g["n"].sum()), 4),
                    "last_issue": str(g["issue_local"].max())[:16],
                    "provisional_share": round(float(g["provisional_share"].mean()), 3)
                                         if g["provisional_share"].notna().any() else None,
                }
        except Exception as exc:
            log(f"verification log read failed: {exc}")
    if blog.exists() and payload["targets"]:
        # reference forecasts (persistence / climatology) scored by verify_imerg.py on
        # the same issues; restricted to the stamps the model table above was built from
        try:
            bdf = pd.read_csv(blog, parse_dates=["issue_local"])
            brecent = bdf[bdf["issue_local"] >= window_start]
            buse = brecent if len(brecent) else bdf
            for (tgt, name), g in buse.groupby(["target", "baseline"]):
                if tgt in payload["targets"]:
                    payload["targets"][tgt].setdefault("baselines", {})[name] = skill(g)
        except Exception as exc:
            log(f"baseline log read failed: {exc}")
    mature_cut = bkk_now() - timedelta(hours=6 + VERIFY_MIN_AGE_H)
    for s in stamps:
        try:
            if datetime.strptime(s, "%Y%m%d_%H%M") <= mature_cut and s not in scored_stamps:
                payload["pending"] += 1
        except ValueError:
            pass
    latest = imerg_frontier()
    if latest is not None:
        payload["imerg_latest"] = str(latest)[:16]
        payload["imerg_lag_days"] = round(
            (bkk_now() - latest).total_seconds() / 86400, 1)
    atomic_write(DATA / "verification.json", json.dumps(payload))


def imerg_frontier():
    """Newest complete-hour IMERG observation in the local table, or None."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", "5432")),
            dbname=os.getenv("PGDATABASE", "postgres"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "Pass1234"))
        try:
            with conn.cursor() as cur:
                cur.execute('SELECT MAX(local_observation_time) '
                            'FROM "IMERG_THAILAND_DATA" WHERE is_complete_hour')
                return cur.fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:
        log(f"imerg frontier check failed: {exc}")
        return None


def _git(*args, timeout=SYNC_TIMEOUT):
    """Run a read-only git command in the project repo.

    GIT_TERMINAL_PROMPT=0 matters: without it a credential prompt on a failed
    fetch would block this thread forever rather than returning an error.
    """
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "1"}
    return subprocess.run(["git", *args], cwd=str(PROJ), env=env,
                          capture_output=True, timeout=timeout,
                          creationflags=NO_WINDOW)


def sync_predictions():
    """Pull prediction CSVs that the GitHub Actions job published to PRED_BRANCH.

    This is how the dashboard covers the hours the workstation slept through: the
    cloud job keeps producing snapshots, and this drags them into PRED_DIR so
    convert_all() can turn them into the JSON the page reads.

    Read-only with respect to the working tree. It updates a remote-tracking ref
    and reads blobs straight out of it, so it never checks anything out, never
    switches branch, and cannot disturb uncommitted work.

    Snapshots older than RETENTION_DAYS are skipped rather than pulled, otherwise
    this and prune_old() would resurrect and re-delete the same files forever.
    """
    with _status_lock:
        STATUS["sync_last_attempt"] = bkk_now().isoformat(timespec="seconds")

    ref = f"refs/remotes/origin/{PRED_BRANCH}"
    # --force is required, not cosmetic. A --depth 1 fetch grafts a parentless
    # commit, so the next one is not a descendant of the last in this repo's view
    # and git rejects the ref update as non-fast-forward. Without it the tracking
    # ref freezes at the first snapshot ever fetched and every later sync fails.
    r = _git("fetch", "--force", "--depth", "1", "origin", f"{PRED_BRANCH}:{ref}")
    if r.returncode != 0:
        raise RuntimeError("fetch failed: "
                           + r.stderr.decode(errors="replace").strip()[-300:])

    r = _git("ls-tree", "--name-only", ref)
    if r.returncode != 0:
        raise RuntimeError("could not list " + PRED_BRANCH)
    names = sorted(n for n in r.stdout.decode(errors="replace").splitlines()
                   if n.startswith("v10_predictions_") and n.endswith(".csv"))

    cutoff = bkk_now() - timedelta(days=RETENTION_DAYS)
    pulled = []
    for name in names:
        dest = PRED_DIR / name
        if dest.exists():
            continue                      # local run already covered this hour
        stamp = name[len("v10_predictions_"):-len(".csv")]
        try:
            ts = datetime.strptime(stamp, "%Y%m%d_%H%M")
        except ValueError:
            continue
        if ts < cutoff:
            continue                      # already pruned locally on purpose
        blob = _git("show", f"{ref}:{name}")
        if blob.returncode != 0 or not blob.stdout:
            log(f"sync: could not read {name}")
            continue
        tmp = dest.parent / (dest.name + ".tmp")
        tmp.write_bytes(blob.stdout)      # bytes: no newline translation
        os.replace(tmp, dest)
        pulled.append(stamp)
    return pulled


def sync_loop():
    while True:
        try:
            pulled = sync_predictions()
            with _status_lock:
                STATUS["sync_last_success"] = bkk_now().isoformat(timespec="seconds")
                STATUS["sync_last_error"] = None
                STATUS["sync_pulled_total"] += len(pulled)
            if pulled:
                span = pulled[0] if len(pulled) == 1 else f"{pulled[0]}..{pulled[-1]}"
                log(f"sync: pulled {len(pulled)} cloud snapshot(s) [{span}]")
                convert_all()
            save_status()
        except Exception as exc:
            with _status_lock:
                STATUS["sync_last_error"] = f"{bkk_now():%Y-%m-%d %H:%M} {exc}"
            log(f"sync FAILED: {exc}")
            save_status()
        time.sleep(SYNC_INTERVAL)


def run_verify():
    with _status_lock:
        STATUS["verify_last_attempt"] = bkk_now().isoformat(timespec="seconds")
    save_status()
    log("launching IMERG verification")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "verify_imerg.py"),
             "--min-age-hours", str(VERIFY_MIN_AGE_H)],
            cwd=str(PROJ), capture_output=True, text=True, timeout=VERIFY_TIMEOUT,
            creationflags=NO_WINDOW)
        if result.returncode != 0:
            tail = (result.stdout + result.stderr).strip()[-500:]
            raise RuntimeError(f"verify exited {result.returncode}: {tail}")
        summary = next((ln for ln in result.stdout.splitlines()
                        if ln.startswith("VERIFY_SUMMARY")), "VERIFY_SUMMARY {}")
        log(f"verification done: {summary.split(' ', 1)[1]}")
        with _status_lock:
            STATUS["verify_last_success"] = bkk_now().isoformat(timespec="seconds")
            STATUS["verify_last_error"] = None
    except Exception as exc:
        with _status_lock:
            STATUS["verify_last_error"] = f"{bkk_now():%Y-%m-%d %H:%M} {exc}"
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


def _failure_record(exit_code, stdout, stderr):
    """Save the predictor's complete output under data/ and return a one-line summary."""
    def text(t):
        return t.decode(errors="replace") if isinstance(t, bytes) else (t or "")
    stdout, stderr = text(stdout), text(stderr)
    # shortcut: predict_fail_*.log files are never pruned; add them to prune_old() if they pile up
    path = DATA / f"predict_fail_{bkk_now():%Y%m%d_%H%M%S}.log"
    atomic_write(path, f"exit code: {exit_code}\n\n--- stdout ---\n{stdout}\n\n--- stderr ---\n{stderr}\n")
    lines = [l for l in (stderr if stderr.strip() else stdout).splitlines() if l.strip()]
    last = lines[-1][:300] if lines else "(no output)"
    return f"predict exited {exit_code}: {last} [full output: data/{path.name}]"


def run_predict():
    with _status_lock:
        STATUS["last_attempt"] = bkk_now().isoformat(timespec="seconds")
        STATUS["running"] = True
    save_status()
    log("launching V10 predict")
    try:
        result = subprocess.run(
            [sys.executable, str(PROJ / "Thailand_Rain_V10_deploy.py"), "predict"],
            cwd=str(PROJ), capture_output=True, text=True, timeout=PREDICT_TIMEOUT,
            creationflags=NO_WINDOW)
        if result.returncode != 0:
            raise RuntimeError(_failure_record(result.returncode, result.stdout, result.stderr))
        with _status_lock:
            STATUS["last_success"] = bkk_now().isoformat(timespec="seconds")
            STATUS["last_error"] = None
        log("predict finished ok")
    except Exception as exc:
        msg = str(exc)
        if isinstance(exc, subprocess.TimeoutExpired):
            msg = _failure_record("timeout", exc.stdout, exc.stderr)
        with _status_lock:
            STATUS["last_error"] = f"{bkk_now():%Y-%m-%d %H:%M} {msg}"
        log(f"predict FAILED: {msg}")
    finally:
        with _status_lock:
            STATUS["running"] = False
        convert_all()
        save_status()


def next_quota_reset(now):
    reset = now.replace(hour=QUOTA_RESET_HOUR_BKK, minute=0, second=0, microsecond=0)
    return reset if reset > now else reset + timedelta(days=1)


def predict_tick(state):
    """One pass of the predict loop; returns the seconds to wait before the next pass.

    Launches only when the current Bangkok hour has no CSV yet, at most
    MAX_LAUNCHES_PER_HOUR times per hour (covers both failed runs and runs that
    wrote an older issue hour), with an escalating cool-down after failures: a
    rate-limited or blocked upstream API (Open-Meteo escalated 429s to connection
    refusals on 2026-09-05 after ~10-minutely retry storms) recovers faster when
    we stop hammering it.
    """
    now = bkk_now()
    if now < state.get("hold_until", now):
        return 60                                 # rate-limited: wait for the quota reset
    want = now.strftime("%Y%m%d_%H00")
    if now.minute < RUN_MINUTE or (PRED_DIR / f"v10_predictions_{want}.csv").exists():
        return 60
    launches = state["launched"].get(want, 0)
    if launches >= MAX_LAUNCHES_PER_HOUR:
        return 60
    state["launched"] = {want: launches + 1}      # only the current hour matters
    run_predict()
    with _status_lock:
        failed = STATUS["last_error"] is not None
        rate_limited = failed and "429 Client Error" in STATUS["last_error"]
    if not failed:
        state["fail_streak"] = 0
        return 60
    if rate_limited:
        # shortcut: the hold lives in memory, so a server restart costs one more attempt before it re-arms
        state["hold_until"] = next_quota_reset(now)
        with _status_lock:                        # not an outage: the cloud run covers these hours
            STATUS["last_error"] = None
            STATUS["predict_hold_until"] = state["hold_until"].isoformat(timespec="minutes")
        save_status()
        log(f"rate-limited by Open-Meteo; no local runs until {state['hold_until']:%d %b %H:%M} "
            "(cloud predictions keep syncing)")
        return 60
    state["fail_streak"] += 1
    cooldown = min(3600, 600 * state["fail_streak"])
    capped = launches + 1 >= MAX_LAUNCHES_PER_HOUR
    log(f"predict failed ({state['fail_streak']} in a row); cooling down {cooldown // 60} min"
        + (f"; no further launches for {want}" if capped else ""))
    return cooldown


def predict_loop():
    state = {"fail_streak": 0, "launched": {}}
    while True:
        try:
            delay = predict_tick(state)
        except Exception as exc:
            log(f"predict loop error: {exc}")
            delay = 60
        time.sleep(delay)


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
    ap.add_argument("--no-sync", action="store_true",
                    help=f"do not pull cloud-produced snapshots from the "
                         f"'{PRED_BRANCH}' branch")
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
    STATUS["sync_enabled"] = not args.no_sync
    save_status()
    log(f"{len(stamps)} prediction snapshot(s) available"
        + (f", latest {stamps[-1]}" if stamps else ""))

    if args.no_predict:
        log("predict loop disabled (--no-predict)")
    else:
        threading.Thread(target=predict_loop, daemon=True).start()
        log(f"predict loop armed: runs when no CSV exists for the current "
            f"hour and minute >= {RUN_MINUTE:02d}")
    if STATUS["sync_enabled"]:
        threading.Thread(target=sync_loop, daemon=True).start()
        log(f"sync loop armed: pulling '{PRED_BRANCH}' every "
            f"{SYNC_INTERVAL // 60} min")
    else:
        log("sync loop disabled (--no-sync)")
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
