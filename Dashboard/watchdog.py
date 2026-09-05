"""Rainwatch watchdog — runs independently of the server (hourly, via Task Scheduler).

Checks that the system is actually alive and raises a Windows toast when it is not:
 - dashboard server unreachable on port 8901
 - newest prediction CSV older than PRED_STALE_H (with a startup grace period, so a
   machine waking from overnight sleep is not flagged while the server catches up)
 - last successful IMERG verification older than VERIFY_STALE_H
 - status.json carrying a standing predict/verify error
 - the IMERG source feed itself more than IMERG_LAG_ALERT_D days behind (NASA/GEE
   ingestion stalls happen; verification pauses until it resumes)

Alerts are debounced to one per kind per DEBOUNCE_H via data/watchdog_state.json.
Runs cleanly under pythonw (no console); the toast subprocess is spawned hidden.

Usage:
    python Dashboard/watchdog.py          # the scheduled check
    python Dashboard/watchdog.py --test   # send a test toast and exit
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT.parent / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
STATE = ROOT / "data" / "watchdog_state.json"
WLOG = ROOT / "data" / "watchdog.log"

PRED_STALE_H = 3
VERIFY_STALE_H = 26
IMERG_LAG_ALERT_D = 3    # toast when the IMERG source feed falls this many days behind
DEBOUNCE_H = {"default": 6, "imerg_lag": 24}   # a stalled feed re-alerts daily, not 4x/day
STARTUP_GRACE_MIN = 30   # server started this recently -> skip staleness checks


def out(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    try:
        WLOG.parent.mkdir(exist_ok=True)
        if WLOG.exists() and WLOG.stat().st_size > 1_000_000:
            WLOG.replace(WLOG.with_name(WLOG.name + ".1"))
        with open(WLOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    if sys.stdout:   # None under pythonw
        print(line)


def toast(title, message):
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-WindowStyle", "Hidden",
         "-File", str(ROOT / "notify.ps1"), "-Title", title, "-Message", message],
        capture_output=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def server_status():
    """Live status.json over HTTP, or None when the server is unreachable."""
    try:
        # generous timeout: heavy local training can starve the server for seconds
        # without it being down (a 5s probe false-alarmed during a 13-fold CV)
        with urllib.request.urlopen(
                "http://localhost:8901/data/status.json", timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def newest_prediction_age_h():
    stamps = []
    for p in PRED_DIR.glob("v10_predictions_*.csv"):
        try:
            stamps.append(datetime.strptime(
                p.stem.replace("v10_predictions_", ""), "%Y%m%d_%H%M"))
        except ValueError:
            pass
    return (datetime.now() - max(stamps)).total_seconds() / 3600 if stamps else None


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def imerg_lag_days():
    """Days between now and the newest complete IMERG hour in the local table."""
    try:
        import psycopg2
        import os
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
                latest = cur.fetchone()[0]
        finally:
            conn.close()
        if latest is None:
            return None
        return (datetime.now() - latest).total_seconds() / 86400
    except Exception:
        return None   # DB down is surfaced through predict errors, not here


def find_problems():
    problems = {}

    lag = imerg_lag_days()
    if lag is not None and lag > IMERG_LAG_ALERT_D:
        problems["imerg_lag"] = (
            f"IMERG source feed is {lag:.1f} days behind — verification is paused "
            f"until NASA/GEE catches up (predictions unaffected).")

    status = server_status()
    if status is None:
        problems["server_down"] = ("Dashboard server unreachable on port 8901 — "
                                   "check the 'Rainwatch Server' scheduled task.")
        return problems

    in_grace = False
    started = status.get("started_at")
    if started:
        try:
            in_grace = datetime.now() - datetime.fromisoformat(started) \
                < timedelta(minutes=STARTUP_GRACE_MIN)
        except ValueError:
            pass

    age = newest_prediction_age_h()
    if age is None:
        problems["no_predictions"] = "No prediction CSVs found at all."
    elif age > PRED_STALE_H and not in_grace:
        problems["stale_predictions"] = (
            f"Newest prediction is {age:.1f} h old although the server is up — "
            f"the predict loop is failing.")

    if status.get("last_error"):
        problems["predict_error"] = f"Predict error: {str(status['last_error'])[:120]}"
    if status.get("verify_enabled"):
        ok = status.get("verify_last_success")
        if ok and not in_grace:
            since = datetime.now() - datetime.fromisoformat(ok)
            if since > timedelta(hours=VERIFY_STALE_H):
                problems["stale_verify"] = (
                    f"Last successful IMERG verification was "
                    f"{since.total_seconds() / 3600:.0f} h ago.")
        if status.get("verify_last_error"):
            problems["verify_error"] = \
                f"Verify error: {str(status['verify_last_error'])[:120]}"
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        toast("Rainwatch watchdog", "Test notification — the watchdog can reach you.")
        out("test toast sent")
        return

    problems = find_problems()
    state = load_json(STATE)
    now = datetime.now()
    sent = 0
    for kind, msg in problems.items():
        last = state.get(kind)
        hours = DEBOUNCE_H.get(kind, DEBOUNCE_H["default"])
        if last and now - datetime.fromisoformat(last) < timedelta(hours=hours):
            continue
        toast("Rainwatch problem", msg)
        state[kind] = now.isoformat(timespec="seconds")
        sent += 1
    for kind in [k for k in state if k not in problems]:
        del state[kind]   # a relapse after recovery alerts immediately
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state), encoding="utf-8")
    out(f"{len(problems)} problem(s), {sent} toast(s) sent"
        + (f": {', '.join(problems)}" if problems else ""))


if __name__ == "__main__":
    main()
