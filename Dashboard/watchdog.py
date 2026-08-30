"""Rainwatch watchdog — runs independently of the server (hourly, via Task Scheduler).

Checks that the system is actually alive and raises a Windows toast when it is not:
 - newest prediction CSV older than PRED_STALE_H  -> server dead or predict failing
 - last successful IMERG verification older than VERIFY_STALE_H
 - status.json carrying a standing predict/verify error

Alerts are debounced to one per kind per DEBOUNCE_H via data/watchdog_state.json,
so a broken night does not produce a toast storm.

Usage:
    python Dashboard/watchdog.py          # the scheduled check
    python Dashboard/watchdog.py --test   # send a test toast and exit
"""

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT.parent / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
STATE = ROOT / "data" / "watchdog_state.json"
STATUS = ROOT / "data" / "status.json"

PRED_STALE_H = 3
VERIFY_STALE_H = 26
DEBOUNCE_H = 6


def toast(title, message):
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(ROOT / "notify.ps1"), "-Title", title, "-Message", message],
        capture_output=True, timeout=60)


def newest_prediction_age_h():
    stamps = []
    for p in PRED_DIR.glob("v10_predictions_*.csv"):
        try:
            stamps.append(datetime.strptime(
                p.stem.replace("v10_predictions_", ""), "%Y%m%d_%H%M"))
        except ValueError:
            pass
    if not stamps:
        return None
    return (datetime.now() - max(stamps)).total_seconds() / 3600


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        toast("Rainwatch watchdog", "Test notification — the watchdog can reach you.")
        print("test toast sent")
        return

    problems = {}
    age = newest_prediction_age_h()
    if age is None:
        problems["no_predictions"] = "No prediction CSVs found at all."
    elif age > PRED_STALE_H:
        problems["stale_predictions"] = (
            f"Newest prediction is {age:.1f} h old — server down or predict failing.")

    status = load_json(STATUS)
    if status.get("last_error"):
        problems["predict_error"] = f"Predict error: {str(status['last_error'])[:120]}"
    if status.get("verify_enabled"):
        ok = status.get("verify_last_success")
        if ok:
            since = datetime.now() - datetime.fromisoformat(ok)
            if since > timedelta(hours=VERIFY_STALE_H):
                problems["stale_verify"] = (
                    f"Last successful IMERG verification was {since.total_seconds()/3600:.0f} h ago.")
        if status.get("verify_last_error"):
            problems["verify_error"] = f"Verify error: {str(status['verify_last_error'])[:120]}"

    state = load_json(STATE)
    now = datetime.now()
    sent = 0
    for kind, msg in problems.items():
        last = state.get(kind)
        if last and now - datetime.fromisoformat(last) < timedelta(hours=DEBOUNCE_H):
            continue
        toast("Rainwatch problem", msg)
        state[kind] = now.isoformat(timespec="seconds")
        sent += 1
    # clear resolved kinds so a relapse alerts immediately
    for kind in [k for k in state if k not in problems]:
        del state[kind]
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state), encoding="utf-8")
    print(f"{len(problems)} problem(s), {sent} toast(s) sent"
          + (f": {', '.join(problems)}" if problems else ""))


if __name__ == "__main__":
    main()
