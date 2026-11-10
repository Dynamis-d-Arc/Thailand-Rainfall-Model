"""Weekly, from Task Scheduler ("TMD Radar Weekly Sync"): mirror the cloud radar archive to the
laptop, then run each frozen forward test (STACK_TEST_PLAN.md) whose month is complete and not
yet scored. Safe to run by hand at any time. Everything it does is logged to
<mirror>/weekly_run.log.

A test that is not ready (labels or radar do not reach the end of its month yet) exits non-zero
with the reason, writes nothing, and is tried again the next week.
"""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIRROR = Path(r"C:\Users\Brandon\Documents\TMD_Radar_Archive_gcp")
VM_HOST = "136.117.100.120"
CUTS = ("2026-12-01", "2027-01-01", "2027-02-01")      # tests December, January, February
LOG = MIRROR / "weekly_run.log"


def log(*lines):
    with LOG.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}] {line}\n")


def run(*cmd):
    """Run from the worktree root, log the tail of the output, return True on exit 0."""
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT.parent, capture_output=True, text=True,
                       timeout=3600, env={**os.environ, "TMD_RADAR_ARCHIVE": str(MIRROR)},
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail = (r.stdout + r.stderr).strip().splitlines()[-25:]
    log(f"{' '.join(str(c) for c in cmd[1:])} -> exit {r.returncode}", *("    " + line for line in tail))
    return r.returncode == 0


def main():
    py = sys.executable
    if not run("powershell", "-NoProfile", "-File", ROOT / "gcp" / "sync_from_vm.ps1", "-VmHost", VM_HOST):
        log("sync failed, tests skipped")
        sys.exit(1)
    due = [c for c in CUTS
           if not (ROOT / "outputs" / f"radar_stack_summary_forward_{c.replace('-', '')}.csv").exists()]
    done = [c for c in due if run(py, "-W", "ignore", ROOT / "radar_stack_test.py", "forward", c)]
    if done:                                   # refresh the two supporting outputs alongside
        run(py, ROOT / "tmd_radar_collect.py", "report")
        run(py, "-W", "ignore", ROOT / "score_radar_persistence.py")
    log(f"done: tests scored this run {done or 'none'}, still due {[c for c in due if c not in done] or 'none'}")


if __name__ == "__main__":
    main()
