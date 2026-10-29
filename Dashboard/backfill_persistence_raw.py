"""One-off: add `persistence_raw` rows for scored issue hours whose prediction file is gone.

verify_imerg.py backfills persistence_raw on its own for every prediction file still on disk.
Older scored hours lost their CSVs to the 30-day retention, but persistence needs only the
cell list, not the forecast. So this scores them through the same score_issue() with a stub
prediction file, copying each hour's model_build from its existing persistence row. It
appends to the baseline log, and re-running adds nothing.

Usage:
    python Dashboard/backfill_persistence_raw.py
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

import verify_imerg as v


def stub_scores(old, pmax, rt, done_base):
    """persistence_raw rows for the (stamp, target) persistence rows in `old`."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for stamp, g in old.groupby("stamp"):
            stub = pd.DataFrame({"grid_number": np.arange(1, v.N_CELLS + 1)})
            for t in g.target:
                stub[f"p_{t}"], stub[f"flag_{t}"] = np.nan, 0
            stub["model_build"] = g.model_build.iloc[0]
            path = Path(tmp) / f"{stamp}.csv"
            stub.to_csv(path, index=False)
            done = {(stamp, t) for t in g.target}               # no model rows
            _, base, _ = v.score_issue(stamp, datetime.strptime(stamp, "%Y%m%d_%H%M"), path,
                                       pmax, rt, done, done_base, None, {})
            rows += [r for r in base if r["baseline"] == "persistence_raw"]
    return rows


def main():
    b = pd.read_csv(v.BASELINE_LOG, dtype={"stamp": str, "model_build": str})
    have_raw = set(b.loc[b.baseline == "persistence_raw", "stamp"])
    on_disk = {s for s, _, _ in v.prediction_issues()}
    old = b[(b.baseline == "persistence") & ~b.stamp.isin(have_raw | on_disk)]
    if old.empty:
        v.log("every scored hour without a prediction file already has persistence_raw")
        return
    issues = [datetime.strptime(s, "%Y%m%d_%H%M") for s in old.stamp.unique()]
    conn = psycopg2.connect(**v.DB_CONFIG)
    try:
        pmax, rt = v.load_labels(conn, min(issues) - timedelta(hours=v.MAX_LEAD),
                                 max(issues) + timedelta(hours=v.MAX_LEAD))
    finally:
        conn.close()
    rows = stub_scores(old, pmax, rt, v.scored_baselines())
    n = v.append_csv(v.BASELINE_LOG, rows, ["stamp", "target", "baseline"])
    v.log(f"{old.stamp.nunique()} issue hours without a prediction file: appended {n} persistence_raw rows "
          f"(of {len(old)} persistence rows; the rest no longer have complete labels)")


if __name__ == "__main__":
    main()
