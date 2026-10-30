"""How far into its own window each live forecast was published.

The +h label window starts AT the issue hour ([t, t+h)), but a forecast only exists once
predict has written it, so the first minutes of every window have already passed. This
measures that delay per issue for both publishers and writes
om_thailand_rain_v10_deploy/v10_publication_delay.csv:

- local: when the workstation's own predict wrote the CSV, from its `written_local` column
  (logged since 2026-10-30) or else the file's modification time. Files the server pulled
  from the cloud (local_kind "pulled") get no local delay: `written_by` says so on new
  files, and on older ones a modification time after the cloud commit does. Archived
  snapshots (> 30 days) lost their mtime and are not read.
- cloud: the commit time of the first `prediction <stamp>` commit on the `predictions`
  branch (fetched as a blobless clone, so no CSVs are downloaded).
- first: the earlier of the two, when the forecast first existed anywhere.

    python Dashboard/publication_delay.py              # clones predictions history to a temp dir
    python Dashboard/publication_delay.py --repo DIR   # reuse a clone that has the branch
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
PRED_DIR = PROJ / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
OUT = PRED_DIR / "v10_publication_delay.csv"
TZ = "Asia/Bangkok"


def to_local(epoch_s):
    return pd.to_datetime(epoch_s, unit="s", utc=True).tz_convert(TZ).tz_localize(None)


def local_times():
    rows = []
    for p in PRED_DIR.glob("v10_predictions_*.csv"):
        cols = pd.read_csv(p, nrows=1)
        if "written_local" in cols:
            rows.append((p.stem[16:], pd.Timestamp(cols["written_local"].iloc[0]), "written_local",
                         cols["written_by"].iloc[0]))
        else:
            rows.append((p.stem[16:], to_local(p.stat().st_mtime), "mtime", None))
    return pd.DataFrame(rows, columns=["stamp", "local_written", "local_source", "written_by"])


def cloud_times(repo):
    log = subprocess.run(["git", "-C", str(repo), "log", "--format=%ct|%s", "predictions"],
                         capture_output=True, text=True, check=True).stdout
    rows = [(s.split()[1], to_local(int(t))) for t, s in (ln.split("|", 1) for ln in log.splitlines())
            if s.startswith("prediction ")]
    df = pd.DataFrame(rows, columns=["stamp", "cloud_committed"])
    return df.groupby("stamp", as_index=False)["cloud_committed"].min()   # re-runs: first one counts


def delays(local, cloud):
    df = local.merge(cloud, on="stamp", how="outer")
    df.insert(1, "issue_local", pd.to_datetime(df["stamp"], format="%Y%m%d_%H%M"))
    mins = lambda col: ((df[col] - df["issue_local"]).dt.total_seconds() / 60).round(1)
    pulled = (df["written_by"] == "github-actions") | (
        df["written_by"].isna() & (df["local_written"] >= df["cloud_committed"]))
    df["local_kind"] = pulled.map({True: "pulled", False: "own"}).where(df["local_written"].notna())
    df["local_delay_min"] = mins("local_written").where(df["local_kind"] == "own")
    df["cloud_delay_min"] = mins("cloud_committed")
    df["first_delay_min"] = df[["local_delay_min", "cloud_delay_min"]].min(axis=1)
    return df.sort_values("stamp").reset_index(drop=True)


def summary(df):
    out = []
    for col in ("local_delay_min", "cloud_delay_min", "first_delay_min"):
        s = df[col].dropna()
        out.append({"publisher": col.replace("_delay_min", ""), "issues": len(s),
                    "p10": s.quantile(0.1), "median": s.median(), "p90": s.quantile(0.9),
                    "share_over_20": round((s > 20).mean(), 3), "share_over_60": round((s > 60).mean(), 3)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="a git dir that already has the predictions branch history")
    args = ap.parse_args()
    if args.repo:
        cloud = cloud_times(args.repo)
    else:
        url = subprocess.run(["git", "-C", str(PROJ), "remote", "get-url", "origin"],
                             capture_output=True, text=True, check=True).stdout.strip()
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "clone", "-q", "--bare", "--single-branch", "--branch", "predictions",
                            "--filter=blob:none", url, tmp], check=True)
            cloud = cloud_times(tmp)
    df = delays(local_times(), cloud)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {len(df)} issues, {df.stamp.min()} to {df.stamp.max()}")
    print(summary(df).to_string(index=False))


if __name__ == "__main__":
    main()
