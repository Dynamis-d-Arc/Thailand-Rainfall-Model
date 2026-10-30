"""Stored outputs for the Phase 3 (operation) figures in the Form E1 report (NFR4).

    python Data_Analysis/phase3_evidence.py

Reads the live logs under ML_Model_V2/trained_models/om_thailand_rain_v10_deploy, the
prediction files on disk, in git HEAD and on the `predictions` branch, and the GitHub
Actions run list (via `gh`). Writes phase3_*.csv into Data_Analysis/report_evidence/.
No database needed.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
OUT = Path(__file__).resolve().parent / "report_evidence"
TARGETS = ["h1_0.1mm", "h1_1.0mm", "h3_0.1mm", "h3_1.0mm", "h6_0.1mm", "h6_1.0mm"]
# label-dependent figures read the labels_v2 logs (window [t, t+h), wettest-pixel label) and
# write *_labels_v2.csv; the unsuffixed phase3 files are the pre-fix evidence, kept as is
LABELS = "labels_v2"

# eras of the live service (Bangkok local issue hours)
LAPTOP_ERA = ("2026-08-29 14:00", "2026-09-28 16:00")   # dashboard launch -> cloud job
CLOUD_ERA = ("2026-09-28 16:00", "2026-10-27 16:00")    # cloud job -> report snapshot
WINDOW_30D = ("2026-09-27 16:00", "2026-10-27 16:00")   # NFR6 window


def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(f"wrote {name}: {len(df)} rows")


def git_stamps(ref, path=""):
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref] + ([path] if path else []),
                         capture_output=True, text=True, cwd=ROOT).stdout.split()
    return {m.group(1) for f in out for m in [re.search(r"v10_predictions_(\d{8}_\d{4})", f)] if m}


def live_skill():
    v = pd.read_csv(DEPLOY / f"v10_verification_log_{LABELS}.csv")
    b = pd.read_csv(DEPLOY / f"v10_baseline_log_{LABELS}.csv")
    m = v.rename(columns={"roc_auc": "model_roc", "brier": "model_brier"})
    p = b[b.baseline == "persistence"][["stamp", "target", "roc_auc", "brier"]] \
        .rename(columns={"roc_auc": "persistence_roc", "brier": "persistence_brier"})
    c = b[b.baseline == "climatology"][["stamp", "target", "roc_auc", "brier"]] \
        .rename(columns={"roc_auc": "climatology_roc", "brier": "climatology_brier"})
    j = m.merge(p, on=["stamp", "target"]).merge(c, on=["stamp", "target"])
    j = j.dropna(subset=["model_roc", "persistence_roc", "climatology_roc"])
    def skill_table(j):
        rows = []
        for t in TARGETS:
            d = j[j.target == t]
            if d.empty:
                continue
            rows.append({
                "target": t, "issue_hours": len(d),
                "first_issue": d.issue_local.min(), "last_issue": d.issue_local.max(),
                "mean_base_rate": round(d.base_rate.mean(), 3),
                "model_roc": round(d.model_roc.mean(), 3),
                "persistence_roc": round(d.persistence_roc.mean(), 3),
                "climatology_roc": round(d.climatology_roc.mean(), 3),
                "model_brier": round(d.model_brier.mean(), 3),
                "persistence_brier": round(d.persistence_brier.mean(), 3),
                "climatology_brier": round(d.climatology_brier.mean(), 3),
                "pct_hours_model_beats_climatology_roc": round(100 * (d.model_roc > d.climatology_roc).mean(), 1),
                "pct_hours_model_beats_persistence_roc": round(100 * (d.model_roc > d.persistence_roc).mean(), 1),
                "model_pod": round(d.pod.mean(), 2), "model_far": round(d.far.mean(), 2),
                "model_csi": round(d.csi.mean(), 2),
                "provisional_label_share": round(d.provisional_share.mean(), 3),
            })
        return pd.DataFrame(rows)

    save(skill_table(j), f"phase3_live_skill_vs_baselines_{LABELS}.csv")
    # scores produced by the corrected-label models (live since 2026-10-28): prediction files
    # carry model_build from that swap on and verify_imerg copies it into the log
    if "model_build" in j.columns:
        new = j[j.model_build.fillna("").astype(str).str.len() > 0]
        if len(new):
            save(skill_table(new), f"phase3_live_skill_vs_baselines_{LABELS}_corrected_models.csv")
    # model only, every verified issue hour (starts 11 Aug, before the baselines existed)
    g = v.groupby("target")
    save(pd.DataFrame({"target": g.size().index, "issue_hours": g.size().values,
                       "first_issue": g.issue_local.min().values, "last_issue": g.issue_local.max().values,
                       "model_roc": g.roc_auc.mean().round(3).values,
                       "model_brier": g.brier.mean().round(3).values}),
         f"phase3_live_skill_all_verified_{LABELS}.csv")
    # month by month for the headline target
    d = j[j.target == "h1_1.0mm"].copy()
    d["month"] = pd.to_datetime(d.issue_local).dt.to_period("M").astype(str)
    gm = d.groupby("month")
    save(pd.DataFrame({"month": gm.size().index, "issue_hours": gm.size().values,
                       "model_roc": gm.model_roc.mean().round(3).values,
                       "persistence_roc": gm.persistence_roc.mean().round(3).values,
                       "climatology_roc": gm.climatology_roc.mean().round(3).values}),
         f"phase3_live_skill_h1_1mm_by_month_{LABELS}.csv")


def lagged_persistence():
    """Persistence six hours old (the baseline IMERG's latency allows) on the issue hours of the
    stored live-skill table, so the report can quote it beside Table 23 without moving that table."""
    t23 = pd.read_csv(OUT / f"phase3_live_skill_vs_baselines_{LABELS}.csv").set_index("target")
    v = pd.read_csv(DEPLOY / f"v10_verification_log_{LABELS}.csv")
    b = pd.read_csv(DEPLOY / f"v10_baseline_log_{LABELS}.csv")
    wide = b.pivot_table(index=["stamp", "target"], columns="baseline", values=["roc_auc", "brier"])
    wide.columns = [f"{base}_{stat}" for stat, base in wide.columns]
    j = v.merge(wide.reset_index(), on=["stamp", "target"])
    j = j.dropna(subset=["roc_auc", "persistence_roc_auc", "climatology_roc_auc", "persistence_lag6h_roc_auc"])
    rows = []
    for t in TARGETS:
        d = j[(j.target == t) & j.issue_local.between(t23.loc[t].first_issue, t23.loc[t].last_issue)]
        rows.append({
            "target": t, "issue_hours": len(d), "table23_issue_hours": int(t23.loc[t].issue_hours),
            "first_issue": d.issue_local.min(), "last_issue": d.issue_local.max(),
            "model_roc": round(d.roc_auc.mean(), 3),
            "persistence_lag6h_roc": round(d.persistence_lag6h_roc_auc.mean(), 3),
            "climatology_roc": round(d.climatology_roc_auc.mean(), 3),
            "model_brier": round(d.brier.mean(), 3),
            "persistence_lag6h_brier": round(d.persistence_lag6h_brier.mean(), 3),
            "climatology_brier": round(d.climatology_brier.mean(), 3),
            "pct_hours_model_beats_persistence_lag6h_roc": round(100 * (d.roc_auc > d.persistence_lag6h_roc_auc).mean(), 1),
        })
    save(pd.DataFrame(rows), f"phase3_live_lagged_persistence_{LABELS}.csv")


def coverage():
    on_disk = {m.group(1) for f in os.listdir(DEPLOY) for m in [re.search(r"v10_predictions_(\d{8}_\d{4})", f)] if m}
    tracked = git_stamps("HEAD", "ML_Model_V2/trained_models/om_thailand_rain_v10_deploy")
    cloud = git_stamps("origin/predictions")
    health = pd.read_csv(DEPLOY / "v10_health_log.csv").issue_local   # appended by laptop runs only
    verified = pd.read_csv(DEPLOY / "v10_verification_log.csv").issue_local
    conv = lambda s: set(pd.to_datetime(pd.Series(sorted(s)), format="%Y%m%d_%H%M"))
    laptop = conv(on_disk) | conv(tracked) | set(pd.to_datetime(health))
    everything = laptop | conv(cloud) | set(pd.to_datetime(verified))

    def cov(stamps, era):
        a, b = pd.Timestamp(era[0]), pd.Timestamp(era[1])
        n = sum(a <= x < b for x in stamps)
        hours = int((b - a) / pd.Timedelta("1h"))
        return n, hours, round(100 * n / hours, 1)

    rows = []
    for name, era in [("laptop only (29 Aug to 28 Sep)", LAPTOP_ERA), ("laptop + cloud (28 Sep to 27 Oct)", CLOUD_ERA),
                      ("last 30 days to 27 Oct 16:00 (NFR6)", WINDOW_30D)]:
        for src, st in [("all sources", everything), ("cloud branch only", conv(cloud))]:
            n, h, pct = cov(st, era)
            rows.append({"era": name, "source": src, "start": era[0], "end": era[1],
                         "hours_with_prediction": n, "hours_in_era": h, "coverage_pct": pct})
    save(pd.DataFrame(rows), "phase3_hourly_coverage.csv")
    s = pd.Series(sorted(x for x in everything if pd.Timestamp("2026-08-29") <= x < pd.Timestamp("2026-10-27")))
    daily = s.dt.date.value_counts().sort_index()
    save(pd.DataFrame({"date": daily.index, "issue_hours_with_prediction": daily.values}),
         "phase3_hourly_coverage_by_day.csv")


def github_actions():
    raw = subprocess.run(["gh", "run", "list", "--workflow", "hourly-predict.yml", "--limit", "1000",
                          "--json", "conclusion,event,createdAt,startedAt,updatedAt"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    df = pd.DataFrame(json.loads(raw))
    for c in ("createdAt", "startedAt", "updatedAt"):
        df[c] = pd.to_datetime(df[c])
    df["minutes"] = (df.updatedAt - df.startedAt).dt.total_seconds() / 60
    g = df.groupby(["event", "conclusion"])
    summary = pd.DataFrame({"runs": g.size(), "median_minutes": g.minutes.median().round(1)}).reset_index()
    summary["first_run_utc"] = df.createdAt.min()
    summary["last_run_utc"] = df.createdAt.max()
    save(summary, "phase3_github_actions_runs.csv")
    save(df[df.conclusion != "success"][["createdAt", "event", "conclusion"]], "phase3_github_actions_failures.csv")


def ir_health():
    hl = pd.read_csv(DEPLOY / f"v10_health_labeled_live_{LABELS}.csv")
    hl["month"] = pd.to_datetime(hl.issue_local).dt.to_period("M").astype(str)
    g = hl.groupby("month")
    save(pd.DataFrame({"month": g.size().index, "issue_hours": g.size().values,
                       "wet_cells": g.wet_cells.sum().values,
                       "cold_top_share_of_rain_weighted": g.apply(
                           lambda x: np.average(x.cold235_given_rain, weights=x.wet_cells)).round(3).values}),
         f"phase3_ir_health_live_labeled_by_month_{LABELS}.csv")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    sections = sys.argv[1:] or ["live_skill", "coverage", "github_actions", "ir_health"]
    for name in sections:
        globals()[name]()
