"""Phase 4 POC, test step 1: does radar persistence beat the live model? No model is trained.

For every Bangkok issue hour t with archived radar for the hours before it, scores these on
the same cells and the same labels_v2 labels the live verify loop uses (IMERG wettest-pixel
hourly total >= threshold in any hour of [t, t+h)):
  model          the live prediction CSV for t (V13 refit, read from the main checkout)
  imerg_persist  IMERG max over [t-h, t): the verify loop's persistence baseline. Not
                 deployable, because IMERG arrives hours late; it is the ceiling radar chases.
  radar_max      radar persistence: each pixel's accumulation per hour T, summed from the
  radar_mean     frames stamped T, T+15, T+30, T+45 (rate x 0.25 h); per cell the max or mean
                 over its pixels, then the max over the hours of [t-h, t). Every frame used is
                 stamped <= t-15 min, so it existed when predict ran.
Plus an agreement check with no forecast in it, target "same_hour_*": radar for hour T against
the IMERG label of the same hour T.

Writes Radar_POC/outputs/radar_persistence_scores.csv (one row per issue x target x forecaster)
and prints, per target, mean per-hour ROC on the hours all forecasters share and radar minus
model with a 95% CI over day means (t distribution).

Usage:
    python Radar_POC/score_radar_persistence.py
    python Radar_POC/score_radar_persistence.py selftest
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "Dashboard"))
from verify_imerg import DB_CONFIG, MIN_COVERAGE, N_CELLS, load_labels, metrics  # noqa: E402
from assess_radar_fit import ARCHIVE, cell_lookup, load_nc  # noqa: E402

OUT = ROOT / "outputs" / "radar_persistence_scores.csv"
# live predictions accumulate in the main checkout, not in this worktree
LIVE_PRED_DIR = Path(os.getenv("LIVE_PRED_DIR", r"C:\Users\Brandon\Documents\Phase 1\ML_Model_V2"
                                               r"\trained_models\om_thailand_rain_v10_deploy"))
BKK = timedelta(hours=7)
FRAME_MIN = (0, 15, 30, 45)
TARGETS = [(f"h{h}_{thr}mm", h, thr) for thr in (0.1, 1.0) for h in (1, 3, 6)]
RADAR = ("radar_max", "radar_mean")


def complete_hours(nc_paths):
    """{Bangkok local hour start: [4 archive paths]} for UTC hours with all four frames."""
    starts = {datetime.strptime(s, "%Y-%m-%dT%H:%MZ").replace(minute=0) for s in nc_paths}
    out = {}
    for T in sorted(starts):
        stamps = [(T + timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%MZ") for m in FRAME_MIN]
        if all(s in nc_paths for s in stamps):
            out[T + BKK] = [nc_paths[s] for s in stamps]
    return out


def hour_stats(frames, lut, n_cells):
    """Per-cell (max, mean) of the hour's per-pixel accumulation in mm; no echo counts as 0."""
    acc = sum(np.nan_to_num(f) for f in frames) * 0.25
    cell, val = lut.ravel(), acc.ravel()
    inside = cell > 0
    mx = np.zeros(n_cells + 1)
    np.maximum.at(mx, cell[inside], val[inside])
    mean = np.bincount(cell[inside], val[inside], minlength=n_cells + 1) / \
        np.maximum(np.bincount(cell[inside], minlength=n_cells + 1), 1)
    return mx[1:], mean[1:]


def day_ci(diff_by_hour):
    """(mean, lo, hi, n_days) of the per-day mean differences; CI needs two or more days."""
    d = diff_by_hour.groupby(diff_by_hour.index.date).mean()
    if len(d) < 2:
        return d.mean(), np.nan, np.nan, len(d)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - half, d.mean() + half, len(d)


def score(y, s, thr):
    # ROC ranks the raw score, so radar's 1 km scale can't change it; metrics() clips for Brier,
    # as the verify loop does. The live baseline log ranks the clipped score instead, which ties
    # every wet cell, so its persistence ROC reads a little lower than imerg_persist here.
    return metrics(y, s / thr, s >= thr)


def load_radar():
    """{Bangkok hour: {"radar_max": per-cell array, "radar_mean": ...}}, cells in grid_number order."""
    man = pd.read_csv(ARCHIVE / "manifest.csv")
    nc = man[man["product"] == "nc"]
    g, lut = cell_lookup()
    return {T: dict(zip(RADAR, hour_stats([load_nc(ARCHIVE / p) for p in paths], lut, len(g))))
            for T, paths in complete_hours(dict(zip(nc.stamp_utc, nc.path))).items()}


def main():
    radar = load_radar()
    if not radar:
        print("no hour with all four NetCDF frames archived yet")
        return
    print(f"{len(radar)} complete radar hours (Bangkok) {min(radar)} to {max(radar)}")
    lo, hi = min(radar) - timedelta(hours=6), max(radar) + timedelta(hours=7)
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        pmax, _ = load_labels(conn, lo, hi)
    finally:
        conn.close()
    if pmax is None:
        print(f"no complete IMERG hours in the DB between {lo} and {hi} yet")
        return
    print(f"IMERG labels in the DB up to {pmax.columns.max()}")
    rows, waiting = score_all(radar, pmax)
    print(f"{waiting} (issue, target) pairs have radar but are waiting for IMERG labels")
    if not rows:
        print("nothing to score yet")
        return
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)\n")
    summarise(df)


def score_all(radar, pmax):
    """Score rows for every hour radar and labels both cover -> (rows, pairs still waiting)."""
    idx = pd.Index(np.arange(1, N_CELLS + 1), name="grid_number")

    def labels(hrs):
        return pd.DataFrame({h: pmax[h].reindex(idx) if h in pmax.columns else np.nan for h in hrs},
                            index=idx)

    rows = []
    for T in sorted(radar):                                   # agreement: radar vs IMERG, same hour
        lab = labels([T])[T]
        ok = lab.notna().to_numpy()
        if ok.sum() < MIN_COVERAGE * N_CELLS:
            continue
        for thr in (0.1, 1.0):
            y = (lab[ok] >= thr).astype(int).to_numpy()
            for fc in RADAR:
                rows.append({"stamp": f"{T:%Y%m%d_%H%M}", "issue_local": T, "target": f"same_hour_{thr}mm",
                             "forecaster": fc, **score(y, radar[T][fc][ok], thr)})

    waiting = 0
    for t in sorted({T + timedelta(hours=1) for T in radar}):
        csv = LIVE_PRED_DIR / f"v10_predictions_{t:%Y%m%d_%H%M}.csv"
        pred = pd.read_csv(csv).set_index("grid_number").reindex(idx) if csv.exists() else None
        for name, h, thr in TARGETS:
            hist_hrs = [t - timedelta(hours=k) for k in range(h, 0, -1)]
            if not all(T in radar for T in hist_hrs):
                continue
            lead, hist = labels([t + timedelta(hours=k) for k in range(h)]), labels(hist_hrs)
            ok = (lead.notna().all(axis=1) & hist.notna().all(axis=1)).to_numpy()
            if ok.sum() < MIN_COVERAGE * N_CELLS:
                waiting += 1
                continue
            y = (lead[ok] >= thr).any(axis=1).astype(int).to_numpy()
            common = {"stamp": f"{t:%Y%m%d_%H%M}", "issue_local": t, "target": name}
            rows.append({**common, "forecaster": "imerg_persist", **score(y, hist[ok].max(axis=1).to_numpy(), thr)})
            for fc in RADAR:
                s = np.max([radar[T][fc] for T in hist_hrs], axis=0)[ok]
                rows.append({**common, "forecaster": fc, **score(y, s, thr)})
            if pred is not None and pred[f"p_{name}"][ok].notna().all():
                rows.append({**common, "forecaster": "model",
                             **metrics(y, pred[f"p_{name}"][ok].to_numpy(), pred[f"flag_{name}"][ok].to_numpy())})
    return rows, waiting


def summarise(df):
    roc = df.pivot_table(index=["issue_local", "target"], columns="forecaster", values="roc_auc").reset_index()
    print("Agreement, radar vs IMERG for the same hour (mean per-hour ROC):")
    for name in ("same_hour_0.1mm", "same_hour_1.0mm"):
        x = roc[roc.target == name]
        if len(x):
            print(f"  {name}: radar_max {x.radar_max.mean():.3f}, radar_mean {x.radar_mean.mean():.3f} "
                  f"({x.radar_max.notna().sum()} hours)")
    print("\nPaired, on hours where every forecaster has a ROC (mean per-hour ROC):")
    cols = ["model", "imerg_persist", *RADAR]
    for name, _, _ in TARGETS:
        x = roc[roc.target == name].dropna(subset=[c for c in cols if c in roc.columns])
        if "model" not in x.columns or x.empty:
            continue
        x = x.set_index("issue_local")
        means = ", ".join(f"{c} {x[c].mean():.3f}" for c in cols)
        print(f"  {name} ({len(x)} hours): {means}")
        for fc in RADAR:
            m, a, b, n = day_ci(x[fc] - x["model"])
            ci = f"95% CI {a:+.3f} to {b:+.3f}" if np.isfinite(a) else "CI needs 2+ days"
            print(f"      {fc} - model: {m:+.3f} over {n} day(s), {ci}")


def selftest():
    # 07:00-07:45 UTC complete, 08:00 missing a frame: only 14:00 Bangkok counts
    paths = {f"2026-10-29T07:{m:02d}Z": f"p{m}" for m in FRAME_MIN}
    paths.update({f"2026-10-29T08:{m:02d}Z": f"q{m}" for m in (0, 15, 30)})
    hrs = complete_hours(paths)
    assert list(hrs) == [datetime(2026, 10, 29, 14, 0)], hrs
    assert hrs[datetime(2026, 10, 29, 14, 0)] == ["p0", "p15", "p30", "p45"]

    lut = np.array([[1, 1], [2, 0]], dtype=np.int16)          # 2 cells, one pixel outside both
    f1 = np.array([[4.0, np.nan], [8.0, 100.0]])
    f2 = np.array([[4.0, 0.0], [np.nan, 100.0]])
    mx, mean = hour_stats([f1, f2, f1, f2], lut, 2)            # mm = sum(rate) x 0.25
    assert np.allclose(mx, [4.0, 4.0]) and np.allclose(mean, [2.0, 4.0]), (mx, mean)

    s = pd.Series([0.02, 0.04, -0.01], index=pd.to_datetime(["2026-11-01 01:00", "2026-11-01 02:00",
                                                              "2026-11-02 01:00"]))
    m, a, b, n = day_ci(s)
    assert n == 2 and np.isclose(m, 0.01) and a < m < b, (m, a, b, n)
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else main()
