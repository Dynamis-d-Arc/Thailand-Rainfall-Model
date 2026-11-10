"""Phase 4 POC, test step 2 in its cheapest form: does radar add skill on top of the live model?

The live model is not refitted or touched. For every archived issue hour t (Bangkok) and cell,
two small stacked models are fitted on the same rows and compared out of fold:
  control  the live model's six probabilities for t (from its prediction CSV), plus the cell's
           latitude, longitude and the hour of day
  radar    the same, plus radar features from the 12 frames stamped t-180 .. t-15 min
           (all published before the hourly predict runs; the frame stamped t is not used)
Labels are the verify loop's labels_v2 (IMERG wettest-pixel hourly total >= threshold in any
hour of [t, t+h)). Two splits, both with a 6 h embargo so label windows never straddle train
and test:
  default  leave one Bangkok day out
  forward  train on every issue hour before a cut date, test on the period after it (one month
           unless an end date is given). This is the frozen monthly test, see STACK_TEST_PLAN.md.

Writes outputs/radar_stack_summary.csv (target x slice) and outputs/radar_stack_hourly.csv
(per issue hour ROC), with a _forward_YYYYMMDD suffix in forward mode, and prints radar minus
control with a 95% CI over day means.

shortcut: a stack on the live probabilities, not a refit of the V13 feature set with radar
added, because the archive (days, one season) is far too short to fit V13 on. Upgrade to the
full refit once the archive spans a wet season.

Usage:
    python Radar_POC/radar_stack_test.py
    python Radar_POC/radar_stack_test.py forward 2026-12-01              # tests December
    python Radar_POC/radar_stack_test.py forward 2026-11-06 2026-11-11   # tests 6 to 10 Nov
    python Radar_POC/radar_stack_test.py selftest
"""

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "Dashboard"))
from verify_imerg import DB_CONFIG, load_labels  # noqa: E402
from assess_radar_fit import ARCHIVE, cell_lookup, load_nc  # noqa: E402
from score_radar_persistence import BKK, LIVE_PRED_DIR, TARGETS, day_ci  # noqa: E402

OUT = ROOT / "outputs"
CACHE = OUT / "radar_cell_frames.npz"
EMBARGO = timedelta(hours=6)
P_COLS = [f"p_{name}" for name, _, _ in TARGETS]
# place and hour go in both arms: radar echo share is a proxy for where a cell is, so without
# them the radar arm could win by learning these few days' regional climate, not from radar
BASE = P_COLS + ["latitude", "longitude", "hour"]
SOUTH_LAT = 11.0          # the peninsula, the only reliably wet region in Nov


def frame_stats(g, lut):
    """(stamps, [frames, cells, 4]): per-cell max rate, mean rate, wet share, echo share. Cached."""
    man = pd.read_csv(ARCHIVE / "manifest.csv")
    nc = man[man["product"] == "nc"].sort_values("stamp_utc")
    stamps = nc.stamp_utc.to_numpy(dtype=str)
    if CACHE.exists():
        z = np.load(CACHE)
        if np.array_equal(z["stamps"], stamps):
            return stamps, z["stats"]
    cell = lut.ravel()
    inside = cell > 0
    c, n = cell[inside], len(g) + 1
    npx = np.maximum(np.bincount(c, minlength=n), 1)
    out = np.zeros((len(nc), n, 4), dtype=np.float32)
    for k, p in enumerate(nc.path):
        v = load_nc(ARCHIVE / p).ravel()[inside]
        r = np.nan_to_num(v)                                  # no echo counts as 0 mm/h
        np.maximum.at(out[k, :, 0], c, r)
        out[k, :, 1] = np.bincount(c, r, minlength=n) / npx
        out[k, :, 2] = np.bincount(c, r > 0, minlength=n) / npx
        out[k, :, 3] = np.bincount(c, ~np.isnan(v), minlength=n) / npx
    out = out[:, 1:]
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, stamps=stamps, stats=out)
    return stamps, out


def env(x, g, k, fn):
    """fn (np.nanmax or np.nanmean) of per-cell x over each cell's (2k+1) x (2k+1) neighbourhood."""
    r = g.grid_row.to_numpy() - g.grid_row.min() + k
    c = g.grid_column.to_numpy() - g.grid_column.min() + k
    grid = np.full((r.max() + k + 1, c.max() + k + 1), np.nan, dtype=np.float32)
    grid[r, c] = x
    return fn(np.stack([grid[r + i, c + j] for i in range(-k, k + 1) for j in range(-k, k + 1)]), axis=0)


def radar_features(F, g):
    """F = [12 frames oldest first, cells, 4] -> {name: per-cell array}."""
    h3, h2, h1 = F[:4], F[4:8], F[8:]
    last = h1[-1]
    f = {"r_last_max": last[:, 0], "r_last_mean": last[:, 1], "r_last_wet": last[:, 2],
         "r_last_echo": last[:, 3], "r_trend": last[:, 1] - h1[0, :, 1]}
    for tag, h in (("h1", h1), ("h2", h2), ("h3", h3)):
        f[f"r_{tag}_max"], f[f"r_{tag}_mean"] = h[..., 0].max(0), h[..., 1].mean(0)
    f["r_h1_wet"], f["r_h1_echo"] = h1[..., 2].mean(0), h1[..., 3].mean(0)
    for k in (1, 2):                                          # rain moves about a cell an hour
        w = 2 * k + 1
        f[f"r_env{w}_max"] = env(f["r_h1_max"], g, k, np.nanmax)
        f[f"r_env{w}_mean"] = env(f["r_h1_mean"], g, k, np.nanmean)
        f[f"r_env{w}_echo"] = env(f["r_h1_echo"], g, k, np.nanmean)
    return f


def build_panel(g, stamps, S, pmax):
    """One row per (issue hour, cell): live probabilities, radar features, labels (NaN if unknown)."""
    at = {s: i for i, s in enumerate(stamps)}
    idx = pd.Index(g.grid_number, name="grid_number")
    first = pd.Timestamp(stamps[0][:-1]) + BKK
    last = pd.Timestamp(stamps[-1][:-1]) + BKK
    parts = []
    for t in pd.date_range(first.ceil("h"), last.ceil("h"), freq="h"):
        ks = [at.get(f"{t - BKK - timedelta(minutes=m):%Y-%m-%dT%H:%MZ}") for m in range(180, 0, -15)]
        csv = LIVE_PRED_DIR / f"v10_predictions_{t:%Y%m%d_%H%M}.csv"
        if None in ks or not csv.exists():
            continue
        d = pd.DataFrame(radar_features(S[ks], g), index=idx)
        d[P_COLS] = pd.read_csv(csv).set_index("grid_number").reindex(idx)[P_COLS]
        for name, h, thr in TARGETS:
            hrs = [t + timedelta(hours=k) for k in range(h)]
            if all(x in pmax.columns for x in hrs):
                lead = pmax[hrs].reindex(idx)
                d[f"y_{name}"] = (lead >= thr).any(axis=1).astype(float).where(lead.notna().all(axis=1))
            else:
                d[f"y_{name}"] = np.nan
        d["issue_local"] = t
        parts.append(d.reset_index())
    return pd.concat(parts, ignore_index=True)


def splits(issue, cut=None, end=None):
    """(train, test) masks: one forward split at `cut` (test up to `end`), else leave one day out."""
    if cut is not None:
        yield issue < cut - EMBARGO, (issue >= cut) & (issue < end)
        return
    day = issue.dt.floor("D")
    for D in day.unique():
        yield ~((issue >= D - EMBARGO) & (issue < D + timedelta(days=1) + EMBARGO)), day == D


def oof(d, cols, y, cut=None, end=None):
    """Held-out probabilities over splits(); NaN for rows never tested."""
    out = pd.Series(np.nan, index=d.index)
    for train, test in splits(d.issue_local, cut, end):
        if y[train].nunique() < 2:
            continue
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=15,
                                           l2_regularization=1.0, random_state=0)
        m.fit(d.loc[train, cols], y[train])
        out[test] = m.predict_proba(d.loc[test, cols])[:, 1]
    return out


def hourly_roc(x, arms):
    rows = {}
    for t, grp in x.groupby("issue_local"):
        if 0 < grp.y.mean() < 1:
            rows[t] = {a: roc_auc_score(grp.y, grp[a]) for a in arms}
    return pd.DataFrame.from_dict(rows, orient="index")


def main(cut=None, end=None):
    g, lut = cell_lookup()
    stamps, S = frame_stats(g, lut)
    lo = pd.Timestamp(stamps[0][:-1]) + BKK
    hi = pd.Timestamp(stamps[-1][:-1]) + BKK + timedelta(hours=7)
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        pmax, _ = load_labels(conn, lo.to_pydatetime(), hi.to_pydatetime())
    finally:
        conn.close()
    d = build_panel(g, stamps, S, pmax).dropna(subset=P_COLS)
    d = d.merge(g[["grid_number", "latitude", "longitude"]], on="grid_number")
    d["hour"] = d.issue_local.dt.hour
    r_cols = [c for c in d.columns if c.startswith("r_")]
    echo = pd.Series(S[..., 3].mean(axis=0), index=g.grid_number)          # long-run echo share
    slices = {"all": pd.Series(True, index=d.index),
              "south": d.latitude < SOUTH_LAT,
              # slice definition only, never a feature: cells in the better-lit half of the grid
              "radar_lit": d.grid_number.map(echo >= echo.median())}
    print(f"{d.issue_local.nunique()} issue hours {d.issue_local.min()} to {d.issue_local.max()}, "
          f"{len(d)} rows, {len(r_cols)} radar features")

    arms = ("live", "control", "radar")
    summary, hourly = [], []
    for name, _, _ in TARGETS:
        x = d[d[f"y_{name}"].notna()].copy()
        x["y"] = x[f"y_{name}"].astype(int)
        x["live"] = x[f"p_{name}"]
        x["control"] = oof(x, BASE, x.y, cut, end)
        x["radar"] = oof(x, BASE + r_cols, x.y, cut, end)
        x = x.dropna(subset=["control", "radar"])
        for sl, mask in slices.items():
            s = x[mask.reindex(x.index)]
            hr = hourly_roc(s, arms)
            m, a, b, n = day_ci(hr.radar - hr.control)
            row = {"target": name, "slice": sl, "rows": len(s), "base_rate": round(s.y.mean(), 4),
                   "hours": len(hr), "days": n}
            for arm in arms:
                row[f"roc_{arm}"] = round(roc_auc_score(s.y, s[arm]), 4)
                row[f"prauc_{arm}"] = round(average_precision_score(s.y, s[arm]), 4)
                row[f"brier_{arm}"] = round(float(np.mean((s[arm] - s.y) ** 2)), 4)
                row[f"hourly_roc_{arm}"] = round(hr[arm].mean(), 4)
            row.update(hourly_gain=round(m, 4), ci_lo=round(a, 4), ci_hi=round(b, 4))
            summary.append(row)
            hourly.append(hr.round(4).assign(target=name, slice=sl).rename_axis("issue_local").reset_index())
    summary = pd.DataFrame(summary)
    tag = f"_forward_{cut:%Y%m%d}" if cut is not None else ""
    summary.to_csv(OUT / f"radar_stack_summary{tag}.csv", index=False)
    pd.concat(hourly).to_csv(OUT / f"radar_stack_hourly{tag}.csv", index=False)
    show = ["target", "slice", "hours", "base_rate", "roc_live", "roc_control", "roc_radar",
            "prauc_control", "prauc_radar", "brier_control", "brier_radar", "hourly_gain", "ci_lo", "ci_hi"]
    print(summary[show].to_string(index=False))


def selftest():
    g = pd.DataFrame({"grid_number": [1, 2, 3], "grid_row": [5, 5, 6], "grid_column": [7, 8, 9]})
    x = np.array([1.0, 4.0, 10.0], dtype=np.float32)
    # cell 3 is diagonal to cell 2 only; cell 1 and 3 are two columns apart
    assert np.allclose(env(x, g, 1, np.nanmax), [4, 10, 10])
    assert np.allclose(env(x, g, 1, np.nanmean), [2.5, 5.0, 7.0])
    assert np.allclose(env(x, g, 2, np.nanmax), [10, 10, 10])

    F = np.zeros((12, 3, 4), dtype=np.float32)
    F[8:, :, 1] = [[1], [2], [3], [4]]                        # last hour's mean rate rises 1 -> 4
    F[0, :, 0] = 9                                            # oldest hour's max
    f = radar_features(F, g)
    assert np.allclose(f["r_last_mean"], 4) and np.allclose(f["r_trend"], 3)
    assert np.allclose(f["r_h1_mean"], 2.5) and np.allclose(f["r_h3_max"], 9) and np.allclose(f["r_h2_max"], 0)

    issue = pd.Series(pd.date_range("2026-11-01", periods=72, freq="h"))
    (train, test), = splits(issue, pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-03"))
    assert issue[train].max() == pd.Timestamp("2026-11-01 17:00") and test.sum() == 24
    lodo = list(splits(issue))
    assert len(lodo) == 3 and all(not (tr & te).any() for tr, te in lodo)
    assert issue[lodo[1][0]].tolist() == issue[(issue.dt.hour < 18) & (issue.dt.day == 1) |
                                              (issue.dt.hour >= 6) & (issue.dt.day == 3)].tolist()
    print("selftest ok")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["selftest"]:
        selftest()
    elif args[:1] == ["forward"]:
        cut = pd.Timestamp(args[1])
        main(cut, pd.Timestamp(args[2]) if len(args) > 2 else cut + pd.DateOffset(months=1))
    else:
        main()
