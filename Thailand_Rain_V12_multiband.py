# %% [markdown]
# # Thailand_Rain_V12 — multi-band Himawari vs the warm-rain ceiling
#
# V9 closed the 2026-decay case with a physical mechanism: 10.4 um IR (B13) cannot see
# warm/shallow rain, and p(cold-top | rain) fell 0.85 -> 0.70. This experiment asks whether
# other AHI 2-km bands recover part of what B13 misses:
#
# | band | wavelength | why it might see warm rain |
# |---|---|---|
# | B07 | 3.9 um  | low-cloud/drizzle microphysics (night); solar-contaminated by day |
# | B11 | 8.6 um  | B11-B13 BTD discriminates cloud phase — water tops vs ice tops |
# | B15 | 12.4 um | B13-B15 split window tracks optical depth / low warm cloud |
#
# Pilot-first, exactly like V7: fetch the three bands ONLY for the warm-rain regime months
# (May-Jul 2026), then a no-training screen on the decisive slice — labeled rows with
# **no cold top at all** (hw_cold235_env_lag1 == 0), the rain B13 cannot rank by
# construction. Pre-registered go rule: any BTD feature (or the rank combo) reaches
# ROC >= 0.60 on that warm slice for h1_1.0mm. If go, fetch the full panel span and run
# the V7 CV protocol with an extended feature block (phases `build` / `cv`, written after
# the pilot decides).
#
# Usage:
#     python Thailand_Rain_V12_multiband.py fetch B11 2026-05-01 2026-08-01
#     python Thailand_Rain_V12_multiband.py screen

# %%
import bz2
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from BKK_Rain_V3 import TARGET_NAMES, log
from Thailand_Rain_V3 import CACHE
from Thailand_Rain_V7_himawari_ir import (
    GRID_CSV, HW_DIR, HW_TMP, UTC_OFFSET_HOURS,
    s3_client, slot_candidates,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "ML_Model_V2" / "trained_models" / \
    "om_thailand_rain_v12_multiband"
CROP_LL_BBOX = (97.0, 5.3, 106.0, 20.9)
MB_STAT_NAMES = ["tb_mean", "tb_min", "tb_mean_env", "tb_min_env"]
BANDS = ["B07", "B11", "B15"]
PILOT_LOCAL = ("2026-05-01", "2026-08-01")
GO_RULE = "warm-slice ROC >= 0.60 for h1_1.0mm on any BTD feature or the rank combo"


def band_dir(band):
    return HW_DIR / f"mb_{band}"


# %% [markdown]
# ## Fetch — V7's day-sweep, parameterized by band
#
# Same buckets, segments, masks and crop as the B13 fetch (every AHI IR band is 2 km /
# R20, so the cached pixel masks apply unchanged). Stats per slot are leaner than V7's:
# mean/min over the cell and the 0.5 deg environment — BTDs are built from these later.

# %%
def segment_key_band(band, satcode, t_utc, segment):
    ts = pd.Timestamp(t_utc)
    return (f"AHI-L1b-FLDK/{ts.year}/{ts.month:02d}/{ts.day:02d}/{ts.hour:02d}00/"
            f"HS_{satcode}_{ts.year}{ts.month:02d}{ts.day:02d}_{ts.hour:02d}00_"
            f"{band}_FLDK_R20_S{segment:02d}10.DAT")


def download_segments_band(client, band, t_utc, segments):
    HW_TMP.mkdir(parents=True, exist_ok=True)
    for bucket, satcode in slot_candidates(t_utc):
        paths = []
        try:
            for seg in segments:
                key = segment_key_band(band, satcode, t_utc, seg)
                raw = HW_TMP / (Path(key).name + ".bz2")
                client.download_file(bucket, key + ".bz2", str(raw))
                out = HW_TMP / Path(key).name
                out.write_bytes(bz2.decompress(raw.read_bytes()))
                raw.unlink()
                paths.append(out)
            return paths
        except Exception:
            for p in paths:
                p.unlink(missing_ok=True)
            continue
    return None


def load_tb_band(band, paths):
    from satpy import Scene
    scn = Scene(reader="ahi_hsd", filenames=[str(p) for p in paths])
    scn.load([band])
    scn = scn.crop(ll_bbox=CROP_LL_BBOX)
    tb = scn[band].values.astype("float32")
    lons, lats = scn[band].attrs["area"].get_lonlats()
    return tb, lons.astype("float32"), lats.astype("float32")


def mb_slot_stats(tb, masks):
    flat = tb.ravel()
    out = np.full((len(masks["cell"]), len(MB_STAT_NAMES)), np.nan, dtype="float32")
    for c, (cell_ix, env_ix) in enumerate(zip(masks["cell"], masks["env"])):
        cell = flat[cell_ix]
        env = flat[env_ix]
        cell = cell[np.isfinite(cell)]
        env = env[np.isfinite(env)]
        if len(cell):
            out[c, 0] = cell.mean()
            out[c, 1] = cell.min()
        if len(env):
            out[c, 2] = env.mean()
            out[c, 3] = env.min()
    return out


_worker = {}


def _mb_worker_init(band, segments, mask_path):
    mask_file = np.load(mask_path)
    n_cells = sum(1 for k in mask_file.files if k.startswith("cell_"))
    _worker["masks"] = {"shape": tuple(mask_file["shape"]),
                        "cell": [mask_file[f"cell_{i}"] for i in range(n_cells)],
                        "env": [mask_file[f"env_{i}"] for i in range(n_cells)]}
    _worker["band"] = band
    _worker["segments"] = segments
    _worker["client"] = s3_client()


def _mb_fetch_one_slot(t_utc):
    paths = download_segments_band(_worker["client"], _worker["band"], t_utc,
                                   _worker["segments"])
    if paths is None:
        return None
    try:
        tb, lons, lats = load_tb_band(_worker["band"], paths)
        if lons.shape != _worker["masks"]["shape"]:
            return None   # unexpected geometry; count as failed rather than misalign
        return mb_slot_stats(tb, _worker["masks"])
    except Exception:
        return None
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


def phase_fetch(band, local_start, local_end):
    from concurrent.futures import ProcessPoolExecutor
    assert band in BANDS, f"band must be one of {BANDS}"
    cfg = json.loads((HW_DIR / "fetch_config.json").read_text(encoding="utf-8"))
    segments = cfg["segments"]
    out_dir = band_dir(band)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = int(os.getenv("HW_FETCH_WORKERS", "4"))

    # dense hourly UTC sweep; lag features need HW_LOOKBACK hours before the window
    start_utc = pd.Timestamp(local_start) - pd.Timedelta(hours=UTC_OFFSET_HOURS + 4)
    end_utc = pd.Timestamp(local_end) - pd.Timedelta(hours=UTC_OFFSET_HOURS)
    hours = pd.date_range(start_utc, end_utc, freq="h", inclusive="left")
    days = pd.DatetimeIndex(hours.normalize()).unique().sort_values()
    log(f"{band}: {len(hours):,} UTC hours across {len(days):,} days, "
        f"segments {segments}, workers {workers}")

    done = failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_mb_worker_init,
            initargs=(band, segments, str(HW_DIR / "pixel_masks.npz"))) as pool:
        for day in days:
            day_file = out_dir / f"day_{pd.Timestamp(day):%Y%m%d}.npz"
            if day_file.exists():
                continue
            day_hours = hours[(hours >= day) & (hours < day + pd.Timedelta(days=1))]
            stats = np.full((len(day_hours), 833, len(MB_STAT_NAMES)), np.nan,
                            dtype="float32")
            for j, result in enumerate(pool.map(_mb_fetch_one_slot, day_hours)):
                if result is None:
                    failed += 1
                else:
                    stats[j] = result
            np.savez_compressed(day_file,
                                hours=day_hours.to_numpy(dtype="datetime64[s]"),
                                stats=stats)
            done += 1
            rate = done / max((time.time() - t0) / 3600, 1e-9)
            print(f"  {band} {pd.Timestamp(day):%Y-%m-%d}  {done}/{len(days)} days  "
                  f"failed slots {failed:,}  {rate:.1f} days/h  "
                  f"eta {(len(days) - done) / max(rate, 1e-9):.1f} h", flush=True)
    log(f"{band} fetch complete: {done:,} new days, {failed:,} failed slots")


# %% [markdown]
# ## Screen — the pre-registered no-training go/no-go
#
# Rows: panel rows in the pilot window with valid B13 (hw_block) and valid new-band
# stats at lag 1. The decisive subset is `warm`: rows with hw_cold235_env_lag1 == 0 —
# zero cold-top fraction anywhere in the 0.5 deg environment, so B13's cold-top features
# carry no signal. If a BTD ranks rain there, it sees what B13 cannot.

# %%
def load_band_lookup(band):
    """{utc datetime64[s] -> (833, 4) stats} from the band's day files."""
    lookup = {}
    for f in sorted(band_dir(band).glob("day_*.npz")):
        z = np.load(f)
        for h, s in zip(z["hours"], z["stats"]):
            lookup[np.datetime64(h, "s")] = s
    return lookup


def rows_lag1_stats(lookup, t_local, unit_ix):
    """Per-row lag-1 band stats aligned to panel rows (NaN when the slot is missing)."""
    utc = (t_local - np.timedelta64(UTC_OFFSET_HOURS + 1, "h")).astype("datetime64[s]")
    out = np.full((len(t_local), len(MB_STAT_NAMES)), np.nan, dtype="float32")
    uniq, inverse = np.unique(utc, return_inverse=True)
    for k, u in enumerate(uniq):
        s = lookup.get(u)
        if s is None:
            continue
        rows = np.flatnonzero(inverse == k)
        out[rows] = s[unit_ix[rows]]
    return out


def auc(y, x):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(x)
    if m.sum() < 200 or len(np.unique(y[m])) < 2:
        return np.nan, int(m.sum())
    return float(roc_auc_score(y[m], x[m])), int(m.sum())


def phase_screen():
    from Thailand_Rain_V7_himawari_ir import himawari_feature_names
    hw_names = himawari_feature_names()
    tf = np.load(CACHE / "forecast_time.npy")
    y = np.load(CACHE / "y.npy")
    unit = np.load(CACHE / "unit_id.npy")
    hw = np.load(CACHE / "hw_block_offset0.npy")
    grid = pd.read_csv(GRID_CSV).sort_values("grid_number")
    unit_ix = np.searchsorted(grid["grid_number"].to_numpy(), unit)

    lo, hi = (np.datetime64(PILOT_LOCAL[0]), np.datetime64(PILOT_LOCAL[1]))
    in_pilot = (tf >= lo) & (tf < hi)
    b13_mean = hw[:, hw_names.index("hw_tb_mean_lag1")]
    b13_envmin = hw[:, hw_names.index("hw_tb_min_env_lag1")]
    cold235 = hw[:, hw_names.index("hw_cold235_env_lag1")]
    base = in_pilot & np.isfinite(b13_mean) & np.isfinite(cold235)
    log(f"pilot rows {in_pilot.sum():,}, with valid B13 {base.sum():,}")

    stats = {}
    for band in BANDS:
        lookup = load_band_lookup(band)
        log(f"{band}: {len(lookup):,} slots cached")
        stats[band] = rows_lag1_stats(lookup, tf, unit_ix)

    MEAN, ENVMIN = MB_STAT_NAMES.index("tb_mean"), MB_STAT_NAMES.index("tb_min_env")
    feats = {
        "btd_11_13_mean": stats["B11"][:, MEAN] - b13_mean,
        "btd_13_15_mean": b13_mean - stats["B15"][:, MEAN],
        "btd_07_13_mean": stats["B07"][:, MEAN] - b13_mean,
        "btd_11_13_envmin": stats["B11"][:, ENVMIN] - b13_envmin,
        "b11_tb_mean": stats["B11"][:, MEAN],
        "b15_tb_mean": stats["B15"][:, MEAN],
        "b07_tb_mean": stats["B07"][:, MEAN],
        "b13_tb_mean_control": b13_mean,       # sanity: known-good signal on `all`
        "cold235_control": cold235,            # sanity: ~0.5 on the warm slice
    }

    ti = {n: i for i, n in enumerate(TARGET_NAMES)}
    results = []
    for tgt in ["h1_1.0mm", "h1_0.1mm", "h3_1.0mm"]:
        yy = y[:, ti[tgt]]
        slices = {
            "all": base,
            "warm": base & (cold235 == 0),
            "cold": base & (cold235 > 0),
        }
        for sname, m in slices.items():
            for fname, x in feats.items():
                a, n = auc(yy[m], x[m])
                results.append({"target": tgt, "slice": sname, "feature": fname,
                                "n": n, "base_rate": round(float(yy[m].mean()), 4),
                                "roc": round(a, 4) if np.isfinite(a) else np.nan,
                                "roc_oriented": round(max(a, 1 - a), 4)
                                                if np.isfinite(a) else np.nan})
        # rank combo of the three mean-BTDs, each oriented on the same slice (screen
        # optimism is acceptable: this decides a download, not a headline)
        for sname, m in slices.items():
            combo = np.zeros(int(m.sum()), dtype="float64")
            ok = np.ones(int(m.sum()), dtype=bool)
            yy_m = yy[m]
            for fname in ["btd_11_13_mean", "btd_13_15_mean", "btd_07_13_mean"]:
                x = feats[fname][m]
                ok &= np.isfinite(x)
                a, _ = auc(yy_m, x)
                sign = 1.0 if (np.isfinite(a) and a >= 0.5) else -1.0
                r = pd.Series(np.where(np.isfinite(x), x, np.nan)).rank(pct=True).to_numpy()
                combo += sign * np.where(np.isfinite(r), r, 0.5)
            a, n = auc(yy_m[ok], combo[ok])
            results.append({"target": tgt, "slice": sname, "feature": "btd_rank_combo",
                            "n": n, "base_rate": round(float(yy_m[ok].mean()), 4)
                                                 if ok.sum() else np.nan,
                            "roc": round(a, 4) if np.isfinite(a) else np.nan,
                            "roc_oriented": round(max(a, 1 - a), 4)
                                            if np.isfinite(a) else np.nan})

    out = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v12_screen.csv", index=False)
    log(f"saved {OUTPUT_DIR / 'v12_screen.csv'}")
    show = out[(out["target"] == "h1_1.0mm")]
    for sname in ["warm", "cold", "all"]:
        s = show[show["slice"] == sname].sort_values("roc_oriented", ascending=False)
        log(f"=== h1_1.0mm {sname} (n={s['n'].max():,}, "
            f"base {s['base_rate'].iloc[0]}) ===")
        print(s[["feature", "roc", "roc_oriented", "n"]].to_string(index=False))
    log(f"GO RULE: {GO_RULE}")


# %% [markdown]
# ## Phases `build` / `cv` / `report` — the full experiment (B07 only, per the pilot)
#
# `build` mirrors V7's build_features: hour-aligned stat series for B07 (mb_ day files)
# and B13 (V7 day files), derived per-hour features, then per-row extraction. The block
# is deliberately small — the screen validated the lag-1 mean BTD; trends and the env
# variant ride along. `cv` runs the V7 protocol over all 13 blocks for config `hw_b07`
# (weather + B13 block + B07 block); `report` scores it against `hw_allseason` OOF on
# identical rows, sliced by regime — the warm slice (no cold top anywhere) is the
# pre-registered readout, since that is the rain B13 cannot see.

# %%
def b07_feature_names():
    return ["b07_tb_mean_lag1", "b07_tb_min_env_lag1",
            "btd_07_13_mean_lag1", "btd_07_13_envmin_lag1",
            "btd_07_13_mean_lag2", "btd_07_13_mean_drop3", "b07_valid"]


def load_mb_series(band):
    """(hours_utc, stats) hour-aligned arrays from a band's day files."""
    hours, stats = [], []
    for f in sorted(band_dir(band).glob("day_*.npz")):
        z = np.load(f)
        hours.append(z["hours"])
        stats.append(z["stats"])
    return np.concatenate(hours), np.concatenate(stats).astype("float32")


def phase_build():
    from Thailand_Rain_V7_himawari_ir import load_stat_series, shift_down, STAT_NAMES
    # B13: V7's series (local-hour aligned). B07: UTC day files -> convert to local.
    hours13_local, s13 = load_stat_series()
    hours07_utc, s07 = load_mb_series("B07")
    hours07_local = hours07_utc + np.timedelta64(UTC_OFFSET_HOURS, "h")
    log(f"B13 series {s13.shape}; B07 series {s07.shape}, "
        f"finite {np.isfinite(s07[..., 0]).mean():.1%}")

    # align B07 onto the B13 hour axis
    pos = {np.datetime64(h): i for i, h in enumerate(hours07_local)}
    idx = np.array([pos.get(np.datetime64(h), -1) for h in hours13_local])
    b07 = np.full((len(hours13_local), s07.shape[1], s07.shape[2]), np.nan, "float32")
    have = idx >= 0
    b07[have] = s07[idx[have]]
    log(f"aligned: {have.mean():.1%} of {len(hours13_local):,} panel hours have B07")
    del s07

    MEAN, ENVMIN = MB_STAT_NAMES.index("tb_mean"), MB_STAT_NAMES.index("tb_min_env")
    b13_mean, b13_envmin = s13[..., STAT_NAMES.index("tb_mean")], \
        s13[..., STAT_NAMES.index("tb_min_env")]
    btd_mean = b07[..., MEAN] - b13_mean
    btd_envmin = b07[..., ENVMIN] - b13_envmin
    with np.errstate(all="ignore"):
        m1, m2, m3 = (shift_down(btd_mean, k) for k in (1, 2, 3))
        hourly = np.stack([
            shift_down(b07[..., MEAN], 1), shift_down(b07[..., ENVMIN], 1),
            m1, shift_down(btd_envmin, 1),
            m2, m3 - m1,
            np.isfinite(shift_down(b07[..., MEAN], 1)).astype("float32"),
        ], axis=-1).astype("float32")
    del b07, btd_mean, btd_envmin

    import pandas as _pd
    from Thailand_Rain_V7_himawari_ir import load_grid
    grid = load_grid()
    unit = np.load(CACHE / "unit_id.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    hour_pos = {np.datetime64(h): i for i, h in enumerate(hours13_local)}
    cell_pos = {g: i for i, g in enumerate(grid["grid_number"])}
    row_hour = np.array([hour_pos.get(np.datetime64(t), -1) for t in tf])
    row_cell = np.array([cell_pos[int(g)] for g in unit])
    outside = row_hour < 0
    row_hour[outside] = 0
    block = hourly[row_hour, row_cell]
    block[outside] = np.nan
    block[outside, -1] = 0.0
    np.save(CACHE / "b07_block.npy", block)
    share = _pd.Series(np.isnan(block).mean(axis=0), index=b07_feature_names())
    log(f"b07_block {block.shape}, NaN share:")
    print(share.round(4).to_string())


def phase_cv():
    import Thailand_Rain_V3 as v3
    from Thailand_Rain_V7_himawari_ir import himawari_feature_names
    v3.CONFIG_BLOCKS["hw_b07"] = [("hw_block_offset0.npy", himawari_feature_names),
                                  ("b07_block.npy", b07_feature_names)]
    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    fold_key = v3.block_keys(tf)
    folds = [int(v) for v in np.unique(fold_key)
             if (fold_key == v).sum() >= v3.MIN_FOLD_ROWS]
    log(f"hw_b07 CV over {len(folds)} folds")
    results, fit_log = v3.run_cv("hw_b07", y, tf, fold_key, folds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "v12_cv_results.csv", index=False)
    pd.DataFrame(fit_log).to_csv(OUTPUT_DIR / "v12_fit_log.csv", index=False)


def phase_report():
    from sklearn.metrics import average_precision_score, roc_auc_score
    from Thailand_Rain_V7_himawari_ir import himawari_feature_names
    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    hw = np.load(CACHE / "hw_block_offset0.npy", mmap_mode="r")
    cold = np.asarray(hw[:, himawari_feature_names().index("hw_cold235_env_lag1")])
    del hw
    base_oof = np.load(CACHE / "oof_cal_hw_allseason.npy")
    new_oof = np.load(CACHE / "oof_cal_hw_b07.npy")

    months = pd.DatetimeIndex(tf).month.to_numpy()
    years = pd.DatetimeIndex(tf).year.to_numpy()
    wet = np.isin(months, [5, 6, 7, 8, 9, 10])
    slices = {
        "ALL": np.ones(len(y), bool),
        "WET": wet,
        "DRY": ~wet,
        "WARM (no cold top)": np.isfinite(cold) & (cold == 0),
        "2026 MAY-JUL": (years == 2026) & np.isin(months, [5, 6, 7]),
    }
    rows = []
    for sname, sm in slices.items():
        for i, name in enumerate(TARGET_NAMES):
            m = sm & np.isfinite(base_oof[:, i]) & np.isfinite(new_oof[:, i])
            if m.sum() < 5000 or len(np.unique(y[m, i])) < 2:
                continue
            yy = y[m, i]
            rows.append({
                "slice": sname, "target": name, "rows": int(m.sum()),
                "base_rate": round(float(yy.mean()), 4),
                "hw_roc": round(float(roc_auc_score(yy, base_oof[m, i])), 4),
                "b07_roc": round(float(roc_auc_score(yy, new_oof[m, i])), 4),
                "hw_pr": round(float(average_precision_score(yy, base_oof[m, i])), 4),
                "b07_pr": round(float(average_precision_score(yy, new_oof[m, i])), 4),
            })
    out = pd.DataFrame(rows)
    out["roc_gain"] = (out["b07_roc"] - out["hw_roc"]).round(4)
    out["pr_gain"] = (out["b07_pr"] - out["hw_pr"]).round(4)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v12_report.csv", index=False)
    log("=== hw_b07 vs hw_allseason on identical rows ===")
    print(out[["slice", "target", "rows", "base_rate", "hw_roc", "b07_roc",
               "roc_gain", "pr_gain"]].to_string(index=False))
    log(f"saved {OUTPUT_DIR / 'v12_report.csv'}")


# %%
if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "screen"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "fetch":
        band = sys.argv[2]
        start = sys.argv[3] if len(sys.argv) > 3 else PILOT_LOCAL[0]
        end = sys.argv[4] if len(sys.argv) > 4 else PILOT_LOCAL[1]
        phase_fetch(band, start, end)
    elif phase == "screen":
        phase_screen()
    elif phase == "build":
        phase_build()
    elif phase == "cv":
        phase_cv()
    elif phase == "report":
        phase_report()
    else:
        raise SystemExit(f"unknown phase {phase!r}")
    log(f"phase {phase} done in {(time.time() - t0) / 60:.1f} min")
