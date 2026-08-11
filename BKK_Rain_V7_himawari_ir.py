# %% [markdown]
# # BKK_Rain_V7 — Himawari IR: the area-covering minutes-latency feed
#
# The V4 latency sweep set the requirement (observed-rain value halves per hour of latency,
# dead by 3 h) and V5 showed TMD gauges — the only minutes-latency source already in the
# database — recover just ~30% of the ceiling, capped by a sparse network measuring points
# against an 11 km cell label. The remaining ~70% needs an *area-covering* source at
# minutes latency. Thai radar has no public historical archive, so it cannot supply the
# 5 years of latency-matched training features this protocol requires. Himawari-8/9 can:
# NOAA mirrors the full JMA archive back to 2015 on AWS (s3://noaa-himawari8,
# s3://noaa-himawari9, anonymous access), updated within ~10–20 min of scan time.
#
# V6 uses AHI Band 13 (10.4 um thermal IR, 2 km): cold cloud tops are the classic
# convection proxy, valid day and night, and observed per pixel over every grid cell —
# no point-vs-cell mismatch. IR is not rain (cirrus is cold but dry), so the ceiling is
# unknown; that is what the run measures.
#
# | config | features | question |
# |---|---|---|
# | `om_only` (from V3, not re-run) | 66 Open-Meteo | control |
# | `hw_offset0h` | 66 + 12 Himawari IR (freshest stamp t-1) | the area-covering deployable gain |
# | `hw_tmd_offset0h` | 66 + 12 Himawari + 16 gauge (V5 block) | the full deployable stack |
#
# Reference points on identical rows: IMERG offset 0 gains +.017..+.050 ROC (ceiling),
# gauges alone +.001..+.015 (V5). Protocol is imported from V3 verbatim, as in V4/V5.
#
# Extra dependencies (fetch phases only): pip install satpy boto3
#
# Phases:
#     python BKK_Rain_V7_himawari_ir.py probe          # one slot end-to-end; picks segments (~5 min)
#     python BKK_Rain_V7_himawari_ir.py fetch [START END]  # hourly slots, resumable; optional
#                                                          # range, e.g. fetch 2025-05-01 2025-11-01
#     python BKK_Rain_V7_himawari_ir.py screen         # NO-TRAINING go/no-go: rank-skill of raw IR
#                                                      # vs gauge/IMERG persistence on fetched rows
#     python BKK_Rain_V7_himawari_ir.py build_features # hourly cell stats -> panel blocks (~5 min)
#     python BKK_Rain_V7_himawari_ir.py cv             # 2 configs x 6 folds x 6 targets (~1.5 h)
#     python BKK_Rain_V7_himawari_ir.py report         # vs om_only + IMERG/gauge references
#
# Recommended pilot before committing to the full 2-7 day fetch (~7% of the download):
#     probe  ->  fetch 2025-05-01 2025-11-01  (~4,400 slots, ~25-35 GB, overnight)  ->  screen
# If screen puts the best IR signal's ROC at or above the gauge band (0.72-0.79 at h1), the
# full fetch is justified; near 0.60-0.65 (the stale-IMERG level), kill V7.

# %%
import bz2
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from BKK_Rain_V3 import (
    CACHE, FEATURE_COLUMNS, MIN_FOLD_ROWS, MODEL_PLAN, PROJECT_ROOT, PURGE_HOURS,
    RANDOM_STATE, TARGETS, TARGET_NAMES,
    fit_predict_calibrated, log, purged_train_mask, score_probabilities,
)
from BKK_Rain_V5_gauge_nowcast import gauge_feature_names

V3_OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v3_imerg_history"
V5_OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v5_gauge_nowcast"
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v7_himawari_ir"
HW_DIR = CACHE / "himawari"          # per-day stat files + pixel-mask cache
HW_TMP = CACHE / "himawari_tmp"      # raw segment downloads, deleted per slot

UTC_OFFSET_HOURS = 7                 # panel is Asia/Bangkok local time
HANDOVER = np.datetime64("2022-12-13")   # Himawari-9 took over operations from Himawari-8
BAND = "B13"                         # 10.4 um clean IR window, 2 km
CANDIDATE_SEGMENTS = [3, 4, 5]       # probe picks which full-disk segment(s) cover Bangkok
CROP_LL_BBOX = (100.0, 12.9, 101.3, 14.5)   # lon_min, lat_min, lon_max, lat_max
CELL_RADIUS_DEG = 0.08               # ~cell scale (9 km grid)
ENV_RADIUS_DEG = 0.35                # ~40 km convection-approach neighbourhood
COLD_220K = 220.0                    # deep convection threshold
COLD_235K = 235.0                    # classic GPI convection threshold
STAT_NAMES = ["tb_mean", "tb_min", "tb_min_env", "cold220_env", "cold235_env", "cold235_scene"]
HW_LOOKBACK = 4                      # deepest stamp used: lag 3 at offset 1


def himawari_feature_names():
    return [
        "hw_tb_mean_lag1", "hw_tb_min_lag1", "hw_tb_min_env_lag1",
        "hw_cold220_env_lag1", "hw_cold235_env_lag1", "hw_cold235_scene_lag1",
        "hw_tb_min_env_lag2", "hw_tb_min_env_lag3",
        "hw_tb_min_env_drop3",   # lag3 - lag1: positive = cloud tops cooling = intensifying
        "hw_tb_min_env_min3", "hw_cold235_env_max3",
        "hw_valid",
    ]


# %% [markdown]
# ## Fetch machinery — one B13 segment per hourly slot, stats computed then raw deleted
#
# A full-disk B13 scan is 10 segments; Bangkok sits in one of them (the probe verifies
# which, rather than trusting hand-done geostationary geometry). Only that segment is
# downloaded — ~6–8 MB per slot instead of ~60. Slot cadence is hourly at :00 (AHI scans
# every 10 min; hourly matches the panel and cuts the download 6x). Pixel->cell index
# masks are computed once from the first decoded slot and cached; every later slot is
# pure array indexing.

# %%
def s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                            retries={"max_attempts": 3}))


def slot_candidates(t_utc):
    """(bucket, satcode) preference order for a UTC timestamp."""
    primary = ("noaa-himawari9", "H09") if t_utc >= HANDOVER else ("noaa-himawari8", "H08")
    backup = ("noaa-himawari8", "H08") if primary[1] == "H09" else ("noaa-himawari9", "H09")
    return [primary, backup]


def segment_key(satcode, t_utc, segment):
    ts = pd.Timestamp(t_utc)
    return (f"AHI-L1b-FLDK/{ts.year}/{ts.month:02d}/{ts.day:02d}/{ts.hour:02d}00/"
            f"HS_{satcode}_{ts.year}{ts.month:02d}{ts.day:02d}_{ts.hour:02d}00_"
            f"{BAND}_FLDK_R20_S{segment:02d}10.DAT")


def download_segments(client, t_utc, segments):
    """Download+decompress the needed segments for one slot; None if unavailable."""
    HW_TMP.mkdir(parents=True, exist_ok=True)
    for bucket, satcode in slot_candidates(t_utc):
        paths = []
        try:
            for seg in segments:
                key = segment_key(satcode, t_utc, seg)
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


def load_tb(paths):
    """Decode segment files -> (brightness_temp_2d, lons, lats), cropped to Bangkok."""
    from satpy import Scene
    scn = Scene(reader="ahi_hsd", filenames=[str(p) for p in paths])
    scn.load([BAND])
    scn = scn.crop(ll_bbox=CROP_LL_BBOX)
    tb = scn[BAND].values.astype("float32")
    lons, lats = scn[BAND].attrs["area"].get_lonlats()
    return tb, lons.astype("float32"), lats.astype("float32")


def build_pixel_masks(lons, lats, grid):
    masks = {"shape": lons.shape, "cell": [], "env": []}
    for _, row in grid.iterrows():
        d2 = (lons - row["longitude"]) ** 2 + (lats - row["latitude"]) ** 2
        masks["cell"].append(np.flatnonzero(d2.ravel() <= CELL_RADIUS_DEG ** 2))
        masks["env"].append(np.flatnonzero(d2.ravel() <= ENV_RADIUS_DEG ** 2))
    return masks


def slot_stats(tb, masks):
    flat = tb.ravel()
    out = np.full((len(masks["cell"]), len(STAT_NAMES)), np.nan, dtype="float32")
    finite_scene = flat[np.isfinite(flat)]
    scene_cold = float((finite_scene < COLD_235K).mean()) if len(finite_scene) else np.nan
    for c, (cell_ix, env_ix) in enumerate(zip(masks["cell"], masks["env"])):
        cell = flat[cell_ix]
        env = flat[env_ix]
        cell = cell[np.isfinite(cell)]
        env = env[np.isfinite(env)]
        if len(cell):
            out[c, 0] = cell.mean()
            out[c, 1] = cell.min()
        if len(env):
            out[c, 2] = env.min()
            out[c, 3] = (env < COLD_220K).mean()
            out[c, 4] = (env < COLD_235K).mean()
        out[c, 5] = scene_cold
    return out


def needed_utc_hours():
    tf = np.load(CACHE / "forecast_time.npy")
    local = pd.DatetimeIndex(tf).unique().sort_values()
    start = local.min() - pd.Timedelta(hours=HW_LOOKBACK)
    hours = pd.date_range(start, local.max(), freq="h")
    return (hours - pd.Timedelta(hours=UTC_OFFSET_HOURS))


# %% [markdown]
# ## Phase `probe` — validate the whole pipeline on a single slot before committing days
#
# Downloads each candidate segment for one recent slot, decodes it, checks which cover the
# Bangkok bbox, saves the needed segment list + pixel masks, and prints the per-cell stats
# for eyeball sanity (dry-season slot should be warm Tb, near-zero cold fractions).

# %%
def phase_probe():
    grid = pd.read_csv(CACHE / "tg_grid.csv")          # written by V5's load_gauges
    client = s3_client()
    probe_t = needed_utc_hours()[len(needed_utc_hours()) // 2]
    log(f"probe slot (UTC): {probe_t}")
    covering = []
    for seg in CANDIDATE_SEGMENTS:
        paths = download_segments(client, probe_t, [seg])
        if paths is None:
            log(f"  segment {seg}: download failed")
            continue
        try:
            tb, lons, lats = load_tb(paths)
            n_finite = int(np.isfinite(tb).sum())
            log(f"  segment {seg}: crop shape {tb.shape}, finite pixels {n_finite:,}")
            if n_finite > 0:
                covering.append((seg, n_finite, tb, lons, lats))
        except Exception as e:
            log(f"  segment {seg}: decode failed ({e})")
        finally:
            for p in paths or []:
                p.unlink(missing_ok=True)
    if not covering:
        raise SystemExit("no candidate segment covers Bangkok — check geometry/keys")

    # keep the minimal set: one segment if it alone covers the full bbox, else the pair
    covering.sort(key=lambda item: -item[1])
    best = covering[0]
    total = sum(item[1] for item in covering)
    segments = [best[0]] if best[1] >= 0.99 * total else sorted(i[0] for i in covering[:2])
    log(f"segments needed: {segments}")

    paths = download_segments(client, probe_t, segments)
    tb, lons, lats = load_tb(paths)
    masks = build_pixel_masks(lons, lats, grid)
    stats = slot_stats(tb, masks)
    for p in paths:
        p.unlink(missing_ok=True)

    HW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(HW_DIR / "pixel_masks.npz",
                        shape=np.array(masks["shape"]),
                        **{f"cell_{i}": m for i, m in enumerate(masks["cell"])},
                        **{f"env_{i}": m for i, m in enumerate(masks["env"])})
    (HW_DIR / "fetch_config.json").write_text(json.dumps(
        {"segments": segments, "band": BAND, "probe_slot_utc": str(probe_t),
         "crop_ll_bbox": CROP_LL_BBOX}, indent=2), encoding="utf-8")
    print(pd.DataFrame(stats, columns=STAT_NAMES).describe().round(2).to_string())
    log("probe OK — masks + segment choice saved; run fetch next")


# %% [markdown]
# ## Phase `fetch` — resumable day-by-day sweep of the archive
#
# One .npz per UTC day under `himawari/`; existing day files are skipped, so the phase can
# be killed and relaunched freely. A failed slot leaves NaN stats for that hour (logged),
# which flows through to NaN features — same convention as gauge outages in V5.

# %%
# Worker-process state for the parallel fetch: each process owns an S3 client and a copy
# of the pixel masks, initialized once. Whole slots (download + decode + stats) run in the
# pool, which sidesteps both the single-stream TCP latency limit and the GIL.
_worker = {}


def _fetch_worker_init(segments, mask_path, grid_csv):
    mask_file = np.load(mask_path)
    n_cells = sum(1 for k in mask_file.files if k.startswith("cell_"))
    _worker["masks"] = {"shape": tuple(mask_file["shape"]),
                        "cell": [mask_file[f"cell_{i}"] for i in range(n_cells)],
                        "env": [mask_file[f"env_{i}"] for i in range(n_cells)]}
    _worker["segments"] = segments
    _worker["grid_csv"] = grid_csv
    _worker["client"] = s3_client()


def _fetch_one_slot(t_utc):
    """Returns the (n_cells, n_stats) array for one slot, or None on failure."""
    paths = download_segments(_worker["client"], t_utc, _worker["segments"])
    if paths is None:
        return None
    try:
        tb, lons, lats = load_tb(paths)
        masks = _worker["masks"]
        if lons.shape != masks["shape"]:          # area drifted (rare) — rebuild masks
            masks = build_pixel_masks(lons, lats, pd.read_csv(_worker["grid_csv"]))
            _worker["masks"] = masks
        return slot_stats(tb, masks)
    except Exception:
        return None
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


def phase_fetch(start=None, end=None):
    from concurrent.futures import ProcessPoolExecutor
    cfg = json.loads((HW_DIR / "fetch_config.json").read_text(encoding="utf-8"))
    segments = cfg["segments"]
    mask_file = np.load(HW_DIR / "pixel_masks.npz")
    n_cells = sum(1 for k in mask_file.files if k.startswith("cell_"))
    workers = int(os.getenv("HW_FETCH_WORKERS", "4"))

    hours = needed_utc_hours()
    if start is not None:
        hours = hours[hours >= pd.Timestamp(start)]
    if end is not None:
        hours = hours[hours < pd.Timestamp(end)]
    days = pd.DatetimeIndex(hours.normalize()).unique().sort_values()
    log(f"UTC hours {len(hours):,} across {len(days):,} days, "
        f"segments {segments}, workers {workers}")

    done = failed_slots = 0
    t_start = time.time()
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_fetch_worker_init,
            initargs=(segments, str(HW_DIR / "pixel_masks.npz"),
                      str(CACHE / "tg_grid.csv"))) as pool:
        for day in days:
            day_file = HW_DIR / f"day_{pd.Timestamp(day):%Y%m%d}.npz"
            if day_file.exists():
                continue
            day_hours = hours[(hours >= day) & (hours < day + pd.Timedelta(days=1))]
            stats = np.full((len(day_hours), n_cells, len(STAT_NAMES)), np.nan,
                            dtype="float32")
            for j, result in enumerate(pool.map(_fetch_one_slot, day_hours)):
                if result is None:
                    failed_slots += 1
                else:
                    stats[j] = result
            np.savez_compressed(day_file, hours=day_hours.to_numpy(dtype="datetime64[s]"),
                                stats=stats)
            done += 1
            rate = done / max((time.time() - t_start) / 3600, 1e-9)
            print(f"  {pd.Timestamp(day):%Y-%m-%d}  days done {done:,}  "
                  f"failed slots {failed_slots:,}  {rate:.1f} days/h", flush=True)
    log(f"fetch complete: {done:,} new days, {failed_slots:,} failed slots")


# %% [markdown]
# ## Phase `screen` — the go/no-go pilot test: no training, minutes of compute
#
# Works on whatever day files exist (intended after a wet-season pilot fetch). Scores raw
# IR signals at stamp t-1 as rank-predictors of the six targets — negated brightness-temp
# minimum (colder tops = more rain) and the 235 K cold fraction — next to the persistence
# yardsticks cached by V3/V5 (fresh IMERG and gauge IDW), restricted to the SAME rows.
# Yardsticks at h1: fresh IMERG ROC 0.79-0.85, gauges 0.72-0.79, 6 h-stale IMERG 0.59-0.65.
# Best IR signal at or above the gauge band justifies the full fetch; at the stale-IMERG
# level, kill V7.

# %%
def load_stat_series():
    files = sorted(HW_DIR.glob("day_*.npz"))
    if not files:
        raise SystemExit("no fetched day files — run probe + fetch first")
    hours_list, stats_list = [], []
    for f in files:
        d = np.load(f)
        hours_list.append(d["hours"])
        stats_list.append(d["stats"])
    hours_utc = np.concatenate(hours_list)
    stats = np.concatenate(stats_list)
    order = np.argsort(hours_utc)
    hours_utc, stats = hours_utc[order], stats[order]
    return hours_utc + np.timedelta64(UTC_OFFSET_HOURS, "h"), stats


def phase_screen():
    from sklearn.metrics import average_precision_score, roc_auc_score
    hours_local, stats = load_stat_series()
    s = {name: stats[..., i] for i, name in enumerate(STAT_NAMES)}
    grid = pd.read_csv(CACHE / "tg_grid.csv")
    unit = np.load(CACHE / "unit_id.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    y = np.load(CACHE / "y.npy")
    hour_pos = {np.datetime64(h): i for i, h in enumerate(hours_local)}
    cell_pos = {g: i for i, g in enumerate(grid["grid_number"])}
    row_hour = np.array([hour_pos.get(np.datetime64(t), -1) for t in tf], dtype="int64")
    row_cell = np.array([cell_pos[int(g)] for g in unit], dtype="int64")
    inside = row_hour >= 0

    candidates = {
        "hw_neg_tbmin_env_lag1": -shift_down(s["tb_min_env"], 1),
        "hw_cold235_env_lag1": shift_down(s["cold235_env"], 1),
    }
    scores = {}
    for name, hourly in candidates.items():
        v = np.full(len(tf), np.nan, dtype="float32")
        v[inside] = hourly[row_hour[inside], row_cell[inside]]
        scores[name] = v
    for name, fname in [("imerg_fresh_persist", "persist_research.npy"),
                        ("gauge_idw_persist", "tg_persist_offset0.npy")]:
        if (CACHE / fname).exists():
            scores[name] = np.load(CACHE / fname)

    valid = np.isfinite(scores["hw_neg_tbmin_env_lag1"])
    log(f"screen rows (IR lag-1 available): {valid.sum():,} / {len(tf):,} "
        f"({pd.Timestamp(tf[valid].min())} .. {pd.Timestamp(tf[valid].max())})")
    rows = []
    for i, tname in enumerate(TARGET_NAMES):
        for name, v in scores.items():
            m = valid & np.isfinite(v)
            base = float(y[m, i].mean())
            rows.append({
                "target": tname, "score": name, "rows": int(m.sum()), "base_rate": base,
                "roc_auc": float(roc_auc_score(y[m, i], v[m])),
                "pr_auc_lift": float(average_precision_score(y[m, i], v[m])) / base,
            })
    out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_DIR / "v7_screen.csv", index=False)
    pivot = out.pivot_table(index="target", columns="score", values="roc_auc")
    log("=== screening ROC-AUC on identical rows (no training) ===")
    print(pivot.round(4).to_string())
    log(f"saved to {OUTPUT_DIR / 'v7_screen.csv'}")


# %% [markdown]
# ## Phase `build_features` — hourly stat series -> panel-aligned feature blocks
#
# Mirrors V5: stamp convention is V3's (offset o => freshest stamp t-(o+1)); the block is
# indexed onto panel rows by (local hour, grid cell). `hw_valid` marks rows whose freshest
# stamp decoded successfully — missingness is signal, and it also isolates any archive gaps.

# %%
def shift_down(arr, k):
    out = np.full_like(arr, np.nan)
    if k < len(arr):
        out[k:] = arr[:-k] if k else arr
    return out


def phase_build_features(offsets=(0, 1)):
    files = sorted(HW_DIR.glob("day_*.npz"))
    if not files:
        raise SystemExit("no fetched day files — run probe + fetch first")
    hours_list, stats_list = [], []
    for f in files:
        d = np.load(f)
        hours_list.append(d["hours"])
        stats_list.append(d["stats"])
    hours_utc = np.concatenate(hours_list)
    stats = np.concatenate(stats_list)                 # (n_hours, n_cells, n_stats)
    order = np.argsort(hours_utc)
    hours_utc, stats = hours_utc[order], stats[order]
    hours_local = hours_utc + np.timedelta64(UTC_OFFSET_HOURS, "h")
    log(f"stat series {stats.shape}, finite share "
        f"{np.isfinite(stats[..., 1]).mean():.1%}")

    grid = pd.read_csv(CACHE / "tg_grid.csv")
    unit = np.load(CACHE / "unit_id.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    hour_pos = {np.datetime64(h): i for i, h in enumerate(hours_local)}
    cell_pos = {g: i for i, g in enumerate(grid["grid_number"])}
    row_hour = np.array([hour_pos.get(np.datetime64(t), -1) for t in tf], dtype="int64")
    row_cell = np.array([cell_pos[int(g)] for g in unit], dtype="int64")
    outside = row_hour < 0
    log(f"panel rows outside fetched range: {outside.sum():,} ({outside.mean():.2%})")
    row_hour[outside] = 0

    s = {name: stats[..., i] for i, name in enumerate(STAT_NAMES)}
    for offset in offsets:
        o = offset
        env1 = shift_down(s["tb_min_env"], o + 1)
        env2 = shift_down(s["tb_min_env"], o + 2)
        env3 = shift_down(s["tb_min_env"], o + 3)
        cold1 = shift_down(s["cold235_env"], o + 1)
        cold2 = shift_down(s["cold235_env"], o + 2)
        cold3 = shift_down(s["cold235_env"], o + 3)
        env_stack = np.stack([env1, env2, env3])
        cold_stack = np.stack([cold1, cold2, cold3])
        with np.errstate(all="ignore"):
            hourly = np.stack([
                shift_down(s["tb_mean"], o + 1), shift_down(s["tb_min"], o + 1),
                env1, shift_down(s["cold220_env"], o + 1), cold1,
                shift_down(s["cold235_scene"], o + 1),
                env2, env3,
                env3 - env1,                       # cooling cloud tops = intensifying
                np.nanmin(env_stack, axis=0), np.nanmax(cold_stack, axis=0),
                np.isfinite(env1).astype("float32"),
            ], axis=-1).astype("float32")          # (n_hours, n_cells, 12)
        block = hourly[row_hour, row_cell]
        block[outside] = np.nan
        block[outside, -1] = 0.0
        np.save(CACHE / f"hw_block_offset{offset}.npy", block)
        share = pd.Series(np.isnan(block).mean(axis=0), index=himawari_feature_names())
        log(f"offset {offset}: block {block.shape}, NaN share:")
        print(share.round(4).to_string())
        del hourly, block
        gc.collect()
    log("saved himawari blocks")


# %% [markdown]
# ## Phase `deploy_fit` — the final deployable artifacts (first ever in this project)
#
# Fits the winning config (66 OM + 12 Himawari + 16 gauge features) on ALL rows, one model
# per target, with the protocol's calibration recipe (12% holdout -> isotonic) and the
# F1-optimal operating threshold selected on the honest OOF predictions from the CV phase.
# Saves one .joblib per target: {model, calibrator, threshold, feature_names, metadata}.

# %%
def phase_deploy_fit():
    import joblib
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from BKK_Rain_V3 import CALIBRATION_FRACTION, pick_threshold

    y = np.load(CACHE / "y.npy")
    parts = [np.load(CACHE / "base_X.npy"), np.load(CACHE / "hw_block_offset0.npy"),
             np.load(CACHE / "tg_block_offset0.npy")]
    x = np.ascontiguousarray(np.concatenate(parts, axis=1))
    del parts
    gc.collect()
    feature_names = (FEATURE_COLUMNS + himawari_feature_names() + gauge_feature_names())
    oof = np.load(CACHE / "oof_cal_hw_tmd_offset0h.npy")
    log(f"deploy fit: X {x.shape}, {len(feature_names)} features")

    deploy_dir = OUTPUT_DIR / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    summary = []
    for i, (horizon, threshold_mm) in enumerate(TARGETS):
        t0 = time.time()
        n = len(y)
        shuffled = rng.permutation(n)
        calib_ix = shuffled[:int(n * CALIBRATION_FRACTION)]
        fit_ix = shuffled[int(n * CALIBRATION_FRACTION):]
        model_name = MODEL_PLAN[horizon]
        if model_name == "lightgbm":
            spw = float((y[fit_ix, i] == 0).sum() / max((y[fit_ix, i] == 1).sum(), 1))
            model = LGBMClassifier(
                objective="binary", n_estimators=500, learning_rate=0.04, num_leaves=63,
                min_child_samples=80, subsample=0.85, colsample_bytree=0.85,
                reg_lambda=1.0, scale_pos_weight=spw, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1)
        else:
            model = HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=250, max_leaf_nodes=31,
                l2_regularization=0.05, random_state=RANDOM_STATE, early_stopping=False)
        model.fit(x[fit_ix], y[fit_ix, i])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(model.predict_proba(x[calib_ix])[:, 1], y[calib_ix, i])
        valid = np.isfinite(oof[:, i])
        op_threshold = pick_threshold(oof[valid, i], y[valid, i])
        artifact = {
            "model": model, "calibrator": iso, "threshold": op_threshold,
            "feature_names": feature_names, "target": TARGET_NAMES[i],
            "horizon_h": horizon, "rain_threshold_mm": threshold_mm,
            "trained_rows": int(len(fit_ix)),
            "notes": "config hw_tmd_offset0h: OM forecast + Himawari B13 IR (stamp t-1) "
                     "+ TMD gauge IDW (stamp t-1); apply calibrator to predict_proba, "
                     "alert when calibrated p >= threshold",
        }
        joblib.dump(artifact, deploy_dir / f"bkk_rain_{TARGET_NAMES[i]}.joblib")
        summary.append({"target": TARGET_NAMES[i], "model": model_name,
                        "threshold": op_threshold, "fit_min": (time.time() - t0) / 60})
        log(f"  {TARGET_NAMES[i]:12} {model_name:22} thr {op_threshold:.2f} "
            f"{(time.time()-t0)/60:.1f} min")
    pd.DataFrame(summary).to_csv(deploy_dir / "deploy_summary.csv", index=False)
    log(f"saved 6 deployable artifacts to {deploy_dir}")


# %% [markdown]
# ## Phase `cv` — V3 protocol; IR alone, then IR + gauges (the full deployable stack)

# %%
CONFIG_BLOCKS = {
    "hw_offset0h": [("hw_block_offset0.npy", himawari_feature_names)],
    "hw_tmd_offset0h": [("hw_block_offset0.npy", himawari_feature_names),
                        ("tg_block_offset0.npy", gauge_feature_names)],
}


def run_hw_cv(config, y, forecast_time, year_key, year_folds):
    parts = [np.load(CACHE / "base_X.npy")]
    for fname, _ in CONFIG_BLOCKS[config]:
        parts.append(np.load(CACHE / fname))
    x = np.ascontiguousarray(np.concatenate(parts, axis=1))
    del parts
    gc.collect()

    log(f"config {config}: X {x.shape}")
    purge = np.timedelta64(PURGE_HOURS, "h")
    rng = np.random.default_rng(RANDOM_STATE)
    oof_raw = np.full((x.shape[0], len(TARGETS)), np.nan, dtype="float32")
    oof_cal = np.full((x.shape[0], len(TARGETS)), np.nan, dtype="float32")
    fit_log = []
    for year in year_folds:
        test_mask = year_key == year
        train_mask = purged_train_mask(test_mask, forecast_time, purge)
        x_train, x_test = x[train_mask], x[test_mask]
        log(f"  fold {year}: test {test_mask.sum():,} train {train_mask.sum():,}")
        for i, (horizon, threshold) in enumerate(TARGETS):
            model_name = MODEL_PLAN[horizon]
            t0 = time.time()
            raw, cal = fit_predict_calibrated(x_train, y[train_mask, i], x_test, model_name, rng)
            oof_raw[test_mask, i] = raw
            oof_cal[test_mask, i] = cal
            fit_log.append({"config": config, "fold": str(year), "target": TARGET_NAMES[i],
                            "model": model_name, "train_rows": int(train_mask.sum()),
                            "fit_seconds": time.time() - t0})
            log(f"    {TARGET_NAMES[i]:12} {model_name:22} {(time.time()-t0)/60:.1f} min")
        del x_train, x_test
        gc.collect()
    del x
    gc.collect()
    np.save(CACHE / f"oof_cal_{config}.npy", oof_cal)
    np.save(CACHE / f"oof_raw_{config}.npy", oof_raw)
    results = score_probabilities(oof_cal, y, year_key, year_folds, "calibrated", config)
    results += score_probabilities(oof_raw, y, year_key, year_folds, "raw", config)
    return results, fit_log


def phase_cv():
    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    stamps = pd.to_datetime(forecast_time)
    year_key = stamps.year.to_numpy()
    year_folds = [int(v) for v in np.unique(year_key) if (year_key == v).sum() >= MIN_FOLD_ROWS]
    log(f"folds: {year_folds}   rows {len(y):,}")

    all_results, all_fitlog = [], []
    results_path = OUTPUT_DIR / "v7_cv_results.csv"
    done = set()
    if results_path.exists():
        prior = pd.read_csv(results_path)
        all_results = prior.to_dict(orient="records")
        done = set(prior["config"].unique())
        log(f"resuming; already done: {sorted(done)}")
    for config in CONFIG_BLOCKS:
        if config in done:
            continue
        results, fit_log = run_hw_cv(config, y, forecast_time, year_key, year_folds)
        all_results += results
        all_fitlog += fit_log
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        pd.DataFrame(all_fitlog).to_csv(OUTPUT_DIR / "v7_fit_log.csv", index=False)
    log("cv phase complete")


# %% [markdown]
# ## Phase `report` — the verdict table across every source tried in this project

# %%
def phase_report():
    v7 = pd.read_csv(OUTPUT_DIR / "v7_cv_results.csv")
    v3 = pd.read_csv(V3_OUTPUT_DIR / "v3_oof_results.csv")
    v5 = pd.read_csv(V5_OUTPUT_DIR / "v5_cv_results.csv")
    refs = pd.concat([
        v3[v3["config"].isin(["om_only", "imerg_research", "imerg_deploy"])],
        v5[v5["config"] == "tmd_offset0h"],
    ], ignore_index=True).replace(
        {"config": {"imerg_research": "imerg_offset0h", "imerg_deploy": "imerg_offset6h"}})
    combined = pd.concat([refs, v7], ignore_index=True)
    cal = combined[combined["probabilities"] == "calibrated"]

    om_roc = cal[cal["config"] == "om_only"].set_index("target")["roc_auc"]
    om_f1 = cal[cal["config"] == "om_only"].set_index("target")["f1"]
    col_order = [c for c in ["hw_offset0h", "hw_tmd_offset0h", "tmd_offset0h",
                             "imerg_offset0h", "imerg_offset6h"]
                 if c in cal["config"].unique()]
    roc_gain = cal.pivot_table(index="target", columns="config",
                               values="roc_auc")[col_order].sub(om_roc, axis=0)
    f1_gain = cal.pivot_table(index="target", columns="config",
                              values="f1")[col_order].sub(om_f1, axis=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "v7_results_combined.csv", index=False)
    roc_gain.to_csv(OUTPUT_DIR / "v7_roc_gain_vs_references.csv")
    f1_gain.to_csv(OUTPUT_DIR / "v7_f1_gain_vs_references.csv")

    metadata = {
        "purpose": "Himawari B13 IR history as the area-covering minutes-latency feed; "
                   "measures whether cold-cloud-top features close the gap between the "
                   "gauge gain (V5, ~30% of ceiling) and the IMERG offset-0 ceiling (V3/V4)",
        "protocol": "identical to V3/V4/V5 (imported): leave-one-year-out CV, 24 h purge, "
                    "isotonic calibration, IMERG labels, F1-optimal leave-one-year-out thresholds",
        "source": "s3://noaa-himawari8 + s3://noaa-himawari9, AHI-L1b-FLDK B13 (10.4 um, "
                  "2 km), hourly :00 slots, one/two segments per slot",
        "himawari_features": himawari_feature_names(),
        "configs": {c: [f for _, names in CONFIG_BLOCKS[c] for f in names()]
                    for c in CONFIG_BLOCKS},
        "roc_gain_vs_references": roc_gain.reset_index().to_dict(orient="records"),
        "f1_gain_vs_references": f1_gain.reset_index().to_dict(orient="records"),
        "caveats": [
            "IR sees cloud tops, not rain: cold cirrus inflates cold fractions without "
            "precipitation, so the IR ceiling may sit well below the IMERG offset-0 ceiling.",
            "Real-time NOAA bucket latency is ~10-20 min after scan; stamp t-1 at forecast "
            "time t is comfortably honest.",
            "Slots that failed to download/decode carry NaN features (hw_valid=0).",
            "IMERG labels: an IR-graded or gauge-graded evaluation answers a different "
            "question (three-way label disagreement).",
        ],
    }
    (OUTPUT_DIR / "v7_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log("=== ROC-AUC gain over om_only: Himawari vs every reference ===")
    print(roc_gain.round(4).to_string())
    log("=== F1 gain over om_only ===")
    print(f1_gain.round(4).to_string())
    log(f"saved V7 outputs to {OUTPUT_DIR}")


# %%
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "cv"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "probe":
        phase_probe()
    elif phase == "fetch":
        phase_fetch(*sys.argv[2:4])
    elif phase == "screen":
        phase_screen()
    elif phase == "build_features":
        phase_build_features()
    elif phase == "cv":
        phase_cv()
    elif phase == "report":
        phase_report()
    elif phase == "deploy_fit":
        phase_deploy_fit()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use probe | fetch | screen | "
                         f"build_features | cv | report")
    log(f"phase {phase} done in {(time.time()-t0)/60:.1f} min")
