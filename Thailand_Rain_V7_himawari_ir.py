# %% [markdown]
# # Thailand_Rain_V7 — Himawari B13 IR over the whole country
#
# Thailand port of `BKK_Rain_V7_himawari_ir.py`. The Bangkok run established that cold-cloud-top
# features from the free NOAA AHI archive recover 60-75% of the observed-rain ceiling and are
# deployable at minutes latency. This asks whether that survives at 25 km over 833 cells spanning
# sea level to 2,500 m, maritime peninsula to continental Isaan — and, unlike Bangkok, against
# labels that are genuinely independent of the features.
#
# Everything geometric was measured, not assumed (probe, 2026-08-11):
#
# | | Bangkok | Thailand | note |
# |---|---|---|---|
# | segments | 4 | **4 + 5** | 3 and 6 decode to zero finite pixels over the bbox |
# | crop | 72 x 89 px | **797 x 452** = 360,244 px | 56x the area |
# | cell radius | 0.08 deg (9 km) | **0.1126** (25 km) | matches the grid |
# | env radius | 0.35 deg (~39 km) | **0.50** (~55 km) | must stay well wider than one cell |
# | per slot | ~1.8 s | **5.5 s** | 3.8 s network, 1.5 s decode, 0.2 s stats |
#
# Full panel is 17,976 hourly slots ~ 7 h at 4 workers, 113 GB transferred, ~150 MB kept.
# The pilot below is 61 days for ~35 min and answers go/no-go first.
#
# Phases:
#     python Thailand_Rain_V7_himawari_ir.py probe                       # verify segments, cache masks
#     python Thailand_Rain_V7_himawari_ir.py fetch [START END]           # resumable, day-by-day
#     python Thailand_Rain_V7_himawari_ir.py screen                      # NO-TRAINING go/no-go
#     python Thailand_Rain_V7_himawari_ir.py build_features              # -> panel-aligned blocks
#     python Thailand_Rain_V7_himawari_ir.py cv                          # hw config vs om_only
#
# Pre-registered pilot rule (set before looking at any number): trained `om_only` scores
# ROC 0.77-0.80 on wet-season blocks. A single *untrained* IR signal clearing **0.70** there
# implies real independent information and justifies the remaining ~6 h of download. Below
# ~0.65 — roughly what a stale observation gives — stop.

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
import psycopg2

from Thailand_Rain_V3 import CACHE, DB_CONFIG, IMERG_TABLE_NAME, OUTPUT_DIR as V3_OUTPUT_DIR
from BKK_Rain_V3 import PROJECT_ROOT, TARGET_NAMES, log

OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v7_himawari_ir"
HW_DIR = CACHE / "himawari"
HW_TMP = CACHE / "himawari_tmp"

UTC_OFFSET_HOURS = 7
HANDOVER = np.datetime64("2022-12-13")     # Himawari-9 took over; this panel is entirely after
BAND = "B13"                                # 10.4 um clean IR window, 2 km
CANDIDATE_SEGMENTS = [3, 4, 5, 6]
CROP_LL_BBOX = (97.0, 5.3, 106.0, 20.9)     # Thailand grid extent, padded for the env radius
CELL_RADIUS_DEG = 0.1126
ENV_RADIUS_DEG = 0.50
COLD_220K = 220.0
COLD_235K = 235.0
STAT_NAMES = ["tb_mean", "tb_min", "tb_min_env", "cold220_env", "cold235_env", "cold235_scene"]
HW_LOOKBACK = 4

GRID_TABLE = '"Thailand_Grid_25km"'
GRID_CSV = CACHE / "th_grid.csv"


def himawari_feature_names():
    return [
        "hw_tb_mean_lag1", "hw_tb_min_lag1", "hw_tb_min_env_lag1",
        "hw_cold220_env_lag1", "hw_cold235_env_lag1", "hw_cold235_scene_lag1",
        "hw_tb_min_env_lag2", "hw_tb_min_env_lag3",
        "hw_tb_min_env_drop3",
        "hw_tb_min_env_min3", "hw_cold235_env_max3",
        "hw_valid",
    ]


def load_grid():
    """Thailand grid as a csv the fetch workers can each read cheaply."""
    if GRID_CSV.exists():
        return pd.read_csv(GRID_CSV)
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT grid_number, longitude, latitude FROM {GRID_TABLE} "
                    f"ORDER BY grid_number;")
        grid = pd.DataFrame(cur.fetchall(), columns=["grid_number", "longitude", "latitude"])
    finally:
        conn.close()
    GRID_CSV.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(GRID_CSV, index=False)
    return grid


# %% [markdown]
# ## Fetch machinery
#
# Identical in shape to the Bangkok version; only the segment list and crop differ. Stats are
# computed per slot and the raw .DAT deleted immediately, so 113 GB of transfer leaves ~150 MB
# on disk. Pixel->cell index masks cost ~12 s to build and are cached once.

# %%
def s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED,
                                            retries={"max_attempts": 3}))


def slot_candidates(t_utc):
    primary = ("noaa-himawari9", "H09") if t_utc >= HANDOVER else ("noaa-himawari8", "H08")
    backup = ("noaa-himawari8", "H08") if primary[1] == "H09" else ("noaa-himawari9", "H09")
    return [primary, backup]


def segment_key(satcode, t_utc, segment):
    ts = pd.Timestamp(t_utc)
    return (f"AHI-L1b-FLDK/{ts.year}/{ts.month:02d}/{ts.day:02d}/{ts.hour:02d}00/"
            f"HS_{satcode}_{ts.year}{ts.month:02d}{ts.day:02d}_{ts.hour:02d}00_"
            f"{BAND}_FLDK_R20_S{segment:02d}10.DAT")


def download_segments(client, t_utc, segments):
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
    """Every hour in the panel span, not just the stride hours.

    The panel keeps local hours 0/4/8/12/16/20, but each row reads IR at t-1..t-4, and the
    union of those offsets over the stride covers all 24 hours. So the fetch is dense.
    """
    tf = np.load(CACHE / "forecast_time.npy")
    local = pd.DatetimeIndex(tf).unique().sort_values()
    start = local.min() - pd.Timedelta(hours=HW_LOOKBACK)
    hours = pd.date_range(start, local.max(), freq="h")
    return hours - pd.Timedelta(hours=UTC_OFFSET_HOURS)


# %% [markdown]
# ## Phase `probe` — confirm the segment set and cache the masks

# %%
def phase_probe():
    grid = load_grid()
    client = s3_client()
    hours = needed_utc_hours()
    probe_t = hours[len(hours) // 2]
    log(f"probe slot (UTC): {probe_t};  grid cells {len(grid)}")
    covering = []
    for seg in CANDIDATE_SEGMENTS:
        paths = download_segments(client, probe_t, [seg])
        if paths is None:
            log(f"  segment {seg}: download failed")
            continue
        try:
            tb, lons, lats = load_tb(paths)
            n_finite = int(np.isfinite(tb).sum())
            log(f"  segment {seg}: crop {tb.shape}, finite pixels {n_finite:,}")
            if n_finite > 0:
                covering.append(seg)
        except Exception as e:
            log(f"  segment {seg}: decode failed ({e})")
        finally:
            for p in paths or []:
                p.unlink(missing_ok=True)
    if not covering:
        raise SystemExit("no candidate segment covers Thailand — check geometry/keys")
    segments = sorted(covering)
    log(f"segments needed: {segments}")

    paths = download_segments(client, probe_t, segments)
    tb, lons, lats = load_tb(paths)
    masks = build_pixel_masks(lons, lats, grid)
    stats = slot_stats(tb, masks)
    for p in paths:
        p.unlink(missing_ok=True)

    covered = int(np.isfinite(stats[:, 0]).sum())
    log(f"cells with data: {covered}/{len(grid)}   "
        f"px/cell {np.mean([len(m) for m in masks['cell']]):.0f}, "
        f"px/env {np.mean([len(m) for m in masks['env']]):.0f}")
    if covered < len(grid):
        log("WARNING: some cells have no IR pixels — widen CROP_LL_BBOX")

    HW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(HW_DIR / "pixel_masks.npz",
                        shape=np.array(masks["shape"]),
                        **{f"cell_{i}": m for i, m in enumerate(masks["cell"])},
                        **{f"env_{i}": m for i, m in enumerate(masks["env"])})
    (HW_DIR / "fetch_config.json").write_text(json.dumps(
        {"segments": segments, "band": BAND, "probe_slot_utc": str(probe_t),
         "crop_ll_bbox": CROP_LL_BBOX, "cell_radius_deg": CELL_RADIUS_DEG,
         "env_radius_deg": ENV_RADIUS_DEG, "n_cells": int(len(grid))}, indent=2),
        encoding="utf-8")
    print(pd.DataFrame(stats, columns=STAT_NAMES).describe().round(2).to_string())
    log("probe OK — masks + segment choice saved; run fetch next")


# %% [markdown]
# ## Phase `fetch` — resumable day-by-day sweep

# %%
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
    paths = download_segments(_worker["client"], t_utc, _worker["segments"])
    if paths is None:
        return None
    try:
        tb, lons, lats = load_tb(paths)
        masks = _worker["masks"]
        if lons.shape != masks["shape"]:
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
        f"segments {segments}, cells {n_cells}, workers {workers}")

    done = failed_slots = 0
    t_start = time.time()
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_fetch_worker_init,
            initargs=(segments, str(HW_DIR / "pixel_masks.npz"), str(GRID_CSV))) as pool:
        for day in days:
            day_file = HW_DIR / f"day_{pd.Timestamp(day):%Y%m%d}.npz"
            if day_file.exists():
                continue
            day_hours = hours[(hours >= day) & (hours < day + pd.Timedelta(days=1))]
            stats = np.full((len(day_hours), n_cells, len(STAT_NAMES)), np.nan, dtype="float32")
            for j, result in enumerate(pool.map(_fetch_one_slot, day_hours)):
                if result is None:
                    failed_slots += 1
                else:
                    stats[j] = result
            np.savez_compressed(day_file, hours=day_hours.to_numpy(dtype="datetime64[s]"),
                                stats=stats)
            done += 1
            rate = done / max((time.time() - t_start) / 3600, 1e-9)
            eta_h = (len(days) - done) / max(rate, 1e-9)
            print(f"  {pd.Timestamp(day):%Y-%m-%d}  days {done}/{len(days)}  "
                  f"failed slots {failed_slots:,}  {rate:.1f} days/h  eta {eta_h:.1f} h",
                  flush=True)
    log(f"fetch complete: {done:,} new days, {failed_slots:,} failed slots")


# %% [markdown]
# ## Phase `screen` — the no-training go/no-go
#
# Bangkok compared raw IR against cached gauge and IMERG persistence. Thailand has no
# nationwide gauge block, so the yardstick is IMERG persistence alone: observed cell-max rain
# in the hour ending at t. That is a 0 h-latency observation and therefore an upper bound no
# deployable feed can reach — if raw IR lands anywhere near it, the signal is real.

# %%
def shift_down(arr, k):
    out = np.full_like(arr, np.nan)
    if k < len(arr):
        out[k:] = arr[:-k] if k else arr
    return out


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


def imerg_persistence(unit, tf, lo, hi):
    """Observed cell-max rain in the hour ending at t, for panel rows in [lo, hi]."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT grid_number, local_observation_time, precipitation_max_mm
            FROM {IMERG_TABLE_NAME}
            WHERE is_complete_hour
              AND local_observation_time BETWEEN %s AND %s
        """, (pd.Timestamp(lo).to_pydatetime(), pd.Timestamp(hi).to_pydatetime()))
        rows = cur.fetchall()
    finally:
        conn.close()
    lookup = {(int(g), np.datetime64(t, "s")): (np.nan if v is None else float(v))
              for g, t, v in rows}
    out = np.full(len(tf), np.nan, dtype="float32")
    for i in range(len(tf)):
        out[i] = lookup.get((int(unit[i]), np.datetime64(tf[i], "s")), np.nan)
    return out


def phase_screen():
    from sklearn.metrics import average_precision_score, roc_auc_score
    hours_local, stats = load_stat_series()
    s = {name: stats[..., i] for i, name in enumerate(STAT_NAMES)}
    grid = load_grid()
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
        "hw_neg_tbmin_cell_lag1": -shift_down(s["tb_min"], 1),
    }
    scores = {}
    for name, hourly in candidates.items():
        v = np.full(len(tf), np.nan, dtype="float32")
        v[inside] = hourly[row_hour[inside], row_cell[inside]]
        scores[name] = v

    valid = np.isfinite(scores["hw_neg_tbmin_env_lag1"])
    if not valid.any():
        raise SystemExit("no panel rows overlap the fetched IR range")
    lo, hi = pd.Timestamp(tf[valid].min()), pd.Timestamp(tf[valid].max())
    log(f"screen rows (IR lag-1 available): {valid.sum():,} / {len(tf):,}  ({lo} .. {hi})")
    scores["imerg_fresh_persist"] = imerg_persistence(unit, tf, lo, hi)

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
    out.to_csv(OUTPUT_DIR / "th_v7_screen.csv", index=False)
    pivot = out.pivot_table(index="target", columns="score", values="roc_auc")
    log("=== screening ROC-AUC on identical rows (no training) ===")
    print(pivot.round(4).to_string())
    best = out[out["score"].str.startswith("hw_")].groupby("target")["roc_auc"].max()
    log(f"best IR signal per target: {best.round(4).to_dict()}")
    log("pre-registered rule: >= 0.70 on wet-season rows => proceed with the full fetch")
    log(f"saved to {OUTPUT_DIR / 'th_v7_screen.csv'}")


# %% [markdown]
# ## Phase `build_features` — hourly stats -> panel-aligned block

# %%
def phase_build_features(offsets=(0,)):
    hours_local, stats = load_stat_series()
    log(f"stat series {stats.shape}, finite share {np.isfinite(stats[..., 1]).mean():.1%}")
    grid = load_grid()
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
        env1, env2, env3 = (shift_down(s["tb_min_env"], o + k) for k in (1, 2, 3))
        cold1, cold2, cold3 = (shift_down(s["cold235_env"], o + k) for k in (1, 2, 3))
        with np.errstate(all="ignore"):
            hourly = np.stack([
                shift_down(s["tb_mean"], o + 1), shift_down(s["tb_min"], o + 1),
                env1, shift_down(s["cold220_env"], o + 1), cold1,
                shift_down(s["cold235_scene"], o + 1),
                env2, env3,
                env3 - env1,
                np.nanmin(np.stack([env1, env2, env3]), axis=0),
                np.nanmax(np.stack([cold1, cold2, cold3]), axis=0),
                np.isfinite(env1).astype("float32"),
            ], axis=-1).astype("float32")
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
# ## Phase `cv` — IR config against the om_only control, on IR-covered blocks only
#
# The fetch is wet-season only, so 7 of the 13 blocks have no IR at all. Training those folds
# would burn ~20 min each to compare NaN features against NaN features. `covered_folds` keeps
# only blocks where `hw_valid` is high, and the report compares against `om_only` restricted to
# exactly the same rows — the V3 OOF array is already on disk, so the control needs no refit.
#
# Reuses Thailand_Rain_V3.run_cv by registering the config there, which means the per-fit
# checkpointing applies to this run too.

# %%
MIN_FOLD_IR_COVERAGE = 0.90


def covered_folds(block, fold_key, folds):
    """Blocks whose rows mostly have a valid freshest IR stamp."""
    valid = block[:, -1] > 0        # hw_valid
    keep = []
    for fold in folds:
        share = float(valid[fold_key == fold].mean())
        flag = "keep" if share >= MIN_FOLD_IR_COVERAGE else "skip"
        log(f"  block {fold:2d}: IR coverage {share:6.1%}  {flag}")
        if share >= MIN_FOLD_IR_COVERAGE:
            keep.append(fold)
    return keep


def phase_cv():
    import Thailand_Rain_V3 as v3

    v3.CONFIG_BLOCKS["hw_offset0h"] = [("hw_block_offset0.npy", himawari_feature_names)]

    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    fold_key = v3.block_keys(forecast_time)
    all_folds = [int(v) for v in np.unique(fold_key)
                 if (fold_key == v).sum() >= v3.MIN_FOLD_ROWS]

    block = np.load(CACHE / "hw_block_offset0.npy", mmap_mode="r")
    log(f"IR block {block.shape}; selecting blocks with >= {MIN_FOLD_IR_COVERAGE:.0%} coverage")
    folds = covered_folds(np.asarray(block[:, -1:]), fold_key, all_folds)
    del block
    gc.collect()
    if not folds:
        raise SystemExit("no block has enough IR coverage — check build_features")
    log(f"training on {len(folds)} of {len(all_folds)} blocks: {folds}")

    results, fit_log = v3.run_cv("hw_offset0h", y, forecast_time, fold_key, folds)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "th_v7_cv_results.csv", index=False)
    pd.DataFrame(fit_log).to_csv(OUTPUT_DIR / "th_v7_fit_log.csv", index=False)
    json.dump({"folds": folds, "min_ir_coverage": MIN_FOLD_IR_COVERAGE},
              open(OUTPUT_DIR / "th_v7_folds.json", "w"), indent=2)
    log("cv phase complete")


# %% [markdown]
# ## Phase `report` — IR gain over om_only on identical rows

# %%
def phase_report():
    from sklearn.metrics import roc_auc_score

    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    import Thailand_Rain_V3 as v3
    fold_key = v3.block_keys(forecast_time)
    folds = json.loads((OUTPUT_DIR / "th_v7_folds.json").read_text())["folds"]
    in_scope = np.isin(fold_key, folds)

    hw = np.load(CACHE / "oof_cal_hw_offset0h.npy")
    om = np.load(CACHE / "oof_cal_om_only.npy")
    rows = []
    for i, name in enumerate(TARGET_NAMES):
        m = in_scope & np.isfinite(hw[:, i]) & np.isfinite(om[:, i])
        rows.append({"target": name, "rows": int(m.sum()),
                     "base_rate": float(y[m, i].mean()),
                     "om_only": float(roc_auc_score(y[m, i], om[m, i])),
                     "hw_offset0h": float(roc_auc_score(y[m, i], hw[m, i]))})
    out = pd.DataFrame(rows)
    out["roc_gain"] = out["hw_offset0h"] - out["om_only"]
    out.to_csv(OUTPUT_DIR / "th_v7_gain_vs_om_only.csv", index=False)
    log("=== IR gain over om_only, identical rows, IR-covered blocks ===")
    print(out.round(4).to_string(index=False))
    per_block = []
    for fold in folds:
        for i, name in enumerate(TARGET_NAMES):
            m = (fold_key == fold) & np.isfinite(hw[:, i]) & np.isfinite(om[:, i])
            if m.sum() < v3.MIN_FOLD_ROWS or len(np.unique(y[m, i])) < 2:
                continue
            per_block.append({"block": int(fold), "span": v3.block_label(fold, forecast_time),
                              "target": name,
                              "om_only": float(roc_auc_score(y[m, i], om[m, i])),
                              "hw_offset0h": float(roc_auc_score(y[m, i], hw[m, i]))})
    pb = pd.DataFrame(per_block)
    pb["roc_gain"] = pb["hw_offset0h"] - pb["om_only"]
    pb.to_csv(OUTPUT_DIR / "th_v7_per_block.csv", index=False)
    log("=== per-block gain (h1_1.0mm) ===")
    print(pb[pb["target"] == "h1_1.0mm"].round(4).to_string(index=False))
    log(f"saved to {OUTPUT_DIR}")


# %%
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "probe"
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
    else:
        raise SystemExit(f"unknown phase {phase!r}; use probe | fetch | screen | "
                         f"build_features | cv | report")
    log(f"phase {phase} done in {(time.time()-t0)/60:.1f} min")
