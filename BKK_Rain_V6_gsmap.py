# %% [markdown]
# # BKK_Rain_V6 — GSMaP: satellite *precipitation* at nowcast latency
#
# The V4 latency sweep set the requirement (observed-rain value halves per hour, dead by
# 3 h) and V5's gauges recovered only ~30% of the ceiling, capped by point-vs-cell
# sampling. GSMaP is the highest-probability way to reach the rest: it is actual satellite
# precipitation (same genus as the IMERG labels) on an hourly 0.1-degree grid covering every
# cell, with a real-time member of the family:
#
#   - GSMaP_NRT: ~4 h latency, deep archive on JAXA's servers — the *training* record.
#   - GSMaP_NOW: ~0 h latency in the Himawari region — the *deployment* feed.
#
# V6 trains on the NRT archive at offsets 0/1 to emulate a NOW-fed deployment. The stated
# caveat: NOW is an extrapolated, lower-quality cousin of NRT, so the offset-0 result is an
# upper bound on a NOW-fed model; the offset-1 config is the conservative reading. (Feeding
# NRT itself at its honest ~4 h latency is already answered by the V4 curve: worthless.)
#
# | config | features | question |
# |---|---|---|
# | `om_only` (from V3, not re-run) | 66 Open-Meteo | control |
# | `gs_offset0h` | 66 + 16 GSMaP history (freshest stamp t-1) | the NOW-emulating gain |
# | `gs_offset1h` | 66 + 16 GSMaP history (freshest stamp t-2) | conservative reading |
# | `gs_tmd_offset0h` | 66 + 16 GSMaP + 16 gauge (V5 block) | the full deployable stack |
#
# Reference points on identical rows: IMERG offset 0 = +.017..+.050 ROC (ceiling), IMERG
# offset 1 = +.007..+.021, gauges (V5) = +.001..+.015. If `gs_offset0h` lands in the IMERG
# offset-0..1 band, the ceiling is reachable with a real-time-honest feed and V6 is the
# model to ship; Himawari IR (V7 script) then becomes optional.
#
# Data access: JAXA P-Tree / EORC "Global Rainfall Watch" FTP, free registration at
# https://sharaku.eorc.jaxa.jp/GSMaP/registration.html — set GSMAP_FTP_USER /
# GSMAP_FTP_PASS (and optionally GSMAP_FTP_HOST) before the fetch phases. Files are
# global 0.1-degree hourly flat binaries (3600 x 1200 float32, ~2-4 MB gzipped); only a
# 16 x 13-pixel Bangkok window is kept, so persistent storage is ~50 MB for 5 years.
#
# Phases:
#     python BKK_Rain_V6_gsmap.py probe          # FTP layout discovery + one-file decode (~5 min)
#     python BKK_Rain_V6_gsmap.py fetch [START END]   # hourly files, resumable; optional date
#                                                     # range, e.g. fetch 2025-05-01 2025-11-01
#     python BKK_Rain_V6_gsmap.py screen         # NO-TRAINING go/no-go: rank-skill of raw GSMaP
#                                                # lag-1 vs gauge/IMERG persistence on fetched rows
#     python BKK_Rain_V6_gsmap.py build_features # windows -> panel blocks (~5 min)
#     python BKK_Rain_V6_gsmap.py cv             # 3 configs x 6 folds x 6 targets (~2 h)
#     python BKK_Rain_V6_gsmap.py report         # vs om_only + IMERG/gauge references
#
# Recommended pilot before committing to the full 1-2 day fetch (~5% of the download):
#     probe  ->  fetch 2025-05-01 2025-11-01  (~4,400 files, 2-3 h)  ->  screen
# If screen puts GSMaP lag-1 ROC at or above the gauge band (0.72-0.79 at h1), the full
# fetch is justified; near 0.60-0.65 (the 6 h-stale IMERG level), kill it.

# %%
import gc
import gzip
import json
import os
import sys
import time
from ftplib import FTP
from io import BytesIO

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
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v6_gsmap"
GS_DIR = CACHE / "gsmap"                 # per-day Bangkok windows + fetch config

FTP_HOST = os.getenv("GSMAP_FTP_HOST", "hokusai.eorc.jaxa.jp")
FTP_USER = os.getenv("GSMAP_FTP_USER", "")
FTP_PASS = os.getenv("GSMAP_FTP_PASS", "")
# Candidate directory layouts, newest algorithm first; the probe discovers which exist and
# in what version, and records the working template(s) per date range in fetch_config.json.
PATH_TEMPLATES = [
    "/realtime_ver/v8/hourly/{y}/{m:02d}/{d:02d}/gsmap_now.{y}{m:02d}{d:02d}.{h:02d}00.dat.gz",
    "/realtime_ver/v8/hourly/{y}/{m:02d}/{d:02d}/gsmap_nrt.{y}{m:02d}{d:02d}.{h:02d}00.dat.gz",
    "/realtime_ver/v7/hourly/{y}/{m:02d}/{d:02d}/gsmap_nrt.{y}{m:02d}{d:02d}.{h:02d}00.dat.gz",
    "/realtime/hourly/{y}/{m:02d}/{d:02d}/gsmap_nrt.{y}{m:02d}{d:02d}.{h:02d}00.dat.gz",
]

UTC_OFFSET_HOURS = 7                     # panel is Asia/Bangkok local time
GS_NLON, GS_NLAT = 3600, 1200           # global 0.1 deg: lon 0.05..359.95, lat 59.95..-59.95
WIN_LAT = (12.9, 14.5)                  # Bangkok window, matches V7's crop
WIN_LON = (100.0, 101.3)
RAIN_MM = 0.1
GS_LOOKBACK = 26                        # deepest stamp: offset 1 + 24 h rolling sum
GS_OFFSETS = [0, 1]


def gsmap_feature_names():
    return [
        "gs_precip_lag1", "gs_precip_lag2", "gs_precip_lag3", "gs_precip_lag6",
        "gs_precip_sum3", "gs_precip_sum6", "gs_precip_sum12", "gs_precip_sum24",
        "gs_rain_lag1", "gs_rain_count3", "gs_rain_count6", "gs_hours_since_rain6",
        "gs_nbr_mean_lag1", "gs_nbr_max_lag1", "gs_far_max_lag1", "gs_valid",
    ]


def window_indices():
    """Row/col slices of the Bangkok window in the global grid (rows run north->south)."""
    lat_hi = int(round((59.95 - WIN_LAT[1]) / 0.1))
    lat_lo = int(round((59.95 - WIN_LAT[0]) / 0.1)) + 1
    lon_lo = int(round((WIN_LON[0] - 0.05) / 0.1))
    lon_hi = int(round((WIN_LON[1] - 0.05) / 0.1)) + 1
    return slice(lat_hi, lat_lo), slice(lon_lo, lon_hi)


def decode_global(buf):
    """Gzipped flat little-endian float32 global grid -> (1200, 3600); negatives = missing."""
    raw = gzip.decompress(buf)
    grid = np.frombuffer(raw, dtype="<f4")
    if grid.size != GS_NLON * GS_NLAT:
        raise ValueError(f"unexpected grid size {grid.size}")
    grid = grid.reshape(GS_NLAT, GS_NLON).copy()
    grid[grid < 0] = np.nan
    return grid


def fetch_window(ftp, template, t_utc):
    ts = pd.Timestamp(t_utc)
    path = template.format(y=ts.year, m=ts.month, d=ts.day, h=ts.hour)
    buf = BytesIO()
    ftp.retrbinary(f"RETR {path}", buf.write)
    grid = decode_global(buf.getvalue())
    rows, cols = window_indices()
    return grid[rows, cols].astype("float32")


def needed_utc_hours():
    tf = np.load(CACHE / "forecast_time.npy")
    local = pd.DatetimeIndex(tf).unique().sort_values()
    start = local.min() - pd.Timedelta(hours=GS_LOOKBACK)
    hours = pd.date_range(start, local.max(), freq="h")
    return hours - pd.Timedelta(hours=UTC_OFFSET_HOURS)


def connect_ftp():
    if not FTP_USER:
        raise SystemExit("set GSMAP_FTP_USER / GSMAP_FTP_PASS (register at "
                         "https://sharaku.eorc.jaxa.jp/GSMaP/registration.html)")
    ftp = FTP(FTP_HOST, timeout=60)
    ftp.login(FTP_USER, FTP_PASS)
    return ftp


# %% [markdown]
# ## Phase `probe` — discover the server layout, decode one file, sanity-check the window
#
# JAXA has moved directories across algorithm versions (v7 -> v8, nrt -> now naming), so
# the probe tries each candidate template on three dates spanning the panel (early, middle,
# late) and records which template works for which era. It then decodes one wet-season file
# and prints the Bangkok window stats — a monsoon afternoon should show nonzero rain.

# %%
def phase_probe():
    hours = needed_utc_hours()
    probes = [hours[10], hours[len(hours) // 2], hours[-10]]
    ftp = connect_ftp()
    working = []
    for t in probes:
        found = None
        for template in PATH_TEMPLATES:
            try:
                window = fetch_window(ftp, template, t)
                found = template
                log(f"  {t}  OK via {template.split('/')[1:3]}  "
                    f"window {window.shape} finite {np.isfinite(window).mean():.0%} "
                    f"max {np.nanmax(window):.2f} mm/h")
                break
            except Exception:
                continue
        if found is None:
            log(f"  {t}  NO template worked — check account/paths with an FTP client")
        working.append({"probe_utc": str(t), "template": found})
    ftp.quit()
    if not any(w["template"] for w in working):
        raise SystemExit("no candidate path template matched — inspect the FTP tree "
                         "manually and add the layout to PATH_TEMPLATES")
    GS_DIR.mkdir(parents=True, exist_ok=True)
    (GS_DIR / "fetch_config.json").write_text(json.dumps(
        {"host": FTP_HOST, "working": working,
         "templates": PATH_TEMPLATES}, indent=2), encoding="utf-8")
    log("probe done — template map saved; run fetch next")


# %% [markdown]
# ## Phase `fetch` — resumable day-by-day sweep of the NRT archive
#
# One .npz per UTC day holding the 24 hourly Bangkok windows (~20 KB/day); existing day
# files are skipped so the phase can be killed and relaunched freely. Each slot tries the
# templates in order (probe-verified first), so version boundaries are handled per file.
# Failed slots stay NaN and are logged — same missingness convention as V5/V7.

# %%
def phase_fetch(start=None, end=None):
    cfg = json.loads((GS_DIR / "fetch_config.json").read_text(encoding="utf-8"))
    preferred = [w["template"] for w in cfg["working"] if w["template"]]
    templates = list(dict.fromkeys(preferred + PATH_TEMPLATES))

    hours = needed_utc_hours()
    if start is not None:
        hours = hours[hours >= pd.Timestamp(start)]
    if end is not None:
        hours = hours[hours < pd.Timestamp(end)]
    days = pd.DatetimeIndex(hours.normalize()).unique().sort_values()
    rows, cols = window_indices()
    n_lat = rows.stop - rows.start
    n_lon = cols.stop - cols.start
    log(f"UTC hours {len(hours):,} across {len(days):,} days; window {n_lat}x{n_lon} px")

    ftp = connect_ftp()
    done = failed = 0
    t_start = time.time()
    for day in days:
        day_file = GS_DIR / f"day_{pd.Timestamp(day):%Y%m%d}.npz"
        if day_file.exists():
            continue
        day_hours = hours[(hours >= day) & (hours < day + pd.Timedelta(days=1))]
        windows = np.full((len(day_hours), n_lat, n_lon), np.nan, dtype="float32")
        for j, t_utc in enumerate(day_hours):
            for attempt in range(2):                      # one reconnect per slot on drop
                try:
                    for template in templates:
                        try:
                            windows[j] = fetch_window(ftp, template, t_utc)
                            break
                        except Exception:
                            continue
                    else:
                        failed += 1
                    break
                except (ConnectionError, EOFError, OSError):
                    try:
                        ftp.quit()
                    except Exception:
                        pass
                    ftp = connect_ftp()
            else:
                failed += 1
        np.savez_compressed(day_file, hours=day_hours.to_numpy(dtype="datetime64[s]"),
                            windows=windows)
        done += 1
        rate = done / max((time.time() - t_start) / 3600, 1e-9)
        print(f"  {pd.Timestamp(day):%Y-%m-%d}  days done {done:,}  "
              f"failed slots {failed:,}  {rate:.1f} days/h", flush=True)
    ftp.quit()
    log(f"fetch complete: {done:,} new days, {failed:,} failed slots")


# %% [markdown]
# ## Phase `screen` — the go/no-go pilot test: no training, minutes of compute
#
# Works on whatever day files exist (intended after a wet-season pilot fetch). Scores raw
# GSMaP lag-1 values as rank-predictors of the six targets, next to the two persistence
# yardsticks already cached by V3/V5 — fresh IMERG (`persist_research.npy`, the ceiling's
# raw signal) and gauge IDW (`tg_persist_offset0.npy`, V5's raw signal) — restricted to the
# SAME rows, so the comparison is apples-to-apples. Yardsticks at h1: fresh IMERG ROC
# 0.79-0.85, gauges 0.72-0.79, 6 h-stale IMERG 0.59-0.65. GSMaP at or above the gauge band
# justifies the full fetch; at the stale-IMERG level, kill V6.

# %%
def phase_screen():
    from sklearn.metrics import average_precision_score, roc_auc_score
    hours_local, s = hourly_cell_series()
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
        "gsmap_center_lag1": shift_down(s["center"], 1),
        "gsmap_nbr_max_lag1": shift_down(s["nbr_max"], 1),
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

    valid = np.isfinite(scores["gsmap_center_lag1"])
    log(f"screen rows (GSMaP lag-1 available): {valid.sum():,} / {len(tf):,} "
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
    out.to_csv(OUTPUT_DIR / "v6_screen.csv", index=False)
    pivot = out.pivot_table(index="target", columns="score", values="roc_auc")
    log("=== screening ROC-AUC on identical rows (no training) ===")
    print(pivot.round(4).to_string())
    log(f"saved to {OUTPUT_DIR / 'v6_screen.csv'}")


# %% [markdown]
# ## Phase `build_features` — Bangkok windows -> hourly cell series -> panel blocks
#
# Three hourly series per cell: precip at the cell's own 0.1-degree pixel, mean and max over
# the surrounding 3x3 pixels (~30 km), and max over 7x7 (~70 km, approaching-storm signal).
# The 16-feature block then mirrors V3's IMERG-history block exactly (same lag/sum/count
# construction, V3 stamp convention: offset o => freshest stamp t-(o+1)), plus the
# neighbourhood features and a validity flag.

# %%
def shift_down(arr, k):
    out = np.full_like(arr, np.nan)
    if k < len(arr):
        out[k:] = arr[:-k] if k else arr
    return out


def hourly_cell_series():
    files = sorted(GS_DIR.glob("day_*.npz"))
    if not files:
        raise SystemExit("no fetched day files — run probe + fetch first")
    hours_list, win_list = [], []
    for f in files:
        d = np.load(f)
        hours_list.append(d["hours"])
        win_list.append(d["windows"])
    hours_utc = np.concatenate(hours_list)
    windows = np.concatenate(win_list)                    # (n_hours, n_lat, n_lon)
    order = np.argsort(hours_utc)
    hours_utc, windows = hours_utc[order], windows[order]
    hours_local = hours_utc + np.timedelta64(UTC_OFFSET_HOURS, "h")
    log(f"windows {windows.shape}, finite share {np.isfinite(windows).mean():.1%}")

    grid = pd.read_csv(CACHE / "tg_grid.csv")             # written by V5's load_gauges
    rows, cols = window_indices()
    cell_r = ((59.95 - grid["latitude"].to_numpy()) / 0.1).round().astype(int) - rows.start
    cell_c = ((grid["longitude"].to_numpy() - 0.05) / 0.1).round().astype(int) - cols.start
    n_hours, n_lat, n_lon = windows.shape

    def patch_stat(radius, fn):
        out = np.empty((n_hours, len(grid)), dtype="float32")
        for i, (r, c) in enumerate(zip(cell_r, cell_c)):
            r0, r1 = max(r - radius, 0), min(r + radius + 1, n_lat)
            c0, c1 = max(c - radius, 0), min(c + radius + 1, n_lon)
            with np.errstate(all="ignore"):
                out[:, i] = fn(windows[:, r0:r1, c0:c1])
        return out

    center = windows[:, np.clip(cell_r, 0, n_lat - 1), np.clip(cell_c, 0, n_lon - 1)]
    nbr_mean = patch_stat(1, lambda w: np.nanmean(w, axis=(1, 2)))
    nbr_max = patch_stat(1, lambda w: np.nanmax(w, axis=(1, 2)))
    far_max = patch_stat(3, lambda w: np.nanmax(w, axis=(1, 2)))
    return hours_local, {"center": center.astype("float32"), "nbr_mean": nbr_mean,
                         "nbr_max": nbr_max, "far_max": far_max}


def phase_build_features():
    hours_local, s = hourly_cell_series()
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

    center = s["center"]
    for offset in GS_OFFSETS:
        o = offset
        lag = {k: shift_down(center, o + k) for k in (1, 2, 3, 6)}
        stack = np.stack([shift_down(center, o + k) for k in range(1, 25)])
        nan_as_zero = np.nan_to_num(stack, nan=0.0)
        sums = {w: nan_as_zero[:w].sum(axis=0) for w in (3, 6, 12, 24)}
        for w in (3, 6, 12, 24):        # all-missing windows revert to NaN, not a dry 0
            sums[w][np.isnan(stack[:w]).all(axis=0)] = np.nan
        rainy = np.stack([(shift_down(center, o + k) >= RAIN_MM).astype("float32")
                          for k in range(1, 7)])
        any6 = rainy.any(axis=0)
        hours_since = np.where(any6, rainy.argmax(axis=0).astype("float32"), 6.0)
        hourly = np.stack([
            lag[1], lag[2], lag[3], lag[6],
            sums[3], sums[6], sums[12], sums[24],
            rainy[0], rainy[:3].sum(axis=0), rainy.sum(axis=0), hours_since,
            shift_down(s["nbr_mean"], o + 1), shift_down(s["nbr_max"], o + 1),
            shift_down(s["far_max"], o + 1),
            np.isfinite(lag[1]).astype("float32"),
        ], axis=-1).astype("float32")                     # (n_hours, n_cells, 16)
        block = hourly[row_hour, row_cell]
        block[outside] = np.nan
        block[outside, -1] = 0.0
        np.save(CACHE / f"gs_block_offset{offset}.npy", block)
        share = pd.Series(np.isnan(block).mean(axis=0), index=gsmap_feature_names())
        log(f"offset {offset}: block {block.shape}, NaN share:")
        print(share.round(4).to_string())
        del hourly, block
        gc.collect()

    # persistence anchor: GSMaP precip at the freshest deployable stamp
    anchor_hourly = shift_down(center, 1)
    anchor = anchor_hourly[row_hour, row_cell].astype("float32")
    anchor[outside] = np.nan
    np.save(CACHE / "gs_persist_offset0.npy", anchor)
    log("saved gsmap blocks + persistence anchor")


# %% [markdown]
# ## Phase `cv` — V3 protocol; GSMaP at 0/1 h, then GSMaP + gauges

# %%
CONFIG_BLOCKS = {
    "gs_offset0h": [("gs_block_offset0.npy", gsmap_feature_names)],
    "gs_offset1h": [("gs_block_offset1.npy", gsmap_feature_names)],
    "gs_tmd_offset0h": [("gs_block_offset0.npy", gsmap_feature_names),
                        ("tg_block_offset0.npy", gauge_feature_names)],
}


def run_gs_cv(config, y, forecast_time, year_key, year_folds):
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
    results_path = OUTPUT_DIR / "v6_cv_results.csv"
    done = set()
    if results_path.exists():
        prior = pd.read_csv(results_path)
        all_results = prior.to_dict(orient="records")
        done = set(prior["config"].unique())
        log(f"resuming; already done: {sorted(done)}")
    for config in CONFIG_BLOCKS:
        if config in done:
            continue
        results, fit_log = run_gs_cv(config, y, forecast_time, year_key, year_folds)
        all_results += results
        all_fitlog += fit_log
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        pd.DataFrame(all_fitlog).to_csv(OUTPUT_DIR / "v6_fit_log.csv", index=False)
    log("cv phase complete")


# %% [markdown]
# ## Phase `report` — GSMaP vs the ceiling, gauges, and the stale floor, on identical rows

# %%
def phase_report():
    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    stamps = pd.to_datetime(forecast_time)
    year_key = stamps.year.to_numpy()
    year_folds = [int(v) for v in np.unique(year_key) if (year_key == v).sum() >= MIN_FOLD_ROWS]

    v6 = pd.read_csv(OUTPUT_DIR / "v6_cv_results.csv")
    v3 = pd.read_csv(V3_OUTPUT_DIR / "v3_oof_results.csv")
    v5 = pd.read_csv(V5_OUTPUT_DIR / "v5_cv_results.csv")
    refs = pd.concat([
        v3[v3["config"].isin(["om_only", "imerg_research", "imerg_deploy"])],
        v5[v5["config"] == "tmd_offset0h"],
    ], ignore_index=True).replace(
        {"config": {"imerg_research": "imerg_offset0h", "imerg_deploy": "imerg_offset6h"}})
    combined = pd.concat([refs, v6], ignore_index=True)
    cal = combined[combined["probabilities"] == "calibrated"]

    om_roc = cal[cal["config"] == "om_only"].set_index("target")["roc_auc"]
    om_f1 = cal[cal["config"] == "om_only"].set_index("target")["f1"]
    col_order = [c for c in ["gs_offset0h", "gs_offset1h", "gs_tmd_offset0h",
                             "tmd_offset0h", "imerg_offset0h", "imerg_offset6h"]
                 if c in cal["config"].unique()]
    roc_gain = cal.pivot_table(index="target", columns="config",
                               values="roc_auc")[col_order].sub(om_roc, axis=0)
    f1_gain = cal.pivot_table(index="target", columns="config",
                              values="f1")[col_order].sub(om_f1, axis=0)

    anchor = np.load(CACHE / "gs_persist_offset0.npy")
    probs = np.repeat(anchor[:, None], len(TARGETS), axis=1)
    persist_df = pd.DataFrame(score_probabilities(
        probs, y, year_key, year_folds, "amount", "gs_persist_offset0"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "v6_results_combined.csv", index=False)
    roc_gain.to_csv(OUTPUT_DIR / "v6_roc_gain_vs_references.csv")
    f1_gain.to_csv(OUTPUT_DIR / "v6_f1_gain_vs_references.csv")
    persist_df.to_csv(OUTPUT_DIR / "v6_gsmap_persistence.csv", index=False)

    metadata = {
        "purpose": "GSMaP hourly satellite precipitation as the area-covering nowcast-latency "
                   "feed; measures how much of the IMERG offset-0 ceiling a NOW-emulating "
                   "GSMaP history recovers, alone and stacked with V5's gauges",
        "protocol": "identical to V3/V4/V5 (imported): leave-one-year-out CV, 24 h purge, "
                    "isotonic calibration, IMERG labels, F1-optimal leave-one-year-out thresholds",
        "source": f"JAXA P-Tree/EORC FTP ({FTP_HOST}), GSMaP NRT archive, hourly 0.1 deg, "
                  f"Bangkok window lat {WIN_LAT} lon {WIN_LON}",
        "gsmap_features": gsmap_feature_names(),
        "configs": {c: [f for _, names in CONFIG_BLOCKS[c] for f in names()]
                    for c in CONFIG_BLOCKS},
        "roc_gain_vs_references": roc_gain.reset_index().to_dict(orient="records"),
        "f1_gain_vs_references": f1_gain.reset_index().to_dict(orient="records"),
        "gsmap_persistence": persist_df.to_dict(orient="records"),
        "caveats": [
            "Training features are the GSMaP_NRT archive; the deployment feed is GSMaP_NOW, "
            "an extrapolated lower-quality product. gs_offset0h is therefore an upper bound "
            "on a NOW-fed model; gs_offset1h is the conservative reading. A pre-deployment "
            "check should compare NOW vs NRT values over a live month.",
            "GSMaP and IMERG estimate rain from an overlapping satellite constellation, so "
            "some of the measured gain is grader affinity; that affinity also exists in "
            "deployment (the labels ARE IMERG), so it is deployment-relevant, not spurious.",
            "GSMaP algorithm versions change across the archive (v7 -> v8); the fetch "
            "records which template served each day, and a version-boundary breakout is "
            "possible from fetch_config.json + day file dates if results look regime-y.",
            "Slots that failed to download stay NaN (gs_valid=0).",
        ],
    }
    (OUTPUT_DIR / "v6_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log("=== ROC-AUC gain over om_only: GSMaP vs every reference ===")
    print(roc_gain.round(4).to_string())
    log("=== F1 gain over om_only ===")
    print(f1_gain.round(4).to_string())
    log("=== GSMaP persistence alone (center-pixel lag-1 amount as score) ===")
    print(persist_df[["target", "roc_auc", "pr_auc_lift", "f1"]].round(4).to_string(index=False))
    log(f"saved V6 outputs to {OUTPUT_DIR}")


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
    else:
        raise SystemExit(f"unknown phase {phase!r}; use probe | fetch | screen | "
                         f"build_features | cv | report")
    log(f"phase {phase} done in {(time.time()-t0)/60:.1f} min")
