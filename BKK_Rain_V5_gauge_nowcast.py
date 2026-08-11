# %% [markdown]
# # BKK_Rain_V5 — can TMD gauges deliver the low-latency observed-rain gain?
#
# The V4 latency sweep made the requirement quantitative: observed-rain history loses
# roughly half its value per hour of latency and is worthless by 3 h. IMERG Early Run
# (~4 h) is disqualified. TMD gauges report within minutes, so on an hourly panel the
# freshest complete gauge hour is stamp t-1 — the "offset 0" column of the V4 curve,
# worth +0.02..+0.05 ROC if gauge rain carries the same signal as satellite rain.
#
# That is the open question V5 answers. Gauges measure a point; the IMERG label is an
# ~11 km cell average, and the two records overlap on only 11–19% of rainy hours. So the
# gauge block may recover the full offset-0 gain, some of it, or none. Everything else
# (panel, labels, CV, purge, calibration, thresholds, scoring) is imported from V3 so the
# feature block is the only variable, exactly as in V4.
#
# | config | features | question |
# |---|---|---|
# | `om_only` (from V3, not re-run) | 66 Open-Meteo | control |
# | `tmd_offset0h` | 66 + 16 gauge-history (freshest stamp t-1) | the deployable gauge gain |
# | `tmd_offset1h` | 66 + 16 gauge-history (freshest stamp t-2) | insurance against stamp-semantics optimism |
#
# Reference points from V3/V4 on identical rows: IMERG history at offset 0 gains
# +.017..+.050 ROC; at offset 1, +.007..+.021. If `tmd_offset0h` lands near the IMERG
# offset-0..1 band, gauges are the real-time feed; if it lands near zero, the
# point-vs-cell mismatch eats the signal and the next candidate is radar/Himawari.
#
# Gauge features per grid cell (16, prefix `tg_`): inverse-distance-weighted (IDW) precip
# over stations within RADIUS_KM — lags 1/2/3/6, rolling sums 3/6/12/24, rain flag, rain
# counts 3/6, hours-since-rain (mirroring V3's 12 IMERG-history features) — plus
# nearest-station precip, max and rainy-fraction within radius, and stations-reporting
# count (coverage is 82% with long outages, so missingness is itself signal).
#
# Run as memory-bounded phases (~3.6 GB free on this box):
#     python BKK_Rain_V5_gauge_nowcast.py load_gauges     # station matrix from Postgres (~3 min)
#     python BKK_Rain_V5_gauge_nowcast.py build_features  # hourly cell features -> panel blocks (~5 min)
#     python BKK_Rain_V5_gauge_nowcast.py cv              # 2 configs x 6 folds x 6 targets (~40 min each)
#     python BKK_Rain_V5_gauge_nowcast.py report          # vs om_only + V3/V4 reference points

# %%
import gc
import json
import sys
import time

import numpy as np
import pandas as pd

from BKK_Rain_V3 import (
    CACHE, FEATURE_COLUMNS, MIN_FOLD_ROWS, MODEL_PLAN, PROJECT_ROOT, PURGE_HOURS,
    RANDOM_STATE, TARGETS, TARGET_NAMES,
    connect, fit_predict_calibrated, log, purged_train_mask, score_probabilities,
)

V3_OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v3_imerg_history"
V4_OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v4_latency_sweep"
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v5_gauge_nowcast"

TMD_TABLE_NAME = '"BKK_TMD_WEATHER_DATA"'
GRID_TABLE_NAME = '"Bangkok_Grid_9km"'
UTC_OFFSET_HOURS = 7      # panel local_forecast_time is Asia/Bangkok; TMD utc_time is UTC
RADIUS_KM = 150.0         # only 5 stations sit within ~30 km of the grid; 150 km reaches ~20
IDW_POWER = 2.0
RAIN_MM = 0.1
MAX_GAUGE_LAG = 30        # deepest stamp needed: offset 1 + 24 h rolling sum ends at t-26
GAUGE_OFFSETS = [0, 1]    # 0 = freshest stamp t-1 (honest for minutes-latency gauges)


def gauge_feature_names():
    return [
        "tg_idw_lag1", "tg_idw_lag2", "tg_idw_lag3", "tg_idw_lag6",
        "tg_idw_sum3", "tg_idw_sum6", "tg_idw_sum12", "tg_idw_sum24",
        "tg_rain_lag1", "tg_rain_count3", "tg_rain_count6", "tg_hours_since_rain6",
        "tg_near_lag1", "tg_max_lag1", "tg_rainfrac_lag1", "tg_nrep_lag1",
    ]


# %% [markdown]
# ## Phase `load_gauges` — station-hour precip matrix on a gap-free hourly local index
#
# Negative precip values are sentinels (min in the table is −3174 mm): nulled, not clipped,
# per the TMD_BKK_V1 finding that clipping fabricates dry hours. Duplicate station-hours are
# averaged. The matrix is reindexed onto a gap-free hourly grid so a positional shift never
# splices across an outage — missing hours stay NaN.

# %%
def phase_load_gauges():
    with connect() as conn:
        stations = pd.read_sql(
            f"""SELECT station, avg(latitude) AS lat, avg(longitude) AS lon
                FROM {TMD_TABLE_NAME} GROUP BY station ORDER BY station""", conn)
        obs = pd.read_sql(
            f"""SELECT station,
                       utc_time + interval '{UTC_OFFSET_HOURS} hour' AS t_local,
                       avg(precipitation_mm) AS precip_mm
                FROM {TMD_TABLE_NAME}
                WHERE precipitation_mm >= 0
                GROUP BY station, t_local""", conn)
        grid = pd.read_sql(
            f"SELECT grid_number, latitude, longitude FROM {GRID_TABLE_NAME} "
            f"ORDER BY grid_number", conn)
    log(f"stations {len(stations)}, obs rows {len(obs):,}, grid cells {len(grid)}")

    obs["t_local"] = pd.to_datetime(obs["t_local"])
    wide = obs.pivot(index="t_local", columns="station", values="precip_mm")
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    wide = wide.reindex(full_index)[stations["station"].tolist()]
    log(f"gauge matrix {wide.shape} ({wide.index.min()} .. {wide.index.max()}), "
        f"non-null {np.isfinite(wide.to_numpy(dtype='float32')).mean():.1%}")

    np.save(CACHE / "tg_station_matrix.npy", wide.to_numpy(dtype="float32"))
    np.save(CACHE / "tg_hour_index.npy", wide.index.to_numpy(dtype="datetime64[s]"))
    stations.to_csv(CACHE / "tg_stations.csv", index=False)
    grid.to_csv(CACHE / "tg_grid.csv", index=False)
    log("saved gauge matrix + station/grid coordinates")


# %% [markdown]
# ## Phase `build_features` — hourly per-cell gauge series, then panel-aligned blocks
#
# Five hourly per-cell series are built first (IDW precip, nearest-station precip, max and
# rainy-fraction within radius, stations-reporting count), each `(n_hours, 56)`. All lag and
# rolling features are then positional shifts of those series, mirroring V3's
# `build_history_block` stamp convention: with offset `o`, the freshest stamp used is
# t-(o+1), lags are stamps t-(o+k), and rolling sums span stamps t-(o+1)..t-(o+w).

# %%
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def hourly_cell_series():
    v = np.load(CACHE / "tg_station_matrix.npy")            # (n_hours, n_stations)
    stations = pd.read_csv(CACHE / "tg_stations.csv")
    grid = pd.read_csv(CACHE / "tg_grid.csv")
    dist = haversine_km(grid["latitude"].to_numpy()[:, None],
                        grid["longitude"].to_numpy()[:, None],
                        stations["lat"].to_numpy()[None, :],
                        stations["lon"].to_numpy()[None, :])   # (n_cells, n_stations)
    in_radius = dist <= RADIUS_KM
    log(f"stations within {RADIUS_KM:.0f} km per cell: "
        f"min {in_radius.sum(1).min()}, max {in_radius.sum(1).max()}")
    weights = np.where(in_radius, 1.0 / np.maximum(dist, 1.0) ** IDW_POWER, 0.0)

    n_hours, n_cells = v.shape[0], len(grid)
    finite = np.isfinite(v)
    v_zero = np.where(finite, v, 0.0)

    # IDW numerator/denominator with per-hour missing-station masking
    numer = v_zero @ weights.T                                # (n_hours, n_cells)
    denom = finite.astype("float32") @ weights.T.astype("float32")
    idw = np.where(denom > 0, numer / np.maximum(denom, 1e-12), np.nan).astype("float32")

    nrep = (finite.astype("float32") @ in_radius.T.astype("float32")).astype("float32")
    rainy_st = (finite & (v >= RAIN_MM)).astype("float32")
    rainfrac = np.where(nrep > 0, (rainy_st @ in_radius.T.astype("float32")) / np.maximum(nrep, 1), np.nan).astype("float32")

    mx = np.full((n_hours, n_cells), np.nan, dtype="float32")
    near = np.full((n_hours, n_cells), np.nan, dtype="float32")
    order = np.argsort(dist, axis=1)
    for c in range(n_cells):
        members = np.flatnonzero(in_radius[c])
        if len(members):
            sub = v[:, members]
            with np.errstate(all="ignore"):
                mx[:, c] = np.nanmax(sub, axis=1)
        # nearest *reporting* station within radius, searched in distance order
        for s in order[c]:
            if not in_radius[c, s]:
                break
            near_col = v[:, s]
            fill = np.isnan(near[:, c]) & np.isfinite(near_col)
            near[fill, c] = near_col[fill]
    return {"idw": idw, "near": near, "mx": mx, "rainfrac": rainfrac, "nrep": nrep}


def shift_down(arr, k):
    """Value at stamp t-k aligned to row t (rows are consecutive hours)."""
    out = np.full_like(arr, np.nan)
    if k < len(arr):
        out[k:] = arr[:-k] if k else arr
    return out


def build_gauge_block_hourly(series, offset):
    idw = series["idw"]
    o = offset
    lag = {k: shift_down(idw, o + k) for k in (1, 2, 3, 6)}
    stack = np.stack([shift_down(idw, o + k) for k in range(1, 25)])  # stamps t-(o+1)..t-(o+24)
    nan_as_zero = np.nan_to_num(stack, nan=0.0)
    sums = {w: nan_as_zero[:w].sum(axis=0) for w in (3, 6, 12, 24)}
    # windows with no reporting hour at all revert to NaN rather than a fabricated dry 0
    for w in (3, 6, 12, 24):
        sums[w][np.isnan(stack[:w]).all(axis=0)] = np.nan
    rainy = np.stack([(shift_down(idw, o + k) >= RAIN_MM).astype("float32")
                      for k in range(1, 7)])                          # stamps t-(o+1)..t-(o+6)
    rain_count3 = rainy[:3].sum(axis=0)
    rain_count6 = rainy.sum(axis=0)
    any6 = rainy.any(axis=0)
    hours_since = np.where(any6, rainy.argmax(axis=0).astype("float32"), 6.0)
    block = np.stack([
        lag[1], lag[2], lag[3], lag[6],
        sums[3], sums[6], sums[12], sums[24],
        rainy[0], rain_count3, rain_count6, hours_since,
        shift_down(series["near"], o + 1), shift_down(series["mx"], o + 1),
        shift_down(series["rainfrac"], o + 1), shift_down(series["nrep"], o + 1),
    ], axis=-1).astype("float32")                                     # (n_hours, n_cells, 16)
    return block


def phase_build_features():
    series = hourly_cell_series()
    hour_index = np.load(CACHE / "tg_hour_index.npy")
    grid = pd.read_csv(CACHE / "tg_grid.csv")

    unit = np.load(CACHE / "unit_id.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    hour_pos = {np.datetime64(h): i for i, h in enumerate(hour_index)}
    cell_pos = {g: i for i, g in enumerate(grid["grid_number"])}
    row_hour = np.array([hour_pos.get(np.datetime64(t), -1) for t in tf], dtype="int64")
    row_cell = np.array([cell_pos[int(g)] for g in unit], dtype="int64")
    outside = row_hour < 0
    log(f"panel rows outside gauge hour range: {outside.sum():,} "
        f"({outside.mean():.2%}) -> NaN features")
    row_hour[outside] = 0

    for offset in GAUGE_OFFSETS:
        hourly = build_gauge_block_hourly(series, offset)
        block = hourly[row_hour, row_cell]
        block[outside] = np.nan
        np.save(CACHE / f"tg_block_offset{offset}.npy", block)
        share = pd.Series(np.isnan(block).mean(axis=0), index=gauge_feature_names())
        log(f"offset {offset}: block {block.shape}, NaN share:")
        print(share.round(4).to_string())
        del hourly, block
        gc.collect()

    # persistence anchor: IDW gauge precip at the freshest deployable stamp
    anchor_hourly = shift_down(series["idw"], 1)
    anchor = anchor_hourly[row_hour, row_cell].astype("float32")
    anchor[outside] = np.nan
    np.save(CACHE / "tg_persist_offset0.npy", anchor)
    log("saved gauge blocks + persistence anchor")


# %% [markdown]
# ## Phase `cv` — V3 protocol per config, checkpointed and resumable (as V4)

# %%
def run_gauge_cv(offset, y, forecast_time, year_key, year_folds):
    block = np.load(CACHE / f"tg_block_offset{offset}.npy")
    base = np.load(CACHE / "base_X.npy")
    x = np.ascontiguousarray(np.concatenate([base, block], axis=1))
    del base, block
    gc.collect()

    config = f"tmd_offset{offset}h"
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
    results_path = OUTPUT_DIR / "v5_cv_results.csv"
    done = set()
    if results_path.exists():
        prior = pd.read_csv(results_path)
        all_results = prior.to_dict(orient="records")
        done = set(prior["config"].unique())
        log(f"resuming; already done: {sorted(done)}")
    for offset in GAUGE_OFFSETS:
        if f"tmd_offset{offset}h" in done:
            continue
        results, fit_log = run_gauge_cv(offset, y, forecast_time, year_key, year_folds)
        all_results += results
        all_fitlog += fit_log
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        pd.DataFrame(all_fitlog).to_csv(OUTPUT_DIR / "v5_fit_log.csv", index=False)
    log("cv phase complete")


# %% [markdown]
# ## Phase `report` — gauge gain vs the IMERG latency curve, on identical rows
#
# The verdict table puts `tmd_offset{0,1}h` next to three reference points from V3/V4:
# `om_only` (floor), IMERG offset 0–1 (the band a minutes-latency feed should reach), and
# IMERG offset 6 (the stale floor V3 shipped). Gauge persistence (IDW lag-1 amount as a
# rank score) says how much of any gain is just "a gauge nearby is wet right now".

# %%
def phase_report():
    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    stamps = pd.to_datetime(forecast_time)
    year_key = stamps.year.to_numpy()
    year_folds = [int(v) for v in np.unique(year_key) if (year_key == v).sum() >= MIN_FOLD_ROWS]

    v5 = pd.read_csv(OUTPUT_DIR / "v5_cv_results.csv")
    v3 = pd.read_csv(V3_OUTPUT_DIR / "v3_oof_results.csv")
    v4 = pd.read_csv(V4_OUTPUT_DIR / "v4_sweep_results.csv")
    refs = pd.concat([
        v3[v3["config"].isin(["om_only", "imerg_research", "imerg_deploy"])],
        v4[v4["config"] == "offset1h"].assign(config="imerg_offset1h"),
    ], ignore_index=True).replace(
        {"config": {"imerg_research": "imerg_offset0h", "imerg_deploy": "imerg_offset6h"}})
    combined = pd.concat([refs, v5], ignore_index=True)
    cal = combined[combined["probabilities"] == "calibrated"]

    om_roc = cal[cal["config"] == "om_only"].set_index("target")["roc_auc"]
    om_f1 = cal[cal["config"] == "om_only"].set_index("target")["f1"]
    roc = cal.pivot_table(index="target", columns="config", values="roc_auc")
    f1 = cal.pivot_table(index="target", columns="config", values="f1")
    col_order = [c for c in ["om_only", "tmd_offset0h", "tmd_offset1h",
                             "imerg_offset0h", "imerg_offset1h", "imerg_offset6h"]
                 if c in roc.columns]
    roc_gain = roc[col_order].sub(om_roc, axis=0).drop(columns="om_only")
    f1_gain = f1[col_order].sub(om_f1, axis=0).drop(columns="om_only")

    anchor = np.load(CACHE / "tg_persist_offset0.npy")
    probs = np.repeat(anchor[:, None], len(TARGETS), axis=1)
    persist_df = pd.DataFrame(score_probabilities(
        probs, y, year_key, year_folds, "amount", "tg_persist_offset0"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "v5_results_combined.csv", index=False)
    roc_gain.to_csv(OUTPUT_DIR / "v5_roc_gain_vs_references.csv")
    f1_gain.to_csv(OUTPUT_DIR / "v5_f1_gain_vs_references.csv")
    persist_df.to_csv(OUTPUT_DIR / "v5_gauge_persistence.csv", index=False)

    metadata = {
        "purpose": "TMD gauge history as the low-latency observed-rain feed the V4 sweep "
                   "calls for; measures how much of the IMERG offset-0/1 gain gauges "
                   "recover despite the point-vs-cell label mismatch",
        "protocol": "identical to V3/V4 (imported): leave-one-year-out CV, 24 h purge, "
                    "isotonic calibration, IMERG labels, F1-optimal leave-one-year-out "
                    "thresholds",
        "gauge_features": gauge_feature_names(),
        "n_features": len(FEATURE_COLUMNS) + len(gauge_feature_names()),
        "gauge_source": f"{TMD_TABLE_NAME}, 28 stations, negatives nulled, IDW within "
                        f"{RADIUS_KM:.0f} km power {IDW_POWER}",
        "offsets": GAUGE_OFFSETS,
        "roc_gain_vs_references": roc_gain.reset_index().to_dict(orient="records"),
        "f1_gain_vs_references": f1_gain.reset_index().to_dict(orient="records"),
        "gauge_persistence": persist_df.to_dict(orient="records"),
        "caveats": [
            "Gauge stamp semantics assumed: stamp t covers the hour ending at t, and the "
            "freshest deployable stamp at forecast time t is t-1. tmd_offset1h is the "
            "conservative reading; if the two disagree materially, verify TMD stamping "
            "before shipping.",
            "Gauge matrix ends 2026-06-24; panel rows after that carry NaN gauge features.",
            "Labels remain IMERG cells; a gauge-graded evaluation would answer a different "
            "question (see three-way label disagreement).",
        ],
    }
    (OUTPUT_DIR / "v5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log("=== ROC-AUC gain over om_only: gauges vs IMERG reference points ===")
    print(roc_gain.round(4).to_string())
    log("=== F1 gain over om_only ===")
    print(f1_gain.round(4).to_string())
    log("=== gauge persistence alone (IDW lag-1 amount as score) ===")
    print(persist_df[["target", "roc_auc", "pr_auc_lift", "f1"]].round(4).to_string(index=False))
    log(f"saved V5 outputs to {OUTPUT_DIR}")


# %%
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "cv"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "load_gauges":
        phase_load_gauges()
    elif phase == "build_features":
        phase_build_features()
    elif phase == "cv":
        phase_cv()
    elif phase == "report":
        phase_report()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use load_gauges | build_features | cv | report")
    log(f"phase {phase} done in {(time.time()-t0)/60:.1f} min")
