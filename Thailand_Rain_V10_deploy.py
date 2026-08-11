# %% [markdown]
# # Thailand_Rain_V10 — deploy fit: package V7 for real use
#
# Trains the `hw_offset0h` config (66 OM features + 12 Himawari IR features) on the FULL
# labeled panel — no CV holdout; the honest generalization numbers live in the V7 per-block
# table and are not re-estimated here — and packages one artifact per target in the same
# dict schema as the BKK V7 deploy: {model, calibrator, threshold, feature_names, ...}.
#
# What makes these artifacts *actually* deployable, unlike the BKK set when it was written:
#   - V8 (2026-08-11) showed the OM features are ECMWF IFS forecast-family data obtainable
#     live, and bounded the staleness cost at ≤0.015 ROC (hw config, +24 h-stale inputs;
#     real staleness ~7-13 h).
#   - V9 explained the 2026 gain decay as a warm-rain regime effect, not a defect —
#     with p(cold-top | rain) as the health metric to watch in production.
#
# Operating thresholds are F1-optimal on the calibrated V7 OOF probabilities over the
# IR-covered blocks — out-of-fold, so not contaminated by the full-panel refit.
#
# Phases:
#     python Thailand_Rain_V10_deploy.py fit       # ~6 fits, saves deploy/ artifacts
#     python Thailand_Rain_V10_deploy.py verify    # reload artifacts, sanity-score June 2026

# %%
import gc
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

from BKK_Rain_V3 import (
    CALIBRATION_FRACTION, FEATURE_COLUMNS, MODEL_PLAN, PROJECT_ROOT, RANDOM_STATE,
    TARGETS, TARGET_NAMES, log, pick_threshold,
)
from Thailand_Rain_V3 import CACHE, block_keys
from Thailand_Rain_V7_himawari_ir import himawari_feature_names

V7_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v7_himawari_ir"
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v10_deploy"
DEPLOY_DIR = OUTPUT_DIR / "deploy"
COVERED_FOLDS = [0, 1, 5, 6, 7, 11, 12]
ALL_FEATURES = FEATURE_COLUMNS + himawari_feature_names()


def fit_full(x, y_col, model_name, rng):
    """Full-panel fit with the exact V3 estimator params, returning (model, isotonic)."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    n = len(y_col)
    calib_size = int(n * CALIBRATION_FRACTION)
    shuffled = rng.permutation(n)
    calib_index, fit_index = shuffled[:calib_size], shuffled[calib_size:]

    if model_name == "lightgbm":
        scale_pos_weight = float((y_col[fit_index] == 0).sum()
                                 / max((y_col[fit_index] == 1).sum(), 1))
        model = LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.04, num_leaves=63,
            min_child_samples=80, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=250, max_leaf_nodes=31,
            l2_regularization=0.05, random_state=RANDOM_STATE, early_stopping=False,
        )
    model.fit(x[fit_index], y_col[fit_index])
    raw_calib = model.predict_proba(x[calib_index])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_calib, y_col[calib_index])
    return model, isotonic


# %%
def phase_fit():
    import lightgbm
    import sklearn

    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    base_x = np.load(CACHE / "base_X.npy")
    hw = np.load(CACHE / "hw_block_offset0.npy")
    x = np.ascontiguousarray(np.concatenate([base_x, hw], axis=1))
    del base_x, hw
    gc.collect()

    fold_key = block_keys(tf)
    in_scope = np.isin(fold_key, COVERED_FOLDS)
    oof = np.load(CACHE / "oof_cal_hw_offset0h.npy")

    # Full-panel training uses only IR-covered rows: outside them the hw features are all-NaN
    # and teach the trees nothing but a coverage indicator.
    x_fit = x[in_scope]
    log(f"deploy fit rows {in_scope.sum():,} of {len(y):,} (IR-covered blocks {COVERED_FOLDS})")

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, (horizon, threshold_mm) in enumerate(TARGETS):
        tname = TARGET_NAMES[i]
        out_path = DEPLOY_DIR / f"th_rain_{tname}.joblib"
        if out_path.exists():
            log(f"  {tname}: artifact exists, skipping (delete to refit)")
            continue
        t0 = time.time()
        m_oof = in_scope & np.isfinite(oof[:, i])
        op_threshold = pick_threshold(oof[m_oof, i], y[m_oof, i])
        rng = np.random.default_rng([RANDOM_STATE, 100, i])
        model, calibrator = fit_full(x_fit, y[in_scope, i], MODEL_PLAN[horizon], rng)
        joblib.dump({
            "model": model, "calibrator": calibrator, "threshold": op_threshold,
            "feature_names": ALL_FEATURES, "target": tname, "horizon_h": horizon,
            "rain_threshold_mm": threshold_mm, "trained_rows": int(in_scope.sum()),
            "notes": "Thailand V10 deploy fit; see v10_manifest.json for provenance/caveats",
        }, out_path, compress=3)
        mins = (time.time() - t0) / 60
        summary.append({"target": tname, "model": MODEL_PLAN[horizon],
                        "op_threshold": op_threshold, "trained_rows": int(in_scope.sum()),
                        "fit_minutes": round(mins, 2)})
        pd.DataFrame(summary).to_csv(OUTPUT_DIR / "v10_fit_summary.csv", index=False)
        log(f"  {tname}: threshold {op_threshold:.2f}, fit {mins:.1f} min -> {out_path.name}")

    manifest = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": "hw_offset0h (66 OM ecmwf_ifs features + 12 Himawari B13 IR features)",
        "training": f"full IR-covered panel ({int(np.isin(fold_key, COVERED_FOLDS).sum()):,} "
                    f"rows, blocks {COVERED_FOLDS}), 12% isotonic calibration holdout",
        "label": "IMERG cell-max precipitation over 25 km cell (39% provisional Late Run)",
        "thresholds": "F1-optimal on calibrated V7 OOF (out-of-fold, IR-covered blocks)",
        "generalization": "see om_thailand_rain_v7_himawari_ir/th_v7_per_block.csv; "
                          "headline h1_1.0mm OOF ROC 0.8217 (June 2026 single-month check)",
        "deployment_evidence": {
            "input_source": "V8: OM features are archived ECMWF IFS (= live forecast API "
                            "family), NOT ERA5; archive best_match == ecmwf_ifs here",
            "staleness_bound": "V8: +24h-stale inputs cost <=0.015 ROC on this config; real "
                               "staleness ~7-13h",
            "regime_dependence": "V9: IR gain tracks deep-convection share; monitor "
                                 "p(cold235>0 | rain) monthly (fell 0.85 -> 0.72 in 2026)",
        },
        "versions": {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                     "numpy": np.__version__},
        "serving_requirements": [
            "OM features from live forecast API (models=ecmwf_ifs), 30 h of hourly history "
            "per cell for lags/rolling sums, neighbor aggregates over the 25 km grid",
            "Himawari B13 slots t-1..t-4 via NOAA S3 (segments/masks in "
            "_th_cache/himawari/fetch_config.json)",
            "hour_sin/cos, month_sin/cos from local (Asia/Bangkok) issue time",
        ],
    }
    (OUTPUT_DIR / "v10_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")
    log(f"manifest + {len(summary)} artifacts in {DEPLOY_DIR}")


# %%
def phase_verify():
    from sklearn.metrics import roc_auc_score

    y = np.load(CACHE / "y.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    base_x = np.load(CACHE / "base_X.npy")
    hw = np.load(CACHE / "hw_block_offset0.npy")
    x = np.concatenate([base_x, hw], axis=1)
    del base_x, hw
    gc.collect()

    june = (tf >= np.datetime64("2026-06-01")) & (tf < np.datetime64("2026-07-01"))
    oof = np.load(CACHE / "oof_cal_hw_offset0h.npy")
    rows = []
    for i, tname in enumerate(TARGET_NAMES):
        d = joblib.load(DEPLOY_DIR / f"th_rain_{tname}.joblib")
        assert d["feature_names"] == ALL_FEATURES, f"{tname}: feature list mismatch"
        prob = d["calibrator"].predict(d["model"].predict_proba(x[june])[:, 1])
        m = np.isfinite(oof[june][:, i])
        rows.append({
            "target": tname, "june_rows": int(june.sum()),
            "roc_deploy_in_sample": float(roc_auc_score(y[june, i], prob)),
            "roc_oof_reference": float(roc_auc_score(y[june][m, i], oof[june][m, i])),
            "op_threshold": d["threshold"],
            "flag_rate_at_threshold": float((prob >= d["threshold"]).mean()),
            "base_rate": float(y[june, i].mean()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "v10_verify.csv", index=False)
    log("=== deploy artifacts sanity (June 2026; deploy is IN-SAMPLE, expect > OOF) ===")
    print(out.round(4).to_string(index=False))
    log("in-sample >> OOF is normal here; generalization numbers come from the V7 CV")


# %% [markdown]
# ## Live-inference feature builder
#
# A Python re-implementation of the SQL precompute for ONE issue hour, fed by a raw hourly
# frame (all 833 grids x >= 24 h ending at t). `phase_parity` proves it reproduces the SQL
# table bit-for-bit before `phase_predict` is allowed to matter. The IR side reuses the V7
# slot machinery (same masks, same stats, same lag formula).

# %%
RAIN_THRESHOLD_MM = 0.1
RAW_VARS = ["temperature_2m", "relative_humidity_2m", "pressure_msl", "surface_pressure",
            "dew_point_2m", "precipitation", "cloud_cover", "wind_speed_10m",
            "wind_direction_10m"]


def load_grid_geometry():
    import psycopg2
    from Thailand_Rain_V3 import DB_CONFIG
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        grid = pd.read_sql(
            'SELECT grid_number, grid_row, grid_column, longitude, latitude '
            'FROM "Thailand_Grid_25km" ORDER BY grid_number', conn)
    finally:
        conn.close()
    return grid


def build_adjacency(grid):
    """8-neighborhood pairs, same as OM_THAILAND_GRID_PAIRS."""
    by_rc = {(r, c): g for g, r, c in
             zip(grid["grid_number"], grid["grid_row"], grid["grid_column"])}
    pairs = []
    for g, r, c in zip(grid["grid_number"], grid["grid_row"], grid["grid_column"]):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                n = by_rc.get((r + dr, c + dc))
                if n is not None:
                    pairs.append((g, n, dr, dc))
    return pd.DataFrame(pairs, columns=["center", "neighbor", "dr", "dc"])


def build_om_features(raw, grid, adjacency, issue_local):
    """The 66 FEATURE_COLUMNS for every grid at one local issue hour.

    raw: DataFrame with grid_number, local_forecast_time and the 9 RAW_VARS, hourly,
    covering at least issue_local-23h .. issue_local for all grids.
    """
    t = pd.Timestamp(issue_local)
    hours_needed = pd.date_range(t - pd.Timedelta(hours=23), t, freq="h")
    r = raw[raw["local_forecast_time"].isin(hours_needed)].copy()
    wide = {v: r.pivot_table(index="local_forecast_time", columns="grid_number",
                             values=v, aggfunc="first").reindex(hours_needed)
            for v in RAW_VARS}
    gnums = grid["grid_number"].to_numpy()
    for v in RAW_VARS:
        wide[v] = wide[v].reindex(columns=gnums)

    def at(v, k=0):
        return wide[v].iloc[len(hours_needed) - 1 - k].to_numpy(dtype="float64")

    def past_sum(v, k):
        return wide[v].iloc[len(hours_needed) - k:].sum(axis=0, skipna=False).to_numpy(dtype="float64")

    f = {}
    for v in RAW_VARS:
        f[v] = at(v)
    f["temperature_dew_point_spread"] = f["temperature_2m"] - f["dew_point_2m"]
    f["pressure_msl_change_3h"] = at("pressure_msl") - at("pressure_msl", 3)
    f["pressure_msl_change_6h"] = at("pressure_msl") - at("pressure_msl", 6)
    for k in (1, 2, 3, 6):
        f[f"precipitation_lag_{k}h"] = at("precipitation", k)
    for k, nh in (("3h", 3), ("6h", 6), ("12h", 12), ("24h", 24)):
        f[f"precipitation_sum_past_{k}"] = past_sum("precipitation", nh)
    for k in (1, 3, 6):
        f[f"cloud_cover_lag_{k}h"] = at("cloud_cover", k)
        f[f"humidity_lag_{k}h"] = at("relative_humidity_2m", k)
    for k in (1, 3):
        f[f"wind_speed_lag_{k}h"] = at("wind_speed_10m", k)
    f["hour_sin"] = np.full(len(gnums), np.sin(2 * np.pi * t.hour / 24.0))
    f["hour_cos"] = np.full(len(gnums), np.cos(2 * np.pi * t.hour / 24.0))
    f["month_sin"] = np.full(len(gnums), np.sin(2 * np.pi * t.month / 12.0))
    f["month_cos"] = np.full(len(gnums), np.cos(2 * np.pi * t.month / 12.0))
    for c in ("grid_row", "grid_column", "latitude", "longitude"):
        f[c] = grid[c].to_numpy(dtype="float64")

    # neighbor aggregates at the issue hour
    now = pd.DataFrame({v: at(v) for v in
                        ("precipitation", "cloud_cover", "relative_humidity_2m",
                         "pressure_msl", "temperature_2m", "dew_point_2m",
                         "wind_speed_10m")}, index=gnums)
    nb = adjacency.join(now, on="neighbor")
    nb["spread"] = nb["temperature_2m"] - nb["dew_point_2m"]
    nb["is_rain"] = (nb["precipitation"] >= RAIN_THRESHOLD_MM).astype(float)
    g = nb.groupby("center")
    agg = pd.DataFrame({
        "neighbor_count": g.size().astype(float),
        "neighbor_precipitation_mean": g["precipitation"].mean(),
        "neighbor_precipitation_max": g["precipitation"].max(),
        "neighbor_precipitation_sum": g["precipitation"].sum(),
        "neighbor_rain_count": g["is_rain"].sum(),
        "neighbor_rain_rate": g["is_rain"].mean(),
        "neighbor_cloud_cover_mean": g["cloud_cover"].mean(),
        "neighbor_cloud_cover_max": g["cloud_cover"].max(),
        "neighbor_relative_humidity_mean": g["relative_humidity_2m"].mean(),
        "neighbor_relative_humidity_max": g["relative_humidity_2m"].max(),
        "neighbor_pressure_msl_mean": g["pressure_msl"].mean(),
        "neighbor_pressure_msl_min": g["pressure_msl"].min(),
        "neighbor_pressure_msl_max": g["pressure_msl"].max(),
        "neighbor_temperature_2m_mean": g["temperature_2m"].mean(),
        "neighbor_dew_point_2m_mean": g["dew_point_2m"].mean(),
        "neighbor_temperature_dew_point_spread_mean": g["spread"].mean(),
        "neighbor_wind_speed_10m_mean": g["wind_speed_10m"].mean(),
        "neighbor_wind_speed_10m_max": g["wind_speed_10m"].max(),
    }).reindex(gnums)
    for direction, mask in (("row_minus", nb["dr"] == -1), ("row_plus", nb["dr"] == 1),
                            ("column_minus", nb["dc"] == -1), ("column_plus", nb["dc"] == 1)):
        sub = nb[mask].groupby("center")
        agg[f"{direction}_precipitation_mean"] = sub["precipitation"].mean().reindex(gnums)
        agg[f"{direction}_cloud_cover_mean"] = sub["cloud_cover"].mean().reindex(gnums)

    center_precip = pd.Series(f["precipitation"], index=gnums)
    center_cloud = pd.Series(f["cloud_cover"], index=gnums)
    center_rh = pd.Series(f["relative_humidity_2m"], index=gnums)
    center_pmsl = pd.Series(f["pressure_msl"], index=gnums)
    for d in ("row_minus", "row_plus", "column_minus", "column_plus"):
        agg[f"{d}_precipitation_mean"] = agg[f"{d}_precipitation_mean"].fillna(0.0)
        agg[f"{d}_cloud_cover_mean"] = agg[f"{d}_cloud_cover_mean"].fillna(center_cloud)
    agg["neighbor_precipitation_mean_minus_center"] = \
        agg["neighbor_precipitation_mean"] - center_precip
    agg["neighbor_cloud_cover_mean_minus_center"] = \
        agg["neighbor_cloud_cover_mean"] - center_cloud
    agg["neighbor_relative_humidity_mean_minus_center"] = \
        agg["neighbor_relative_humidity_mean"] - center_rh
    agg["center_pressure_msl_minus_neighbor_mean"] = \
        center_pmsl - agg["neighbor_pressure_msl_mean"]

    for c in agg.columns:
        f[c] = agg[c].to_numpy(dtype="float64")
    out = pd.DataFrame({c: f[c] for c in FEATURE_COLUMNS}, index=gnums)
    return out


def build_ir_features(stats_by_slot, cell_index):
    """The 12 himawari features for one issue hour from per-slot stat arrays.

    stats_by_slot: {lag_hours: ndarray (n_cells, 6)} for lags 1..3 (V7 STAT_NAMES order);
    missing slots may be absent or all-NaN.
    """
    n = len(cell_index)
    nanrow = np.full(n, np.nan, dtype="float32")

    def stat(lag, name_ix):
        s = stats_by_slot.get(lag)
        return s[:, name_ix].astype("float32") if s is not None else nanrow.copy()

    env1, env2, env3 = stat(1, 2), stat(2, 2), stat(3, 2)
    cold1, cold2, cold3 = stat(1, 4), stat(2, 4), stat(3, 4)
    with np.errstate(all="ignore"):
        block = np.stack([
            stat(1, 0), stat(1, 1),
            env1, stat(1, 3), cold1,
            stat(1, 5),
            env2, env3,
            env3 - env1,
            np.nanmin(np.stack([env1, env2, env3]), axis=0),
            np.nanmax(np.stack([cold1, cold2, cold3]), axis=0),
            np.isfinite(env1).astype("float32"),
        ], axis=-1).astype("float32")
    return block


# %% [markdown]
# ## Phase `parity` — prove the Python builder matches the SQL precompute
#
# Runs entirely offline: raw June rows come from "OM_Thailand_Forecast_Data", the reference
# values from "OM_THAILAND_FORECAST_PRECOMPUTE" (identical data to the training table, per
# V8), and the IR reference from the cached hw_block row. Any feature off by >1e-4 fails.

# %%
PARITY_HOURS = ["2026-06-15 12:00:00", "2026-06-02 00:00:00", "2026-06-28 20:00:00"]


def phase_parity():
    import psycopg2
    from Thailand_Rain_V3 import DB_CONFIG

    grid = load_grid_geometry()
    adjacency = build_adjacency(grid)
    conn = psycopg2.connect(**DB_CONFIG)
    worst = 0.0
    try:
        for hour in PARITY_HOURS:
            t = pd.Timestamp(hour)
            raw = pd.read_sql(f"""
                SELECT grid_number, local_forecast_time, {", ".join(RAW_VARS)}
                FROM "OM_Thailand_Forecast_Data"
                WHERE local_forecast_time BETWEEN %(lo)s AND %(hi)s
            """, conn, params={"lo": t - pd.Timedelta(hours=23), "hi": t})
            mine = build_om_features(raw, grid, adjacency, t)
            ref = pd.read_sql(f"""
                SELECT grid_number, {", ".join(FEATURE_COLUMNS)}
                FROM "OM_THAILAND_FORECAST_PRECOMPUTE"
                WHERE local_forecast_time = %(t)s
            """, conn, params={"t": t}).set_index("grid_number").reindex(mine.index)
            diff = (mine - ref).abs().max()
            log(f"  {hour}: max |python - sql| over 66 features x 833 grids = {diff.max():.2e}"
                f"   (worst feature: {diff.idxmax()})")
            worst = max(worst, float(diff.max()))
    finally:
        conn.close()

    # IR side: rebuild one panel row's hw features from the day files
    from Thailand_Rain_V7_himawari_ir import load_stat_series
    tf = np.load(CACHE / "forecast_time.npy")
    unit = np.load(CACHE / "unit_id.npy")
    hw_ref = np.load(CACHE / "hw_block_offset0.npy")
    hours_local, stats = load_stat_series()
    hour_pos = {np.datetime64(h): i for i, h in enumerate(hours_local)}
    t = np.datetime64(PARITY_HOURS[0].replace(" ", "T"))
    stats_by_slot = {k: stats[hour_pos[t - np.timedelta64(k, "h")]] for k in (1, 2, 3)}
    mine_ir = build_ir_features(stats_by_slot, grid["grid_number"].to_numpy())
    rows = np.flatnonzero(tf == t)
    cell_pos = {g: i for i, g in enumerate(grid["grid_number"])}
    ref_ir = hw_ref[rows]
    mine_rows = mine_ir[[cell_pos[int(g)] for g in unit[rows]]]
    ir_diff = np.nanmax(np.abs(mine_rows - ref_ir))
    log(f"  IR features {PARITY_HOURS[0]}: max |python - cache| = {ir_diff:.2e} "
        f"({len(rows)} cells)")

    if worst > 1e-4 or ir_diff > 1e-4:
        raise SystemExit("PARITY FAILED — do not use phase_predict until this is fixed")
    log("parity OK: live builder reproduces the training features")


# %% [markdown]
# ## Phase `predict` — one live nationwide inference
#
# Issue time = the most recent fully-published local hour. OM inputs come from the live
# forecast API (`models=ecmwf_ifs`, past_days=2 -> 30+ h of history per cell); IR slots
# t-1..t-3 come from the NOAA S3 bucket with the cached V7 masks. Output: per-cell calibrated
# probability + flag for each of the 6 targets.

# %%
def fetch_live_om(grid, batch_size=50, sleep_s=1.0, retries=6):
    import requests
    frames = []
    points = grid.to_dict("records")
    for lo in range(0, len(points), batch_size):
        batch = points[lo:lo + batch_size]
        for attempt in range(1, retries + 1):
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": ",".join(str(p["latitude"]) for p in batch),
                "longitude": ",".join(str(p["longitude"]) for p in batch),
                "hourly": ",".join(RAW_VARS), "models": "ecmwf_ifs",
                "past_days": 2, "forecast_days": 1, "timezone": "Asia/Bangkok",
            }, timeout=120)
            if resp.status_code == 429 and attempt < retries:
                wait = min(60 * attempt, 300)
                log(f"  batch {lo // batch_size + 1}: rate-limited, retrying in {wait}s "
                    f"({attempt}/{retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        payloads = resp.json()
        if isinstance(payloads, dict):
            payloads = [payloads]
        for p, pl in zip(batch, payloads):
            h = pl["hourly"]
            df = pd.DataFrame({v: h[v] for v in RAW_VARS})
            df["local_forecast_time"] = pd.to_datetime(h["time"])
            df["grid_number"] = p["grid_number"]
            frames.append(df)
        time.sleep(sleep_s)
    return pd.concat(frames, ignore_index=True)


def fetch_live_ir(grid, issue_local):
    import json as _json
    from Thailand_Rain_V7_himawari_ir import (
        HW_DIR, UTC_OFFSET_HOURS, download_segments, load_tb, s3_client, slot_stats)
    cfg = _json.loads((HW_DIR / "fetch_config.json").read_text(encoding="utf-8"))
    mask_file = np.load(HW_DIR / "pixel_masks.npz")
    n_cells = sum(1 for k in mask_file.files if k.startswith("cell_"))
    masks = {"shape": tuple(mask_file["shape"]),
             "cell": [mask_file[f"cell_{i}"] for i in range(n_cells)],
             "env": [mask_file[f"env_{i}"] for i in range(n_cells)]}
    client = s3_client()
    stats_by_slot = {}
    for lag in (1, 2, 3):
        t_utc = pd.Timestamp(issue_local) - pd.Timedelta(hours=lag + UTC_OFFSET_HOURS)
        paths = download_segments(client, t_utc, cfg["segments"])
        if paths is None:
            log(f"  IR slot t-{lag}: unavailable")
            continue
        try:
            tb, lons, lats = load_tb(paths)
            stats_by_slot[lag] = slot_stats(tb, masks)
        finally:
            for p in paths:
                p.unlink(missing_ok=True)
        log(f"  IR slot t-{lag}: ok")
    return stats_by_slot


def phase_predict():
    grid = load_grid_geometry()
    adjacency = build_adjacency(grid)

    log("fetching live OM history (833 cells, ecmwf_ifs)")
    raw = fetch_live_om(grid)
    now_local = pd.Timestamp.now(tz="Asia/Bangkok").tz_localize(None)
    have = raw.groupby("local_forecast_time")["grid_number"].nunique()
    complete = have[(have == len(grid)) & (have.index <= now_local)]
    issue = complete.index.max()
    log(f"issue hour (latest complete local hour): {issue}")

    om = build_om_features(raw, grid, adjacency, issue)
    nan_share = float(om.isna().any(axis=1).mean())
    if nan_share > 0:
        log(f"WARNING: {nan_share:.1%} of cells have NaN OM features")

    log("fetching live Himawari slots")
    stats_by_slot = fetch_live_ir(grid, issue)
    ir = build_ir_features(stats_by_slot, grid["grid_number"].to_numpy())

    x = np.concatenate([om.to_numpy(dtype="float32"), ir], axis=1)
    out = grid[["grid_number", "longitude", "latitude"]].copy()
    out["issue_local"] = issue
    for tname in TARGET_NAMES:
        d = joblib.load(DEPLOY_DIR / f"th_rain_{tname}.joblib")
        prob = d["calibrator"].predict(d["model"].predict_proba(x)[:, 1])
        out[f"p_{tname}"] = np.round(prob, 4)
        out[f"flag_{tname}"] = (prob >= d["threshold"]).astype(int)
    stamp = pd.Timestamp(issue).strftime("%Y%m%d_%H%M")
    path = OUTPUT_DIR / f"v10_predictions_{stamp}.csv"
    out.to_csv(path, index=False)
    log(f"=== live nationwide prediction @ {issue} ===")
    log(f"IR available: {sorted(stats_by_slot)}  "
        f"flagged cells h1_1.0mm: {int(out['flag_h1_1.0mm'].sum())}/{len(out)}  "
        f"max p: {out['p_h1_1.0mm'].max():.3f}")
    log(f"saved {path}")


# %%
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "fit"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "fit":
        phase_fit()
    elif phase == "verify":
        phase_verify()
    elif phase == "parity":
        phase_parity()
    elif phase == "predict":
        phase_predict()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use fit | verify | parity | predict")
    log(f"phase {phase} done in {(time.time() - t0) / 60:.1f} min")
