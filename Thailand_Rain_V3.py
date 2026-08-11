# %% [markdown]
# # Thailand_Rain_V3 — the nationwide panel under the V3 protocol
#
# Named for the protocol, not a lineage: this is the Thailand counterpart of
# `BKK_Rain_V3.py`, and it imports that module's CV machinery verbatim (leave-one-*block*-out
# instead of leave-one-year-out, 24 h purge, isotonic calibration, F1-optimal thresholds
# chosen on the other folds) so nationwide results stay readable next to the Bangkok ladder.
#
# It exists because the Thailand work so far graded itself against Open-Meteo's own ERA5
# `precipitation` — the same column that produces `precipitation`, `precipitation_lag_*` and
# `precipitation_sum_past_*` on the feature side. On identical grid/hour rows, IMERG cell-mean
# correlates with that column at **r = 0.12**: similar wet-hour base rates (44.9% vs 48.5% in
# the September 2025 pilot), near-total disagreement about *which* hours. So `om_only` was
# partly learning ERA5's own persistence, and no observed-rain feature could have been
# measured honestly against it. This module switches the labels to observed IMERG.
#
# Four deliberate departures from the Bangkok build, each forced by scale or geography:
#
# | | Bangkok V3 | Thailand V3 | why |
# |---|---|---|---|
# | label | IMERG point sample | IMERG **cell-max** over the 25 km footprint | a 25 km cell spans ~10 IMERG samples; the max is "rain anywhere in the cell" |
# | rows | all hours, 2.45 M | **every 4th hour**, ~3.7 M of 14.9 M | 14.9 M x 66 float32 peaks ~14 GB through the concatenate path; this box has 15.6 GB |
# | folds | leave-one-year-out | leave-one-**2-month-block**-out | the panel's three "years" are Jul-Dec 2024, all of 2025, Jan-Jul 2026 — year folds confound season with fold |
# | run_type | 16% provisional | **39% provisional** | stored per row; the 2026 tail is entirely Late Run |
#
# The hour stride subsamples *forecast issue times* only. Labels are built by LEAD over the
# full hourly IMERG series and observed-rain features (V4/V5/V7-style) are per-row lookups
# into full hourly series, so nothing downstream sees a gappy history because of it.
#
# `cv` checkpoints after **every fit**, not every config. V3/V7 resume at config granularity,
# which was fine at 6 folds x 6 targets (~40 min of exposure) but not at 13 x 6: an interrupted
# run there would discard hours. Each fold keeps an `oof_fold_<config>_<fold>.npz` holding the
# test-row predictions plus a per-target `done` flag, so a kill costs at most one partial fit
# (~5 min) and a resume skips straight past finished folds without even slicing the train set.
# Per-fit rng seeding (below) makes a resumed run bit-identical to an uninterrupted one.
#
# Phases:
#     python Thailand_Rain_V3.py load_panel   # DB -> base_X / y / forecast_time / unit_id
#     python Thailand_Rain_V3.py cv           # om_only control, 6 targets x N blocks; resumable
#     python Thailand_Rain_V3.py report       # scores + per-block and run_type breakouts

# %%
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from BKK_Rain_V3 import (
    FEATURE_COLUMNS, MAX_LEAD, MIN_FOLD_ROWS, MODEL_PLAN, PROJECT_ROOT,
    PURGE_HOURS, RANDOM_STATE, TARGETS, TARGET_NAMES,
    fit_predict_calibrated, log, purged_train_mask, score_probabilities,
)

# %% [markdown]
# ## 1. Configuration

# %%
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

PRECOMPUTE_TABLE_NAME = '"OM_THAILAND_DATA_PRECOMPUTE"'
IMERG_TABLE_NAME = '"IMERG_THAILAND_DATA"'
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_thailand_rain_v3"
CACHE = Path(os.getenv("TH_CACHE", str(PROJECT_ROOT / "ML_Model_V2" / "_th_cache")))
CACHE.mkdir(parents=True, exist_ok=True)

# "rain anywhere in the 25 km cell". IMERG_THAILAND_DATA also carries precipitation_mm
# (cell mean, 17.8% wet vs 26.9% for the max) if the cell-average definition is ever wanted.
IMERG_LABEL_COLUMN = "precipitation_max_mm"

HOUR_STRIDE = 4          # keep local hours 0, 4, 8, 12, 16, 20 — diurnally balanced
BLOCK_MONTHS = 2         # CV fold width

# Same row filter as V3: every lag/rolling feature must be present, and the cell must have
# neighbours. Applied to the Thailand precompute table, which carries the identical columns.
ROW_FILTER_SQL = """
      om.pressure_msl_change_6h IS NOT NULL
      AND om.precipitation_lag_6h IS NOT NULL
      AND om.precipitation_sum_past_24h IS NOT NULL
      AND om.cloud_cover_lag_6h IS NOT NULL
      AND om.humidity_lag_6h IS NOT NULL
      AND om.wind_speed_lag_3h IS NOT NULL
      AND om.neighbor_count > 0
      AND MOD(EXTRACT(HOUR FROM om.local_forecast_time)::int, {stride}) = 0
""".replace("{stride}", str(HOUR_STRIDE))
# MOD() rather than the % operator on purpose: these queries go through
# cur.execute(sql) with no parameters, so psycopg2 does no %-interpolation and a
# literal %% would reach Postgres unescaped as a syntax error.


def connect():
    return psycopg2.connect(**DB_CONFIG)


# %% [markdown]
# ## 2. Phase `load_panel` — Open-Meteo features + observed-IMERG targets
#
# Structure is V3's `build_panel_query` with the Thailand tables and the cell-max label. The
# `im` CTE runs LEAD over the *complete* hourly IMERG series (no stride), and each future
# stamp is guarded against gaps: if `t{k}` is not exactly `t + k hours` the value becomes NaN
# rather than silently borrowing across a hole. The single missing archive hour
# (local 2026-06-01 12:00) therefore invalidates the rows that would have depended on it
# instead of mislabelling them.

# %%
def build_panel_query():
    feature_sql = ",\n        ".join(
        f"COALESCE(om.{c}::real, 'NaN'::real) AS {c}" for c in FEATURE_COLUMNS)
    lead_sql = ",\n            ".join(
        f"LEAD({IMERG_LABEL_COLUMN}, {k}) OVER w AS p{k}, "
        f"LEAD(local_observation_time, {k}) OVER w AS t{k}"
        for k in range(1, MAX_LEAD + 1))
    guarded_sql = ",\n        ".join(
        f"CASE WHEN im.t{k} = im.t_local + interval '{k} hour' "
        f"THEN im.p{k}::real ELSE 'NaN'::real END AS p{k}"
        for k in range(1, MAX_LEAD + 1))
    return f"""
    WITH im AS (
        SELECT grid_number, local_observation_time AS t_local, run_type,
            {lead_sql}
        FROM {IMERG_TABLE_NAME}
        WHERE is_complete_hour
        WINDOW w AS (PARTITION BY grid_number ORDER BY local_observation_time)
    )
    SELECT om.grid_number, om.local_forecast_time,
        {feature_sql},
        {guarded_sql},
        im.run_type
    FROM {PRECOMPUTE_TABLE_NAME} om
    JOIN im ON im.grid_number = om.grid_number AND im.t_local = om.local_forecast_time
    WHERE {ROW_FILTER_SQL}
    ORDER BY om.local_forecast_time, om.grid_number
    """


def count_panel_rows(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT count(*) FROM {PRECOMPUTE_TABLE_NAME} om
            JOIN (SELECT grid_number, local_observation_time AS t_local
                  FROM {IMERG_TABLE_NAME} WHERE is_complete_hour) im
              ON im.grid_number = om.grid_number AND im.t_local = om.local_forecast_time
            WHERE {ROW_FILTER_SQL}""")
        return int(cur.fetchone()[0])


def phase_load_panel(batch_size=200_000):
    n_feat = len(FEATURE_COLUMNS)
    conn = connect()
    try:
        n_rows = count_panel_rows(conn)
        log(f"panel rows {n_rows:,} x {n_feat} feat "
            f"({n_rows * n_feat * 4 / 1e9:.2f} GB) + {MAX_LEAD} future cols "
            f"(hour stride {HOUR_STRIDE}, label {IMERG_LABEL_COLUMN})")

        x = np.empty((n_rows, n_feat), dtype="float32")
        future = np.empty((n_rows, MAX_LEAD), dtype="float32")
        t = np.empty(n_rows, dtype="datetime64[s]")
        unit = np.empty(n_rows, dtype="int32")
        run_type = np.empty(n_rows, dtype="object")
        filled = 0
        with conn.cursor(name="th_v3_panel") as cur:
            cur.itersize = batch_size
            cur.execute(build_panel_query())
            while filled < n_rows:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                block = np.asarray(batch, dtype="object")
                take = len(batch)
                unit[filled:filled + take] = block[:, 0].astype("int32")
                t[filled:filled + take] = block[:, 1].astype("datetime64[s]")
                x[filled:filled + take] = block[:, 2:2 + n_feat].astype("float32")
                future[filled:filled + take] = \
                    block[:, 2 + n_feat:2 + n_feat + MAX_LEAD].astype("float32")
                run_type[filled:filled + take] = block[:, -1]
                filled += take
                print(f"  {filled:,} / {n_rows:,}", end="\r", flush=True)
        print()
    finally:
        conn.close()

    x, future, t, unit, run_type = (a[:filled] for a in (x, future, t, unit, run_type))

    # A row is usable only with the full MAX_LEAD future window present.
    complete = ~np.isnan(future).any(axis=1)
    x = np.ascontiguousarray(x[complete])
    future, t, unit, run_type = future[complete], t[complete], unit[complete], run_type[complete]
    log(f"complete future window: {complete.sum():,} ({complete.mean():.1%})")

    y = np.empty((x.shape[0], len(TARGETS)), dtype="int8")
    for i, (horizon, threshold) in enumerate(TARGETS):
        y[:, i] = (future[:, :horizon] >= threshold).any(axis=1).astype("int8")
        log(f"  {TARGET_NAMES[i]:12} base rate {y[:, i].mean():.4f}")

    np.save(CACHE / "base_X.npy", x)
    np.save(CACHE / "y.npy", y)
    np.save(CACHE / "forecast_time.npy", t)
    np.save(CACHE / "unit_id.npy", unit)
    np.save(CACHE / "run_type.npy", run_type.astype("U12"))
    log(f"saved panel caches to {CACHE}")


# %% [markdown]
# ## 3. Blocked CV keys
#
# Leave-one-year-out would hand the folds different seasons: 2024 is Jul-Dec (wet-heavy),
# 2026 is Jan-Jul (dry-heavy) and entirely provisional IMERG. Contiguous 2-month blocks give
# ~13 folds that each see one season, and the 24 h purge already prevents leakage across a
# block edge, so the protocol's guarantees are unchanged.

# %%
def block_keys(forecast_time):
    stamps = pd.to_datetime(forecast_time)
    months = stamps.year.to_numpy() * 12 + (stamps.month.to_numpy() - 1)
    return (months - months.min()) // BLOCK_MONTHS


def block_label(key, forecast_time):
    stamps = pd.to_datetime(forecast_time[block_keys(forecast_time) == key])
    return f"{stamps.min():%Y-%m}..{stamps.max():%Y-%m}"


# %% [markdown]
# ## 4. Phase `cv` — the om_only control
#
# Nothing to compare against yet: this establishes the Thailand baseline that every later
# observed-rain block (Himawari IR, gauges) is measured against, on identical rows.

# %%
CONFIG_BLOCKS = {
    "om_only": [],
}


def fold_ckpt_path(config, fold):
    return CACHE / f"oof_fold_{config}_{int(fold)}.npz"


def load_fold_ckpt(config, fold, n_test):
    """Restore a partially-finished fold, or start it fresh.

    Returns (raw, cal, done) where `done[i]` marks target i as already fitted. A checkpoint
    whose test-row count disagrees with the current fold definition is discarded rather than
    trusted -- that is what a changed HOUR_STRIDE or BLOCK_MONTHS looks like on disk.
    """
    blank = (np.full((n_test, len(TARGETS)), np.nan, dtype="float32"),
             np.full((n_test, len(TARGETS)), np.nan, dtype="float32"),
             np.zeros(len(TARGETS), dtype=bool))
    path = fold_ckpt_path(config, fold)
    if not path.exists():
        return blank
    saved = np.load(path)
    if int(saved["n_test"]) != n_test:
        log(f"  fold {fold}: checkpoint has {int(saved['n_test']):,} test rows, "
            f"expected {n_test:,}; discarding")
        return blank
    return saved["raw"].copy(), saved["cal"].copy(), saved["done"].copy()


def run_cv(config, y, forecast_time, fold_key, folds):
    parts = [np.load(CACHE / "base_X.npy")]
    for fname, _ in CONFIG_BLOCKS[config]:
        parts.append(np.load(CACHE / fname))
    x = np.ascontiguousarray(np.concatenate(parts, axis=1)) if len(parts) > 1 else parts[0]
    del parts
    gc.collect()

    log(f"config {config}: X {x.shape} ({x.nbytes / 1e9:.2f} GB)")
    purge = np.timedelta64(PURGE_HOURS, "h")
    oof_raw = np.full((x.shape[0], len(TARGETS)), np.nan, dtype="float32")
    oof_cal = np.full((x.shape[0], len(TARGETS)), np.nan, dtype="float32")
    fit_log = []
    for fold in folds:
        test_mask = fold_key == fold
        n_test = int(test_mask.sum())
        fold_raw, fold_cal, done = load_fold_ckpt(config, fold, n_test)
        if done.all():
            oof_raw[test_mask], oof_cal[test_mask] = fold_raw, fold_cal
            log(f"  fold {fold} ({block_label(fold, forecast_time)}): "
                f"restored from checkpoint")
            continue

        train_mask = purged_train_mask(test_mask, forecast_time, purge)
        x_train, x_test = x[train_mask], x[test_mask]
        log(f"  fold {fold} ({block_label(fold, forecast_time)}): "
            f"test {n_test:,} train {train_mask.sum():,}"
            + (f" ({int(done.sum())}/{len(TARGETS)} targets already done)"
               if done.any() else ""))
        for i, (horizon, threshold) in enumerate(TARGETS):
            if done[i]:
                continue
            model_name = MODEL_PLAN[horizon]
            t0 = time.time()
            # Seeded per (fold, target) rather than from one long stream, so a resumed run
            # reproduces an uninterrupted one exactly. The rng only picks the calibration
            # holdout, but a shifted stream would silently change every later fold.
            rng = np.random.default_rng([RANDOM_STATE, int(fold), i])
            raw, cal = fit_predict_calibrated(
                x_train, y[train_mask, i], x_test, model_name, rng)
            fold_raw[:, i], fold_cal[:, i] = raw, cal
            done[i] = True
            np.savez(fold_ckpt_path(config, fold), raw=fold_raw, cal=fold_cal,
                     done=done, n_test=n_test)
            fit_log.append({"config": config, "fold": int(fold),
                            "span": block_label(fold, forecast_time),
                            "target": TARGET_NAMES[i], "model": model_name,
                            "train_rows": int(train_mask.sum()),
                            "fit_seconds": time.time() - t0})
            log(f"    {TARGET_NAMES[i]:12} {model_name:22} {(time.time()-t0)/60:.1f} min")
        oof_raw[test_mask], oof_cal[test_mask] = fold_raw, fold_cal
        del x_train, x_test
        gc.collect()
    del x
    gc.collect()
    np.save(CACHE / f"oof_cal_{config}.npy", oof_cal)
    np.save(CACHE / f"oof_raw_{config}.npy", oof_raw)
    results = score_probabilities(oof_cal, y, fold_key, folds, "calibrated", config)
    results += score_probabilities(oof_raw, y, fold_key, folds, "raw", config)
    return results, fit_log


def phase_cv():
    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    fold_key = block_keys(forecast_time)
    folds = [int(v) for v in np.unique(fold_key) if (fold_key == v).sum() >= MIN_FOLD_ROWS]
    log(f"blocks: {len(folds)} of {len(np.unique(fold_key))} "
        f"(>= {MIN_FOLD_ROWS:,} rows)   rows {len(y):,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "th_v3_cv_results.csv"
    fitlog_path = OUTPUT_DIR / "th_v3_fit_log.csv"
    all_results, all_fitlog, done = [], [], set()
    if results_path.exists():
        prior = pd.read_csv(results_path)
        all_results = prior.to_dict(orient="records")
        done = set(prior["config"].unique())
        log(f"resuming; configs already scored: {sorted(done)}")
    if fitlog_path.exists():
        # timings from earlier attempts survive a resume; restored folds add no new rows
        all_fitlog = pd.read_csv(fitlog_path).to_dict(orient="records")
    for config in CONFIG_BLOCKS:
        if config in done:
            continue
        results, fit_log = run_cv(config, y, forecast_time, fold_key, folds)
        all_results += results
        all_fitlog += fit_log
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        pd.DataFrame(all_fitlog).to_csv(OUTPUT_DIR / "th_v3_fit_log.csv", index=False)
    log("cv phase complete")


# %% [markdown]
# ## 5. Phase `report` — headline scores, per-block spread, and the provisional breakout
#
# 39% of this panel is Late Run IMERG (everything from October 2025), against 16% at Bangkok.
# Late Run has lower detection, so a fold sitting in that era scores worse for reasons that
# have nothing to do with the model. The breakout keeps that visible instead of letting it
# masquerade as a seasonal effect.

# %%
def phase_report():
    from sklearn.metrics import roc_auc_score

    results = pd.read_csv(OUTPUT_DIR / "th_v3_cv_results.csv")
    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    run_type = np.load(CACHE / "run_type.npy")
    fold_key = block_keys(forecast_time)

    cal = results[results["probabilities"] == "calibrated"]
    headline = cal.pivot_table(index="target", columns="config", values="roc_auc")

    rows = []
    for config in cal["config"].unique():
        oof = np.load(CACHE / f"oof_cal_{config}.npy")
        for i, name in enumerate(TARGET_NAMES):
            for label, mask in [("permanent", run_type == "permanent"),
                                ("provisional", run_type == "provisional")]:
                m = mask & np.isfinite(oof[:, i])
                if m.sum() < MIN_FOLD_ROWS or len(np.unique(y[m, i])) < 2:
                    continue
                rows.append({"config": config, "target": name, "run_type": label,
                             "rows": int(m.sum()), "base_rate": float(y[m, i].mean()),
                             "roc_auc": float(roc_auc_score(y[m, i], oof[m, i]))})
    breakout = pd.DataFrame(rows)

    per_block = []
    for config in cal["config"].unique():
        oof = np.load(CACHE / f"oof_cal_{config}.npy")
        for fold in np.unique(fold_key):
            for i, name in enumerate(TARGET_NAMES):
                m = (fold_key == fold) & np.isfinite(oof[:, i])
                if m.sum() < MIN_FOLD_ROWS or len(np.unique(y[m, i])) < 2:
                    continue
                per_block.append({"config": config, "block": int(fold),
                                  "span": block_label(fold, forecast_time),
                                  "target": name, "rows": int(m.sum()),
                                  "base_rate": float(y[m, i].mean()),
                                  "roc_auc": float(roc_auc_score(y[m, i], oof[m, i]))})
    per_block = pd.DataFrame(per_block)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    breakout.to_csv(OUTPUT_DIR / "th_v3_run_type_breakout.csv", index=False)
    per_block.to_csv(OUTPUT_DIR / "th_v3_per_block.csv", index=False)
    (OUTPUT_DIR / "th_v3_metadata.json").write_text(json.dumps({
        "purpose": "nationwide Thailand baseline under the V3 protocol, graded on observed "
                   "IMERG instead of the ERA5 precipitation column the features come from",
        "label": f"{IMERG_LABEL_COLUMN} (cell-max over the 25 km footprint)",
        "hour_stride": HOUR_STRIDE,
        "cv": f"leave-one-{BLOCK_MONTHS}-month-block-out, {PURGE_HOURS} h purge",
        "features": FEATURE_COLUMNS,
        "caveats": [
            "39% of rows are provisional (Late Run) IMERG; see the run_type breakout.",
            "Features are ERA5 reanalysis (source_api=open_meteo_historical_weather), so "
            "these scores are an upper bound on what a forecast-fed deployment would reach.",
            "Hour stride 4 subsamples issue times; labels and any observed-rain features "
            "are built from the full hourly series.",
            "One IMERG archive hour is missing (local 2026-06-01 12:00); rows depending on "
            "it are dropped by the future-window guard, not mislabelled.",
        ],
    }, indent=2), encoding="utf-8")

    log("=== ROC-AUC (calibrated) ===")
    print(headline.round(4).to_string())
    if not breakout.empty:
        log("=== permanent vs provisional IMERG ===")
        print(breakout.pivot_table(index="target", columns="run_type",
                                   values="roc_auc").round(4).to_string())
    if not per_block.empty:
        log("=== per-block spread (h1_1.0mm) ===")
        print(per_block[per_block["target"] == "h1_1.0mm"]
              [["span", "rows", "base_rate", "roc_auc"]].round(4).to_string(index=False))
    log(f"saved Thailand V3 outputs to {OUTPUT_DIR}")


# %%
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "load_panel"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "load_panel":
        phase_load_panel()
    elif phase == "cv":
        phase_cv()
    elif phase == "report":
        phase_report()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use load_panel | cv | report")
    log(f"phase {phase} done in {(time.time()-t0)/60:.1f} min")
