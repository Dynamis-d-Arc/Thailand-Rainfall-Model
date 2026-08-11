# %% [markdown]
# # BKK_Rain_V4 latency sweep — at what observation latency does the gain die?
#
# V3 measured two points: observed IMERG history at 0 h lag lifts ROC +0.017..+0.050
# (`imerg_research`), and at 6 h lag the lift is gone (`imerg_deploy`). This sweep fills in
# offsets 1..5 h with the identical protocol, producing the latency-decay curve. The output
# is the requirement any real-time feed must beat: the maximum latency at which observed
# rain still adds value over the 66 Open-Meteo features.
#
# Everything (features, CV, purge, calibration, thresholds, scoring) is imported from
# BKK_Rain_V3, so the only variable is the offset. V3's cached panel is reused; only the raw
# 30-lag matrix must be rebuilt (V3 saved the derived blocks, not the lags).
#
# Phases (memory-bounded, ~3.6 GB free on this box):
#     python BKK_Rain_V4_latency_sweep.py load_lags   # rebuild lags.npy from Postgres (~5 min)
#     python BKK_Rain_V4_latency_sweep.py cv          # offsets 1..5, ~30 min each
#     python BKK_Rain_V4_latency_sweep.py report      # merge with V3's 0 h / 6 h endpoints

# %%
import gc
import json
import sys
import time

import numpy as np
import pandas as pd

from BKK_Rain_V3 import (
    CACHE, FEATURE_COLUMNS, MAX_HISTORY_LAG, MIN_FOLD_ROWS, MODEL_PLAN, PROJECT_ROOT,
    PURGE_HOURS, RANDOM_STATE, TARGETS, TARGET_NAMES,
    build_history_block, build_history_query, connect, fit_predict_calibrated,
    history_feature_names, log, purged_train_mask, score_probabilities,
)

V3_OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v3_imerg_history"
OUTPUT_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v4_latency_sweep"

SWEEP_OFFSETS = [1, 2, 3, 4, 5]  # 0 and 6 come from V3 (imerg_research / imerg_deploy)


# %% [markdown]
# ## Phase `load_lags` — V3's history loader, but saving the raw lag matrix

# %%
def phase_load_lags(batch_size=200_000):
    unit = np.load(CACHE / "unit_id.npy")
    tf = np.load(CACHE / "forecast_time.npy")
    log(f"panel keys: {len(unit):,}")

    key_index = {}
    for go, gr, ro in zip(unit, tf, range(len(unit))):
        key_index[(int(go), np.datetime64(gr))] = ro

    n_lag = MAX_HISTORY_LAG
    lags = np.full((len(unit), n_lag), np.nan, dtype="float32")
    matched = 0
    with connect() as conn:
        with conn.cursor(name="bkk_v4_lags") as cur:
            cur.itersize = batch_size
            cur.execute(build_history_query())
            seen = 0
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                block = np.asarray(batch, dtype="object")
                grids = block[:, 0].astype("int64")
                times = block[:, 1].astype("datetime64[s]")
                vals = block[:, 3:3 + n_lag].astype("float32")
                for j in range(len(block)):
                    ro = key_index.get((int(grids[j]), np.datetime64(times[j])))
                    if ro is not None:
                        lags[ro] = vals[j]
                        matched += 1
                seen += len(block)
                print(f"  scanned {seen:,}  matched {matched:,}", end="\r", flush=True)
        print()
    log(f"matched {matched:,} / {len(unit):,} panel rows")
    np.save(CACHE / "lags.npy", lags)

    # sanity: offset-0 and offset-6 blocks rebuilt from these lags must equal V3's cached blocks
    for offset, ref in [(0, "imerg_research.npy"), (6, "imerg_deploy.npy")]:
        rebuilt = build_history_block(lags, offset=offset)
        cached = np.load(CACHE / ref)
        same = np.array_equal(np.nan_to_num(rebuilt, nan=-9e9), np.nan_to_num(cached, nan=-9e9))
        log(f"  rebuilt offset {offset} == V3 {ref}: {same}")
        del rebuilt, cached
    log(f"saved lags {lags.shape}")


# %% [markdown]
# ## Phase `cv` — one full V3-protocol CV per offset, checkpointed per offset

# %%
def run_offset_cv(offset, y, forecast_time, year_key, year_folds):
    lags = np.load(CACHE / "lags.npy")
    block = build_history_block(lags, offset=offset)
    del lags
    gc.collect()
    base = np.load(CACHE / "base_X.npy")
    x = np.ascontiguousarray(np.concatenate([base, block], axis=1))
    del base, block
    gc.collect()

    config = f"offset{offset}h"
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
    results_path = OUTPUT_DIR / "v4_sweep_results.csv"
    done = set()
    if results_path.exists():  # resume support: skip offsets already scored
        prior = pd.read_csv(results_path)
        all_results = prior.to_dict(orient="records")
        done = set(prior["config"].unique())
        log(f"resuming; already done: {sorted(done)}")
    for offset in SWEEP_OFFSETS:
        if f"offset{offset}h" in done:
            continue
        results, fit_log = run_offset_cv(offset, y, forecast_time, year_key, year_folds)
        all_results += results
        all_fitlog += fit_log
        pd.DataFrame(all_results).to_csv(results_path, index=False)  # checkpoint per offset
        pd.DataFrame(all_fitlog).to_csv(OUTPUT_DIR / "v4_fit_log.csv", index=False)
    log("cv phase complete")


# %% [markdown]
# ## Phase `report` — the decay curve: sweep offsets merged with V3's 0 h and 6 h endpoints
#
# Persistence at every offset is also scored here (no training, one column of `lags` each),
# so the curve shows both "model + history at lag L" and "history at lag L alone".

# %%
def phase_report():
    y = np.load(CACHE / "y.npy")
    forecast_time = np.load(CACHE / "forecast_time.npy")
    stamps = pd.to_datetime(forecast_time)
    year_key = stamps.year.to_numpy()
    year_folds = [int(v) for v in np.unique(year_key) if (year_key == v).sum() >= MIN_FOLD_ROWS]

    sweep = pd.read_csv(OUTPUT_DIR / "v4_sweep_results.csv")
    v3 = pd.read_csv(V3_OUTPUT_DIR / "v3_oof_results.csv")
    v3 = v3.rename(columns={"config": "config"})
    v3_map = {"imerg_research": "offset0h", "imerg_deploy": "offset6h"}
    v3_part = v3[v3["config"].isin(v3_map)].copy()
    v3_part["config"] = v3_part["config"].map(v3_map)
    combined = pd.concat([sweep, v3_part], ignore_index=True)
    cal = combined[combined["probabilities"] == "calibrated"].copy()
    cal["offset_h"] = cal["config"].str.extract(r"offset(\d+)h").astype(int)

    om = v3[(v3["config"] == "om_only") & (v3["probabilities"] == "calibrated")]
    om_roc = om.set_index("target")["roc_auc"]
    om_f1 = om.set_index("target")["f1"]

    roc = cal.pivot_table(index="target", columns="offset_h", values="roc_auc")
    f1 = cal.pivot_table(index="target", columns="offset_h", values="f1")
    roc_gain = roc.sub(om_roc, axis=0)
    f1_gain = f1.sub(om_f1, axis=0)

    # persistence decay: observed precip amount at t-(offset+1) as the score, no model
    lags = np.load(CACHE / "lags.npy")
    persist_rows = []
    for offset in range(0, 7):
        anchor = lags[:, offset]
        probs = np.repeat(anchor[:, None], len(TARGETS), axis=1)
        persist_rows += score_probabilities(probs, y, year_key, year_folds, "amount",
                                            f"persist_offset{offset}h",
                                            extra={"offset_h": offset})
    del lags
    persist_df = pd.DataFrame(persist_rows)
    persist_roc = persist_df.pivot_table(index="target", columns="offset_h", values="roc_auc")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "v4_sweep_results_combined.csv", index=False)
    persist_df.to_csv(OUTPUT_DIR / "v4_persistence_decay.csv", index=False)
    roc_gain.to_csv(OUTPUT_DIR / "v4_roc_gain_by_offset.csv")
    f1_gain.to_csv(OUTPUT_DIR / "v4_f1_gain_by_offset.csv")

    metadata = {
        "purpose": "Latency-decay curve for observed IMERG history: V3 protocol at offsets "
                   "1..5 h merged with V3's 0 h (imerg_research) and 6 h (imerg_deploy) "
                   "endpoints. Gains are relative to V3's om_only control.",
        "protocol": "identical to V3 (imported): leave-one-year-out CV, 24 h purge, isotonic "
                    "calibration, IMERG labels, F1-optimal leave-one-year-out thresholds",
        "offsets_h": list(range(0, 7)),
        "history_features": history_feature_names(),
        "n_features": len(FEATURE_COLUMNS) + len(history_feature_names()),
        "roc_gain_by_offset": roc_gain.reset_index().to_dict(orient="records"),
        "f1_gain_by_offset": f1_gain.reset_index().to_dict(orient="records"),
        "persistence_roc_by_offset": persist_roc.reset_index().to_dict(orient="records"),
    }
    (OUTPUT_DIR / "v4_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    log("=== ROC-AUC gain over om_only, by observation latency (h) ===")
    print(roc_gain.round(4).to_string())
    log("=== F1 gain over om_only, by observation latency (h) ===")
    print(f1_gain.round(4).to_string())
    log("=== persistence-alone ROC-AUC by latency ===")
    print(persist_roc.round(4).to_string())
    log(f"saved sweep outputs to {OUTPUT_DIR}")


# %%
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "cv"
    log(f"phase: {phase}")
    t0 = time.time()
    if phase == "load_lags":
        phase_load_lags()
    elif phase == "cv":
        phase_cv()
    elif phase == "report":
        phase_report()
    else:
        raise SystemExit(f"unknown phase {phase!r}; use load_lags | cv | report")
    log(f"phase {phase} done in {(time.time()-t0)/60:.1f} min")
