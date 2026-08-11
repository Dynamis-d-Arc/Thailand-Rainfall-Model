"""Replace the single global operating threshold with a per-month one, for all six targets.

`validate_by_season.py` showed that one global F1-optimal cut behaves incoherently across the
year: at `h6_0.1mm` it fires on ~85% of wet-season cell-hours (close to a constant "yes") while
missing 77% of dry-season rain. The ranking is fine - wet-season ROC-AUC is 0.75+ - so the fix
belongs at the operating point, not in the model.

Two things happen here.

**Out-of-fold probabilities get saved.** The deploy run scored them and discarded them, so every
later threshold question needed a 25-minute rerun. They are written to `oof_probabilities.npz`
and everything downstream reads from there.

**Thresholds are chosen per calendar month, leave-one-year-out.** For each (year, month) the cut
comes from that month in *other* years, so the reported metrics never see the rows they were tuned
on. That also repairs the selection leak in the deploy run, where the global threshold was picked
on the same OOF rows its F1 was reported against. The value that ships for month M is fitted on
all years of month M; the leave-one-year-out figures are the honest estimate of how it will do.

The six trained models are not refitted - only their operating point changes.

Usage:
    python ML_Model_V2/fit_seasonal_thresholds.py
    python ML_Model_V2/fit_seasonal_thresholds.py --reuse-oof     # skip the CV, load the .npz
"""

import argparse
import gc
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v2_deploy"
OOF_PATH = MODEL_DIR / "oof_probabilities.npz"

HORIZONS = [1, 3, 6]
RAIN_THRESHOLDS_MM = [0.1, 1.0]
MAX_LEAD = max(HORIZONS)
MODEL_PLAN = {1: "lightgbm", 3: "hist_gradient_boosting", 6: "hist_gradient_boosting"}
RANDOM_STATE = 42
PURGE_HOURS = 24
MIN_FOLD_ROWS = 50_000
TARGETS = [(h, thr) for thr in RAIN_THRESHOLDS_MM for h in HORIZONS]
TARGET_NAMES = [f"h{h}_{thr}mm" for h, thr in TARGETS]

BASELINE_FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "pressure_msl", "surface_pressure",
    "dew_point_2m", "precipitation", "cloud_cover", "wind_speed_10m",
    "wind_direction_10m", "temperature_dew_point_spread", "pressure_msl_change_3h",
    "pressure_msl_change_6h", "precipitation_lag_1h", "precipitation_lag_2h",
    "precipitation_lag_3h", "precipitation_lag_6h", "precipitation_sum_past_3h",
    "precipitation_sum_past_6h", "precipitation_sum_past_12h", "precipitation_sum_past_24h",
    "cloud_cover_lag_1h", "cloud_cover_lag_3h", "cloud_cover_lag_6h",
    "humidity_lag_1h", "humidity_lag_3h", "humidity_lag_6h",
    "wind_speed_lag_1h", "wind_speed_lag_3h", "hour_sin", "hour_cos",
    "month_sin", "month_cos", "grid_row", "grid_column", "latitude", "longitude",
]
NEIGHBOR_FEATURE_COLUMNS = [
    "neighbor_count", "neighbor_precipitation_mean", "neighbor_precipitation_max",
    "neighbor_precipitation_sum", "neighbor_rain_count", "neighbor_rain_rate",
    "neighbor_cloud_cover_mean", "neighbor_cloud_cover_max", "neighbor_relative_humidity_mean",
    "neighbor_relative_humidity_max", "neighbor_pressure_msl_mean", "neighbor_pressure_msl_min",
    "neighbor_pressure_msl_max", "neighbor_temperature_2m_mean", "neighbor_dew_point_2m_mean",
    "neighbor_temperature_dew_point_spread_mean", "neighbor_wind_speed_10m_mean",
    "neighbor_wind_speed_10m_max", "row_minus_precipitation_mean", "row_plus_precipitation_mean",
    "column_minus_precipitation_mean", "column_plus_precipitation_mean",
    "row_minus_cloud_cover_mean", "row_plus_cloud_cover_mean", "column_minus_cloud_cover_mean",
    "column_plus_cloud_cover_mean", "neighbor_precipitation_mean_minus_center",
    "neighbor_cloud_cover_mean_minus_center", "neighbor_relative_humidity_mean_minus_center",
    "center_pressure_msl_minus_neighbor_mean",
]
FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + NEIGHBOR_FEATURE_COLUMNS

ROW_FILTER_SQL = '''
      om.pressure_msl_change_6h IS NOT NULL
      AND om.precipitation_lag_6h IS NOT NULL
      AND om.precipitation_sum_past_24h IS NOT NULL
      AND om.cloud_cover_lag_6h IS NOT NULL
      AND om.humidity_lag_6h IS NOT NULL
      AND om.wind_speed_lag_3h IS NOT NULL
      AND om.neighbor_count > 0
'''


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# --------------------------------------------------------------------------- data

def read_training_data(batch_size=200_000):
    feature_sql = ",\n        ".join(
        f"COALESCE(om.{c}::real, 'NaN'::real) AS {c}" for c in FEATURE_COLUMNS)
    lead_sql = ",\n            ".join(
        f"LEAD(precipitation_mm, {k}) OVER w AS p{k}, "
        f"LEAD(local_observation_time, {k}) OVER w AS t{k}" for k in range(1, MAX_LEAD + 1))
    guarded_sql = ",\n        ".join(
        f"CASE WHEN im.t{k} = im.t_local + interval '{k} hour' "
        f"THEN im.p{k}::real ELSE 'NaN'::real END AS p{k}" for k in range(1, MAX_LEAD + 1))
    query = f'''
    WITH im AS (
        SELECT grid_number, local_observation_time AS t_local, {lead_sql}
        FROM "IMERG_BKK_DATA" WHERE is_complete_hour
        WINDOW w AS (PARTITION BY grid_number ORDER BY local_observation_time))
    SELECT om.grid_number, om.local_forecast_time, {feature_sql}, {guarded_sql}
    FROM "OM_BKK_DATA_PRECOMPUTE" om
    JOIN im ON im.grid_number = om.grid_number AND im.t_local = om.local_forecast_time
    WHERE {ROW_FILTER_SQL}
    ORDER BY om.local_forecast_time, om.grid_number'''

    n_feat = len(FEATURE_COLUMNS)
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(f'''SELECT count(*) FROM "OM_BKK_DATA_PRECOMPUTE" om
                JOIN (SELECT grid_number, local_observation_time AS t_local
                      FROM "IMERG_BKK_DATA" WHERE is_complete_hour) im
                  ON im.grid_number = om.grid_number AND im.t_local = om.local_forecast_time
                WHERE {ROW_FILTER_SQL}''')
            n_rows = cur.fetchone()[0]
        log(f"allocating {n_rows:,} x {n_feat} float32 ({n_rows * n_feat * 4 / 1e9:.2f} GB)")
        x = np.empty((n_rows, n_feat), dtype="float32")
        future = np.empty((n_rows, MAX_LEAD), dtype="float32")
        t = np.empty(n_rows, dtype="datetime64[s]")
        filled = 0
        with conn.cursor(name="bkk_seasonal") as cur:
            cur.itersize = batch_size
            cur.execute(query)
            while filled < n_rows:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                block = np.asarray(batch, dtype="object")
                take = len(batch)
                t[filled:filled + take] = block[:, 1].astype("datetime64[s]")
                x[filled:filled + take] = block[:, 2:2 + n_feat].astype("float32")
                future[filled:filled + take] = block[:, 2 + n_feat:].astype("float32")
                filled += take
                del block
        gc.collect()
    return x[:filled], future[:filled], t[:filled]


def build_model(model_name, y_fit):
    if model_name == "lightgbm":
        scale_pos_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
        return LGBMClassifier(
            objective="binary", n_estimators=500, learning_rate=0.04, num_leaves=63,
            min_child_samples=80, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=250, max_leaf_nodes=31,
        l2_regularization=0.05, random_state=RANDOM_STATE, early_stopping=False)


def compute_oof(x_all, y_all, forecast_time, year_key):
    purge = np.timedelta64(PURGE_HOURS, "h")

    def purged_train_mask(test_mask):
        test_times = np.unique(forecast_time[test_mask])
        position = np.searchsorted(test_times, forecast_time)
        left = test_times[np.clip(position - 1, 0, len(test_times) - 1)]
        right = test_times[np.clip(position, 0, len(test_times) - 1)]
        distance = np.minimum(np.abs(forecast_time - left), np.abs(forecast_time - right))
        return ~test_mask & (distance > purge)

    folds = [int(y) for y in np.unique(year_key) if (year_key == y).sum() >= MIN_FOLD_ROWS]
    oof = np.full((x_all.shape[0], len(TARGETS)), np.nan, dtype="float32")
    started = time.time()
    for year in folds:
        test_mask = year_key == year
        train_mask = purged_train_mask(test_mask)
        x_train, x_test = x_all[train_mask], x_all[test_mask]
        log(f"fold {year}: test {test_mask.sum():,} train {train_mask.sum():,}")
        for i, (horizon, _) in enumerate(TARGETS):
            y_train = y_all[train_mask, i]
            model = build_model(MODEL_PLAN[horizon], y_train)
            model.fit(x_train, y_train)
            oof[test_mask, i] = model.predict_proba(x_test)[:, 1].astype("float32")
            del model, y_train
            gc.collect()
        del x_train, x_test
        gc.collect()
    log(f"OOF finished in {(time.time() - started) / 60:.1f} min; "
        f"coverage {np.isfinite(oof).all(axis=1).mean():.4f}")
    return oof, folds


# --------------------------------------------------------- thresholds and scoring

def sweep(probabilities, y_true, objective="f1"):
    """Exact optimal cut for `probability >= threshold`, via one sort.

    `f1` maximises 2PR/(P+R). It chases the base rate: where positives are common, recall is
    cheap and the optimum slides toward firing on everything - which is what made a single
    global cut fire on 85% of wet-season cell-hours.

    `tss` maximises the True Skill Statistic, TPR - FPR (Hanssen-Kuipers). Each term is
    normalised by its own class count, so the optimum does not move with the base rate. This is
    the standard choice for meteorological warnings for exactly that reason.
    """
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    order = np.argsort(probabilities, kind="stable")[::-1]
    p_sorted = probabilities[order]
    tp = np.cumsum(y_true[order].astype("int64"))
    predicted_positive = np.arange(1, len(p_sorted) + 1, dtype="int64")
    positives = int(tp[-1])
    negatives = len(y_true) - positives
    if objective.startswith("precision"):
        # Lowest cut whose precision clears the floor - i.e. the most recall available at an
        # alert reliability the user can actually live with. Both failure modes above are
        # precision failures, so constraining precision directly is the honest fix.
        floor = float(objective.split(":")[1]) if ":" in objective else 0.5
        precision = tp / predicted_positive
        eligible = (precision >= floor) & (predicted_positive >= MIN_ALERTS)
        if not eligible.any():
            # unreachable this month: fall back to the most reliable cut with real support
            supported = predicted_positive >= MIN_ALERTS
            best = int(np.where(supported, precision, -np.inf).argmax())
            threshold = float(p_sorted[best])
            del order, p_sorted, tp, predicted_positive, precision, eligible
            gc.collect()
            return threshold
        best = int(np.max(np.flatnonzero(eligible)))    # largest k => lowest cut => most recall
        threshold = float(p_sorted[best])
        del order, p_sorted, tp, predicted_positive, precision, eligible
        gc.collect()
        return threshold
    if objective == "tss":
        false_positives = predicted_positive - tp
        criterion = tp / positives - false_positives / negatives
    else:
        precision = tp / predicted_positive
        recall = tp / positives
        criterion = np.divide(2 * precision * recall, precision + recall,
                              out=np.zeros_like(precision), where=(precision + recall) > 0)
        del precision, recall
    realisable = np.empty(len(p_sorted), dtype=bool)
    realisable[:-1] = p_sorted[:-1] > p_sorted[1:]
    realisable[-1] = True
    best = int(np.where(realisable, criterion, -np.inf).argmax())
    threshold = float(p_sorted[best])
    del order, p_sorted, tp, predicted_positive, criterion, realisable
    gc.collect()
    return threshold


# Bangkok's southwest monsoon runs roughly May to October; the rest of the year is dry.
WET_MONTHS = frozenset(range(5, 11))

# A cut is only trusted if it still raises this many alerts in the month it was fitted on.
# Without it the precision-floor search drifts to the extreme tail, where one lucky alert
# reads as precision 1.0.
MIN_ALERTS = 200


def score(y_true, predictions):
    """Precision, recall, F1 and TSS from one pass over the confusion matrix."""
    predicted = predictions == 1
    actual = y_true == 1
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0
    base = (tp + fn) / len(y_true) if len(y_true) else 0.0
    return {
        "rows": int(len(y_true)),
        "base_rate": base,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tss": recall - false_alarm_rate,
        "fired_share": float(predicted.mean()) if len(y_true) else 0.0,
        "f1_always_yes": 2 * base / (base + 1) if base else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-oof", action="store_true",
                        help="load oof_probabilities.npz instead of re-running the CV")
    parser.add_argument("--ship", default="monthly_precision",
                        choices=["global_f1", "monthly_f1", "monthly_tss", "monthly_precision"],
                        help="which operating-point scheme to write into the bundles")
    parser.add_argument("--precision-floor", type=float, default=0.5,
                        help="alert reliability the monthly_precision scheme must clear")
    args = parser.parse_args()

    if args.reuse_oof and OOF_PATH.exists():
        log(f"loading {OOF_PATH}")
        store = np.load(OOF_PATH, allow_pickle=False)
        oof_raw, y_all = store["oof_raw"], store["y_all"]
        forecast_time = store["forecast_time"].astype("datetime64[s]")
    else:
        x_all, future_rain, forecast_time = read_training_data()
        complete = ~np.isnan(future_rain).any(axis=1)
        x_all = np.ascontiguousarray(x_all[complete])
        future_rain, forecast_time = future_rain[complete], forecast_time[complete]
        del complete
        gc.collect()
        y_all = np.empty((x_all.shape[0], len(TARGETS)), dtype="int8")
        for i, (h, thr) in enumerate(TARGETS):
            y_all[:, i] = (future_rain[:, :h].max(axis=1) >= thr).astype("int8")
        del future_rain
        gc.collect()
        year_key = pd.to_datetime(forecast_time).year.to_numpy()
        oof_raw, _ = compute_oof(x_all, y_all, forecast_time, year_key)
        del x_all
        gc.collect()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OOF_PATH, oof_raw=oof_raw, y_all=y_all,
                            forecast_time=forecast_time.astype("int64"))
        log(f"saved {OOF_PATH} ({OOF_PATH.stat().st_size / 1e6:.0f} MB) - "
            f"future threshold work can use --reuse-oof")

    stamps = pd.to_datetime(forecast_time)
    year_key = stamps.year.to_numpy()
    month_key = stamps.month.to_numpy()
    years = sorted({int(y) for y in np.unique(year_key)
                    if (year_key == y).sum() >= MIN_FOLD_ROWS})

    # Three candidate operating-point schemes, all evaluated leave-one-year-out.
    SCHEMES = ["global_f1", "monthly_f1", "monthly_tss", "monthly_precision"]
    OBJECTIVES = {"monthly_f1": "f1", "monthly_tss": "tss",
                  "monthly_precision": f"precision:{args.precision_floor}"}
    rows, monthly_rows, pending = [], [], []

    for i, name in enumerate(TARGET_NAMES):
        path = next(p for p in MODEL_DIR.glob(f"*_next_{TARGETS[i][0]}h_*")
                    if p.name.endswith(f"rain_threshold_{TARGETS[i][1]}mm.joblib"))
        bundle = joblib.load(path)

        y_true = y_all[:, i]
        raw = oof_raw[:, i]
        valid = np.isfinite(raw)
        calibrated = bundle["calibrator"].predict(raw).astype("float32")

        # what would ship, per scheme: cuts fitted on every year
        global_cut_all = sweep(calibrated[valid], y_true[valid])
        shipped = {"global_f1": {m: global_cut_all for m in range(1, 13)}}
        for scheme, objective in OBJECTIVES.items():
            cuts = {}
            for month in range(1, 13):
                pick = valid & (month_key == month)
                cuts[month] = (sweep(calibrated[pick], y_true[pick], objective)
                               if pick.sum() else 0.5)
            shipped[scheme] = cuts

        # honest evaluation: the cut applied to (year, month) is fitted on other years only
        predictions = {s: np.zeros(len(y_true), dtype="int8") for s in SCHEMES}
        for year in years:
            held = valid & (year_key == year)
            others = valid & (year_key != year)
            global_cut = sweep(calibrated[others], y_true[others])
            predictions["global_f1"][held] = (calibrated[held] >= global_cut).astype("int8")
            for month in range(1, 13):
                cell = held & (month_key == month)
                if not cell.sum():
                    continue
                pool = others & (month_key == month)
                for scheme, objective in OBJECTIVES.items():
                    cut = (sweep(calibrated[pool], y_true[pool], objective)
                           if pool.sum() else global_cut)
                    predictions[scheme][cell] = (calibrated[cell] >= cut).astype("int8")

        wet = valid & np.isin(month_key, list(WET_MONTHS))
        dry = valid & ~np.isin(month_key, list(WET_MONTHS))
        roc = float(roc_auc_score(y_true[valid], calibrated[valid]))
        for scheme in SCHEMES:
            for season, mask in (("all", valid), ("wet", wet), ("dry", dry)):
                entry = score(y_true[mask], predictions[scheme][mask])
                entry.update(target=name, scheme=scheme, season=season, roc_auc=roc)
                rows.append(entry)
        for scheme in SCHEMES:
            for month, cut in shipped[scheme].items():
                pick = valid & (month_key == month)
                monthly_rows.append({
                    "target": name, "scheme": scheme, "month": month, "threshold": cut,
                    "base_rate": float(y_true[pick].mean()) if pick.sum() else np.nan,
                    "rows": int(pick.sum())})

        def line(scheme, season):
            r = next(x for x in rows if x["target"] == name
                     and x["scheme"] == scheme and x["season"] == season)
            return (f"F1 {r['f1']:.3f} TSS {r['tss']:.3f} fires {r['fired_share']:5.1%} "
                    f"prec {r['precision']:.3f}")
        log(f"{name:12} ROC {roc:.4f}")
        for scheme in SCHEMES:
            log(f"   {scheme:12} wet[{line(scheme, 'wet')}]  dry[{line(scheme, 'dry')}]")

        pending.append((name, path, bundle, shipped))
        del calibrated, raw, valid, predictions
        gc.collect()

    frame = pd.DataFrame(rows)[
        ["target", "scheme", "season", "rows", "base_rate", "roc_auc", "precision", "recall",
         "f1", "f1_always_yes", "tss", "fired_share"]]
    monthly_frame = pd.DataFrame(monthly_rows)
    frame.to_csv(MODEL_DIR / "threshold_scheme_comparison.csv", index=False)
    monthly_frame.to_csv(MODEL_DIR / "seasonal_thresholds_by_month.csv", index=False)

    print()
    for season in ("wet", "dry"):
        print(f"=== {season} season ===")
        pivot = frame[frame["season"] == season].pivot_table(
            index="target", columns="scheme", values=["f1", "tss", "fired_share"])
        print(pivot.to_string())
        print()

    # Ship the requested scheme.
    chosen = args.ship
    for name, path, bundle, shipped in pending:
        cuts = shipped[chosen]
        bundle["seasonal_thresholds"] = {int(k): float(v) for k, v in cuts.items()}
        bundle["threshold_mode"] = chosen
        bundle["threshold_objective"] = "tss" if chosen.endswith("tss") else "f1"
        bundle["threshold_selection"] = (
            f"{chosen}: per calendar month, exact "
            f"{'TSS' if chosen.endswith('tss') else 'F1'}-optimal on calibrated OOF "
            "probabilities; reported metrics use leave-one-year-out cuts, so no row is scored "
            "at a cut fitted on it")
        bundle["threshold_scheme_metrics"] = {
            s: {r["season"]: {k: r[k] for k in
                              ("precision", "recall", "f1", "tss", "fired_share")}
                for r in rows if r["target"] == name and r["scheme"] == s}
            for s in SCHEMES}
        chosen_all = bundle["threshold_scheme_metrics"][chosen]["all"]
        bundle["metrics"].update({k: chosen_all[k] for k in ("precision", "recall", "f1")})
        bundle["retuned_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        joblib.dump(bundle, path, compress=3)

    (MODEL_DIR / "seasonal_threshold_metadata.json").write_text(json.dumps({
        "shipped_scheme": chosen,
        "schemes_compared": SCHEMES,
        "wet_months": sorted(WET_MONTHS),
        "shipped_cut": "fitted on all years of that month, from calibrated OOF probabilities",
        "reported_metrics": "leave-one-year-out per (year, month) - unbiased",
        "note": ("F1-optimal cuts chase the base rate and collapse toward always-yes in the wet "
                 "season; TSS (TPR - FPR) normalises by each class count and does not."),
        "comparison": frame.to_dict(orient="records"),
        "by_month": monthly_frame.to_dict(orient="records"),
    }, indent=2), encoding="utf-8")
    log(f"shipped scheme '{chosen}' into 6 bundles; comparisons in {MODEL_DIR}")


if __name__ == "__main__":
    main()
