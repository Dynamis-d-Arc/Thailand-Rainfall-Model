"""Head-to-head on IMERG: the V0 baseline arms alongside the V1 full-feature models.

Extends `compare_tmd_vs_om_on_imerg.py` from 2 models to 4:

    BKK_Rain_V0   36 own-cell features        Open-Meteo grid
    BKK_Rain_V1   66 (+30 neighbourhood)      Open-Meteo grid
    TMD_BKK_V0    40 own-station features     TMD station network
    TMD_BKK_V1    63 (+23 neighbourhood)      TMD station network

All four are scored on ONE common yardstick, exactly as the two-model version did:

  same locations  - the 3 TMD stations inside the 56-cell Bangkok grid that "OM_BKK_DATA"
                    and "IMERG_BKK_DATA" share (104, 106, 37), each paired with its cell
  same hours      - the intersection of the notebooks' test splits, Asia/Bangkok local time
  same labels     - rain_any_next_Nh rebuilt from "IMERG_BKK_DATA", independent of every
                    model's training labels

Because V0's feature list is a strict subset of V1's (asserted in both V0 notebooks), each
side builds its feature frame ONCE at V1 width and every model selects its own columns by
name from that frame. All four models therefore see byte-identical rows, and the IMERG base
rate is identical for all of them by construction - which is what makes precision, recall
and F1 comparable here even though they are not comparable between the source notebooks.

Writes comparison CSVs next to this script. Reads only; trains nothing.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
TRAINED = PROJECT_ROOT / "ML_Model_V2" / "trained_models"
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}
HORIZONS = [1, 2, 3, 6]
RAIN_THRESHOLD_MM = 0.1
LOCAL_OFFSET = pd.Timedelta(hours=7)  # Asia/Bangkok, verified against IMERG_BKK_DATA

# (label, side, model dir, filename glob template, metadata file, key holding feature list)
MODEL_SPECS = [
    ("BKK_Rain_V0", "om", TRAINED / "om_bkk_rain_any_v0_baseline",
     "om_bkk_rain_any_v0_next_{h}h_*.joblib", "v0_metadata.json", "feature_columns"),
    ("BKK_Rain_V1", "om", TRAINED / "om_bkk_rain_any_v1",
     "om_bkk_rain_any_v1_next_{h}h_*.joblib", "bkk_rain_v1_metadata.json", "feature_columns"),
    ("TMD_BKK_V0", "tmd", TRAINED / "tmd_bkk_rain_any_v0_station_only",
     "tmd_bkk_rain_any_v0_next_{h}h_*.joblib", "v0_metadata.json", "feature_columns"),
    ("TMD_BKK_V1", "tmd", TRAINED / "tmd_bkk_rain_any_final_4_neighbor_models",
     "tmd_bkk_rain_any_next_{h}h_*.joblib", "final_4_metadata.json", "feature_columns"),
]


def connect():
    return psycopg2.connect(**DB_CONFIG)


def notebook_namespace(path, upto_code_cell):
    """Exec the first N code cells of a notebook so we reuse its exact feature code."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__notebook__", "display": lambda *a: None, "get_ipython": lambda: None}
    for src in codes[:upto_code_cell]:
        exec(compile(src, str(path), "exec"), ns)
    return ns


def load_spec(spec):
    """Resolve a MODEL_SPEC into its feature list plus per-horizon (model, threshold)."""
    label, side, directory, glob_template, meta_name, feature_key = spec
    meta = json.loads((directory / meta_name).read_text(encoding="utf-8"))
    features = meta[feature_key]
    per_horizon = {}
    for horizon in HORIZONS:
        path = next(directory.glob(glob_template.format(h=horizon)), None)
        if path is None:
            raise FileNotFoundError(f"{label}: no model for {horizon}h in {directory}")
        threshold = float(path.stem.split("prob_threshold_")[1].split("_")[0])
        per_horizon[horizon] = (path, threshold)
    return {"label": label, "side": side, "features": features, "models": per_horizon}


# --------------------------------------------------------------------------------------
# 1. Common locations
# --------------------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))


def build_station_grid_pairs():
    with connect() as conn:
        stations = pd.read_sql_query(
            'SELECT station, station_name, latitude, longitude FROM "BKK_TMD_WEATHER_DATA" '
            "WHERE region = '1' GROUP BY 1,2,3,4", conn)
        grids = pd.read_sql_query(
            'SELECT grid_number, latitude, longitude FROM "OM_BKK_DATA" GROUP BY 1,2,3', conn)

    box = ((stations["latitude"].between(grids["latitude"].min(), grids["latitude"].max()))
           & (stations["longitude"].between(grids["longitude"].min(), grids["longitude"].max())))
    inside = stations[box].copy()

    pairs = []
    for row in inside.itertuples(index=False):
        d = haversine_km(row.latitude, row.longitude, grids["latitude"], grids["longitude"])
        best = grids.loc[d.idxmin()]
        pairs.append({"station": row.station, "station_name": row.station_name,
                      "grid_number": int(best.grid_number),
                      "pair_distance_km": float(d.min())})
    return pd.DataFrame(pairs)


# --------------------------------------------------------------------------------------
# 2. TMD side - build the panel ONCE at V1 width, predict with every TMD model
# --------------------------------------------------------------------------------------
def tmd_frame(stations):
    ns = notebook_namespace(PROJECT_ROOT / "TMD_BKK_V1.ipynb", upto_code_cell=4)

    raw = ns["read_raw_tmd_data"]()
    raw = raw[raw["station"].isin(stations)].reset_index(drop=True)
    panel = ns["to_hourly_panel"](raw)
    panel = ns["add_derived_features"](panel)
    panel = ns["add_lag_features"](panel)
    panel = ns["add_targets"](panel)
    panel = panel.dropna(subset=ns["REQUIRED_NON_NULL_COLUMNS"]).copy()
    panel = panel[panel["nearby_station_count_100km"] > 0]

    # TMD stores UTC; align onto the Asia/Bangkok clock the other two tables use.
    panel["t_local"] = panel["forecast_time"] + LOCAL_OFFSET

    keep = ["station", "t_local"] + [f"rain_any_next_{h}h" for h in HORIZONS]
    out = panel[keep].rename(
        columns={f"rain_any_next_{h}h": f"tmd_label_{h}h" for h in HORIZONS})
    return panel, out


def om_frame(grid_numbers):
    ns = notebook_namespace(PROJECT_ROOT / "BKK_Rain_V1.ipynb", upto_code_cell=4)
    feature_columns = ns["FEATURE_COLUMNS"]  # 66, superset of BKK_Rain_V0's 36

    feature_sql = ",\n        ".join(f"COALESCE({c}::real, 'NaN'::real) AS {c}" for c in feature_columns)
    target_sql = ",\n        ".join(f"rain_any_next_{h}h::smallint AS om_label_{h}h" for h in HORIZONS)
    query = f"""
    SELECT grid_number, local_forecast_time AS t_local,
        {feature_sql},
        {target_sql}
    FROM "OM_BKK_DATA_PRECOMPUTE"
    WHERE grid_number = ANY(%(grids)s)
      AND rain_any_next_6h IS NOT NULL
      AND neighbor_count > 0
    ORDER BY local_forecast_time, grid_number
    """
    with connect() as conn:
        df = pd.read_sql_query(query, conn, params={"grids": list(grid_numbers)},
                               parse_dates=["t_local"])
    keep = ["grid_number", "t_local"] + [f"om_label_{h}h" for h in HORIZONS]
    return df, df[keep].copy()


# --------------------------------------------------------------------------------------
# 3. IMERG labels, built exactly as BKK_Rain_V1 section 11 builds them
# --------------------------------------------------------------------------------------
def imerg_labels(grid_numbers):
    max_h = max(HORIZONS)
    leads = ",\n        ".join(
        f"LEAD(precipitation_mm, {h}) OVER w AS nxt_{h}" for h in range(1, max_h + 1))
    cols = ", ".join(f"nxt_{h}" for h in range(1, max_h + 1))
    query = f"""
    WITH iw AS (
        SELECT grid_number, local_observation_time AS t_local,
        {leads}
        FROM "IMERG_BKK_DATA"
        WHERE is_complete_hour AND grid_number = ANY(%(grids)s)
        WINDOW w AS (PARTITION BY grid_number ORDER BY local_observation_time)
    )
    SELECT grid_number, t_local, {cols} FROM iw WHERE nxt_{max_h} IS NOT NULL
    """
    with connect() as conn:
        imerg = pd.read_sql_query(query, conn, params={"grids": list(grid_numbers)},
                                  parse_dates=["t_local"])
    for horizon in HORIZONS:
        window = [f"nxt_{h}" for h in range(1, horizon + 1)]
        imerg[f"imerg_label_{horizon}h"] = (
            (imerg[window] >= RAIN_THRESHOLD_MM).any(axis=1).astype("int8"))
    return imerg[["grid_number", "t_local"] + [f"imerg_label_{h}h" for h in HORIZONS]]


def score(y_true, probabilities, predictions):
    base = float(np.mean(y_true))
    pr = float(average_precision_score(y_true, probabilities))
    return {
        "base_rate": base,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": pr,
        "pr_auc_lift": pr / base if base else float("nan"),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def contingency_scores(y_true, predictions):
    """Chance-corrected scores, valid to compare across differing base rates."""
    y_true = y_true.astype(bool)
    predictions = predictions.astype(bool)
    tp = int(np.sum(y_true & predictions))
    fp = int(np.sum(~y_true & predictions))
    fn = int(np.sum(y_true & ~predictions))
    tn = int(np.sum(~y_true & ~predictions))
    n = tp + fp + fn + tn
    hits_by_chance = (tp + fp) * (tp + fn) / n if n else 0.0
    ets_denominator = tp + fp + fn - hits_by_chance
    hss_denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "csi": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
        "ets": (tp - hits_by_chance) / ets_denominator if ets_denominator else float("nan"),
        "hss": 2 * (tp * tn - fp * fn) / hss_denominator if hss_denominator else float("nan"),
        "pss": (tp / (tp + fn) if (tp + fn) else np.nan) - (fp / (fp + tn) if (fp + tn) else np.nan),
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
    }


def main():
    specs = [load_spec(s) for s in MODEL_SPECS]
    for s in specs:
        print(f"{s['label']:<14} {len(s['features']):>3} features  "
              f"thresholds={{{', '.join(f'{h}: {t:g}' for h, (_, t) in s['models'].items())}}}")

    pairs = build_station_grid_pairs()
    print("\nCo-located station / grid pairs:")
    print(pairs.to_string(index=False))

    tmd_panel, tmd_keys = tmd_frame(pairs["station"].tolist())
    om_panel, om_keys = om_frame(pairs["grid_number"].tolist())
    imerg = imerg_labels(pairs["grid_number"].tolist())
    print(f"\nrows -> tmd {len(tmd_keys):,} | om {len(om_keys):,} | imerg {len(imerg):,}")

    # Predict on the full per-side frames first, then merge - so every model's probabilities
    # ride along the same join and land on identical rows.
    for spec in specs:
        panel = tmd_panel if spec["side"] == "tmd" else om_panel
        keys = tmd_keys if spec["side"] == "tmd" else om_keys
        x = np.ascontiguousarray(panel[spec["features"]].to_numpy(dtype="float32"))
        for horizon, (path, threshold) in spec["models"].items():
            model = joblib.load(path)
            keys[f"{spec['label']}_prob_{horizon}h"] = model.predict_proba(x)[:, 1]
            keys[f"{spec['label']}_thr_{horizon}h"] = threshold
            del model

    tmd_keys = tmd_keys.merge(pairs[["station", "grid_number"]], on="station", how="inner")
    merged = tmd_keys.merge(om_keys, on=["grid_number", "t_local"], how="inner")
    merged = merged.merge(imerg, on=["grid_number", "t_local"], how="inner")

    # Restrict to the intersection of the notebooks' TEST splits. V0 and V1 share split
    # boundaries on each side (asserted in both V0 notebooks), so the two-model window holds.
    om_meta = json.loads((TRAINED / "om_bkk_rain_any_v1" / "bkk_rain_v1_metadata.json")
                         .read_text(encoding="utf-8"))
    tmd_meta = json.loads((TRAINED / "tmd_bkk_rain_any_final_4_neighbor_models" / "final_4_metadata.json")
                          .read_text(encoding="utf-8"))
    window_start = max(pd.Timestamp(om_meta["validation_end_exclusive"]),
                       pd.Timestamp(tmd_meta["validation_end_exclusive"]) + LOCAL_OFFSET)
    merged = merged[merged["t_local"] >= window_start].copy()

    print(f"\ncommon window          : {window_start} .. {merged['t_local'].max()}")
    print(f"common evaluation rows : {len(merged):,} across {merged['station'].nunique()} stations")

    rows = []
    for spec in specs:
        for horizon in HORIZONS:
            probabilities = merged[f"{spec['label']}_prob_{horizon}h"].to_numpy()
            threshold = merged[f"{spec['label']}_thr_{horizon}h"].iloc[0]
            predictions = (probabilities >= threshold).astype("int8")
            for label_name, label_col in [
                ("imerg", f"imerg_label_{horizon}h"),
                ("tmd_gauge", f"tmd_label_{horizon}h"),
                ("open_meteo", f"om_label_{horizon}h"),
            ]:
                y_true = merged[label_col].to_numpy().astype("int8")
                if y_true.min() == y_true.max():
                    continue
                rows.append({
                    "model": spec["label"], "features": len(spec["features"]),
                    "horizon_h": horizon, "label_source": label_name,
                    "threshold": float(threshold), "rows": int(len(y_true)),
                    **score(y_true, probabilities, predictions),
                    **contingency_scores(y_true, predictions),
                })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "v0_v1_common_imerg_comparison.csv", index=False)
    merged.to_csv(OUT_DIR / "v0_v1_common_rows.csv.gz", index=False, compression="gzip")

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:.4f}".format)

    imerg_only = results[results["label_source"] == "imerg"]
    print("\n================ ALL FOUR MODELS, GRADED ON IMERG ================")
    print(imerg_only[["horizon_h", "model", "features", "base_rate", "roc_auc", "pr_auc",
                      "pr_auc_lift", "precision", "recall", "f1", "ets", "hss"]]
          .sort_values(["horizon_h", "model"]).to_string(index=False))

    print("\n================ ROC-AUC ON IMERG ================")
    print(imerg_only.pivot(index="horizon_h", columns="model", values="roc_auc").to_string())

    print("\n================ F1 ON IMERG (comparable here - identical rows) ================")
    print(imerg_only.pivot(index="horizon_h", columns="model", values="f1").to_string())

    print("\n================ EACH MODEL AGAINST EVERY LABEL SOURCE (ROC-AUC) ================")
    print(results.pivot_table(index=["horizon_h", "model"], columns="label_source",
                              values="roc_auc").to_string())

    print(f"\nSaved -> {OUT_DIR / 'v0_v1_common_imerg_comparison.csv'}")


if __name__ == "__main__":
    main()
