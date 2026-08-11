"""Head-to-head: TMD_BKK_V1 vs BKK_Rain_V1, graded against the same IMERG observations.

The two notebooks report metrics against their own label sources, which are not comparable
(Open-Meteo `precipitation` is ECMWF model output; TMD `precipitation_mm` is a gauge).
This script scores both models on ONE common yardstick:

  same locations  - the 3 TMD stations that fall inside the 56-cell Bangkok grid that
                    "OM_BKK_DATA" and "IMERG_BKK_DATA" share (104, 106, 37), each paired
                    with the grid cell containing it
  same hours      - the intersection of the two notebooks' test splits, in Asia/Bangkok
                    local time (TMD is stored UTC; local = UTC + 7h, verified against
                    IMERG's own local/UTC offset column)
  same labels     - rain_any_next_Nh rebuilt from "IMERG_BKK_DATA", the satellite record,
                    which is independent of both models' training labels

Only 3 of 26 TMD stations can be used: IMERG_BKK_DATA and OM_BKK_DATA cover a small box
over Bangkok (13.49-13.96N, 100.33-100.91E) while the TMD network spans the whole central
basin. The other hourly IMERG tables do not help - `imerg_grid_hourly` holds two weeks and
`IMERG_DATA` / `TMD_IMERG_DAILY_DATA` are daily.

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

TMD_MODEL_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "tmd_bkk_rain_any_final_4_neighbor_models"
OM_MODEL_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_any_v1"


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


# --------------------------------------------------------------------------------------
# 1. Common locations: TMD stations inside the shared Bangkok grid, paired to their cell
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
# 2. TMD side: rebuild features with the notebook's own code, predict with saved models
# --------------------------------------------------------------------------------------
def tmd_predictions(stations):
    ns = notebook_namespace(PROJECT_ROOT / "TMD_BKK_V1.ipynb", upto_code_cell=4)
    feature_columns = ns["FEATURE_COLUMNS"]

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

    x = np.ascontiguousarray(panel[feature_columns].to_numpy(dtype="float32"))
    out = panel[["station", "t_local"] + [f"rain_any_next_{h}h" for h in HORIZONS]].copy()
    out = out.rename(columns={f"rain_any_next_{h}h": f"tmd_label_{h}h" for h in HORIZONS})

    for horizon in HORIZONS:
        model_path = next(TMD_MODEL_DIR.glob(f"tmd_bkk_rain_any_next_{horizon}h_*.joblib"))
        threshold = float(model_path.stem.split("prob_threshold_")[1].split("_")[0])
        model = joblib.load(model_path)
        out[f"tmd_prob_{horizon}h"] = model.predict_proba(x)[:, 1]
        out[f"tmd_threshold_{horizon}h"] = threshold
    return out


# --------------------------------------------------------------------------------------
# 3. Open-Meteo side: pull precompute rows for the paired cells, predict with saved models
# --------------------------------------------------------------------------------------
def om_predictions(grid_numbers):
    ns = notebook_namespace(PROJECT_ROOT / "BKK_Rain_V1.ipynb", upto_code_cell=4)
    feature_columns = ns["FEATURE_COLUMNS"]

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

    x = np.ascontiguousarray(df[feature_columns].to_numpy(dtype="float32"))
    out = df[["grid_number", "t_local"] + [f"om_label_{h}h" for h in HORIZONS]].copy()
    for horizon in HORIZONS:
        model_path = next(OM_MODEL_DIR.glob(f"om_bkk_rain_any_v1_next_{horizon}h_*.joblib"))
        threshold = float(model_path.stem.split("prob_threshold_")[1].split("_")[0])
        model = joblib.load(model_path)
        out[f"om_prob_{horizon}h"] = model.predict_proba(x)[:, 1]
        out[f"om_threshold_{horizon}h"] = threshold
    return out


# --------------------------------------------------------------------------------------
# 4. IMERG labels, built exactly as BKK_Rain_V1 section 11 builds them
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


def main():
    pairs = build_station_grid_pairs()
    print("Co-located station / grid pairs:")
    print(pairs.to_string(index=False))

    tmd = tmd_predictions(pairs["station"].tolist())
    om = om_predictions(pairs["grid_number"].tolist())
    imerg = imerg_labels(pairs["grid_number"].tolist())
    print(f"\nrows -> tmd {len(tmd):,} | om {len(om):,} | imerg {len(imerg):,}")

    # Attach each TMD station to its paired grid, then inner-join everything on
    # (grid_number, local hour). The inner join IS the common evaluation set.
    tmd = tmd.merge(pairs[["station", "grid_number"]], on="station", how="inner")
    merged = tmd.merge(om, on=["grid_number", "t_local"], how="inner")
    merged = merged.merge(imerg, on=["grid_number", "t_local"], how="inner")

    # Restrict to the intersection of both notebooks' TEST splits (both models are
    # out-of-sample here). Boundaries come from the saved run metadata.
    om_meta = json.loads((OM_MODEL_DIR / "bkk_rain_v1_metadata.json").read_text(encoding="utf-8"))
    tmd_meta = json.loads((TMD_MODEL_DIR / "final_4_metadata.json").read_text(encoding="utf-8"))
    om_test_start = pd.Timestamp(om_meta["validation_end_exclusive"])
    tmd_test_start = pd.Timestamp(tmd_meta["validation_end_exclusive"]) + LOCAL_OFFSET
    window_start = max(om_test_start, tmd_test_start)
    merged = merged[merged["t_local"] >= window_start].copy()

    print(f"\nOM test starts (local)  : {om_test_start}")
    print(f"TMD test starts (local) : {tmd_test_start}")
    print(f"common window           : {window_start} .. {merged['t_local'].max()}")
    print(f"common evaluation rows  : {len(merged):,} "
          f"across {merged['station'].nunique()} stations")

    rows = []
    for horizon in HORIZONS:
        for name, prob_col, thr_col in [
            ("TMD_BKK_V1", f"tmd_prob_{horizon}h", f"tmd_threshold_{horizon}h"),
            ("BKK_Rain_V1", f"om_prob_{horizon}h", f"om_threshold_{horizon}h"),
        ]:
            probabilities = merged[prob_col].to_numpy()
            predictions = (probabilities >= merged[thr_col].iloc[0]).astype("int8")
            for label_name, label_col in [
                ("imerg", f"imerg_label_{horizon}h"),
                ("tmd_gauge", f"tmd_label_{horizon}h"),
                ("open_meteo", f"om_label_{horizon}h"),
            ]:
                y_true = merged[label_col].to_numpy().astype("int8")
                if y_true.min() == y_true.max():
                    continue
                rows.append({"model": name, "horizon_h": horizon, "label_source": label_name,
                             "threshold": float(merged[thr_col].iloc[0]),
                             "rows": int(len(y_true)), **score(y_true, probabilities, predictions)})

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "tmd_vs_om_common_imerg_comparison.csv", index=False)
    merged.to_csv(OUT_DIR / "tmd_vs_om_common_rows.csv.gz", index=False, compression="gzip")

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", "{:.4f}".format)
    head = results[results["label_source"] == "imerg"]
    print("\n================ SCORED AGAINST IMERG (the common yardstick) ================")
    print(head[["horizon_h", "model", "base_rate", "roc_auc", "pr_auc", "pr_auc_lift",
                "precision", "recall", "f1"]].sort_values(["horizon_h", "model"]).to_string(index=False))

    print("\n================ EACH MODEL AGAINST EVERY LABEL SOURCE ================")
    print(results.pivot_table(index=["horizon_h", "model"], columns="label_source",
                              values=["roc_auc", "pr_auc_lift", "f1"]).to_string())
    print(f"\nSaved -> {OUT_DIR / 'tmd_vs_om_common_imerg_comparison.csv'}")


if __name__ == "__main__":
    main()
