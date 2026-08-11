import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

TRAINING_TABLE = "bkk_weather_lag_feature"
PREDICTION_FEATURE_TABLE = "bkk_weather_lag_feature"
OUTPUT_TABLE = "bkk_future_hourly_rain_probability_predictions"
MODEL_DIR = Path(__file__).with_name("trained_models") / "future_hour_models"
PREDICTION_DIR = Path(__file__).with_name("future_predictions")
MAP_DIR = Path(__file__).with_name("prediction_maps")
RAIN_THRESHOLD_MM = 0.1

FEATURE_COLUMNS = [
    "temperature_2m",
    "dew_point_2m",
    "temperature_dew_point_spread",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_lag_1h",
    "precipitation_lag_3h",
    "precipitation_sum_past_3h",
    "precipitation_sum_past_6h",
    "cloud_cover",
    "cloud_cover_avg_past_3h",
    "pressure_msl",
    "pressure_msl_change_3h",
    "surface_pressure",
    "wind_speed_10m",
    "wind_speed_10m_avg_past_3h",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]


CREATE_OUTPUT_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
    grid_number integer NOT NULL,
    grid_row integer NOT NULL,
    grid_column integer NOT NULL,
    longitude double precision NOT NULL,
    latitude double precision NOT NULL,
    as_of_time timestamp with time zone NOT NULL,
    target_time timestamp with time zone NOT NULL,
    horizon_hours integer NOT NULL,
    rain_probability double precision NOT NULL,
    model_name text NOT NULL,
    model_path text NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    PRIMARY KEY (grid_number, as_of_time, horizon_hours, model_name)
);

CREATE INDEX IF NOT EXISTS bkk_future_hourly_rain_probability_target_time_idx
  ON {OUTPUT_TABLE} (target_time);
"""


INSERT_OUTPUT_SQL = f"""
INSERT INTO {OUTPUT_TABLE} (
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    as_of_time,
    target_time,
    horizon_hours,
    rain_probability,
    model_name,
    model_path
)
VALUES %s
ON CONFLICT (grid_number, as_of_time, horizon_hours, model_name) DO UPDATE SET
    grid_row = EXCLUDED.grid_row,
    grid_column = EXCLUDED.grid_column,
    longitude = EXCLUDED.longitude,
    latitude = EXCLUDED.latitude,
    target_time = EXCLUDED.target_time,
    rain_probability = EXCLUDED.rain_probability,
    model_path = EXCLUDED.model_path,
    created_at = now();
"""


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def connect():
    return psycopg2.connect(**DB_CONFIG)


def parse_horizons(value):
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise argparse.ArgumentTypeError("Use positive hour values, for example: 1,2,3")
    return horizons


def model_path_for(horizon, model_dir):
    return model_dir / f"bkk_rain_exact_next_{horizon}h_hist_gradient_boosting.joblib"


def metadata_path_for(horizon, model_dir):
    return model_dir / f"bkk_rain_exact_next_{horizon}h_hist_gradient_boosting_metadata.json"


def build_training_query(table, horizon, rain_threshold_mm, sample_rows):
    columns = [
        "grid_number",
        "forecast_time",
        *FEATURE_COLUMNS,
        (
            f"(LEAD(precipitation, {horizon}) OVER "
            f"(PARTITION BY grid_number ORDER BY forecast_time) >= {rain_threshold_mm})::integer "
            f"AS rain_exact_next_{horizon}h"
        ),
    ]
    where_parts = [
        "precipitation_lag_3h IS NOT NULL",
        "precipitation_sum_past_6h IS NOT NULL",
        "pressure_msl_change_3h IS NOT NULL",
    ]
    feature_not_null = " AND ".join(f"{column} IS NOT NULL" for column in FEATURE_COLUMNS)
    where_parts.append(feature_not_null)

    query = f"""
    WITH labeled AS (
        SELECT
            {", ".join(columns)}
        FROM {table}
    )
    SELECT *
    FROM labeled
    WHERE rain_exact_next_{horizon}h IS NOT NULL
      AND {" AND ".join(where_parts)}
    ORDER BY forecast_time, grid_number
    """
    if sample_rows:
        query = f"""
        SELECT *
        FROM ({query}) labeled_rows
        ORDER BY random()
        LIMIT {int(sample_rows)}
        """
    return query


def load_training_data(conn, table, horizon, rain_threshold_mm, sample_rows):
    query = build_training_query(table, horizon, rain_threshold_mm, sample_rows)
    return pd.read_sql_query(query, conn, parse_dates=["forecast_time"])


def add_time_split(df, train_fraction, validation_fraction):
    unique_times = np.array(sorted(df["forecast_time"].unique()))
    if len(unique_times) < 3:
        raise RuntimeError("Not enough forecast times to create train/validation/test splits.")

    train_end = unique_times[max(1, int(len(unique_times) * train_fraction))]
    validation_end = unique_times[
        max(2, int(len(unique_times) * (train_fraction + validation_fraction)))
    ]

    df = df.copy()
    df["split"] = "test"
    df.loc[df["forecast_time"] < train_end, "split"] = "train"
    df.loc[
        (df["forecast_time"] >= train_end) & (df["forecast_time"] < validation_end),
        "split",
    ] = "validation"
    return df, train_end, validation_end


def build_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=250,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )


def evaluate(model, x, y, split_name):
    probabilities = model.predict_proba(x)[:, 1]
    metrics = {
        "split": split_name,
        "rows": int(len(y)),
        "rain_rate": float(y.mean()),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }
    if len(set(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y, probabilities))
    return metrics


def train_horizon_model(conn, args, horizon):
    target = f"rain_exact_next_{horizon}h"
    log(f"Loading training rows for exact +{horizon}h rain target...")
    df = load_training_data(
        conn,
        args.training_table,
        horizon,
        args.rain_threshold_mm,
        args.sample_rows,
    )
    if df.empty:
        raise RuntimeError(f"No training rows found for horizon {horizon}h.")

    df, train_end, validation_end = add_time_split(
        df,
        args.train_fraction,
        args.validation_fraction,
    )
    train_df = df[df["split"] == "train"]
    validation_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    x_train = train_df[FEATURE_COLUMNS].astype("float32")
    y_train = train_df[target].astype("int8")

    model = build_model()
    log(f"Training +{horizon}h model on {len(x_train):,} rows...")
    model.fit(x_train, y_train)

    metrics = {
        "validation": evaluate(
            model,
            validation_df[FEATURE_COLUMNS].astype("float32"),
            validation_df[target].astype("int8"),
            "validation",
        ),
        "test": evaluate(
            model,
            test_df[FEATURE_COLUMNS].astype("float32"),
            test_df[target].astype("int8"),
            "test",
        ),
    }

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_path_for(horizon, args.model_dir)
    metadata_path = metadata_path_for(horizon, args.model_dir)
    joblib.dump(model, model_path)

    metadata = {
        "model_path": str(model_path),
        "training_table": args.training_table,
        "target": target,
        "horizon_hours": horizon,
        "rain_threshold_mm": args.rain_threshold_mm,
        "feature_columns": FEATURE_COLUMNS,
        "train_end_exclusive": str(train_end),
        "validation_end_exclusive": str(validation_end),
        "metrics": metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"Saved +{horizon}h model to {model_path}")
    return model_path, metadata_path, metrics


def latest_as_of_time(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(forecast_time) FROM {table};")
        value = cur.fetchone()[0]
    if value is None:
        raise RuntimeError(f"No rows found in {table}.")
    return value


def fetch_as_of_features(conn, table, as_of_time):
    columns = [
        "grid_number",
        "grid_row",
        "grid_column",
        "longitude",
        "latitude",
        "forecast_time",
        *FEATURE_COLUMNS,
    ]
    query = f"""
    SELECT {", ".join(columns)}
    FROM {table}
    WHERE forecast_time = %s
    ORDER BY grid_number;
    """
    return pd.read_sql_query(query, conn, params=[as_of_time], parse_dates=["forecast_time"])


def ensure_output_table(conn, output_table):
    sql = CREATE_OUTPUT_TABLE_SQL.replace(OUTPUT_TABLE, output_table)
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def save_predictions(conn, output_table, predictions):
    rows = [
        (
            int(row.grid_number),
            int(row.grid_row),
            int(row.grid_column),
            float(row.longitude),
            float(row.latitude),
            row.as_of_time.to_pydatetime(),
            row.target_time.to_pydatetime(),
            int(row.horizon_hours),
            float(row.rain_probability),
            row.model_name,
            str(row.model_path),
        )
        for row in predictions.itertuples(index=False)
    ]
    sql = INSERT_OUTPUT_SQL.replace(OUTPUT_TABLE, output_table)
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()


def predict_future_hours(conn, args):
    as_of_time = args.as_of_time or latest_as_of_time(conn, args.prediction_table)
    log(f"Loading latest available feature rows at {as_of_time}...")
    base_df = fetch_as_of_features(conn, args.prediction_table, as_of_time)
    if base_df.empty:
        raise RuntimeError(f"No prediction rows found at as_of_time={as_of_time}.")

    x = base_df[FEATURE_COLUMNS].astype("float32")
    prediction_frames = []

    for horizon in args.horizons:
        model_path = model_path_for(horizon, args.model_dir)
        if not model_path.exists():
            raise RuntimeError(
                f"Missing +{horizon}h model: {model_path}. "
                "Run with --mode train or --mode both first."
            )

        log(f"Predicting exact +{horizon}h rain probability...")
        model = joblib.load(model_path)
        horizon_df = base_df[
            [
                "grid_number",
                "grid_row",
                "grid_column",
                "longitude",
                "latitude",
                "forecast_time",
            ]
        ].copy()
        horizon_df = horizon_df.rename(columns={"forecast_time": "as_of_time"})
        horizon_df["target_time"] = horizon_df["as_of_time"] + pd.to_timedelta(
            horizon,
            unit="h",
        )
        horizon_df["horizon_hours"] = horizon
        horizon_df["rain_probability"] = model.predict_proba(x)[:, 1]
        horizon_df["model_name"] = f"rain_exact_next_{horizon}h_hist_gradient_boosting"
        horizon_df["model_path"] = str(model_path)
        prediction_frames.append(horizon_df)

    predictions = pd.concat(prediction_frames, ignore_index=True)

    if not args.no_db:
        log(f"Saving future predictions into {args.output_table}...")
        ensure_output_table(conn, args.output_table)
        save_predictions(conn, args.output_table, predictions)

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.csv_output, index=False)
        log(f"Wrote CSV predictions to {args.csv_output}")

    return predictions


def build_map_html(predictions):
    center_lat = predictions["latitude"].mean()
    center_lon = predictions["longitude"].mean()
    map_df = predictions.copy()
    map_df["as_of_time_label"] = map_df["as_of_time"].astype(str)
    map_df["target_time_label"] = map_df["target_time"].astype(str)

    horizon_labels = []
    frames = {}
    for horizon, group in map_df.groupby("horizon_hours", sort=True):
        label = f"+{int(horizon)}h"
        horizon_labels.append(label)
        frames[label] = group[
            [
                "grid_number",
                "grid_row",
                "grid_column",
                "longitude",
                "latitude",
                "as_of_time_label",
                "target_time_label",
                "rain_probability",
            ]
        ].to_dict(orient="records")

    all_points = map_df[["grid_number", "longitude", "latitude"]].drop_duplicates()
    all_points = all_points.to_dict(orient="records")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bangkok Future Rain Probability</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
    }}
    .summary, .legend, .horizon-control {{
      background: white;
      border-radius: 6px;
      box-shadow: 0 1px 8px rgba(0, 0, 0, 0.22);
      color: #1f2933;
      font: 14px/1.4 Arial, sans-serif;
      padding: 10px 12px;
    }}
    .summary strong, .legend strong {{
      display: block;
      font-size: 15px;
      margin-bottom: 4px;
    }}
    .legend-row {{
      align-items: center;
      display: flex;
      gap: 8px;
      margin: 4px 0;
    }}
    .swatch {{
      border: 1px solid rgba(0, 0, 0, 0.25);
      display: inline-block;
      height: 12px;
      width: 18px;
    }}
    .horizon-control {{
      min-width: 280px;
    }}
    .horizon-control label {{
      display: block;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .horizon-control input {{
      width: 100%;
    }}
    .horizon-value {{
      margin-top: 6px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const frames = {json.dumps(frames, separators=(",", ":"))};
    const horizonLabels = {json.dumps(horizon_labels, separators=(",", ":"))};
    const allPoints = {json.dumps(all_points, separators=(",", ":"))};
    const map = L.map("map").setView([{center_lat:.6f}, {center_lon:.6f}], 10);

    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}).addTo(map);

    function colorForProbability(probability) {{
      if (probability >= 0.70) return "#dc2626";
      if (probability >= 0.50) return "#f97316";
      if (probability >= 0.30) return "#eab308";
      return "#16a34a";
    }}

    let markers = [];
    function clearMarkers() {{
      markers.forEach((marker) => marker.remove());
      markers = [];
    }}

    function drawFrame(horizonLabel) {{
      clearMarkers();
      const points = frames[horizonLabel] || [];
      points.forEach((point) => {{
        const probability = point.rain_probability;
        const marker = L.circleMarker([point.latitude, point.longitude], {{
          radius: 8 + probability * 10,
          color: "#111827",
          weight: 1,
          fillColor: colorForProbability(probability),
          fillOpacity: 0.86
        }})
          .bindPopup(`
            <strong>Grid ${{point.grid_number}}</strong><br>
            Row: ${{point.grid_row}}<br>
            Column: ${{point.grid_column}}<br>
            Horizon: ${{horizonLabel}}<br>
            As of: ${{point.as_of_time_label}}<br>
            Target: ${{point.target_time_label}}<br>
            Rain probability: ${{(probability * 100).toFixed(1)}}%
          `)
          .addTo(map);
        markers.push(marker);
      }});
      updateSummary(horizonLabel);
    }}

    const bounds = allPoints.map((point) => [point.latitude, point.longitude]);
    if (bounds.length > 0) {{
      map.fitBounds(bounds, {{ padding: [30, 30] }});
    }}

    let summaryDiv;
    function updateSummary(horizonLabel) {{
      if (!summaryDiv) return;
      const points = frames[horizonLabel] || [];
      const probs = points.map((point) => point.rain_probability);
      const mean = probs.reduce((a, b) => a + b, 0) / probs.length;
      const max = Math.max(...probs);
      const first = points[0] || {{}};
      summaryDiv.innerHTML = `
        <strong>Future Rain Probability</strong>
        Horizon: ${{horizonLabel}}<br>
        As of: ${{first.as_of_time_label || ""}}<br>
        Target: ${{first.target_time_label || ""}}<br>
        Grid points: ${{points.length}}<br>
        Mean: ${{(mean * 100).toFixed(1)}}%<br>
        Max: ${{(max * 100).toFixed(1)}}%
      `;
    }}

    const summary = L.control({{ position: "bottomleft" }});
    summary.onAdd = () => {{
      summaryDiv = L.DomUtil.create("div", "summary");
      return summaryDiv;
    }};
    summary.addTo(map);

    const horizonControl = L.control({{ position: "topright" }});
    horizonControl.onAdd = () => {{
      const div = L.DomUtil.create("div", "horizon-control");
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.disableScrollPropagation(div);
      div.innerHTML = `
        <label for="horizon-slider">Future hour</label>
        <input id="horizon-slider" type="range" min="0" max="${{horizonLabels.length - 1}}" value="0" step="1">
        <div id="horizon-value" class="horizon-value"></div>
      `;
      return div;
    }};
    horizonControl.addTo(map);

    const slider = document.getElementById("horizon-slider");
    const horizonValue = document.getElementById("horizon-value");
    function setHorizon(index) {{
      const horizonLabel = horizonLabels[index];
      horizonValue.textContent = horizonLabel;
      drawFrame(horizonLabel);
    }}
    slider.addEventListener("input", (event) => setHorizon(Number(event.target.value)));

    const legend = L.control({{ position: "bottomright" }});
    legend.onAdd = () => {{
      const div = L.DomUtil.create("div", "legend");
      div.innerHTML = `
        <strong>Probability</strong>
        <div class="legend-row"><span class="swatch" style="background:#16a34a"></span>0-29%</div>
        <div class="legend-row"><span class="swatch" style="background:#eab308"></span>30-49%</div>
        <div class="legend-row"><span class="swatch" style="background:#f97316"></span>50-69%</div>
        <div class="legend-row"><span class="swatch" style="background:#dc2626"></span>70%+</div>
      `;
      return div;
    }};
    legend.addTo(map);

    setHorizon(0);
  </script>
</body>
</html>
"""


def save_map(predictions, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_map_html(predictions), encoding="utf-8")


def print_prediction_summary(predictions):
    print("\nFuture prediction summary", flush=True)
    print(f"Rows: {len(predictions):,}", flush=True)
    print(f"As-of time: {predictions['as_of_time'].min()}", flush=True)
    for horizon, group in predictions.groupby("horizon_hours", sort=True):
        target_time = group["target_time"].iloc[0]
        print(
            (
                f"+{horizon}h target {target_time}: "
                f"mean={group['rain_probability'].mean():.3f}, "
                f"max={group['rain_probability'].max():.3f}, "
                f"points>=0.50={(group['rain_probability'] >= 0.50).sum()}"
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train and run direct future-hour Bangkok rainfall models. "
            "Each model uses only weather data available at time T to predict rain at T+h."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "both"],
        default="both",
        help="Train models, run predictions, or do both. Defaults to both.",
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=[1, 2, 3],
        help="Comma-separated future hours to model, for example 1,2,3. Defaults to 1,2,3.",
    )
    parser.add_argument("--training-table", default=TRAINING_TABLE)
    parser.add_argument("--prediction-table", default=PREDICTION_FEATURE_TABLE)
    parser.add_argument("--output-table", default=OUTPUT_TABLE)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--rain-threshold-mm", type=float, default=RAIN_THRESHOLD_MM)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional random row sample per horizon for quick experiments.",
    )
    parser.add_argument(
        "--as-of-time",
        default=None,
        help="Optional exact current/history time to predict from. Defaults to latest forecast_time.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PREDICTION_DIR / "latest_future_hourly_rain_predictions.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write a CSV prediction file.",
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        default=MAP_DIR / "latest_bkk_future_hourly_rain_probability.html",
        help="HTML map output path.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Do not write an HTML prediction map.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Do not save predictions into PostgreSQL.",
    )
    args = parser.parse_args()

    if args.no_csv:
        args.csv_output = None

    log("Connecting to PostgreSQL...")
    with connect() as conn:
        if args.mode in ("train", "both"):
            for horizon in args.horizons:
                _, _, metrics = train_horizon_model(conn, args, horizon)
                print(f"\n+{horizon}h metrics", flush=True)
                print(json.dumps(metrics, indent=2), flush=True)

        if args.mode in ("predict", "both"):
            predictions = predict_future_hours(conn, args)
            print_prediction_summary(predictions)
            if not args.no_map:
                log(f"Writing future prediction map to {args.map_output}...")
                save_map(predictions, args.map_output)
                print(f"\nMap: {args.map_output.resolve()}", flush=True)

    log("Done.")


if __name__ == "__main__":
    main()
