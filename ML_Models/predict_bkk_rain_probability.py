import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

PREDICTION_FEATURE_TABLE = "bkk_rain_prediction_features"
PREDICTION_OUTPUT_TABLE = "bkk_hourly_rain_probability_predictions"
MODEL_DIR = Path(__file__).with_name("trained_models")
DEFAULT_MODEL_PATH = MODEL_DIR / "bkk_rain_next_1h_probability_hist_gradient_boosting.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "bkk_rain_next_1h_probability_hist_gradient_boosting_metadata.json"
MAP_DIR = Path(__file__).with_name("prediction_maps")

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


CREATE_PREDICTION_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {PREDICTION_OUTPUT_TABLE} (
    grid_number integer NOT NULL,
    grid_row integer NOT NULL,
    grid_column integer NOT NULL,
    longitude double precision NOT NULL,
    latitude double precision NOT NULL,
    forecast_time timestamp with time zone NOT NULL,
    rain_probability_next_1h double precision NOT NULL,
    model_name text NOT NULL,
    model_path text NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    PRIMARY KEY (grid_number, forecast_time, model_name)
);

CREATE INDEX IF NOT EXISTS bkk_hourly_rain_probability_predictions_time_idx
  ON {PREDICTION_OUTPUT_TABLE} (forecast_time);
"""


INSERT_PREDICTIONS_SQL = f"""
INSERT INTO {PREDICTION_OUTPUT_TABLE} (
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    forecast_time,
    rain_probability_next_1h,
    model_name,
    model_path
)
VALUES %s
ON CONFLICT (grid_number, forecast_time, model_name) DO UPDATE SET
    grid_row = EXCLUDED.grid_row,
    grid_column = EXCLUDED.grid_column,
    longitude = EXCLUDED.longitude,
    latitude = EXCLUDED.latitude,
    rain_probability_next_1h = EXCLUDED.rain_probability_next_1h,
    model_path = EXCLUDED.model_path,
    created_at = now();
"""


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def connect():
    return psycopg2.connect(**DB_CONFIG)


def load_metadata(metadata_path):
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def latest_forecast_time(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(forecast_time) FROM {table};")
        value = cur.fetchone()[0]
    if value is None:
        raise RuntimeError(f"No rows found in {table}.")
    return value


def latest_forecast_times(conn, table, hours):
    query = f"""
    SELECT DISTINCT forecast_time
    FROM {table}
    ORDER BY forecast_time DESC
    LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, [hours])
        values = [row[0] for row in cur.fetchall()]
    if not values:
        raise RuntimeError(f"No rows found in {table}.")
    return sorted(values)


def fetch_prediction_features(conn, table, forecast_times):
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
    WHERE forecast_time = ANY(%s::timestamptz[])
    ORDER BY forecast_time, grid_number;
    """
    return pd.read_sql_query(query, conn, params=[forecast_times], parse_dates=["forecast_time"])


def ensure_prediction_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_PREDICTION_TABLE_SQL)
    conn.commit()


def save_predictions(conn, df, model_name, model_path):
    rows = [
        (
            int(row.grid_number),
            int(row.grid_row),
            int(row.grid_column),
            float(row.longitude),
            float(row.latitude),
            row.forecast_time.to_pydatetime(),
            float(row.rain_probability_next_1h),
            model_name,
            str(model_path),
        )
        for row in df.itertuples(index=False)
    ]

    with conn.cursor() as cur:
        execute_values(cur, INSERT_PREDICTIONS_SQL, rows, page_size=1000)
    conn.commit()


def probability_summary(df):
    return {
        "rows": int(len(df)),
        "forecast_times": int(df["forecast_time"].nunique()),
        "min_probability": float(df["rain_probability_next_1h"].min()),
        "mean_probability": float(df["rain_probability_next_1h"].mean()),
        "max_probability": float(df["rain_probability_next_1h"].max()),
        "high_risk_points_50": int((df["rain_probability_next_1h"] >= 0.50).sum()),
        "high_risk_points_70": int((df["rain_probability_next_1h"] >= 0.70).sum()),
    }


def build_map_html(df, model_name):
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    map_df = df.copy()
    map_df["forecast_time_label"] = map_df["forecast_time"].astype(str)
    time_labels = sorted(map_df["forecast_time_label"].unique())
    frames = {}
    for time_label, group in map_df.groupby("forecast_time_label", sort=True):
        frames[time_label] = group[
            [
                "grid_number",
                "grid_row",
                "grid_column",
                "longitude",
                "latitude",
                "rain_probability_next_1h",
            ]
        ].to_dict(orient="records")

    all_points = map_df[
        [
            "grid_number",
            "longitude",
            "latitude",
        ]
    ].to_dict(orient="records")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bangkok Rain Probability</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
    }}
    .summary {{
      background: white;
      border-radius: 6px;
      box-shadow: 0 1px 8px rgba(0, 0, 0, 0.22);
      color: #1f2933;
      font: 14px/1.4 Arial, sans-serif;
      padding: 10px 12px;
    }}
    .summary strong {{
      display: block;
      font-size: 15px;
      margin-bottom: 4px;
    }}
    .legend {{
      background: white;
      border-radius: 6px;
      box-shadow: 0 1px 8px rgba(0, 0, 0, 0.22);
      color: #1f2933;
      font: 13px/1.4 Arial, sans-serif;
      padding: 10px 12px;
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
    .time-control {{
      background: white;
      border-radius: 6px;
      box-shadow: 0 1px 8px rgba(0, 0, 0, 0.22);
      color: #1f2933;
      font: 14px/1.4 Arial, sans-serif;
      min-width: 280px;
      padding: 10px 12px;
    }}
    .time-control label {{
      display: block;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .time-control input {{
      width: 100%;
    }}
    .time-value {{
      margin-top: 6px;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const frames = {json.dumps(frames, separators=(",", ":"))};
    const timeLabels = {json.dumps(time_labels, separators=(",", ":"))};
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

    function drawFrame(timeLabel) {{
      clearMarkers();
      const points = frames[timeLabel] || [];
      points.forEach((point) => {{
        const probability = point.rain_probability_next_1h;
        const latLng = [point.latitude, point.longitude];
        const marker = L.circleMarker(latLng, {{
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
            Forecast time: ${{timeLabel}}<br>
            Rain probability next hour: ${{(probability * 100).toFixed(1)}}%
          `)
          .addTo(map);
        markers.push(marker);
      }});
      updateSummary(timeLabel);
    }}

    const bounds = [];
    allPoints.forEach((point) => {{
      const latLng = [point.latitude, point.longitude];
      bounds.push(latLng);
    }});

    if (bounds.length > 0) {{
      map.fitBounds(bounds, {{ padding: [30, 30] }});
    }}

    let summaryDiv;
    function updateSummary(timeLabel) {{
      if (!summaryDiv) return;
      const points = frames[timeLabel] || [];
      const probs = points.map((point) => point.rain_probability_next_1h);
      const mean = probs.reduce((a, b) => a + b, 0) / probs.length;
      const max = Math.max(...probs);
      summaryDiv.innerHTML = `
        <strong>Rain Probability Next Hour</strong>
        Forecast time: ${{timeLabel}}<br>
        Model: {model_name}<br>
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

    const timeControl = L.control({{ position: "topright" }});
    timeControl.onAdd = () => {{
      const div = L.DomUtil.create("div", "time-control");
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.disableScrollPropagation(div);
      div.innerHTML = `
        <label for="time-slider">Forecast hour</label>
        <input id="time-slider" type="range" min="0" max="${{timeLabels.length - 1}}" value="${{timeLabels.length - 1}}" step="1">
        <div id="time-value" class="time-value"></div>
      `;
      return div;
    }};
    timeControl.addTo(map);

    const slider = document.getElementById("time-slider");
    const timeValue = document.getElementById("time-value");
    function setTime(index) {{
      const timeLabel = timeLabels[index];
      timeValue.textContent = timeLabel;
      drawFrame(timeLabel);
    }}
    slider.addEventListener("input", (event) => setTime(Number(event.target.value)));

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

    setTime(timeLabels.length - 1);
  </script>
</body>
</html>
"""


def save_map(df, model_name, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_map_html(df, model_name), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Predict Bangkok grid rainfall probability and create an OpenStreetMap overlay."
    )
    parser.add_argument("--feature-table", default=PREDICTION_FEATURE_TABLE)
    parser.add_argument("--output-table", default=PREDICTION_OUTPUT_TABLE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument(
        "--forecast-time",
        default=None,
        help="Optional exact forecast_time to predict. Overrides --hours.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=6,
        help="Number of latest hourly forecast times to predict and show on the map. Defaults to 6.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name stored with predictions. Defaults to metadata target/model type.",
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        default=None,
        help="Optional HTML map output path. Defaults to ML_Models/prediction_maps/latest_bkk_hourly_rain_probability.html.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Save database predictions only; do not create an HTML map.",
    )
    args = parser.parse_args()

    if not args.model_path.exists():
        raise RuntimeError(f"Model file not found: {args.model_path}")

    metadata = load_metadata(args.metadata_path)
    model_name = args.model_name
    if not model_name:
        target = metadata.get("target", "rain_next_1h")
        model_type = metadata.get("model_type", "model")
        model_name = f"{target}_{model_type}"

    log(f"Loading model from {args.model_path}...")
    model = joblib.load(args.model_path)

    log("Connecting to PostgreSQL...")
    with connect() as conn:
        if args.forecast_time:
            forecast_times = [args.forecast_time]
        else:
            forecast_times = latest_forecast_times(conn, args.feature_table, args.hours)
        log(f"Loading prediction features for {len(forecast_times)} forecast time(s)...")
        df = fetch_prediction_features(conn, args.feature_table, forecast_times)

        if df.empty:
            raise RuntimeError(f"No prediction rows found for forecast_times={forecast_times}.")

        log(f"Predicting probabilities for {len(df):,} grid points...")
        x = df[FEATURE_COLUMNS].astype("float32")
        df["rain_probability_next_1h"] = model.predict_proba(x)[:, 1]

        log(f"Saving predictions into {args.output_table}...")
        ensure_prediction_table(conn)
        save_predictions(conn, df, model_name, args.model_path)

    summary = probability_summary(df)
    print("\nPrediction summary", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)

    if not args.no_map:
        output_path = args.map_output
        if output_path is None:
            output_path = MAP_DIR / "latest_bkk_hourly_rain_probability.html"
        log(f"Writing map to {output_path}...")
        save_map(df, model_name, output_path)
        print(f"\nMap: {output_path.resolve()}", flush=True)

    log("Done.")


if __name__ == "__main__":
    main()
