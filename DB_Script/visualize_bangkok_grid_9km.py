import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

try:
    import psycopg2
except ModuleNotFoundError:
    psycopg2 = None


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "Pass1234",
}

TABLE_NAME = '"Bangkok_Grid_9km"'
DEFAULT_OUTPUT = Path(__file__).with_name("bangkok_grid_9km_map.html")
DEFAULT_PSQL = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"


POINTS_SQL = f"""
SELECT
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    approximate_resolution_km
FROM {TABLE_NAME}
ORDER BY grid_row, grid_column;
"""


def connect():
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Run with --method psql instead.")

    config = DB_CONFIG.copy()
    config["password"] = os.getenv("PGPASSWORD", config["password"])
    return psycopg2.connect(**config)


def resolve_psql_path(psql_path):
    candidates = [
        psql_path,
        os.getenv("PSQL"),
        DEFAULT_PSQL,
        shutil.which("psql"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise RuntimeError(
        "Could not find psql.exe. Pass --psql-path or set the PSQL environment variable."
    )


def run_psql(sql, psql_path):
    command = [
        resolve_psql_path(psql_path),
        "-v",
        "ON_ERROR_STOP=1",
        "-h",
        DB_CONFIG["host"],
        "-p",
        str(DB_CONFIG["port"]),
        "-U",
        DB_CONFIG["user"],
        "-d",
        DB_CONFIG["dbname"],
        "-At",
        "-F",
        "\t",
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = env.get("PGPASSWORD", DB_CONFIG["password"])

    return subprocess.run(
        command,
        input=sql,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        env=env,
    ).stdout


def fetch_points_with_psycopg2():
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(POINTS_SQL)
            return [
                {
                    "grid_number": row[0],
                    "grid_row": row[1],
                    "grid_column": row[2],
                    "longitude": float(row[3]),
                    "latitude": float(row[4]),
                    "approximate_resolution_km": float(row[5]),
                }
                for row in cur.fetchall()
            ]


def fetch_points_with_psql(psql_path):
    output = run_psql(POINTS_SQL, psql_path).strip()
    if not output:
        return []

    points = []
    for line in output.splitlines():
        fields = line.split("\t")
        points.append(
            {
                "grid_number": int(fields[0]),
                "grid_row": int(fields[1]),
                "grid_column": int(fields[2]),
                "longitude": float(fields[3]),
                "latitude": float(fields[4]),
                "approximate_resolution_km": float(fields[5]),
            }
        )
    return points


def fetch_points(method, psql_path):
    if method == "auto":
        method = "psycopg2" if psycopg2 is not None else "psql"
    if method == "psql":
        return fetch_points_with_psql(psql_path)
    return fetch_points_with_psycopg2()


def point_feature(point):
    return {
        "type": "Feature",
        "properties": {
            "grid_number": point["grid_number"],
            "grid_row": point["grid_row"],
            "grid_column": point["grid_column"],
            "resolution_km": point["approximate_resolution_km"],
        },
        "geometry": {
            "type": "Point",
            "coordinates": [point["longitude"], point["latitude"]],
        },
    }


def line_features(points, key, label):
    grouped = {}
    for point in points:
        grouped.setdefault(point[key], []).append(point)

    features = []
    for value, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda p: (p["longitude"], p["latitude"]))
        if key == "grid_column":
            ordered = sorted(group, key=lambda p: (p["latitude"], p["longitude"]))
        features.append(
            {
                "type": "Feature",
                "properties": {"name": f"{label} {value}", key: value},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [point["longitude"], point["latitude"]] for point in ordered
                    ],
                },
            }
        )
    return features


def build_geojson(points):
    return {
        "points": {
            "type": "FeatureCollection",
            "features": [point_feature(point) for point in points],
        },
        "lines": {
            "type": "FeatureCollection",
            "features": line_features(points, "grid_row", "Row")
            + line_features(points, "grid_column", "Column"),
        },
    }


def build_html(points):
    if not points:
        raise RuntimeError(f"No rows found in {TABLE_NAME}.")

    geojson = build_geojson(points)
    min_lon = min(point["longitude"] for point in points)
    max_lon = max(point["longitude"] for point in points)
    min_lat = min(point["latitude"] for point in points)
    max_lat = max(point["latitude"] for point in points)
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bangkok 9 km Grid</title>
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
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const gridPoints = {json.dumps(geojson["points"], separators=(",", ":"))};
    const gridLines = {json.dumps(geojson["lines"], separators=(",", ":"))};

    const map = L.map("map").setView([{center_lat:.6f}, {center_lon:.6f}], 10);

    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }}).addTo(map);

    const lines = L.geoJSON(gridLines, {{
      style: {{
        color: "#2563eb",
        weight: 1.5,
        opacity: 0.68
      }}
    }}).addTo(map);

    const points = L.geoJSON(gridPoints, {{
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{
        radius: 5,
        color: "#111827",
        weight: 1,
        fillColor: "#f97316",
        fillOpacity: 0.92
      }}),
      onEachFeature: (feature, layer) => {{
        const props = feature.properties;
        layer.bindPopup(`
          <strong>Grid ${{props.grid_number}}</strong><br>
          Row: ${{props.grid_row}}<br>
          Column: ${{props.grid_column}}<br>
          Resolution: ${{props.resolution_km}} km
        `);
      }}
    }}).addTo(map);

    const layers = {{
      "Grid points": points,
      "Grid lines": lines
    }};
    L.control.layers(null, layers, {{ collapsed: false }}).addTo(map);

    const bounds = L.latLngBounds(
      [{min_lat:.6f}, {min_lon:.6f}],
      [{max_lat:.6f}, {max_lon:.6f}]
    );
    map.fitBounds(bounds.pad(0.12));

    const summary = L.control({{ position: "bottomleft" }});
    summary.onAdd = () => {{
      const div = L.DomUtil.create("div", "summary");
      div.innerHTML = `
        <strong>Bangkok 9 km Grid</strong>
        {len(points)} points<br>
        Lat {min_lat:.6f} to {max_lat:.6f}<br>
        Lon {min_lon:.6f} to {max_lon:.6f}
      `;
      return div;
    }};
    summary.addTo(map);
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description='Export PostgreSQL table "Bangkok_Grid_9km" to an interactive Leaflet map.'
    )
    parser.add_argument(
        "--method",
        choices=["auto", "psycopg2", "psql"],
        default="auto",
        help="Database read method. Defaults to psycopg2 when available, otherwise psql.",
    )
    parser.add_argument(
        "--psql-path",
        default=None,
        help="Path to psql.exe. Defaults to PSQL env var, PostgreSQL 18 default path, or PATH lookup.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"HTML file to write. Defaults to {DEFAULT_OUTPUT}.",
    )
    args = parser.parse_args()

    points = fetch_points(args.method, args.psql_path)
    html = build_html(points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output.resolve()} with {len(points):,} grid points.")


if __name__ == "__main__":
    main()
