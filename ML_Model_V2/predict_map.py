"""Predict rain probability for all 56 Bangkok grid cells and overlay it on a map.

The 56 cells form a regular 7 x 8 lattice at roughly 0.081 deg latitude x 0.083 deg longitude
(~9 km), and `latitude` / `longitude` are already model features - so every prediction arrives
with its own footprint attached and no spatial join is needed.

Each cell is drawn as the rectangle it actually covers. That is deliberate: the model predicts
"rain somewhere in this cell", so a filled cell is an honest rendering of its resolution. A
smoothed or contoured surface would imply a spatial precision the model does not have.

Usage:
    python ML_Model_V2/predict_map.py                       # newest hour, h6_0.1mm
    python ML_Model_V2/predict_map.py --target h1_1.0mm
    python ML_Model_V2/predict_map.py --at "2026-07-19 18:00"
"""

import argparse
import os
from pathlib import Path

import branca.colormap as cm
import folium
import joblib
import numpy as np
import pandas as pd
import psycopg2

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}
PRECOMPUTE_TABLE_NAME = '"OM_BKK_DATA_PRECOMPUTE"'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "ML_Model_V2" / "trained_models" / "om_bkk_rain_v2_deploy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rain_maps"

# Sequential blue ramp, light -> dark: magnitude encoding, one hue. The domain is pinned to
# 0..1 rather than to each hour's own min/max, so two maps are comparable - a relative scale
# would make a dry hour look identical to a wet one.
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                   "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

ROW_FILTER_SQL = '''
      pressure_msl_change_6h IS NOT NULL
      AND precipitation_lag_6h IS NOT NULL
      AND precipitation_sum_past_24h IS NOT NULL
      AND cloud_cover_lag_6h IS NOT NULL
      AND humidity_lag_6h IS NOT NULL
      AND wind_speed_lag_3h IS NOT NULL
      AND neighbor_count > 0
'''


def find_bundle(target):
    matches = sorted(MODEL_DIR.glob(f"*_next_{target.split('_')[0][1:]}h_*"))
    rain_mm = target.split("_")[1]
    matches = [m for m in matches if m.name.endswith(f"rain_threshold_{rain_mm}.joblib")]
    if not matches:
        raise SystemExit(f"no bundle for target {target} in {MODEL_DIR}")
    return matches[0]


def latest_complete_hour(cur):
    cur.execute(f'''
        SELECT local_forecast_time FROM {PRECOMPUTE_TABLE_NAME}
        WHERE {ROW_FILTER_SQL}
        GROUP BY local_forecast_time HAVING count(*) = 56
        ORDER BY local_forecast_time DESC LIMIT 1''')
    row = cur.fetchone()
    if row is None:
        raise SystemExit("no hour has all 56 cells complete")
    return row[0]


def load_hour(feature_columns, at=None):
    """Return one row per grid cell for a single forecast hour, in feature order."""
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            stamp = at or latest_complete_hour(cur)
            feature_sql = ",\n            ".join(
                f"COALESCE({c}::real, 'NaN'::real)" for c in feature_columns)
            cur.execute(f'''
                SELECT grid_number, latitude, longitude,
                    {feature_sql}
                FROM {PRECOMPUTE_TABLE_NAME}
                WHERE local_forecast_time = %s AND {ROW_FILTER_SQL}
                ORDER BY grid_number''', (stamp,))
            rows = cur.fetchall()
    block = np.asarray(rows, dtype="object")
    cells = pd.DataFrame({
        "grid_number": block[:, 0].astype("int32"),
        "latitude": block[:, 1].astype("float64"),
        "longitude": block[:, 2].astype("float64"),
    })
    x = block[:, 3:].astype("float32")
    return stamp, cells, x


def cell_edges(values):
    """Boundaries halfway between neighbouring centres; outer edges mirror the adjacent gap.

    Derived rather than assumed constant: the northern row of this grid is clipped, so a fixed
    step would draw that row's rectangle past the real cell boundary.
    """
    unique = np.unique(values)
    midpoints = (unique[:-1] + unique[1:]) / 2
    lower = np.r_[unique[0] - (midpoints[0] - unique[0]), midpoints]
    upper = np.r_[midpoints, unique[-1] + (unique[-1] - midpoints[-1])]
    return {v: (lo, hi) for v, lo, hi in zip(unique, lower, upper)}


def operating_threshold(bundle, stamp):
    """The cut for this hour's calendar month, falling back to the old global one.

    A single global cut fires on ~85% of wet-season cell-hours and misses most dry-season rain,
    because the base rate swings from about 0.05 to 0.59 across the year.
    """
    seasonal = bundle.get("seasonal_thresholds")
    if not seasonal:
        return bundle["probability_threshold"], "global"
    return float(seasonal[stamp.month]), f"{stamp:%B}"


def build_map(cells, bundle, stamp, threshold, threshold_label):
    lat_edges = cell_edges(cells["latitude"].to_numpy())
    lon_edges = cell_edges(cells["longitude"].to_numpy())
    ramp = cm.LinearColormap(SEQUENTIAL_BLUE, vmin=0.0, vmax=1.0)

    fmap = folium.Map(
        location=[cells["latitude"].mean(), cells["longitude"].mean()],
        zoom_start=10, tiles="CartoDB positron", control_scale=True)

    hottest = cells["probability"].idxmax()
    for index, cell in cells.iterrows():
        south, north = lat_edges[cell["latitude"]]
        west, east = lon_edges[cell["longitude"]]
        probability = float(cell["probability"])
        firing = probability >= threshold
        tooltip = (f"<b>cell {int(cell['grid_number'])}</b><br>"
                   f"probability {probability:.1%}<br>"
                   f"{'above' if firing else 'below'} threshold ({threshold:.1%})<br>"
                   f"{cell['latitude']:.3f}, {cell['longitude']:.3f}")
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            # a 2px surface-coloured edge separates adjacent fills; cells that clear the
            # operating threshold switch to a dark solid edge, so the decision is carried by
            # structure as well as by colour
            color=TEXT_PRIMARY if firing else SURFACE,
            weight=2, opacity=1.0,
            fill=True, fill_color=ramp(probability), fill_opacity=0.85,
            tooltip=tooltip,
        ).add_to(fmap)

    # one selective direct label rather than a number on all 56 cells
    peak = cells.loc[hottest]
    folium.map.Marker(
        [peak["latitude"], peak["longitude"]],
        icon=folium.DivIcon(html=(
            f'<div style="font:600 12px system-ui;color:{TEXT_PRIMARY};'
            f'background:{SURFACE};border-radius:4px;padding:1px 5px;'
            f'white-space:nowrap;transform:translate(-50%,-50%);'
            f'box-shadow:0 1px 3px rgba(0,0,0,.25)">{peak["probability"]:.0%}</div>')),
    ).add_to(fmap)

    # branca pins its colourbar to the top right, so the caption stays short and the title
    # block sits bottom left - the two never compete for the same corner
    ramp.caption = f"probability, next {bundle['horizon_h']} h"
    ramp.add_to(fmap)

    firing_count = int((cells["probability"] >= threshold).sum())
    title = f'''
    <div style="position:fixed;bottom:42px;left:12px;z-index:9999;background:{SURFACE};
         padding:10px 14px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.25);
         font:13px system-ui;max-width:400px">
      <div style="font-weight:600;color:{TEXT_PRIMARY};font-size:14px">
        Rain probability &mdash; next {bundle['horizon_h']} h, &ge; {bundle['rain_threshold_mm']} mm</div>
      <div style="color:{TEXT_SECONDARY};margin-top:3px">
        {stamp:%Y-%m-%d %H:%M} local &middot; {len(cells)} cells &middot;
        {firing_count} above the {threshold:.1%} threshold for {threshold_label} (dark outline)</div>
      <div style="color:{TEXT_SECONDARY};margin-top:5px;font-size:11.5px">
        Each rectangle is one ~9 km cell. The model predicts rain <i>somewhere</i> in a cell,
        not at a point.</div>
    </div>'''
    fmap.get_root().html.add_child(folium.Element(title))
    return fmap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="h6_0.1mm", help="e.g. h1_1.0mm, h6_0.1mm")
    parser.add_argument("--at", default=None, help="forecast hour, 'YYYY-MM-DD HH:MM'")
    args = parser.parse_args()

    path = find_bundle(args.target)
    bundle = joblib.load(path)
    print(f"model    : {path.name}")

    stamp, cells, x = load_hour(bundle["feature_columns"], args.at)
    if len(cells) != 56:
        print(f"warning  : {len(cells)} cells returned, expected 56")

    raw = bundle["model"].predict_proba(x)[:, 1]
    cells["probability"] = bundle["calibrator"].predict(raw)
    threshold, threshold_label = operating_threshold(bundle, stamp)
    cells["above_threshold"] = cells["probability"] >= threshold

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = f"{bundle['target']}_{stamp:%Y%m%d_%H%M}"
    html_path = OUTPUT_DIR / f"rain_map_{slug}.html"
    csv_path = OUTPUT_DIR / f"rain_map_{slug}.csv"

    build_map(cells, bundle, stamp, threshold, threshold_label).save(str(html_path))
    cells.to_csv(csv_path, index=False)          # the table view of the same numbers

    print(f"hour     : {stamp}")
    print(f"threshold: {threshold:.4f} ({threshold_label})")
    print(f"cells    : {len(cells)}  above threshold: {int(cells['above_threshold'].sum())}")
    print(f"P range  : {cells['probability'].min():.3f} .. {cells['probability'].max():.3f}")
    print(f"map      : {html_path}")
    print(f"table    : {csv_path}")


if __name__ == "__main__":
    main()
