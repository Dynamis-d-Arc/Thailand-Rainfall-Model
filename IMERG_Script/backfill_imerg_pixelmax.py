"""Backfill precipitation_pixelmax_mm (wettest-pixel HOURLY total) in "IMERG_THAILAND_DATA".

The original fetch stored precipitation_max_mm = 0.5*max(first half-hour rate) +
0.5*max(second half-hour rate). Those two maxima can sit on different pixels, so the value
over-states the wettest pixel's hourly total (grid 2, 2026-10-20 14:00 Bangkok: 1.07 mm
stored, 0.87 mm at the wettest pixel). This script recomputes the hourly total per pixel
first and takes the cell max, and writes ONLY the new column; every other column is left
exactly as fetched.

A row is filled only when Earth Engine still reproduces its stored precipitation_max_mm
(|diff| < 0.005 mm). If NASA has revised the product since the original fetch the two
columns would come from different versions, so such rows stay NULL and are counted.
Hours with a missing half-hour are never filled (is_complete_hour gate).

Resumable: a chunk with no NULL complete rows is skipped.

    python IMERG_Script/backfill_imerg_pixelmax.py --start-date 2024-06-30 --end-date 2026-10-27
    python IMERG_Script/backfill_imerg_pixelmax.py --check 2 2026-10-20T07   # one cell-hour, pixel level
"""
import argparse
import sys
import time as time_module
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_imerg_thailand_data_gee import (  # noqa: E402
    CELL_HALF_DEG, DEFAULT_PROJECT_ID, GEE_PIXEL_SCALE_METERS, OUTPUT_TABLE,
    build_cells_feature_collection, chunk_band_metadata, chunk_image_collection, connect,
    date_chunks, ensure_table, fetch_grid_cells, hourly_pixel_max, hourly_sum_bands,
    initialize_earth_engine, log, parse_date, require_ee_dependency, utc_chunk_bounds,
    with_retries,
)

TOLERANCE_MM = 0.005


def pending_rows(conn, start_date, end_date):
    start, stop = utc_chunk_bounds(start_date, end_date)
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT COUNT(*) FROM {OUTPUT_TABLE} WHERE observation_time >= %s AND observation_time < %s "
            f"AND is_complete_hour AND precipitation_pixelmax_mm IS NULL",
            (start, stop))
        return int(cur.fetchone()[0])


def reduce_pixelmax(ee, collection, cells_fc, scale, band_map):
    """[(grid_number, hour_utc, pixelmax_mm, stored_style_mm)] for every complete hour."""
    stacked = collection.toBands()
    hourly, hours = hourly_sum_bands(ee, stacked, band_map)
    if not hourly:
        return []
    image = ee.Image.cat([stacked] + hourly)
    # The same reducer as the original fetch. Combining max with mean makes Earth Engine
    # weight every pixel that touches the cell (the ~9-pixel footprint pixel_count records);
    # a lone max reducer would only see pixels centred inside the cell and mis-reproduce
    # the stored maxima whenever the wettest pixel sits on the cell edge.
    reducer = ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True)
    reduced = image.reduceRegions(collection=cells_fc, reducer=reducer, scale=scale)
    info = with_retries("reduceRegions", reduced.getInfo)
    rows = []
    for feature in info.get("features", []):
        props = feature["properties"]
        grid = int(props["grid_number"])
        for name, (hour, band_a, band_b) in hours.items():
            pm, a, b = props.get(f"{name}_max"), props.get(f"{band_a}_max"), props.get(f"{band_b}_max")
            if pm is None or a is None or b is None:
                continue
            rows.append((grid, hour, float(pm), 0.5 * float(a) + 0.5 * float(b)))
    return rows


def write_rows(conn, rows):
    """UPDATE only the new column, only where the stored max is still reproduced. Returns rows updated."""
    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE stage (grid_number int, observation_time timestamptz, "
                    "pixelmax double precision, stored_style double precision) ON COMMIT DROP")
        execute_values(cur, "INSERT INTO stage VALUES %s", rows, page_size=10000)
        cur.execute(
            f"UPDATE {OUTPUT_TABLE} t SET precipitation_pixelmax_mm = s.pixelmax FROM stage s "
            f"WHERE t.grid_number = s.grid_number AND t.observation_time = s.observation_time "
            f"AND t.is_complete_hour AND t.precipitation_pixelmax_mm IS NULL "
            f"AND abs(t.precipitation_max_mm - s.stored_style) < %s", (TOLERANCE_MM,))
        n = cur.rowcount
    conn.commit()
    return n


def check_one(ee, conn, grid_number, hour_utc, scale):
    """Pixel-level cross-check of one cell-hour: numpy reference vs the server-side reduction."""
    cells = [c for c in fetch_grid_cells(conn) if c["grid_number"] == grid_number]
    if not cells:
        raise SystemExit(f"grid {grid_number} not found")
    cell = cells[0]
    rect = ee.Geometry.Rectangle([cell["longitude"] - CELL_HALF_DEG, cell["latitude"] - CELL_HALF_DEG,
                                  cell["longitude"] + CELL_HALF_DEG, cell["latitude"] + CELL_HALF_DEG],
                                 None, False)
    day = hour_utc.date()
    collection = chunk_image_collection(ee, day, day, "any")
    band_map = chunk_band_metadata(ee, collection)
    halves = {stamp: band for band, (stamp, _s) in band_map.items()
              if stamp.replace(minute=0, second=0, microsecond=0) == hour_utc}
    if len(halves) != 2:
        raise SystemExit(f"{hour_utc}: expected 2 half-hours, found {sorted(halves)}")
    stacked = collection.toBands()
    arrays = []
    for stamp in sorted(halves):
        px = stacked.select(halves[stamp]).sampleRectangle(region=rect, defaultValue=-1).getInfo()
        arr = np.asarray(px["properties"][halves[stamp]], dtype=float)
        arr[arr < 0] = np.nan
        arrays.append(arr)
    numpy_value = hourly_pixel_max(arrays[0], arrays[1])
    stored_style = 0.5 * np.nanmax(arrays[0]) + 0.5 * np.nanmax(arrays[1])
    rows = reduce_pixelmax(ee, collection, build_cells_feature_collection(ee, cells), scale, band_map)
    server = [r for r in rows if r[1] == hour_utc]
    with conn.cursor() as cur:
        cur.execute(f"SELECT precipitation_max_mm, precipitation_pixelmax_mm FROM {OUTPUT_TABLE} "
                    f"WHERE grid_number = %s AND observation_time = %s", (grid_number, hour_utc))
        db = cur.fetchone()
    print(f"grid {grid_number} {hour_utc:%Y-%m-%d %H:%M} UTC: pixels {arrays[0].shape}")
    print(f"  first half max {np.nanmax(arrays[0]):.3f} mm/h at "
          f"{np.unravel_index(np.nanargmax(arrays[0]), arrays[0].shape)}, second half max "
          f"{np.nanmax(arrays[1]):.3f} mm/h at {np.unravel_index(np.nanargmax(arrays[1]), arrays[1].shape)}")
    print(f"  stored-style 0.5*max(a)+0.5*max(b) = {stored_style:.3f} mm   "
          f"(DB precipitation_max_mm = {db[0] if db else None})")
    print(f"  per-pixel hourly max, numpy        = {numpy_value:.3f} mm   "
          f"(DB precipitation_pixelmax_mm = {db[1] if db else None})")
    print(f"  per-pixel hourly max, Earth Engine = {server[0][2]:.3f} mm" if server
          else "  server-side value missing")
    if server and abs(server[0][2] - numpy_value) > TOLERANCE_MM:
        raise SystemExit("MISMATCH between numpy reference and the server-side reduction")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    ap.add_argument("--start-date", type=parse_date, default=parse_date("2024-06-30"))
    ap.add_argument("--end-date", type=parse_date, default=datetime.now(timezone.utc).date())
    ap.add_argument("--chunk-days", type=int, default=3)
    ap.add_argument("--scale", type=int, default=GEE_PIXEL_SCALE_METERS)
    ap.add_argument("--check", nargs=2, metavar=("GRID", "HOUR_UTC"), default=None,
                    help="pixel-level check of one cell-hour, e.g. --check 2 2026-10-20T07")
    args = ap.parse_args()

    ee = require_ee_dependency()
    initialize_earth_engine(ee, args.project_id, False, "localhost")
    conn = connect()
    try:
        ensure_table(conn)   # adds the column on an existing table
        if args.check:
            hour = datetime.strptime(args.check[1], "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            check_one(ee, conn, int(args.check[0]), hour, args.scale)
            return
        cells = fetch_grid_cells(conn)
        cells_fc = build_cells_feature_collection(ee, cells)
        chunks = list(date_chunks(args.start_date, args.end_date, args.chunk_days))
        total_updated = total_pending = 0
        t_run = time_module.time()
        for i, (a, b) in enumerate(chunks, start=1):
            pending = pending_rows(conn, a, b)
            if pending == 0:
                log(f"chunk {i}/{len(chunks)} {a}..{b}: nothing pending, skipped")
                continue
            t0 = time_module.time()
            collection = chunk_image_collection(ee, a, b, "any")
            band_map = chunk_band_metadata(ee, collection)
            rows = reduce_pixelmax(ee, collection, cells_fc, args.scale, band_map) if band_map else []
            t_gee = time_module.time() - t0
            updated = write_rows(conn, rows) if rows else 0
            total_updated += updated
            total_pending += pending
            log(f"chunk {i}/{len(chunks)} {a}..{b}: pending {pending:,}, updated {updated:,}, "
                f"not reproduced/left NULL {pending - updated:,}; gee {t_gee:.0f}s "
                f"total {time_module.time() - t0:.0f}s; {(time_module.time() - t_run) / 60:.0f} min so far")
        log(f"done: updated {total_updated:,} of {total_pending:,} pending rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
