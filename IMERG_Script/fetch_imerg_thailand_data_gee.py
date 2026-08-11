"""Sample NASA/GPM_L3/IMERG_V07 over the 833-cell Thailand 25 km grid via Earth Engine.

Thailand counterpart of fetch_imerg_bkk_data_gee.py. The Bangkok script maps one
sampleRegions call over each half-hour image and flattens, producing
(images x cells) features per request. At Bangkok scale that is 48 x 56 = 2,688,
comfortably inside Earth Engine's 5,000-element collection limit. At Thailand scale
it is 48 x 833 = 39,984 and every request dies with

    Collection query aborted after accumulating over 5000 elements.

So the reduction is transposed: a chunk's half-hour images are stacked into a single
multi-band image with toBands(), and reduceRegions runs once over the 833 cells. The
result is 833 features regardless of chunk length -- element count no longer scales
with time, only the per-feature property count does. Timing measured on this box:
~4 s for a 1-day chunk, ~2.5 s/day at --chunk-days 3.

The second difference is the reduction itself. A 25 km cell spans ~5-6 IMERG pixels,
so each half-hour is reduced with mean AND max over the cell footprint (see
DB_Script/create_imerg_thailand_data_table.py for why both are stored).

Usage:
    python IMERG_Script/fetch_imerg_thailand_data_gee.py --schema-only
    python IMERG_Script/fetch_imerg_thailand_data_gee.py --limit-chunks 3   # pilot
    python IMERG_Script/fetch_imerg_thailand_data_gee.py                    # full panel
"""

import argparse
import os
import sys
import time as time_module
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psycopg2
from psycopg2.extras import execute_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_SCRIPT_DIR = PROJECT_ROOT / "DB_Script"
if str(DB_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DB_SCRIPT_DIR))

from create_imerg_thailand_data_table import CREATE_TABLE_SQL  # noqa: E402


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

GRID_TABLE = '"Thailand_Grid_25km"'
OM_TABLE = '"OM_Thailand_Data"'
OUTPUT_TABLE = '"IMERG_THAILAND_DATA"'
IMERG_COLLECTION = "NASA/GPM_L3/IMERG_V07"
PRODUCT_SHORT_NAME = "GPM_3IMERGHH"
PRODUCT_VERSION = "07"
SOURCE_SERVICE = "Google Earth Engine / NASA GPM IMERG V07"
TIMEZONE_NAME = "Asia/Bangkok"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
DEFAULT_PROJECT_ID = (
    os.getenv("EE_PROJECT_ID")
    or os.getenv("GOOGLE_CLOUD_PROJECT")
    or "gen-lang-client-0449940358"
)

GEE_PIXEL_SCALE_METERS = 11132   # IMERG V07 native 0.1 deg
CELL_HALF_DEG = 0.1126           # 25 km / 2, in degrees
BAND = "precipitation"           # mm/h
MAX_ATTEMPTS = 4


INSERT_SQL = f"""
INSERT INTO {OUTPUT_TABLE} (
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    observation_time,
    local_observation_time,
    timezone,
    precipitation_mm,
    precipitation_max_mm,
    precipitation_rate_mean_mm_h,
    precipitation_rate_max_mm_h,
    precipitation_rate_first_half_hour_mm_h,
    precipitation_rate_second_half_hour_mm_h,
    pixel_count,
    half_hour_count,
    product_short_name,
    product_version,
    run_type,
    source_service
)
VALUES %s
ON CONFLICT (grid_number, observation_time) DO UPDATE SET
    grid_row = EXCLUDED.grid_row,
    grid_column = EXCLUDED.grid_column,
    longitude = EXCLUDED.longitude,
    latitude = EXCLUDED.latitude,
    local_observation_time = EXCLUDED.local_observation_time,
    timezone = EXCLUDED.timezone,
    precipitation_mm = EXCLUDED.precipitation_mm,
    precipitation_max_mm = EXCLUDED.precipitation_max_mm,
    precipitation_rate_mean_mm_h = EXCLUDED.precipitation_rate_mean_mm_h,
    precipitation_rate_max_mm_h = EXCLUDED.precipitation_rate_max_mm_h,
    precipitation_rate_first_half_hour_mm_h = EXCLUDED.precipitation_rate_first_half_hour_mm_h,
    precipitation_rate_second_half_hour_mm_h = EXCLUDED.precipitation_rate_second_half_hour_mm_h,
    pixel_count = EXCLUDED.pixel_count,
    half_hour_count = EXCLUDED.half_hour_count,
    product_short_name = EXCLUDED.product_short_name,
    product_version = EXCLUDED.product_version,
    run_type = EXCLUDED.run_type,
    source_service = EXCLUDED.source_service,
    fetched_at = now();
"""


SUMMARY_SQL = f"""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT grid_number) AS grid_count,
    MIN(observation_time) AS min_time,
    MAX(observation_time) AS max_time,
    COUNT(*) FILTER (WHERE is_complete_hour) AS complete_rows,
    COUNT(*) FILTER (WHERE is_complete_hour AND precipitation_max_mm >= 0.1) AS wet_cell_max_rows,
    COUNT(*) FILTER (WHERE is_complete_hour AND precipitation_mm >= 0.1) AS wet_cell_mean_rows
FROM {OUTPUT_TABLE};
"""


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def require_ee_dependency():
    try:
        import ee
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Earth Engine dependency. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install earthengine-api"
        ) from exc
    return ee


def connect():
    return psycopg2.connect(**DB_CONFIG)


def parse_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; use YYYY-MM-DD.") from exc


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()


def fetch_grid_cells(conn):
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT grid_number, grid_row, grid_column, longitude, latitude
            FROM {GRID_TABLE}
            ORDER BY grid_number;
            """
        )
        return [
            {
                "grid_number": int(row[0]),
                "grid_row": int(row[1]),
                "grid_column": int(row[2]),
                "longitude": float(row[3]),
                "latitude": float(row[4]),
            }
            for row in cur.fetchall()
        ]


def existing_om_date_range(conn):
    with conn.cursor() as cur:
        cur.execute(f"SELECT MIN(forecast_time), MAX(forecast_time) FROM {OM_TABLE};")
        start, end = cur.fetchone()
    if start is None or end is None:
        raise RuntimeError(f"{OM_TABLE} is empty; pass --start-date and --end-date explicitly.")
    return start.date(), end.date()


def resolve_date_range(conn, requested_start, requested_end):
    om_start, om_end = existing_om_date_range(conn)
    start_date = requested_start or om_start
    end_date = requested_end or om_end
    if end_date < start_date:
        raise RuntimeError(f"Resolved end date {end_date} is before start date {start_date}.")
    return start_date, end_date, om_start, om_end


def date_chunks(start_date, end_date, days_per_chunk):
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=days_per_chunk - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def utc_chunk_bounds(start_date, end_date):
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    stop_exclusive = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return start, stop_exclusive


def fetch_complete_row_count(conn, start_date, end_date):
    start, stop = utc_chunk_bounds(start_date, end_date)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM {OUTPUT_TABLE}
            WHERE observation_time >= %s
              AND observation_time < %s
              AND is_complete_hour;
            """,
            (start, stop),
        )
        return int(cur.fetchone()[0])


def delete_date_range(conn, start_date, end_date):
    start, stop = utc_chunk_bounds(start_date, end_date)
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {OUTPUT_TABLE} WHERE observation_time >= %s AND observation_time < %s;",
            (start, stop),
        )
        deleted = cur.rowcount
    conn.commit()
    return deleted


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or value < 0:   # IMERG fill values are negative
        return None
    return value


def initialize_earth_engine(ee, project_id, authenticate, auth_mode):
    if not project_id:
        raise RuntimeError(
            "Earth Engine project ID is required. Pass --project-id or set EE_PROJECT_ID."
        )
    if authenticate:
        log("Opening Earth Engine authentication flow.")
        ee.Authenticate(auth_mode=auth_mode)
    try:
        ee.Initialize(project=project_id)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run once with --authenticate, complete the "
            "browser login, then rerun the same command without --authenticate."
        ) from exc


def build_cells_feature_collection(ee, cells):
    features = [
        ee.Feature(
            ee.Geometry.Rectangle(
                [
                    cell["longitude"] - CELL_HALF_DEG,
                    cell["latitude"] - CELL_HALF_DEG,
                    cell["longitude"] + CELL_HALF_DEG,
                    cell["latitude"] + CELL_HALF_DEG,
                ],
                None,
                False,   # planar rectangle in EPSG:4326; no geodesic densification needed
            ),
            {"grid_number": cell["grid_number"]},
        )
        for cell in cells
    ]
    return ee.FeatureCollection(features)


def chunk_image_collection(ee, start_date, end_date, status):
    start, stop = utc_chunk_bounds(start_date, end_date)
    collection = (
        ee.ImageCollection(IMERG_COLLECTION)
        .filterDate(start.isoformat(), stop.isoformat())
        .select([BAND])
        .sort("system:time_start")
    )
    if status != "any":
        collection = collection.filter(ee.Filter.eq("status", status))
    return collection


def with_retries(label, fn):
    """Earth Engine returns transient 429/500s under sustained load; back off and retry."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 5 * (2 ** (attempt - 1))
            log(f"  {label}: attempt {attempt}/{MAX_ATTEMPTS} failed ({str(exc)[:90]}); "
                f"retrying in {wait}s")
            time_module.sleep(wait)


def chunk_band_metadata(ee, collection):
    """One cheap round trip for the band -> (time, status) map.

    toBands() names each band '<system:index>_precipitation', so the collection's
    index/time/status arrays are all that is needed to decode the reduceRegions output.
    """
    meta = ee.Dictionary(
        {
            "ids": collection.aggregate_array("system:index"),
            "times": collection.aggregate_array("system:time_start"),
            "status": collection.aggregate_array("status"),
        }
    )
    info = with_retries("metadata", meta.getInfo)
    band_map = {}
    for index, millis, status in zip(info["ids"], info["times"], info["status"]):
        band_map[f"{index}_{BAND}"] = (
            datetime.fromtimestamp(int(millis) / 1000, tz=timezone.utc),
            str(status).lower() if status else "unknown",
        )
    return band_map


def reduce_chunk(ee, collection, cells_fc, scale):
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.max(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )
    stacked = collection.toBands()
    reduced = stacked.reduceRegions(collection=cells_fc, reducer=reducer, scale=scale)
    info = with_retries("reduceRegions", reduced.getInfo)
    return info.get("features", [])


def rows_from_features(features, cells_by_number, band_map):
    """Feature properties are '<band>_mean' / '_max' / '_count'; regroup them by hour."""
    rows = []
    suffixes = ("_mean", "_max", "_count")
    for feature in features:
        props = feature.get("properties", {})
        grid_number = int(props["grid_number"])
        cell = cells_by_number[grid_number]

        by_hour = {}
        for key, value in props.items():
            suffix = next((s for s in suffixes if key.endswith(s)), None)
            if suffix is None:
                continue
            band = key[: -len(suffix)]
            stamp_status = band_map.get(band)
            if stamp_status is None:
                continue
            stamp, status = stamp_status
            hour = stamp.replace(minute=0, second=0, microsecond=0)
            slot = by_hour.setdefault(hour, {"status": set(), "halves": {}})
            slot["status"].add(status)
            half = slot["halves"].setdefault(stamp, {})
            half[suffix] = value

        for hour, slot in by_hour.items():
            halves = [slot["halves"][stamp] for stamp in sorted(slot["halves"])]
            means, maxes, counts, first_rate, second_rate = [], [], [], None, None
            for stamp, half in sorted(slot["halves"].items()):
                mean_rate = finite_or_none(half.get("_mean"))
                if mean_rate is None:
                    continue
                means.append(mean_rate)
                max_rate = finite_or_none(half.get("_max"))
                maxes.append(max_rate if max_rate is not None else mean_rate)
                pixels = half.get("_count")
                if pixels is not None:
                    counts.append(float(pixels))
                if stamp.minute < 30:
                    first_rate = mean_rate
                else:
                    second_rate = mean_rate

            half_hour_count = min(len(means), 2)
            complete = half_hour_count == 2
            statuses = sorted(s for s in slot["status"] if s and s != "unknown")
            if not statuses:
                run_type = "unknown"
            elif len(statuses) == 1:
                run_type = statuses[0]
            else:
                run_type = "mixed"

            rows.append(
                (
                    grid_number,
                    cell["grid_row"],
                    cell["grid_column"],
                    cell["longitude"],
                    cell["latitude"],
                    hour,
                    hour.astimezone(LOCAL_TIMEZONE).replace(tzinfo=None),
                    TIMEZONE_NAME,
                    float(sum(r * 0.5 for r in means[:2])) if complete else None,
                    float(sum(r * 0.5 for r in maxes[:2])) if complete else None,
                    float(np.mean(means)) if means else None,
                    float(np.max(maxes)) if maxes else None,
                    first_rate,
                    second_rate,
                    float(np.mean(counts)) if counts else None,
                    half_hour_count,
                    PRODUCT_SHORT_NAME,
                    PRODUCT_VERSION,
                    run_type,
                    SOURCE_SERVICE,
                )
            )
    return rows


def upsert_rows(conn, rows):
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, page_size=10000)
    conn.commit()
    return len(rows)


def fetch_summary(conn):
    with conn.cursor() as cur:
        cur.execute(SUMMARY_SQL)
        return cur.fetchone()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample NASA/GPM_L3/IMERG_V07 from Earth Engine over the Thailand 25 km grid, "
            'reduce each cell with mean+max, and upsert PostgreSQL table "IMERG_THAILAND_DATA".'
        )
    )
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument(
        "--auth-mode",
        default="localhost",
        choices=["localhost", "notebook", "gcloud"],
    )
    parser.add_argument("--start-date", type=parse_date, default=None)
    parser.add_argument("--end-date", type=parse_date, default=None)
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=3,
        help="Days per Earth Engine request. Element count is fixed at 833 regardless; "
             "longer chunks trade a bigger per-feature payload for fewer round trips.",
    )
    parser.add_argument(
        "--status",
        choices=["any", "permanent", "provisional"],
        default="any",
        help="Filter IMERG status metadata. Use permanent for final-only training data.",
    )
    parser.add_argument("--scale", type=int, default=GEE_PIXEL_SCALE_METERS)
    parser.add_argument(
        "--replace-range",
        action="store_true",
        help="Delete the requested date range before loading it again.",
    )
    parser.add_argument(
        "--no-skip-complete-chunks",
        action="store_true",
        help="Reprocess chunks even when all expected complete hourly rows already exist.",
    )
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Process at most this many chunks; use for a short validation run.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.chunk_days < 1:
        raise RuntimeError("--chunk-days must be at least 1.")
    if args.scale < 1:
        raise RuntimeError("--scale must be a positive integer.")
    if args.limit_chunks is not None and args.limit_chunks < 1:
        raise RuntimeError("--limit-chunks must be positive when provided.")


def main():
    args = parse_args()
    validate_args(args)

    conn = connect()
    try:
        ensure_table(conn)
        if args.schema_only:
            log(f"Created/verified {OUTPUT_TABLE}; schema-only run complete.")
            return

        cells = fetch_grid_cells(conn)
        if not cells:
            raise RuntimeError(f"No grid cells found in {GRID_TABLE}.")
        cells_by_number = {cell["grid_number"]: cell for cell in cells}

        start_date, end_date, om_start, om_end = resolve_date_range(
            conn, args.start_date, args.end_date
        )
        chunks = list(date_chunks(start_date, end_date, args.chunk_days))
        if args.limit_chunks is not None:
            chunks = chunks[: args.limit_chunks]
        total_days = (end_date - start_date).days + 1

        log(f"OM history={om_start}..{om_end}; IMERG request={start_date}..{end_date}; "
            f"status={args.status}")
        log(f"Grid cells={len(cells)}; chunk_days={args.chunk_days}; chunks={len(chunks)}; "
            f"expected hourly rows={len(cells) * total_days * 24:,}; project={args.project_id}")
        if args.dry_run:
            for chunk_start, chunk_end in chunks:
                print(f"{chunk_start}..{chunk_end}")
            return

        if args.replace_range:
            deleted = delete_date_range(conn, start_date, end_date)
            log(f"Deleted {deleted:,} existing rows inside the requested date range.")

        ee = require_ee_dependency()
        initialize_earth_engine(ee, args.project_id, args.authenticate, args.auth_mode)
        cells_fc = build_cells_feature_collection(ee, cells)

        total_upserted = 0
        processed_chunks = 0
        run_started = time_module.time()
        for chunk_number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            chunk_days = (chunk_end - chunk_start).days + 1
            expected_chunk_rows = len(cells) * chunk_days * 24
            existing_complete = fetch_complete_row_count(conn, chunk_start, chunk_end)
            if not args.no_skip_complete_chunks and existing_complete >= expected_chunk_rows:
                log(f"Chunk {chunk_number}/{len(chunks)} {chunk_start}..{chunk_end}: "
                    f"skipped ({existing_complete:,}/{expected_chunk_rows:,} complete rows).")
                continue

            t0 = time_module.time()
            collection = chunk_image_collection(ee, chunk_start, chunk_end, args.status)
            band_map = chunk_band_metadata(ee, collection)
            if not band_map:
                log(f"Chunk {chunk_number}/{len(chunks)} {chunk_start}..{chunk_end}: "
                    f"no images; skipped.")
                continue

            features = reduce_chunk(ee, collection, cells_fc, args.scale)
            t_gee = time_module.time() - t0
            rows = rows_from_features(features, cells_by_number, band_map)
            written = upsert_rows(conn, rows)
            total_upserted += written
            processed_chunks += 1

            complete_in_result = sum(row[15] == 2 for row in rows)
            elapsed = time_module.time() - run_started
            rate_days = (processed_chunks * args.chunk_days) / max(elapsed / 3600, 1e-9)
            log(f"Chunk {chunk_number}/{len(chunks)} {chunk_start}..{chunk_end}: "
                f"{len(band_map)} half-hours, upserted {written:,} rows "
                f"(complete={complete_in_result:,}); gee {t_gee:.1f}s "
                f"total {time_module.time() - t0:.1f}s; {rate_days:.0f} days/h")

        summary = fetch_summary(conn)
    finally:
        conn.close()

    log(f"Finished; rows inserted/updated this run={total_upserted:,}.")
    print(
        f"{OUTPUT_TABLE}: rows={summary[0]:,}, grids={summary[1]:,}, "
        f"time={summary[2]}..{summary[3]}, complete={summary[4]:,}, "
        f"cell_max>=0.1mm={summary[5]:,}, cell_mean>=0.1mm={summary[6]:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
