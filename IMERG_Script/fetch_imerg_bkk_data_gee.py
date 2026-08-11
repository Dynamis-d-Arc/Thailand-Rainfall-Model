import argparse
import math
import os
import sys
from collections import defaultdict
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

from create_imerg_bkk_data_table import CREATE_TABLE_SQL  # noqa: E402


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

GRID_TABLE = '"Bangkok_Grid_9km"'
OM_TABLE = '"OM_BKK_DATA"'
OUTPUT_TABLE = '"IMERG_BKK_DATA"'
IMERG_COLLECTION = "NASA/GPM_L3/IMERG_V07"
PRODUCT_SHORT_NAME = "GPM_3IMERGHH"
PRODUCT_VERSION = "07"
SOURCE_SERVICE = "Google Earth Engine / NASA GPM IMERG V07"
TIMEZONE_NAME = "Asia/Bangkok"
LOCAL_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
DEFAULT_PROJECT_ID = os.getenv("EE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
DEFAULT_BANDS = [
    "precipitation",
    "randomError",
    "probabilityLiquidPrecipitation",
]
GEE_PIXEL_SCALE_METERS = 11132


INSERT_SQL = f"""
INSERT INTO {OUTPUT_TABLE} (
    grid_number,
    grid_row,
    grid_column,
    longitude,
    latitude,
    imerg_longitude,
    imerg_latitude,
    observation_time,
    local_observation_time,
    timezone,
    precipitation_mm,
    precipitation_rate_mean_mm_h,
    precipitation_rate_max_mm_h,
    precipitation_rate_first_half_hour_mm_h,
    precipitation_rate_second_half_hour_mm_h,
    random_error_mean_mm_h,
    probability_liquid_precipitation_mean_pct,
    precipitation_quality_index_mean,
    half_hour_count,
    product_short_name,
    product_version,
    run_type,
    source_service,
    source_granules
)
VALUES %s
ON CONFLICT (grid_number, observation_time) DO UPDATE SET
    grid_row = EXCLUDED.grid_row,
    grid_column = EXCLUDED.grid_column,
    longitude = EXCLUDED.longitude,
    latitude = EXCLUDED.latitude,
    imerg_longitude = EXCLUDED.imerg_longitude,
    imerg_latitude = EXCLUDED.imerg_latitude,
    local_observation_time = EXCLUDED.local_observation_time,
    timezone = EXCLUDED.timezone,
    precipitation_mm = EXCLUDED.precipitation_mm,
    precipitation_rate_mean_mm_h = EXCLUDED.precipitation_rate_mean_mm_h,
    precipitation_rate_max_mm_h = EXCLUDED.precipitation_rate_max_mm_h,
    precipitation_rate_first_half_hour_mm_h = EXCLUDED.precipitation_rate_first_half_hour_mm_h,
    precipitation_rate_second_half_hour_mm_h = EXCLUDED.precipitation_rate_second_half_hour_mm_h,
    random_error_mean_mm_h = EXCLUDED.random_error_mean_mm_h,
    probability_liquid_precipitation_mean_pct = EXCLUDED.probability_liquid_precipitation_mean_pct,
    precipitation_quality_index_mean = EXCLUDED.precipitation_quality_index_mean,
    half_hour_count = EXCLUDED.half_hour_count,
    product_short_name = EXCLUDED.product_short_name,
    product_version = EXCLUDED.product_version,
    run_type = EXCLUDED.run_type,
    source_service = EXCLUDED.source_service,
    source_granules = EXCLUDED.source_granules,
    fetched_at = now();
"""


SUMMARY_SQL = f"""
SELECT
    COUNT(*) AS row_count,
    COUNT(DISTINCT grid_number) AS grid_count,
    MIN(observation_time) AS min_time,
    MAX(observation_time) AS max_time,
    COUNT(*) FILTER (WHERE is_complete_hour) AS complete_rows,
    COUNT(*) FILTER (
        WHERE is_complete_hour AND precipitation_rate_max_mm_h >= 2.5
    ) AS moderate_or_heavier_rows,
    COUNT(*) FILTER (
        WHERE is_complete_hour AND precipitation_rate_max_mm_h >= 7.5
    ) AS heavy_candidate_rows
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
            "  .\\.venv\\Scripts\\python.exe -m pip install earthengine-api\n"
            "or:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install -r .\\IMERG_Script\\requirements.txt"
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


def fetch_grid_points(conn):
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
                "imerg_longitude": imerg_cell_center(float(row[3])),
                "imerg_latitude": imerg_cell_center(float(row[4])),
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


def imerg_cell_center(value):
    return round((math.floor(value * 10.0) / 10.0) + 0.05, 6)


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def mean_non_null(values):
    available = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(available)) if available else None


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


def build_points_feature_collection(ee, points):
    features = []
    for point in points:
        features.append(
            ee.Feature(
                ee.Geometry.Point([point["longitude"], point["latitude"]]),
                {
                    "grid_number": point["grid_number"],
                    "grid_row": point["grid_row"],
                    "grid_column": point["grid_column"],
                    "longitude": point["longitude"],
                    "latitude": point["latitude"],
                    "imerg_longitude": point["imerg_longitude"],
                    "imerg_latitude": point["imerg_latitude"],
                },
            )
        )
    return ee.FeatureCollection(features)


def available_bands(ee, start_date, end_date):
    start, stop = utc_chunk_bounds(start_date, end_date)
    collection = ee.ImageCollection(IMERG_COLLECTION).filterDate(start.isoformat(), stop.isoformat())
    first = collection.first()
    if first is None:
        return []
    return first.bandNames().getInfo()


def chunk_image_collection(ee, start_date, end_date, bands, status):
    start, stop = utc_chunk_bounds(start_date, end_date)
    collection = (
        ee.ImageCollection(IMERG_COLLECTION)
        .filterDate(start.isoformat(), stop.isoformat())
        .select(bands)
    )
    if status != "any":
        collection = collection.filter(ee.Filter.eq("status", status))
    return collection


def sample_collection(ee, collection, points_fc, scale):
    def image_to_samples(image):
        samples = image.sampleRegions(
            collection=points_fc,
            scale=scale,
            geometries=False,
        )
        return samples.map(
            lambda feature: feature.set(
                {
                    "sample_time_millis": image.get("system:time_start"),
                    "imerg_status": image.get("status"),
                    "source_image_id": image.id(),
                }
            )
        )

    return ee.FeatureCollection(collection.map(image_to_samples).flatten())


def read_half_hour_samples(features):
    samples = defaultdict(dict)
    for feature in features:
        props = feature.get("properties", {})
        millis = props.get("sample_time_millis")
        if millis is None:
            continue
        sample_time = datetime.fromtimestamp(int(millis) / 1000, tz=timezone.utc)
        hour = sample_time.replace(minute=0, second=0, microsecond=0)
        grid_number = int(props["grid_number"])
        precipitation_rate = finite_or_none(props.get("precipitation"))
        if precipitation_rate is not None and precipitation_rate < 0:
            precipitation_rate = None
        sample = {
            "time": sample_time,
            "point": {
                "grid_number": grid_number,
                "grid_row": int(props["grid_row"]),
                "grid_column": int(props["grid_column"]),
                "longitude": float(props["longitude"]),
                "latitude": float(props["latitude"]),
                "imerg_longitude": float(props["imerg_longitude"]),
                "imerg_latitude": float(props["imerg_latitude"]),
            },
            "precipitation_rate": precipitation_rate,
            "random_error": finite_or_none(props.get("randomError")),
            "liquid_probability": finite_or_none(props.get("probabilityLiquidPrecipitation")),
            "quality_index": None,
            "status": props.get("imerg_status") or "unknown",
            "source_image_id": props.get("source_image_id") or "",
        }
        samples[(grid_number, hour)][sample_time] = sample
    return samples


def run_type_from_samples(hourly_samples):
    statuses = sorted(
        {
            str(sample["status"]).lower()
            for sample in hourly_samples
            if sample.get("status") not in {None, ""}
        }
    )
    if not statuses:
        return "unknown"
    if len(statuses) == 1:
        return statuses[0]
    return "mixed"


def aggregate_hourly_rows(samples, store_source_images=False):
    rows = []
    for (_, hour), samples_by_time in sorted(samples.items()):
        hourly_samples = sorted(samples_by_time.values(), key=lambda sample: sample["time"])
        valid_precipitation = [
            sample for sample in hourly_samples if sample["precipitation_rate"] is not None
        ]
        half_hour_count = min(len(valid_precipitation), 2)
        point = hourly_samples[0]["point"]
        rates = [sample["precipitation_rate"] for sample in valid_precipitation[:2]]
        precipitation_mm = float(sum(rate * 0.5 for rate in rates)) if half_hour_count == 2 else None
        mean_rate = float(np.mean(rates)) if rates else None
        max_rate = float(np.max(rates)) if rates else None
        first_rate = next(
            (
                sample["precipitation_rate"]
                for sample in valid_precipitation
                if sample["time"].minute < 30
            ),
            None,
        )
        second_rate = next(
            (
                sample["precipitation_rate"]
                for sample in valid_precipitation
                if sample["time"].minute >= 30
            ),
            None,
        )
        local_hour = hour.astimezone(LOCAL_TIMEZONE).replace(tzinfo=None)
        source_images = sorted(
            {
                f"{sample['source_image_id']}|{sample['status']}"
                for sample in hourly_samples
                if sample.get("source_image_id")
            }
        )
        rows.append(
            (
                point["grid_number"],
                point["grid_row"],
                point["grid_column"],
                point["longitude"],
                point["latitude"],
                point["imerg_longitude"],
                point["imerg_latitude"],
                hour,
                local_hour,
                TIMEZONE_NAME,
                precipitation_mm,
                mean_rate,
                max_rate,
                first_rate,
                second_rate,
                mean_non_null(sample["random_error"] for sample in hourly_samples),
                mean_non_null(sample["liquid_probability"] for sample in hourly_samples),
                mean_non_null(sample["quality_index"] for sample in hourly_samples),
                half_hour_count,
                PRODUCT_SHORT_NAME,
                PRODUCT_VERSION,
                run_type_from_samples(hourly_samples),
                SOURCE_SERVICE,
                source_images if store_source_images else [],
            )
        )
    return rows


def upsert_rows(conn, rows):
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, page_size=5000)
    conn.commit()
    return len(rows)


def fetch_summary(conn):
    with conn.cursor() as cur:
        cur.execute(SUMMARY_SQL)
        return cur.fetchone()


def parse_bands(value):
    bands = [item.strip() for item in value.split(",") if item.strip()]
    if "precipitation" not in bands:
        raise argparse.ArgumentTypeError("--bands must include precipitation.")
    return bands


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Sample NASA/GPM_L3/IMERG_V07 from Google Earth Engine, aggregate half-hourly '
            'rates into hourly Bangkok-grid rows, and upsert PostgreSQL table "IMERG_BKK_DATA".'
        )
    )
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument(
        "--auth-mode",
        default="localhost",
        choices=["localhost", "notebook", "gcloud"],
        help="Earth Engine auth mode used only with --authenticate.",
    )
    parser.add_argument("--start-date", type=parse_date, default=None)
    parser.add_argument("--end-date", type=parse_date, default=None)
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=1,
        help="Days per Earth Engine getInfo request. One day stays under getInfo feature limits.",
    )
    parser.add_argument(
        "--bands",
        type=parse_bands,
        default=DEFAULT_BANDS,
        help="Comma-separated Earth Engine bands to sample.",
    )
    parser.add_argument(
        "--status",
        choices=["any", "permanent", "provisional"],
        default="any",
        help="Filter the IMERG status metadata. Use permanent for final-only training data.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=GEE_PIXEL_SCALE_METERS,
        help="Earth Engine sample scale in meters.",
    )
    parser.add_argument(
        "--replace-range",
        action="store_true",
        help="Delete only the requested date range before loading it again.",
    )
    parser.add_argument(
        "--no-skip-complete-chunks",
        action="store_true",
        help="Reprocess chunks even when all expected complete hourly rows already exist.",
    )
    parser.add_argument(
        "--store-source-images",
        action="store_true",
        help="Store image IDs and status in source_granules. Disabled by default to save space.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Create/verify IMERG_BKK_DATA and exit without Earth Engine authentication.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show resolved range and chunks without contacting Earth Engine.",
    )
    parser.add_argument(
        "--limit-chunks",
        type=int,
        default=None,
        help="Process at most this many chunks, useful for a short validation run.",
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

    with connect() as conn:
        ensure_table(conn)
        if args.schema_only:
            log(f"Created/verified {OUTPUT_TABLE}; schema-only run complete.")
            return

        points = fetch_grid_points(conn)
        if not points:
            raise RuntimeError(f"No Bangkok grid points found in {GRID_TABLE}.")
        start_date, end_date, om_start, om_end = resolve_date_range(
            conn, args.start_date, args.end_date
        )
        chunks = list(date_chunks(start_date, end_date, args.chunk_days))
        if args.limit_chunks is not None:
            chunks = chunks[: args.limit_chunks]
        expected_total_rows = len(points) * ((end_date - start_date).days + 1) * 24

        log(
            f"OM history={om_start}..{om_end}; IMERG request={start_date}..{end_date}; "
            f"status={args.status}"
        )
        log(
            f"Grid points={len(points)}; chunks={len(chunks)}; "
            f"expected hourly rows={expected_total_rows:,}; project={args.project_id}"
        )
        if args.dry_run:
            for chunk_start, chunk_end in chunks:
                print(f"{chunk_start}..{chunk_end}")
            return

        if args.replace_range:
            deleted = delete_date_range(conn, start_date, end_date)
            log(f"Deleted {deleted:,} existing rows inside the requested date range.")

        ee = require_ee_dependency()
        initialize_earth_engine(ee, args.project_id, args.authenticate, args.auth_mode)
        available = set(available_bands(ee, start_date, end_date))
        requested_bands = [band for band in args.bands if band in available]
        if "precipitation" not in requested_bands:
            raise RuntimeError(
                f"IMERG collection did not expose precipitation. Available bands: {sorted(available)}"
            )
        missing_bands = [band for band in args.bands if band not in available]
        if missing_bands:
            log(f"Skipping bands unavailable in Earth Engine: {missing_bands}")
        points_fc = build_points_feature_collection(ee, points)

        total_upserted = 0
        for chunk_number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            expected_chunk_rows = len(points) * ((chunk_end - chunk_start).days + 1) * 24
            existing_complete = fetch_complete_row_count(conn, chunk_start, chunk_end)
            if not args.no_skip_complete_chunks and existing_complete >= expected_chunk_rows:
                log(
                    f"Chunk {chunk_number}/{len(chunks)} {chunk_start}..{chunk_end}: "
                    f"skipped ({existing_complete:,}/{expected_chunk_rows:,} complete rows)."
                )
                continue

            collection = chunk_image_collection(
                ee,
                chunk_start,
                chunk_end,
                requested_bands,
                args.status,
            )
            image_count = int(collection.size().getInfo())
            log(
                f"Chunk {chunk_number}/{len(chunks)} {chunk_start}..{chunk_end}: "
                f"sampling {image_count:,} half-hour images."
            )
            if image_count == 0:
                continue

            sampled_fc = sample_collection(ee, collection, points_fc, args.scale)
            features = sampled_fc.getInfo().get("features", [])
            samples = read_half_hour_samples(features)
            rows = aggregate_hourly_rows(
                samples,
                store_source_images=args.store_source_images,
            )
            written = upsert_rows(conn, rows)
            total_upserted += written
            complete_in_result = sum(row[18] == 2 for row in rows)
            log(
                f"Chunk {chunk_number}/{len(chunks)}: upserted {written:,} rows; "
                f"complete={complete_in_result:,}; sampled_features={len(features):,}."
            )

        summary = fetch_summary(conn)

    log(f"Finished; rows inserted/updated this run={total_upserted:,}.")
    print(
        f"{OUTPUT_TABLE}: rows={summary[0]:,}, grids={summary[1]:,}, "
        f"time={summary[2]}..{summary[3]}, complete={summary[4]:,}, "
        f"max_rate>=2.5={summary[5]:,}, max_rate>=7.5={summary[6]:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
