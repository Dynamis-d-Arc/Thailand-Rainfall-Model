"""
db_check.py  --  read-only snapshot of the Phase 1 PostgreSQL database.

Run this from your project venv:

    .\.venv\Scripts\python.exe .\db_check.py

It connects to localhost:5432 (override via PG* env vars), runs a handful of
SELECT-only queries, prints a summary, and writes two files next to itself:

    db_status_report.txt          -- human-readable snapshot
    imerg_backfill_progress.csv   -- one-row machine-readable progress

Nothing is modified. Safe to run while the backfill is in progress.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}

HERE = Path(__file__).resolve().parent
REPORT_PATH = HERE / "db_status_report.txt"
CSV_PATH = HERE / "imerg_backfill_progress.csv"

IMERG_TABLE = '"IMERG_BKK_DATA"'

lines = []


def emit(text=""):
    print(text, flush=True)
    lines.append(text)


def run_scalar_queries(cur):
    # --- table inventory ------------------------------------------------
    emit("=== TABLE INVENTORY (public schema) ===")
    cur.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
    )
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{t}";')
            emit(f"  {t:<32} rows={cur.fetchone()[0]:,}")
        except Exception as exc:  # noqa: BLE001
            emit(f"  {t:<32} (count failed: {exc})")
    emit()
    return tables


def run_imerg_progress(cur):
    emit("=== IMERG BACKFILL PROGRESS (2021-07-20 .. 2026-07-19 window) ===")
    progress = {}

    cur.execute(
        f"""
        SELECT COUNT(*),
               COUNT(*) FILTER (WHERE is_complete_hour),
               MIN(observation_time AT TIME ZONE 'UTC'),
               MAX(observation_time AT TIME ZONE 'UTC'),
               COUNT(DISTINCT (observation_time AT TIME ZONE 'UTC')::date)
        FROM {IMERG_TABLE}
        WHERE observation_time >= TIMESTAMPTZ '2021-07-20 00:00:00+00'
          AND observation_time <  TIMESTAMPTZ '2026-07-20 00:00:00+00';
        """
    )
    total, complete, min_t, max_t, distinct_days = cur.fetchone()
    progress.update(
        total_rows=total,
        complete_rows=complete,
        min_observation_utc=str(min_t),
        max_observation_utc=str(max_t),
        distinct_days=distinct_days,
    )
    emit(f"  total rows            : {total:,}")
    emit(f"  complete-hour rows    : {complete:,}")
    emit(f"  min observation (UTC) : {min_t}")
    emit(f"  max observation (UTC) : {max_t}")
    emit(f"  distinct days covered : {distinct_days:,} / 1826 expected")
    if distinct_days:
        emit(f"  day-coverage progress : {distinct_days / 1826 * 100:.1f}%")

    cur.execute(
        f"""
        SELECT COUNT(*) FILTER (WHERE is_complete_hour AND precipitation_rate_max_mm_h >= 2.5),
               COUNT(*) FILTER (WHERE is_complete_hour AND precipitation_rate_max_mm_h >= 7.5),
               MAX(precipitation_mm),
               MAX(precipitation_rate_max_mm_h)
        FROM {IMERG_TABLE}
        WHERE observation_time >= TIMESTAMPTZ '2021-07-20 00:00:00+00'
          AND observation_time <  TIMESTAMPTZ '2026-07-20 00:00:00+00';
        """
    )
    moderate, heavy, max_mm, max_rate = cur.fetchone()
    progress.update(
        moderate_or_heavier_rows=moderate,
        heavy_candidate_rows=heavy,
        max_hourly_precip_mm=max_mm,
        max_halfhour_rate_mm_h=max_rate,
    )
    emit(f"  moderate+ rows (>=2.5): {moderate:,}")
    emit(f"  heavy rows (>=7.5)    : {heavy:,}")
    emit(f"  max hourly precip mm  : {max_mm}")
    emit(f"  max half-hour rate    : {max_rate}")
    emit()
    return progress


def main():
    emit(f"Snapshot generated: {datetime.now(timezone.utc).isoformat()}  (host={DB_CONFIG['host']}:{DB_CONFIG['port']})")
    emit()
    with psycopg2.connect(**DB_CONFIG) as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            run_scalar_queries(cur)
            progress = None
            try:
                progress = run_imerg_progress(cur)
            except Exception as exc:  # noqa: BLE001
                emit(f"(IMERG progress query skipped: {exc})")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    if progress:
        progress["snapshot_utc"] = datetime.now(timezone.utc).isoformat()
        with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(list(progress.keys()))
            writer.writerow(list(progress.values()))

    print(f"\nWrote {REPORT_PATH.name} and {CSV_PATH.name} to {HERE}")


if __name__ == "__main__":
    main()
