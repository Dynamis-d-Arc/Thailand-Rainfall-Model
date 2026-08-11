import argparse
import csv
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener


BASE_URL = "http://www.aws-observation.tmd.go.th"
REPORT_URL = f"{BASE_URL}/rprt/weatherMinute"
DATA_URL = f"{BASE_URL}/rprt/weatherMinuteData"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Data_Analysis" / "tmd_weather_data"

FIELD_MAP = {
    "sectime": "raw_sectime",
    "s00a": "wind_dir_avg_deg",
    "s00m": "max_wind_dir_deg",
    "s01a": "wind_speed_avg_knot",
    "s01m": "max_wind_speed_knot",
    "s02a": "temperature_c",
    "r01m": "precipitation_mm",
    "s04a": "pressure_hpa",
    "s05a": "humidity_percent",
    "s06m": "weather_code",
    "s07a": "visibility_m",
}


def log(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def parse_date(value):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"Invalid date: {value}. Use YYYY-MM-DD, for example 2026-01-01."
    )


def date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def post_form(opener, url, data):
    body = urlencode(data).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    with opener.open(request, timeout=60) as response:
        return response.read().decode("utf-8-sig")


def normalize_number(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def parse_utc_time(raw_sectime):
    if not raw_sectime:
        return None
    text = str(raw_sectime)
    if len(text) != 14:
        return text

    year = int(text[0:4])
    month = int(text[4:6])
    day = int(text[6:8])
    hour = int(text[8:10])
    minute = int(text[10:12])
    second = int(text[12:14])

    if hour == 24:
        dt = datetime(year, month, day, 0, minute, second) + timedelta(days=1)
    else:
        dt = datetime(year, month, day, hour, minute, second)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def clean_row(row, station_meta, region, station):
    cleaned = {
        "utc_time": parse_utc_time(row.get("sectime")),
        "region": region,
        "station": station,
        "station_name": station_meta.get("fname"),
        "latitude": normalize_number(station_meta.get("lat")),
        "longitude": normalize_number(station_meta.get("lon")),
        "altitude_m": normalize_number(station_meta.get("alt")),
    }
    for source_field, output_field in FIELD_MAP.items():
        value = row.get(source_field)
        cleaned[output_field] = value if source_field == "sectime" else normalize_number(value)
    return cleaned


def fetch_day(opener, target_date, region, station, interval_step, rows):
    # Visit report page first so the server creates a normal session.
    post_form(opener, REPORT_URL, {"regions": str(region), "station": str(station)})

    ymd = target_date.strftime("%Y%m%d")
    form_date = target_date.strftime("%Y/%m/%d")
    body = {
        "listwidth": "740",
        "syear": "",
        "smonth": "",
        "sday": "",
        "fdate": f"{ymd}240000",
        "sdate": f"{ymd}000000",
        "fyear": "",
        "fmonth": "",
        "fday": "",
        "gmt": "y",
        "rows": str(rows),
        "regions": str(region),
        "station": str(station),
        "step": str(interval_step),
        "indate": form_date,
        "shr": "00",
        "smin": "00",
        "fhr": "24",
        "fmin": "00",
    }

    raw_text = post_form(opener, DATA_URL, body)
    payload = json.loads(raw_text)
    if str(payload.get("resultCode")) != "200":
        raise RuntimeError(f"TMD returned error for {target_date}: {payload}")

    data = payload.get("data") or {}
    station_meta = data.get("object") or {}
    rows_data = data.get("list") or []
    cleaned_rows = [clean_row(row, station_meta, region, station) for row in rows_data]
    return cleaned_rows, payload


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def try_write_excel(path, rows):
    try:
        import pandas as pd
    except ModuleNotFoundError:
        log("pandas is not installed in this Python environment; skipping Excel output.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch TMD AWS Collection Data by Time Slot data for one station."
    )
    parser.add_argument("--region", default="1", help="TMD region code. Example: 1 = Central.")
    parser.add_argument("--station", default="37", help="TMD AWS station id. Example: 37 = Bangna.")
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument(
        "--interval",
        choices=["1min", "5min", "10min", "15min", "30min", "1hr", "3hr"],
        default="1hr",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--no-excel", action="store_true", help="Only write CSV and raw JSON.")
    parser.add_argument("--save-raw-json", action="store_true", help="Save raw daily JSON responses.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.end_date < args.start_date:
        raise RuntimeError("--end-date must be on or after --start-date.")

    interval_steps = {
        "1min": "100",
        "5min": "500",
        "10min": "1000",
        "15min": "1500",
        "30min": "3000",
        "1hr": "6000",
        "3hr": "18000",
    }
    interval_step = interval_steps[args.interval]
    rows_limit = 2000 if args.interval == "1min" else 500

    output_stem = (
        f"tmd_station_{args.station}_{args.interval}_"
        f"{args.start_date:%Y%m%d}_{args.end_date:%Y%m%d}"
    )
    csv_path = args.output_dir / f"{output_stem}.csv"
    xlsx_path = args.output_dir / f"{output_stem}.xlsx"
    raw_dir = args.output_dir / "raw_json" / output_stem

    opener = build_opener(HTTPCookieProcessor())
    all_rows = []
    failed_dates = []

    for target_date in date_range(args.start_date, args.end_date):
        try:
            log(f"Fetching {target_date} region={args.region} station={args.station} interval={args.interval}...")
            rows, payload = fetch_day(
                opener,
                target_date,
                args.region,
                args.station,
                interval_step,
                rows_limit,
            )
            all_rows.extend(rows)
            log(f"  received {len(rows):,} rows")
            if args.save_raw_json:
                write_json(raw_dir / f"{target_date:%Y%m%d}.json", payload)
        except Exception as exc:
            failed_dates.append({"date": str(target_date), "error": str(exc)})
            log(f"  failed: {exc}")
        time.sleep(args.sleep_seconds)

    if not all_rows:
        raise RuntimeError("No rows were fetched. Check station, region, date range, or network access.")

    write_csv(csv_path, all_rows)
    log(f"Saved CSV: {csv_path}")

    if not args.no_excel:
        if try_write_excel(xlsx_path, all_rows):
            log(f"Saved Excel: {xlsx_path}")

    if failed_dates:
        failed_path = args.output_dir / f"{output_stem}_failed_dates.json"
        write_json(failed_path, failed_dates)
        log(f"Saved failed date list: {failed_path}")


if __name__ == "__main__":
    main()
