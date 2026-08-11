# IMERG Bangkok ingestion

`fetch_imerg_bkk_data.py` retrieves spatially subsetted NASA IMERG Final Run V07
half-hourly precipitation through Earthdata Harmony, maps the nearest IMERG 0.1-degree
cell to each point in `Bangkok_Grid_9km`, aggregates the two half-hour rates to hourly
records, and upserts `IMERG_BKK_DATA`.

`fetch_imerg_bkk_data_gee.py` does the same database load through Google Earth Engine.
Use this path when an Earth Engine education/student account is available; it samples
only the Bangkok grid points and does not download the global IMERG files or local
NetCDF subsets.

The default date range is the overlap with `OM_BKK_DATA`, ending 120 days before the
current date to allow for the approximate 3.5-month IMERG Final Run latency.

## Setup

1. Create a free NASA Earthdata Login account.
2. Install dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r .\IMERG_Script\requirements.txt
   ```

3. Provide authentication through `EARTHDATA_TOKEN`, `EARTHDATA_USERNAME` plus
   `EARTHDATA_PASSWORD`, or a Windows `_netrc` file. Alternatively, use
   `--interactive-login`.

For the Earth Engine route, register a Google Cloud project for Earth Engine access,
then authenticate once from this project environment using a tiny one-day request:

```powershell
.\.venv\Scripts\python.exe .\IMERG_Script\fetch_imerg_bkk_data_gee.py `
  --project-id gen-lang-client-0449940358 `
  --start-date 2025-01-01 --end-date 2025-01-01 --authenticate
```

## Commands

Create or verify the table without contacting NASA:

```powershell
.\.venv\Scripts\python.exe .\IMERG_Script\fetch_imerg_bkk_data.py --schema-only
```

Preview the resolved range and requests:

```powershell
.\.venv\Scripts\python.exe .\IMERG_Script\fetch_imerg_bkk_data.py --dry-run
```

Download the complete default training overlap:

```powershell
.\.venv\Scripts\python.exe .\IMERG_Script\fetch_imerg_bkk_data.py
```

Start with a one-day validation load:

```powershell
.\.venv\Scripts\python.exe .\IMERG_Script\fetch_imerg_bkk_data.py `
  --start-date 2025-01-01 --end-date 2025-01-01 --interactive-login
```

Earth Engine one-day validation load:

```powershell
.\.venv\Scripts\python.exe .\IMERG_Script\fetch_imerg_bkk_data_gee.py `
  --project-id gen-lang-client-0449940358 `
  --start-date 2025-01-01 --end-date 2025-01-01
```

Earth Engine historical load for the training window:

```powershell
.\.venv\Scripts\python.exe .\IMERG_Script\fetch_imerg_bkk_data_gee.py `
  --project-id gen-lang-client-0449940358 `
  --start-date 2021-07-20 --end-date 2026-07-19 --chunk-days 1
```

Completed chunks are skipped automatically. Use `--replace-range` only when the selected
date range should be deleted and rebuilt.

Temporary subset files and repeated source filenames are not retained by default. Use
`--keep-downloads` or `--store-source-granules` only when detailed download provenance is
needed, because both options increase storage use.

For Earth Engine, `run_type` is populated from the IMERG image `status` metadata
(`permanent`, `provisional`, `mixed`, or `unknown`). For final-only model training,
filter to `run_type = 'permanent'`, or load with `--status permanent`.

For model training, filter `is_complete_hour = true`. `precipitation_mm` is the hourly
accumulation, while `precipitation_rate_max_mm_h` retains the stronger of the two
half-hour rates and is especially useful when studying Moderate and Heavy events.
