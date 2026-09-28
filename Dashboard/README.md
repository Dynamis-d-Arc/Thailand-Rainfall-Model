# Rainwatch dashboard

Self-updating map of the V10 nationwide rain predictions.

**It runs itself.** Two Windows scheduled tasks (registered 2026-08-30) keep the
system alive with no manual steps:

- **"Rainwatch Server"** — starts `rainwatch_server.py` (windowless, via
  `pythonw`) at every logon and restarts it up to 10 times if it crashes.
  Output goes to `Dashboard/data/server.log` (rotated at ~5 MB). Postgres is an
  Automatic service, so the whole stack survives reboots.
- **"Rainwatch Watchdog"** — `watchdog.py` hourly: raises a Windows toast
  (via `notify.ps1`) when the newest prediction is >3 h old, verification
  hasn't succeeded in >26 h, the cloud-prediction pull hasn't succeeded in
  >3 h, or status.json carries a standing error.
  Debounced to one toast per problem per 6 h. `--test` sends a test toast.

Manage both in Task Scheduler; `start_rainwatch.bat` remains for manual runs
on a machine without the tasks (it will fail to bind port 8901 if the task's
server is already running). Dashboard: http://localhost:8901.

Raw hourly prediction CSVs are pruned after 30 days (`RETENTION_DAYS`); the
verification and health logs keep the distilled record indefinitely.

What the server does:

- Serves `index.html` — a Leaflet map (CARTO/OSM basemap) of all 833 grid cells,
  with horizon (+1/+3/+6 h) and threshold (0.1/1.0 mm) controls, alert-cell
  outlines from the packaged F1 thresholds, per-cell tooltips, and an issue-time
  selector covering every prediction on disk.
- Once per hour (at minute ≥ 10, when no CSV exists for the current hour) it runs
  `Thailand_Rain_V10_deploy.py predict` and converts the result into
  `Dashboard/data/` (gitignored, regenerable). The page polls every 60 s and
  picks up new predictions automatically; it warns when the newest data is stale.
- Every 10 minutes it pulls any prediction CSVs the GitHub Actions job has
  published to the `predictions` branch (see below) and converts them like
  local ones. This is what fills in the hours the machine spent asleep.
- `--no-predict` serves existing snapshots without touching the live APIs
  (useful offline or when Open-Meteo is rate-limiting); `--no-sync` disables
  the pull; `--port` changes the port. `--no-predict` with sync left on turns
  the dashboard into a pure viewer of cloud-produced predictions.

## Cloud predictions

This machine sleeps (Windows Modern Standby), which used to cost 10-16 of every
24 hourly snapshots - the predict itself was fine, the laptop simply was not
awake. `.github/workflows/hourly-predict.yml` therefore runs the same
`Thailand_Rain_V10_deploy.py predict` on a GitHub runner every hour and commits
the CSV to the `predictions` branch; the server's sync loop pulls those in.

The cloud job needs no database and no secrets: Open-Meteo is public, the NOAA
Himawari buckets are read unsigned, and the static 833-cell grid comes from the
committed `deploy/th_grid_25km.csv` instead of Postgres. Output is bit-identical
to a local run - verified against the same issue hour, max probability
difference 0.00000 across all six targets.

The pull is read-only with respect to the working tree: it updates a
remote-tracking ref and reads blobs out of it, never checking anything out or
switching branch. Snapshots older than `RETENTION_DAYS` are skipped rather than
pulled, so sync and pruning do not fight. When both the laptop and the cloud
produce the same hour, whichever lands first wins and the other is ignored.

## V9 health metric

The page's "IR health" panel tracks p(cold-top ≤235 K near cell | rain) — the V9
regime guard: infrared cannot see warm shallow rain, so when this falls below the
2024–25 band the Himawari gain shrinks (it was 0.85 in 2024–25, ~0.70 by 2026-07).
Two data sources feed it:

- **Labeled (canonical):** `v9_rain_type_by_month.csv` from the V9 analysis —
  monthly, IMERG-labeled. It only extends when IMERG is backfilled and V9 rerun.
- **Live proxy (hourly):** each predict run now appends run-level IR aggregates
  (cold-top share among all cells and among alert cells, min-Tb percentiles) to
  `v10_health_log.csv` next to the prediction CSVs — unlabeled but immediate.

The server merges both into `Dashboard/data/health.json` at every conversion.

## IMERG verification loop

`verify_imerg.py` closes the observation loop: every 6 h the server grades all
mature predictions (issue + 6 h + 7 h IMERG-latency margin) against observed
rain, using the exact training label — IMERG cell-max hourly rain
(`precipitation_max_mm`, complete hours) ≥ threshold in ANY of the next h hours.

Per run it tops up `"IMERG_THAILAND_DATA"` via the Earth Engine fetcher
(idempotent), then appends per-(issue, target) scores — Brier, ROC-AUC, and the
alert confusion (POD/FAR/CSI) — to `v10_verification_log.csv`, and extends the
labeled V9 health metric from the per-cell `hw_cold235_env_lag1` column the
predict phase now writes into each prediction CSV
(`v10_health_labeled_live.csv`). The dashboard shows the rolling 14-day skill
for the active layer ("Skill vs observed rain") and plots live-labeled months
as hollow points on the IR-health chart.

The same run also scores two reference forecasts on exactly the same (issue,
target, cells) and appends them to `v10_baseline_log.csv`, so the model's live
numbers read as a lift rather than a bare score:

- **persistence** — the last h observed hours ending at issue time (IMERG
  cell-max in mm; flag = "≥ threshold now"). It uses observed rain *at issue
  time*, which the live model never has (IMERG latency 4 h+), so it is the
  observation ceiling, not a competitor. It beats the model at +1 h and loses by
  +6 h.
- **climatology** — the cell's observed label frequency for this calendar month
  and hour of day (±1 h pooled) from IMERG 2024-07..2026-07, flagged at the
  deployed model's own probability threshold. The no-skill floor. Built once by
  `build_climatology.py` into `v10_climatology.csv.gz`; rebuild after a major
  IMERG backfill.

The "Skill vs observed rain" panel shows a model / persistence / climatology
table for the active layer; a reference cell is green where the model beats
it and red where it does not. Baselines for issues scored before the log
existed are backfilled automatically on the next cycle.

Issues whose labels haven't reached GEE yet simply stay pending and are retried
next cycle. `--no-verify` disables the loop; `--no-predict` disables both loops.
Requires the machine's persisted Earth Engine login (same as the backfills).
