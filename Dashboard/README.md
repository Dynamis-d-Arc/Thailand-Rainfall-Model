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
  hasn't succeeded in >26 h, or status.json carries a standing error.
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
- `--no-predict` serves existing snapshots without touching the live APIs
  (useful offline or when Open-Meteo is rate-limiting); `--port` changes the port.

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

Issues whose labels haven't reached GEE yet simply stay pending and are retried
next cycle. `--no-verify` disables the loop; `--no-predict` disables both loops.
Requires the machine's persisted Earth Engine login (same as the backfills).
