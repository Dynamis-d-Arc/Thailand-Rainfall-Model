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

Raw hourly prediction CSVs leave the live set after 30 days (`RETENTION_DAYS`)
and move to `om_thailand_rain_v10_deploy/archive/` as `.csv.gz` (about 16 KB each),
where `verify_imerg.py` still finds them, so a label change can rescore every hour.
Until 30 Oct 2026 they were deleted instead, and 231 local-only hours from 6-27 Sep
(before the cloud job existed) are gone for good. The `predictions` branch never prunes.

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
rain, using the exact training label — the wettest IMERG pixel's hourly total
(`precipitation_pixelmax_mm`, complete hours) ≥ threshold in ANY of the h hours
starting at the issue hour, i.e. the window [t, t+h). A prediction for issue
hour t is published about 13–20 minutes after t (laptop runs from :10, the
cloud job from :15, ~2.4 min per run), so the first quarter-hour of the +1 h
window has already elapsed when it appears.

Label history: until 2026-10-27 the loop scored the window [t+1, t+1+h) on
`precipitation_max_mm`, which is 0.5·max(first half-hour) + 0.5·max(second
half-hour) over possibly different pixels and over-states the pixel total.
Those scores stay in `v10_verification_log.csv`, `v10_baseline_log.csv` and
`v10_health_labeled_live.csv` as history and are never rewritten; the corrected
rescoring of the same prediction files lives in the `*_labels_v2.csv` logs,
which the dashboard now reads. The two sets must not be compared with each
other. See `Docs/label_fix_2026-10-27.md`.

Per run it tops up `"IMERG_THAILAND_DATA"` via the Earth Engine fetcher
(idempotent), then fills any hour Earth Engine has not delivered yet straight
from NASA's archive (`IMERG_Script/fetch_imerg_thailand_data_nasa.py`, OPeNDAP
subsets of the GES DISC Late Run files, same footprint and columns; needs the
Earthdata login in `~/_netrc`, and is skipped with a log line without it), then appends per-(issue, target) scores — Brier, ROC-AUC, and the
alert confusion (POD/FAR/CSI) — to `v10_verification_log_labels_v2.csv`, and extends the
labeled V9 health metric from the per-cell `hw_cold235_env_lag1` column the
predict phase now writes into each prediction CSV
(`v10_health_labeled_live_labels_v2.csv`). The dashboard shows the rolling 14-day skill
for the active layer ("Skill vs observed rain") and plots live-labeled months
as hollow points on the IR-health chart.

The same run also scores two reference forecasts on exactly the same (issue,
target, cells) and appends them to `v10_baseline_log_labels_v2.csv`, so the model's live
numbers read as a lift rather than a bare score:

- **persistence** — the last h observed hours ending at the issue hour, the
  window [t-h, t) (IMERG pixel-max in mm; flag = "≥ threshold in that window").
  It uses observed rain *up to issue time*, which the live model never has
  (IMERG latency 4 h+), so it is the observation ceiling, not a competitor.
- **climatology** — the cell's observed label frequency for this calendar month
  and hour of day (±1 h pooled) from IMERG 2024-07..2026-07, flagged at the
  deployed model's own probability threshold. The no-skill floor. Built once by
  `build_climatology.py` into `v10_climatology_labels_v2.csv.gz`; rebuild after
  a major IMERG backfill.

The "Skill vs observed rain" panel shows a model / persistence / climatology
table for the active layer; a reference cell is green where the model beats
it and red where it does not. Baselines for issues scored before the log
existed are backfilled automatically on the next cycle.

Issues whose labels haven't reached GEE yet simply stay pending and are retried
next cycle. `--no-verify` disables the loop; `--no-predict` disables both loops.
Requires the machine's persisted Earth Engine login (same as the backfills).

## Bangkok comparison against public weather services

`compare_providers.py` logs, every hour, the rain probability that the live
model and several public forecast services give for the nine 25 km cells over
the Bangkok metro area (next 1 / 3 / 6 h), then grades everything against IMERG
as labels arrive. Open-Meteo's `precipitation_probability` needs no key; Google
Weather, AccuWeather, WeatherAPI.com, Tomorrow.io and OpenWeatherMap are used
when a key is present in `provider_keys.json` (copy `provider_keys.example.json`;
the real file is git-ignored) or the matching environment variable.

A scheduled task, **"Rainwatch Provider Compare"** (registered 2026-09-24,
hourly at :20 for five days), runs `compare_providers.py run`, which appends to
`provider_forecasts.csv`, scores whatever has labels into
`provider_scores_labels_v2.csv`, and rewrites `provider_comparison_labels_v2.md`
— a paired table (each provider vs the
model on identical issue hours and cells, both the cell-max and the
area-average label) that can be pasted into the report. `python
Dashboard/compare_providers.py report` prints it. Output lives next to the
verification logs; `PROVIDER_COMPARE_DIR` redirects it for testing.

Caveats for the write-up: provider probabilities are point forecasts of
"measurable rain", so they are graded at 0.1 mm only and the area-average label
is the fairer comparison for them; a provider's hourly value refers to the
preceding hour, so hours 1..h cover the label window [issue, issue+h) and the
1 h rows are the primary comparison; multi-hour provider rows are reported as
bounds on P(any rain in the window) — `[max]` (max of the hourly values, lower
bound) and `[1-prod]` (independence product, upper bound) — because the hourly
marginals do not determine that probability; AccuWeather is fetched for the central cell only
(free tier: 50 calls/day); and the machine's overnight sleep limits the sample
to hours the model itself ran.
