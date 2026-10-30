# Thailand Rainfall Model — where to look

Orientation for coding agents. Read this before searching; the repo has ~50
root-level notebooks and scripts and most of them are superseded.

## Where the latest version is

- **This folder, branch `main`**, is the source of truth. Remote:
  https://github.com/Dynamis-d-Arc/Thailand-Rainfall-Model (public).
- Remote branch `predictions` is **output only**: hourly prediction CSVs
  committed by GitHub Actions. Never put code there.
- Nothing is pinned to a release. "Latest" = `git log -1 main`.

## Start here, in this order

1. `Docs/handover_2026-08-12.md` — most recent narrative of what exists, what
   was proven, what is open. Earlier handovers (08-01, 08-11) are history.
2. `Docs/capstone_experiments.md` — the experiment ladder and why each step
   exists. Quote numbers from the stored outputs it names, not from prose.
3. `Dashboard/README.md` — the live system: server, watchdog, cloud predict,
   IMERG verification loop, provider comparison.

## What is live (the only code that runs in production)

| Thing | File |
|---|---|
| Model fit / verify / predict | `Thailand_Rain_V10_deploy.py` (`fit`, `verify`, `predict`) |
| Packaged models (6 targets) | `ML_Model_V2/trained_models/om_thailand_rain_v10_deploy/deploy/th_rain_h{1,3,6}_{0.1,1.0}mm.joblib` (V13 refit on the corrected labels, live since 2026-10-28; earlier fits in `deploy_v13_shifted_window_backup/` and `deploy_v10_wet_only_backup/`, see `v10_manifest.json`) |
| Static 833-cell grid (no DB needed) | `.../om_thailand_rain_v10_deploy/deploy/th_grid_25km.csv` |
| Prediction / health / baseline logs | `.../om_thailand_rain_v10_deploy/v10_*.csv` |
| Dashboard server (localhost:8901) | `Dashboard/rainwatch_server.py`, `Dashboard/index.html` |
| Cloud hourly predict | `.github/workflows/hourly-predict.yml` |
| Verification against observed rain | `Dashboard/verify_imerg.py` |

## The experiment chain (root-level `.py` files, jupytext percent format)

- `BKK_Rain_V3.py` is the **shared protocol module** (CV machinery, feature
  list, targets, thresholds). Every later script imports from it. Change it
  with care: V4–V13 and V10 deploy all depend on it.
- Bangkok ladder: `BKK_Rain_V3..V7_*.py`. Thailand ladder:
  `Thailand_Rain_V3.py` (nationwide panel, IMERG labels), `V7` (Himawari IR,
  the main gain), `V8` (forecast-staleness gate), `V9` (2026 warm-rain decay),
  `V10` (deploy), `V12` (multiband pilot, not deployed), `V13` (dry season),
  `V14` (forward-in-time replay of the deploy recipe: train 2024-07..2025-06, calibrate
  2025-07..12, test 2026; checks the blocked CV and the random calibration split),
  `V15` (no fits: onset/continue/stop slices, day-block bootstrap CIs, reliability figure).
- Each experiment writes to `ML_Model_V2/trained_models/<experiment_name>/`.
  Search there for stored metrics, fold tables and manifests.
- `ML_Model_V2/_th_cache_labels_v2/`, `_v3_cache_labels_v2/`, `cache/` are regenerable fold
  caches, gitignored and multi-GB. Do not read them to answer questions.

## Data pipeline (Postgres, local)

- Connection: `PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD` env vars, defaults
  to localhost / db `postgres` (see `DB_CONFIG` in any `OM_Script/*.py`).
- Fetch + precompute: `OM_Script/` (Open-Meteo, always pass `models=`
  explicitly), `IMERG_Script/` (NASA IMERG via Google Earth Engine, see its
  README; `fetch_imerg_thailand_data_nasa.py` fills hours Earth Engine lags on,
  straight from GES DISC), `DB_Script/` (grids, boundaries, lag features, TMD gauges).
- Main tables: `"OM_Thailand_Data"`, `"OM_Thailand_Forecast_Data"`,
  `"IMERG_THAILAND_DATA"`, `thailand_grid_25km`, and the Bangkok-era
  `"OM_BKK_DATA"`, `"OM_BKK_DATA_PRECOMPUTE"`, `"IMERG_BKK_DATA"`,
  `"BKK_TMD_WEATHER_DATA"`.
- Analysis of label disagreement (gauge vs IMERG vs Open-Meteo): `Data_Analysis/`.

## Superseded — search here only for history

- Every root-level `*.ipynb`, `BKK_3_Class_*`, `BKK_Rain_V0..V2*`,
  `TMD_BKK_*`, `OM_BKK_V1`: July 2026 exploration, replaced by the V3 protocol.
- `ML_Models/`: the first Bangkok model (pre-V2). `outputs/`: old maps.
- `ML_Model_V2/train_bkk_rain_next_*h.py` and `build_bkk_rain_v2_deploy.py`:
  Bangkok V2, replaced by V3+.

## Conventions that bite

- Dashboard code must use `bkk_now()` for time, never the machine clock.
- Labels since 2026-10-27 are `labels_v2`: window `[t, t+h)` on
  `precipitation_pixelmax_mm` (wettest pixel's hourly total). Experiment dirs,
  fold cache and live logs carry a `_labels_v2` suffix; unsuffixed ones are the
  pre-fix history (window `[t+1, t+1+h)` on `precipitation_max_mm`) and are
  never rewritten. See `Docs/label_fix_2026-10-27.md`.
- Prediction stamps are Bangkok local hours (`v10_predictions_YYYYMMDD_HHMM.csv`).
  After 30 days they move to `om_thailand_rain_v10_deploy/archive/*.csv.gz`; never delete them.
- The live build (`model_build` 2026-10-28 02:59:33) is **frozen** for prospective evaluation
  since 2026-10-30: do not refit, replace or re-threshold `deploy/*.joblib` without the user's
  OK. New candidates go in their own experiment dir. The freeze covers `th_rain_*`;
  `deploy/shadow_om_only_*.joblib` are no-IR reference models scored beside it
  (`om_only` baseline), never the live forecast.
- Any claim in a report must trace to a stored output, code, or the DB.
- `*.joblib`, `*.npy`, `*.npz`, `*.log` are gitignored except the `deploy/`
  artifacts the cloud job needs.
