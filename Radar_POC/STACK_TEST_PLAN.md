# Radar stack test: frozen plan, 2026-11-10

Question: does TMD radar add forecast skill on top of the live model? This plan fixes how that
is tested from now until the project's data cutoff, so the later results are out of sample.

## What is frozen

`radar_stack_test.py` as committed with this file: the 19 radar features, the control arm (the
live model's six probabilities, latitude, longitude, hour), the model settings, the 6 h embargo,
the slices and the metrics. No tuning after this commit. A bug fix is allowed, and must be
listed here with its date and its effect on the earlier numbers.

The live model is not part of this. It stays frozen as it is (`model_build` 2026-10-28 02:59:33)
and the test only reads its prediction CSVs.

## The pilot (exploratory, not evidence)

The recipe was chosen on 29 Oct to 10 Nov 2026, so these numbers are in sample for the design:
- `outputs/radar_persistence_scores.csv`: radar persistence alone is 0.09 to 0.13 ROC below the
  live model on every target.
- `outputs/radar_stack_summary.csv` (leave one day out): radar minus control, mean per-hour ROC,
  +0.034 at h1 @ 0.1 mm and +0.040 at h1 @ 1.0 mm.
- `outputs/radar_stack_summary_probs_only.csv`: the same test before latitude, longitude and
  hour were added to both arms.
- `outputs/radar_stack_summary_forward_20261106.csv`: train to 5 Nov, test 6 to 10 Nov.

## The monthly forward test (the evidence)

Each test month is scored once, by a model trained only on earlier issue hours:

| Run after | Command | Trains on | Tests |
|---|---|---|---|
| labels for 31 Dec are in (about 3 Jan) | `forward 2026-12-01` | 29 Oct to 30 Nov | December |
| about 3 Feb | `forward 2027-01-01` | 29 Oct to 31 Dec | January |
| about 3 Mar | `forward 2027-02-01` | 29 Oct to 31 Jan | February |

Data cutoff: 28 Feb 2027. Nothing after it goes in the report.

Report, whatever the sign:
- all six targets, slices `all` and `south`;
- radar minus control as mean per-hour ROC with its 95% CI over day means, and pooled PR-AUC;
- the number of test hours and the base rate, because January and February are dry outside the
  south and may leave few rainy hours.

What it can support: radar as a second-stage correction to the live model in November to
February. What it cannot: the May to October wet season, or radar inside the main model. Both
need an archive that covers a wet season.

## How it runs

The Windows task `TMD Radar Weekly Sync` runs `weekly_sync_and_test.py` every Wednesday at
12:00, or at the next start-up if the laptop was off. It is registered once, by hand, with
`powershell -File Radar_POC\register_weekly_task.ps1`. Each run:

1. syncs the cloud archive to `Documents\TMD_Radar_Archive_gcp` (the only second copy);
2. runs every forward test in the table above that has no summary file yet. A test whose month
   is not complete (labels or radar stop short of its end) refuses to run and is tried again
   the next week;
3. after a test is scored, refreshes the collection report and the persistence scores.

The log is `Documents\TMD_Radar_Archive_gcp\weekly_run.log`. To run it by hand, from the
worktree root:

```powershell
& 'C:\Users\Brandon\Documents\Phase 1\.venv\Scripts\python.exe' Radar_POC\weekly_sync_and_test.py
```

It needs the laptop on with Postgres running, the Rainwatch server having filled the IMERG
labels and pulled the month's prediction files, and the worktree left in place.

## Changes since the freeze

- 2026-11-10: forward mode now refuses to score a period whose labels or radar are incomplete.
  No effect on any number. The 6 to 10 Nov pilot file was written before this, with labels up
  to 9 Nov 17:00, so re-running that command later would score more hours than the stored file.
