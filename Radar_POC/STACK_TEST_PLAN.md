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

## Each month

From the worktree root, with the main checkout's `.venv`:

```powershell
powershell -File Radar_POC\gcp\sync_from_vm.ps1 -VmHost 136.117.100.120
$env:TMD_RADAR_ARCHIVE = 'C:\Users\Brandon\Documents\TMD_Radar_Archive_gcp'
$py = 'C:\Users\Brandon\Documents\Phase 1\.venv\Scripts\python.exe'
& $py Radar_POC\tmd_radar_collect.py report          # gaps, layout changes
& $py Radar_POC\score_radar_persistence.py
& $py Radar_POC\radar_stack_test.py forward 2026-12-01
```

The sync is also the only second copy of the archive, so run it even in a month with no test.
