# Report evidence

Stored outputs for figures in the Form E1 progress report that previously had none (NFR4).
Regenerate with `python Data_Analysis/report_evidence.py [section ...]`. The parity output comes from
`python Thailand_Rain_V10_deploy.py parity`. All files were written on 27 October 2026.

| Report figure | Evidence file | Stored value |
|---|---|---|
| IMERG vs ECMWF precipitation r = 0.21, 0.07–0.38 by month | `imerg_vs_om_precip_correlation.csv` | r = 0.215 over 14,973,175 joined rows; monthly 0.074–0.384 |
| Late Run detection roughly 12% lower vs the three in-grid gauges, season controlled | `imerg_late_run_pod_vs_gauges.csv` | POD 0.774 (Final) vs 0.676 (Late) on the five calendar months both eras share = −12.7%. Small sample: 490 Late Run gauge-rain hours |
| Everything after 1 October 2025 is Late Run | `imerg_run_type_periods.csv` | provisional starts 2025-10-01 07:00 local (00:00 UTC) in both IMERG tables |
| Parity within 2.3 × 10⁻¹³ | `../ML_Model_V2/trained_models/om_thailand_rain_v10_deploy/v10_parity_output.txt` | 2.27e-13 at all three test hours (66 features × 833 cells); IR features exact |
| Himawari scan available about 12 minutes after it starts | `himawari_b13_upload_latency.csv` (+ `_per_scan`) | median 12.1 min, range 11.9–12.7, 72 hourly scans 24–27 Oct 2026 (measured now, not in August) |
| Archive matches the May 2026 live-scraped pilot | `may_2026_pilot_vs_archive.csv` (+ `_per_cell`) | 19 of the 21 cells with a pilot point within 0.5 km share its model point. On 6,216 cell-hours, humidity, precipitation, pressure, cloud and wind are identical (r = 1.000); temperature and dew point are within 0.1 °C (elevation adjustment) |
| Newest IFS forecast is 7–13 hours old at issue time | `ecmwf_ifs_forecast_age.csv` (+ `ecmwf_ifs_meta_snapshot.json`) | the run became available 6.7 h after initialisation, with runs every 6 h, so 6.7–12.7 h. One snapshot, 27 Oct 2026 |
| Open-Meteo rain in 530,436 of 2,420,544 hours (21.9%) | `om_bkk_rain_hour_share.csv` | exact: 56 cells, 2021-07-20 to 2026-06-24, ≥ 0.1 mm |
| TMD lists 227 stations across six regions | `tmd_station_catalogue.csv` | exact: 26 + 17 + 59 + 42 + 71 + 12 |
| 26 central stations 37–97% complete depending on the year | `tmd_central_completeness_by_year.csv` | 2020–2025: 37% (2021) to 97% (2024). The 2026 part-year is 18% because the scrape stopped early |
| The 85 stations collected so far cover only the central and northern regions | `tmd_merge_station_regions.csv` | exact: Central 26 + Northern 59 |
| About one second per station-day | `tmd_scrape_seconds_per_station_day.csv` | median 0.85 s (p10 0.80, p90 1.04) from `loaded_at`. At that rate 26 stations × 5 years ≈ 11 h and 227 stations ≈ 4 days (the report says roughly 13 h and about five days, i.e. 1 s exactly) |

## Phase 3 (operation) figures, added 27 October 2026

Regenerate with `python Data_Analysis/phase3_evidence.py` (needs `gh` and `git fetch origin predictions`).
Used by the V9.1 report, section 3.7, Tables 22 and 23, Limitations and Risks.

| Report figure | Evidence file | Stored value |
|---|---|---|
| Live model vs persistence vs climatology, mean per issue hour | `phase3_live_skill_vs_baselines.csv` | 883–912 issue hours per target, 29 Aug–26 Oct 2026; model ROC 0.766–0.814, climatology 0.672–0.690, persistence 0.674–0.907 |
| Model skill on every verified issue hour (from 11 Aug) | `phase3_live_skill_all_verified.csv` | 936 issue hours at 1 h |
| 1 h 1.0 mm skill by month | `phase3_live_skill_h1_1mm_by_month.csv` | Aug 0.69 (27 h), Sep 0.76 (354), Oct 0.85 (524) |
| Hourly coverage: laptop only 45 %, with the cloud job 99.1 %, last 30 days 97.1 % (NFR6) | `phase3_hourly_coverage.csv`, `phase3_hourly_coverage_by_day.csv` | from prediction files on disk, in git HEAD, on the `predictions` branch, and the health and verification logs |
| GitHub Actions: 760 runs, 756 successes, 137 from GitHub's own schedule, about 8 min per run | `phase3_github_actions_runs.csv`, `phase3_github_actions_failures.csv` | `gh run list --workflow hourly-predict.yml` on 27 Oct 2026 |
| Live-labelled cold-top share of rain by month | `phase3_ir_health_live_labeled_by_month.csv` | Aug 0.78, Sep 0.79, Oct 0.73 (weighted by wet cells) |

## Corrected labels (`labels_v2`), added 27 October 2026

Window [t, t+h) on the wettest pixel's hourly total; see `Docs/label_fix_2026-10-27.md`. The
unsuffixed files above are the pre-fix evidence and are kept as they were. Used by the V9.2
(corrected labels) report copy.

| Report figure | Evidence file | Stored value |
|---|---|---|
| How much the correction changes each label | `label_change_labels_v2_summary.csv` (+ `label_change_labels_v2.csv` by month) | 4.8–11.8 % of labels change, almost all from the window shift; the pixel max removes 0.5–3.9 % of the old positives and never adds any (16,485,070 cell-hours) |
| Live model vs persistence vs climatology, corrected labels, rescored retrospectively | `phase3_live_skill_vs_baselines_labels_v2.csv` | 721–745 issue hours per target, 11 Aug–26 Oct 2026; model ROC 0.800–0.871, climatology 0.688–0.701, persistence 0.687–0.923 |
| Model skill on every rescored issue hour | `phase3_live_skill_all_verified_labels_v2.csv` | 749 issue hours scored; 231 verified hours (6–27 Sep) have no prediction file left and cannot be rescored |
| 1 h 1.0 mm skill by month, corrected labels | `phase3_live_skill_h1_1mm_by_month_labels_v2.csv` | Aug 0.74 (28 h), Sep 0.83 (123), Oct 0.89 (584) |
| Live-labelled cold-top share of rain by month, corrected labels | `phase3_ir_health_live_labeled_by_month_labels_v2.csv` | Aug 0.82, Sep 0.73, Oct 0.82 |
| Bangkok provider pilot, aligned intervals, 3 h as bounds | `../../ML_Model_V2/trained_models/om_thailand_rain_v10_deploy/provider_scores_labels_v2.csv` | 46 issue hours 24–29 Sep 2026; area-average 1 h: model 0.89 vs Open-Meteo 0.76 |
| Cross-validation on corrected labels, Tables 21 and 24 of the V9.2 copy | `../../ML_Model_V2/trained_models/om_thailand_rain_v3_labels_v2/`, `om_thailand_rain_v13_allseason_labels_v2/v13_dry_wet_report.csv` | 3,740,170 rows; infrared gain wet +0.040 to +0.074, dry +0.019 to +0.031, pooled +0.019 to +0.047 ROC-AUC |
| Old vs corrected cross-validation on identical rows | `../../ML_Model_V2/trained_models/om_thailand_rain_v13_allseason_labels_v2/cv_old_vs_new_labels_v2.csv` | om_only within 0.01 of the old run; infrared +0.011 to +0.020 pooled, of which +0.005 to +0.014 is the label change alone |
