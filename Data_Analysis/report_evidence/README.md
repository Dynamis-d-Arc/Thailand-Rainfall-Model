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
