# Can TMD radar feed the 1–6 h model? First assessment, 2026-10-29

Scope: whether the archived data could become features, and how to test that. No model was
trained or changed. This doesn't assume radar helps: section 5 is the test that decides.

Numbers come from:
- `outputs/radar_png_vs_nc.csv`, `outputs/radar_cell_coverage.csv` (`assess_radar_fit.py` on the
  first 25 stamps, 02:15–08:15 UTC on 29 Oct);
- the archive's `manifest.csv`;
- the live logs `v10_verification_log_labels_v2.csv` and `v10_baseline_log_labels_v2.csv`.

## 1. What the data contains

- **Quantitative rain rate, not just pictures.** `composite.nc` holds mm/h on a 0.01° grid, from
  a 2 km PCAPPI mosaic through Z = 200 R^1.6.
- **Reflectivity is not published, but it can be recovered.** Because the Z–R is fixed,
  dBZ = 10·log10(200·R^1.6) exactly for R > 0. Below about 1 mm/h (≈ 23 dBZ) the rate is stored
  as 0, and those pixels are weak echo with no intensity.
- **The PNG frames are a lossy copy that stays up 6 h.** They match the NetCDF's rain/no-rain
  for 99% of pixels, and the intensity to about one colour bin. That is enough to backfill a
  missed slot for presence features, but too coarse for accumulation.
- Wrong-stamp control: the 08:15 PNG against the 08:00 NetCDF drops to 74% / 61%, so the
  list-to-frame mapping is right.

## 2. Spatial coverage over the 833 cells

- **All 833 cells sit inside the composite box.** Each holds 506–552 radar pixels (about 23 ×
  23), so per-cell statistics like the Himawari ones (cell max, mean, wet share, plus a 3 × 3
  "env" ring) are well sampled.
- **The real footprint is not known yet.** The file marks only echo, so an empty pixel means
  "dry" or "not covered" alike. Over the first 6 h (one wet afternoon):
  - 693 of 833 cells showed echo somewhere;
  - 469 had rain > 0 in at least one frame;
  - 175 had echo over more than half their area.
- That is mostly a picture of where it rained: the wet southern peninsula is fully lit, and the
  dry centre is sparse (`outputs/radar_coverage_map.png`).
- How to get the footprint: re-run `assess_radar_fit.py` after a few weeks. The echo union then
  converges on it, and cells that never light up after a wet spell are the blind spots. A list of
  TMD radar sites with 240 km rings would give an independent check.
- Known artefacts to mask before any feature:
  - permanent clutter near radar sites, and sea clutter off the coasts;
  - radial interference spokes (one was visible in the north-east on 29 Oct);
  - the cone of silence over each radar;
  - beam overshoot of shallow rain at long range.

## 3. Time alignment with prediction stamps

| | Live model today | TMD radar |
|---|---|---|
| Issue hour `t` | Bangkok local, latest complete hour (`phase_predict`) | stamps are UTC: t_utc = t − 7 h |
| Freshest input | Himawari IR at t−1 h, t−2 h, t−3 h (`fetch_live_ir`) | frames every 15 min, available 11–17 min after their stamp |
| Label window | `[t, t+h)` on IMERG wettest-pixel hourly totals | — |

- Predict runs at t+10 min on the laptop (`RUN_MINUTE`) and t+15–25 min in the cloud (cron `:15`,
  keeper `:25`). By then the frames stamped up to t−15 min are published, and t itself is
  borderline. That is **45 min fresher than the IR the model already uses.**
- The closest radar match to the label's "rain in the hour" is the 4 frames stamped t−60, t−45,
  t−30 and t−15 min (UTC). Each covers 15 min, so the sum of rate × 0.25 h approximates the
  accumulation over `[t−1h, t)`.
- **Training must copy live availability.** Use only frames stamped ≤ t_utc − 15 min, never
  t_utc itself, unless the manifest's `fetched_utc` shows it arrived before that hour's predict
  ran. Otherwise the experiment leaks the future, the same trap the IR lag-1 rule avoids.
- Frames align with the grid with no regridding beyond pixel-to-cell assignment (the lookup in
  `assess_radar_fit.py`). The PNGs need the Web-Mercator inverse, which is also in that script.

## 4. Why it could matter, and why that isn't proven

Over 11 Aug–28 Oct the live logs show this (mean per-issue-hour ROC, paired hours, about 800 per
target). Persistence here is observed IMERG of the previous hour carried forward.

| target | live model | IMERG persistence |
|---|---:|---:|
| h1 @ 0.1 mm | 0.838 | 0.860 |
| h1 @ 1.0 mm | 0.873 | 0.924 |
| h3 @ 0.1 mm | 0.823 | 0.756 |
| h3 @ 1.0 mm | 0.841 | 0.801 |
| h6 @ 0.1 mm | 0.806 | 0.692 |
| h6 @ 1.0 mm | 0.812 | 0.717 |

- Persistence beats the model at h1, but it can't be deployed: IMERG arrives hours late.
- Radar is a low-latency observation of the same thing, so the hypothesis is that radar
  persistence recovers part of the h1 gap. Advected radar (motion vectors between frames) might
  also help h3.
- The Bangkok latency sweep (`om_bkk_rain_v4_latency_sweep_labels_v2/v4_roc_gain_by_offset.csv`)
  measured the h1 ROC gain from observed-rain features at different delays:

  | delay | h1 @ 0.1 mm | h1 @ 1.0 mm |
  |---|---:|---:|
  | 0 h | +0.080 | +0.087 |
  | 1 h | +0.036 | +0.045 |
  | 2 h | +0.018 | +0.019 |

  The gain roughly halves for every hour of delay. A 15–30 min-old frame falls in the range where
  it should still pay off. That was a Bangkok-only panel with IMERG as the observation.
- What could kill it:
  - **The labels are IMERG.** Gauge, IMERG and Open-Meteo agree on only a small share of rainy
    hours. Radar may disagree with IMERG as much as gauges do, and then a model graded on IMERG
    gains little even if radar is closer to the truth.
  - **The archive starts in the dry season** for the north, centre and north-east. Only the
    south is reliably wet before the project ends, so positives will be few and regional.
  - **Sleep gaps** shrink the usable sample (README, Limitations).
  - The rate floor at ~1 mm/h hides light rain, which counts for the 0.1 mm targets. Echo
    presence, including zeros, may stand in for it.

## 5. Proposed test (Phase 4 proper, after enough archive)

1. **Radar-vs-IMERG agreement (no model).** For each archived hour, make per-cell radar
   1-h accumulation and echo-presence fields. Score them against the `labels_v2` labels as a
   third live baseline next to climatology and persistence.
   - Go: radar persistence clearly beats the model at h1 on the same hours and cells.
   - No-go: if it doesn't, a radar feature is unlikely to add much on IMERG-graded targets.
2. **Feature experiment, only if step 1 is a go.** Take the V13 config and add radar features:
   - per cell and its env ring: last-frame max and mean rate;
   - wet share above 1 mm/h, and echo share;
   - 1-h accumulation and 1-h trend;
   - a `radar_valid` flag for missing frames.

   Then fit that against the V13 control on a time-blocked split of the archived period, using
   the `BKK_Rain_V3` protocol, and report ROC and PR-AUC per target with a paired bootstrap.
   Advection or optical-flow features are a separate rung, after the static ones.
3. Report all six targets and the regional split, and say so if the gain sits only in the south.

## 6. What it takes to keep the archive usable

- **Coverage is the main risk, not access.** Every slot the laptop is asleep for more than 6 h is
  permanent loss. `collection_report.md` will show the real coverage after a few days.
- If coverage is too low, the choices are:
  - (a) a wake timer on the task;
  - (b) keep the machine awake;
  - (c) run the collector off the laptop. A GitHub Actions copy would need a workflow on `main`,
    which this POC was told not to touch, and it is untested whether TMD answers non-Thai IPs.
- Re-run `tmd_radar_collect.py report` and `assess_radar_fit.py` weekly. Watch for
  `layout_changed`, `stamp_mismatch` and `duplicate_content`: TMD could change the product
  without notice.
