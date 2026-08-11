# Hourly rainfall nowcasting for Bangkok — experiment record

Eight experiments, in the order they were run. Each exists because the previous one produced
a result that demanded it. This is the narrative spine for a methodology chapter.

Compiled 23 July 2026 · PostgreSQL · LightGBM · HistGradientBoosting
Published copy: https://claude.ai/code/artifact/37dd89c2-e9cb-45c1-9597-934188c47416

---

## Overview

Three independent rainfall records were available: an ECMWF forecast (Open-Meteo), a satellite
observation (IMERG), and a station gauge network (TMD). The first four experiments built models
on each source and reported strong headline metrics. The middle experiments established that
those headlines were not comparable to one another, and that one of them was not measuring
rainfall skill at all. The final experiment corrected the label source and produced a controlled
demonstration that the correction improved out-of-sample performance at every target.

**The through-line:** the headline metric of a rainfall model is determined as much by which
record grades it as by the model itself.

---

## Phase one — baseline models

### 01. BKK_Rain_V1 — Open-Meteo features, Open-Meteo labels

- **Why:** establish a Bangkok baseline on the densest available grid — 56 cells at 9 km, five years.
- **Setup:** 66 features (36 baseline + 30 neighbourhood), targets `rain_any_next_{1,2,3,6}h` at
  0.1 mm, chronological 70/15/15 split, LightGBM at 1–2 h and HistGradientBoosting at 3–6 h.
- **Result:** test ROC-AUC 0.925 at 1 h, F1 0.681. Best-looking model in the project.
- **Artifacts:** `ML_Model_V2/trained_models/om_bkk_rain_any_v1/`

### 02. Thailand_Rain_V1 — same design, national grid

- **Why:** test whether the Bangkok approach scales to 833 cells covering Thailand.
- **Result:** test ROC-AUC 0.914 at 1 h, F1 0.787 — apparently the strongest F1 in the project.
- **Finding:** both the F1 and the base rate are inflated by the split landing on the monsoon.
  Its test window rains 37.4% of hours against a validation window at 7.9% — a 4.7x difference.
  Only 11 months of data exist, so no chronological split can be seasonally representative.
- **Artifacts:** `ML_Model_V2/trained_models/om_thailand_rain_any_final_4_neighbor_models/`

### 03. TMD_BKK_V1 — station observations, gauge labels

- **Why:** Open-Meteo is model output, not measurement. Train on physically measured data
  instead: 26 TMD stations, 902,356 rows, 6.5 years.
- **Setup:** 63 features. Three data decisions worth reporting:
  1. the panel is reindexed onto a gap-free hourly grid per station before any lag (coverage is
     82%, and the holes are long outages, so a row-order `shift` would splice lags across them);
  2. 62 negative-precipitation sentinel rows are nulled rather than clipped to zero, which would
     fabricate dry labels;
  3. features that are ~80% null are left as NaN because the missingness is itself the signal.
- **Result:** test ROC-AUC 0.899 at 1 h, F1 0.562.
- **Artifacts:** `ML_Model_V2/trained_models/tmd_bkk_rain_any_final_4_neighbor_models/`

### 04. OM_BKK_V1 — three-class intensity

- **Why:** rain/no-rain is less useful than knowing how hard it will rain. Reframe as
  Light / Moderate / Heavy.
- **Result:** macro-F1 0.405, which conceals near-total failure on the rare classes.

| Class (1 h) | Support | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| Light rain | 55,920 | 0.962 | 0.843 | 0.979 |
| Moderate rain | 3,516 | 0.167 | 0.427 | 0.165 |
| Heavy rain | 387 | **0.047** | 0.220 | **0.043** |

- **Finding:** heavy-rain precision of 0.047 means roughly 20 false alarms per correct warning.
  This line of work was not pursued further — see the intensity limitation below.

---

## Phase two — are the headline numbers real?

### 05. Thailand_Rain_Seasonal_CV — month-blocked cross-validation

- **Why:** experiment 02 showed the single chronological split hands validation and test entirely
  different seasons. Replace it with 4 folds x 3 months, 24-hour purge, full out-of-fold coverage.
- **Result:** ranking skill held (CV mean ROC 0.938 vs headline 0.914) but F1 did not: mean 0.714
  against a headline 0.787, worst month 0.533. The F1-optimal threshold nearly doubled across
  seasons, 0.45 to 0.85.
- **Left open, and stated explicitly:** with only 12 months of data, every held-out block is *also*
  a season the model never trained on. Degradation confounds "harder season" with "unfamiliar
  season."
- **Artifacts:** `ML_Model_V2/trained_models/om_thailand_rain_any_seasonal_cv/`

### 06. BKK_RAIN_CV1 & TMD_BKK_CV1 — separating the two effects

- **Why:** resolve the confound left by 05. The Bangkok datasets span 5 and 6.5 years, so a season
  can be held out in one year while remaining in training in others.
- **Setup:** Part A holds out one calendar *year* (season seen elsewhere). Part B holds out one
  calendar *quarter* across all years (season never seen). Part C is A−B on identical rows. A
  volume-matched control rules out the training-set size difference.

| Cost of an unseen season (ROC-AUC) | 1 h | 3 h | 6 h |
|---|---|---|---|
| BKK_RAIN_CV1 | +0.0017 | +0.0026 | +0.0037 |
| TMD_BKK_CV1 | +0.0016 | +0.0006 | −0.0012 |

- **Finding:** holding out an entire season costs essentially nothing — replicated across two
  datasets with different sources, labels and features. Thailand's seasonal degradation was
  intrinsic month difficulty, not unfamiliarity. The volume control attributes 0.0002 ROC-AUC to
  training-set size.
- **Second finding:** TMD_BKK_V1's Brier skill is **−1.03** at 1 h and −0.83 at 2 h under honest
  CV — worse than always predicting the base rate. Confirmed as intrinsic to `scale_pos_weight`,
  not an artefact of the split. This motivated calibration in experiment 08.
- **Artifacts:** `ML_Model_V2/trained_models/om_bkk_rain_any_seasonal_cv/` and
  `.../tmd_bkk_rain_any_seasonal_cv/`

---

## Phase three — correcting the label source

### 07. Cross-source label agreement

- **Why:** experiments 01–03 report metrics against three different records. Establish whether
  those records agree before treating the metrics as comparable.
- **Result:** they agree poorly. On rainy hours the three overlap on only 11–19% of cases.
  Open-Meteo and IMERG correlate at 0.18 and peak about five hours apart. Model rankings flip
  depending on which record grades them.

> **ARTIFACTS DELETED.** This analysis was reverted and its output files no longer exist on disk.
> The findings are sound but **not currently reproducible**. Re-run before citing any specific
> agreement statistic in the report — the scripts are deterministic and take about two minutes.

### 08. BKK_Rain_V2 — same features, observed labels

- **Why:** V1 trains Open-Meteo `precipitation` -> Open-Meteo rain labels. The label is ECMWF
  forecast output, and it also feeds the model's own `precipitation*` features — so the model is
  substantially graded on reproducing a forecast derived from its own inputs. IMERG is an
  independent satellite observation.
- **Changed:** labels moved to IMERG; thresholds 0.1 mm *and* 1.0 mm both built; isotonic
  calibration fitted inside each fold; leave-one-year-out CV replacing the single split; operating
  thresholds selected on other years only. **Features and hyperparameters unchanged**, so the label
  is the sole variable.

| Graded on IMERG, V1's own test window | V1 ROC | V2 ROC | Δ | V1 lift | V2 lift |
|---|---|---|---|---|---|
| 1 h @ 0.1 mm | 0.760 | **0.828** | +0.068 | 2.67 | 3.32 |
| 1 h @ 1.0 mm | 0.813 | **0.862** | +0.049 | 3.92 | 5.47 |
| 3 h @ 0.1 mm | 0.769 | **0.826** | +0.057 | 2.35 | 2.68 |
| 3 h @ 1.0 mm | 0.807 | **0.849** | +0.042 | 3.60 | 4.61 |
| 6 h @ 0.1 mm | 0.767 | **0.830** | +0.062 | 2.06 | 2.29 |
| 6 h @ 1.0 mm | 0.793 | **0.833** | +0.041 | 2.96 | 3.63 |

- **Finding:** V2 wins at all six targets on identical out-of-sample rows (367,808 rows, both
  models strictly out-of-sample). V1 scores 0.76–0.81 here against its own reported 0.925 — that
  gap is the cost of grading a model on a forecast partly derived from its own inputs.
- **Calibration:** isotonic rescued exactly the models predicted to need it. `h1_1.0mm` Brier skill
  moved from **−1.71** to **+0.12**, and `h1_0.1mm` from −0.07 to +0.26, while the
  HistGradientBoosting targets were untouched — confirming `scale_pos_weight` as the cause. ROC
  cost was nil (0.8721 -> 0.8716).
- **Artifacts:** `ML_Model_V2/trained_models/om_bkk_rain_v2_imerg/`

---

## Final V2 performance

Out-of-fold across 2,453,360 rows, calibrated probabilities, leave-one-year-out thresholds.

| Target | Base rate | ROC | PR-AUC | Lift | Precision | Recall | F1 | Brier skill |
|---|---|---|---|---|---|---|---|---|
| h1 @ 0.1 mm | 0.178 | 0.850 | 0.551 | 3.10 | 0.450 | 0.687 | 0.544 | +0.255 |
| h3 @ 0.1 mm | 0.275 | 0.852 | 0.675 | 2.46 | 0.550 | 0.782 | 0.646 | +0.319 |
| h6 @ 0.1 mm | 0.373 | 0.862 | 0.774 | 2.07 | 0.654 | 0.819 | **0.727** | +0.375 |
| h1 @ 1.0 mm | 0.048 | **0.872** | 0.271 | **5.66** | 0.270 | 0.432 | 0.333 | +0.116 |
| h3 @ 1.0 mm | 0.081 | 0.859 | 0.344 | 4.26 | 0.320 | 0.517 | 0.396 | +0.158 |
| h6 @ 1.0 mm | 0.124 | 0.844 | 0.415 | 3.35 | 0.354 | 0.640 | 0.456 | +0.186 |

Threshold stability: leave-one-year-out selection picked the identical threshold in all six years
for five of six targets.

---

## Methodological note — which metrics may be compared across targets

The single most defensible methodological point in this project, and the one most likely to be
challenged. PR-AUC's floor *is* the base rate, and precision, recall and F1 all move with class
balance. Only base-rate-independent measures can be compared between targets with different rain
frequencies.

| Metric | Comparable across base rates? |
|---|---|
| ROC-AUC | Yes |
| PR-AUC / base rate (lift) | Yes |
| Brier *skill* | Yes |
| PR-AUC, F1, precision, recall, raw Brier | **No** |

**Supporting simulation.** A synthetic model with separability fixed by construction was scored at
both V2 base rates. ROC held flat (0.863 -> 0.865, confirming skill was unchanged) while F1 fell
from 0.588 to 0.382 — a 35% drop caused by rarity alone. V2's observed drop across the same base
rates is 39%, with recall landing on 0.432 in both cases.

This licenses the report's key claim: **the lower F1 at 1.0 mm reflects a 4.8% base rate, not
reduced model skill.** ROC rises (0.850 -> 0.872) and lift nearly doubles (3.10 -> 5.66) over the
same change.

---

## Limitations — state these before you are asked

- **Spatial mismatch.** IMERG measures area-averaged rain over an ~11 km cell; a gauge measures a
  point. These are different physical quantities. V2 predicts "rain somewhere in the cell", not
  "rain at this location."
- **No intensity capability.** For gauge-hours above 10 mm, IMERG reports at least half the gauge
  amount only 2.9% of the time. This model can distinguish rain from no-rain; it can never become
  a heavy-rain warning system. Experiment 04 is the empirical confirmation.
- **Provisional data.** Everything after 2025-10-01 is IMERG Late Run, with roughly 12% lower
  detection when controlled for season. That covers the final CV fold and all future data.
- **Features remain forecast output.** V2 corrects the label but its inputs are still ECMWF. The
  model inherits any systematic bias in that forecast; it has simply stopped being graded on
  reproducing it.
- **Single region.** All V2 conclusions are Bangkok-specific, 56 grid cells. The national grid was
  not re-run with IMERG labels.

---

## Not yet done

- **No deployable model exists.** The V2 cross-validation fits were used for scoring and discarded
  — the output directory contains metrics but zero `.joblib` files. A final fit on all rows, with
  its calibrator and threshold, is the immediate next step.
- **Provisional-period validation.** Whether V2's skill holds on Late Run data has not been tested
  separately.
- **TMD calibration.** The isotonic recipe from experiment 08 transfers directly to TMD_BKK_V1,
  whose probabilities remain unusable as probabilities.
- **Version control.** The project is not under source control; the `.git` directory is empty.
