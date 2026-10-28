# 16/10/26 (CAPSTONE REFLECTION)

**TASK AND ACTION**

Over the past two weeks, I compared the three rainfall data sources — TMD gauges, IMERG satellite observations, and Open-Meteo model rainfall — on overlapping hours. They agreed on which hours were rainy only 11–19% of the time, and model rankings flipped depending on which source was treated as the truth: each model scored around 0.90 ROC-AUC on its own labels but only around 0.72 on an independent source.

**RESULTS AND LEARNING**

Through this task, I learned that "which model is better" is undefined until you fix what counts as rain, and that a model graded on its own labels can look about 0.20 ROC-AUC better than it really is. From this point on, I always compared models against the same independent reference. This experience strengthened my scientific rigour and evaluation skills.

# 30/10/26 (CAPSTONE REFLECTION)

**TASK AND ACTION**

Since satellite rainfall observations are too delayed for real-time use, I explored TMD weather station gauges, which report within minutes, as a low-latency substitute. I engineered 16 gauge-history features by interpolating readings from 28 stations within 150 km of the Bangkok grid, and added them to the baseline model on identical training rows for a fair comparison.

**RESULTS AND LEARNING**

The gauge features improved ROC-AUC by 0.007–0.015 at the one-hour horizon — only about 30% of the zero-latency ceiling, but five to seven times better than delayed satellite features, making this the first genuinely deployable gain. The cap came from gauges measuring a point while labels describe a grid cell. I learned that an honest, usable improvement beats a larger unusable one. This strengthened my feature engineering and root-cause analysis.

# 13/11/26 (CAPSTONE REFLECTION)

**TASK AND ACTION**

I turned to the Himawari weather satellite, whose infrared images are published freely within minutes. Before downloading five years of data, I ran a two-month pilot with a pre-registered decision rule requiring the raw infrared signal to at least match the gauge benchmark. It passed, so I downloaded the full history and engineered 12 "cold cloud top" features, since very cold cloud tops indicate deep storms.

**RESULTS AND LEARNING**

The satellite features recovered roughly 65–75% of the ceiling set by perfect rainfall observations — more than double the gauges' gain — and essentially reached it at the six-hour horizon, because cold cloud tops appear before rain falls. I learned to de-risk expensive commitments with cheap pre-registered pilots, and that domain knowledge guides feature design better than blind experimentation. This strengthened my experimental design and scientific reasoning skills.

# 27/11/26 (CAPSTONE REFLECTION)

**TASK AND ACTION**

While preparing for deployment, I feared the Open-Meteo archive behind all my training features served reanalysis data — produced days after the fact — which no live system could obtain. Instead of accepting the assumption, I verified it directly by probing the API with explicit dataset parameters and comparing the returned values numerically against the forecast archive and a live-scraped sample.

**RESULTS AND LEARNING**

The fear was unfounded: the archive's default source resolves to the ECMWF IFS forecast family, matching live-obtainable data at a correlation of 1.000; only a 7–13 hour forecast lead-time gap remained. I learned to always specify data sources explicitly rather than trust API defaults, and to answer project-threatening doubts with decisive tests. This strengthened my debugging discipline and understanding of data provenance.

# 11/12/26 (CAPSTONE REFLECTION)

**TASK AND ACTION**

I scaled the project up to a nationwide model covering Thailand with 833 grid cells. This exposed two problems: the old nationwide labels correlated with observed rainfall at only r = 0.12, so I built a 15-million-row IMERG label dataset; and the Bangkok Google Earth Engine sampling method exceeded platform limits at national scale, which I fixed by restructuring the computation to reduce over all cells once per image stack.

**RESULTS AND LEARNING**

Through this task, I learned that methods which work at small scale can fail structurally — not just slowly — at larger scale, and that the fix is often restructuring the computation rather than batching harder. I also stored both cell-mean and cell-maximum rainfall to avoid repeating the earlier point-versus-area mismatch. This strengthened my skills in cloud-based geospatial processing and scalable data pipelines.

# 25/12/26 (CAPSTONE REFLECTION)

**TASK AND ACTION**

Over the past two weeks, I ported the Himawari infrared features to the nationwide 833-cell model and evaluated them against the new IMERG labels on nearly two million wet-season rows. I also re-measured the satellite geometry for the wider domain rather than inheriting Bangkok's settings, finding a wider "environment ring" worked best — convection approaching a cell matters as much as convection above it.

**RESULTS AND LEARNING**

The satellite features improved ROC-AUC by +0.040 and PR-AUC by +0.084 at the one-hour heavy-rain target — better than Bangkok, with no gauges at all — with the gain coming from precision: suppressing false alarms under clear skies. However, the 2026 gain was roughly half of 2024–2025's, which I recorded as an open problem. I learned to decompose headline metrics and document anomalies honestly. This strengthened my analytical thoroughness.

# 8/1/27 (CAPSTONE REFLECTION)

**TASK AND ACTION**

Before declaring the model deployable, I measured the worst-case cost of real-world data staleness, since live forecast inputs arrive 7–13 hours older than the fresh values used in training. Using an API that serves previous forecast runs, I fed the trained model inputs a full 24 hours stale — deliberately worse than reality — and measured the degradation on a held-out month.

**RESULTS AND LEARNING**

The forecast-only model lost up to 0.036 ROC-AUC, but the satellite-equipped model lost only 0.008–0.015 — the real-time observations absorbed most of the damage, and even with stale inputs it still beat the fresh-input baseline, so the +0.040 gain survives deployment. I learned to prove deployability with an upper bound: if the worst case is acceptable, reality must be too. This strengthened my operational experiment design skills.

# 22/1/27 (CAPSTONE REFLECTION)

**TASK AND ACTION**

I investigated why the satellite gain halved in 2026 by systematically eliminating explanations one by one: seasonality (via calendar-matched comparisons), satellite feed health, label version changes, and provisional label quality. The remaining evidence pointed to meteorology: in the weak 2026 wet season, the share of rain accompanied by cold cloud tops fell from 85% to 72%, and cloud tops over rain events warmed significantly.

**RESULTS AND LEARNING**

I concluded that 2026 produced more "warm rain" from shallow clouds, which infrared physically cannot see, so the gain is regime-dependent rather than broken — nothing needed retraining. I learned that some performance drops reflect real sensor limits, where the right response is monitoring (I proposed the cold-cloud-top share as a live health metric), not endless tweaking. This strengthened my elimination-based diagnostic skills and domain understanding.

# 5/2/27 (CAPSTONE REFLECTION)

**TASK AND ACTION**

I packaged the nationwide model for real use: training final models for all six prediction targets, bundling each with its calibrator, threshold and feature list, and writing a live pipeline that fetches current forecasts and fresh satellite imagery on demand. The key step was a parity test proving the live feature builder reproduces the training-time computation to within 2×10⁻¹³, eliminating train/serve skew.

**RESULTS AND LEARNING**

The live demonstration worked end to end: nationwide probabilities for all 833 cells in 2.4 minutes, with meteorologically sensible output. I learned that deployment is a distinct discipline from modelling — feature parity, rate-limit handling and robust packaging never appear in a research notebook — and that train/serve skew must be proven absent, not assumed. This strengthened my software engineering and MLOps skills.

# 19/2/27 (CAPSTONE REFLECTION)

**TASK AND ACTION**

I built "Rainwatch", a self-updating dashboard that re-runs the live pipeline hourly and maps all 833 cells with horizon, threshold and alert controls. I made it fully unattended with scheduled tasks, crash restarts and a staleness watchdog, and added an automatic loop that grades every mature prediction against IMERG observations once available (about 31 hours later), logging accuracy alongside the cloud-top health metric.

**RESULTS AND LEARNING**

The system has run unattended hourly cycles since day one, and the dashboard's alert counts exactly match the prediction logs. I learned that a model only becomes a system when it monitors itself: automated verification, health metrics and failure recovery matter as much as accuracy, and operational issues like observation latency never appear in research. This strengthened my full-stack development and operational engineering skills.

# 5/3/27 (CAPSTONE REFLECTION)

**TASK AND ACTION**

I addressed two robustness gaps. The deployed models were trained only on wet-season satellite data, so I backfilled 362 missing dry-season days and retrained with all-season cross-validation. I also piloted a 3.9 µm shortwave-infrared band that can theoretically detect the warm rain the main band misses, using the same pre-registered pilot methodology as before.

**RESULTS AND LEARNING**

The infrared features actively help in the dry season (+0.014–0.020 ROC-AUC), so I fitted new all-season models. The warm-rain pilot passed its screen, but day/night confounding had inflated the result; full cross-validation showed only +0.004 and nothing in the problem 2026 months, so I did not deploy it. I learned to hunt for confounders in promising results, and that not deploying can be correct. This strengthened my critical evaluation skills.

# 19/3/27 (CAPSTONE REFLECTION)

**TASK AND ACTION**

To conclude the evaluation honestly, I added two reference forecasts to the live verification loop: climatology as a no-skill floor, and persistence — the latest observed rainfall projected forward — as an oracle ceiling the live model cannot access due to data latency. Over 143 verified live runs, the model scored 0.72 ROC-AUC at one hour, above climatology's 0.61 but below persistence's 0.86.

**RESULTS AND LEARNING**

This framing quantifies the achievement precisely: genuine live skill over climatology, with the 0.14 gap to the oracle being the exact value of a real-time observation feed that no modelling can replace. Across the project, the biggest gains came from questioning labels and data provenance, not model complexity, and honest evaluation separates defensible results from optimistic ones. Overall, this capstone strengthened my machine learning, engineering, and scientific reasoning skills.
