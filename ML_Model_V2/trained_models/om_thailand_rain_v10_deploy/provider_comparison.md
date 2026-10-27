# Bangkok: live model vs public weather services

Generated 2026-09-29 00:20. Nine 25 km cells over the Bangkok metro area, graded against IMERG. Each provider row is paired with the model on exactly the same issue hours and cells; 'model' columns are the model on that provider's sample. Flags use p >= 0.5 for every forecaster. Providers are graded at 0.1 mm only (their probability means 'measurable rain at the point').

## Label: rain anywhere in the cell (project label, IMERG cell-max)

### Threshold 0.1 mm

| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | climatology | 30 | 270 | 0.82 | 0.56 | 0.172 | 0.94 | 0.17 | 0.79 | 0.85 | 0.129 |
| 1 h | model | 30 | 270 | 0.82 | 0.85 | 0.129 | 1.00 | 0.18 | 0.82 | 0.85 | 0.129 |
| 1 h | open_meteo | 30 | 270 | 0.82 | 0.73 | 0.126 | 1.00 | 0.16 | 0.84 | 0.85 | 0.129 |
| 1 h | persistence | 30 | 270 | 0.82 | 0.89 | 0.066 | 0.97 | 0.07 | 0.91 | 0.85 | 0.129 |
| 3 h | climatology | 30 | 270 | 0.94 | 0.66 | 0.067 | 1.00 | 0.06 | 0.94 | 0.82 | 0.053 |
| 3 h | model | 30 | 270 | 0.94 | 0.82 | 0.053 | 1.00 | 0.06 | 0.94 | 0.82 | 0.053 |
| 3 h | open_meteo | 30 | 270 | 0.94 | 0.67 | 0.063 | 1.00 | 0.06 | 0.94 | 0.82 | 0.053 |
| 3 h | persistence | 30 | 270 | 0.94 | 0.70 | 0.096 | 0.91 | 0.04 | 0.88 | 0.82 | 0.053 |
| 6 h | climatology | 30 | 270 | 1.00 | — | 0.006 | 1.00 | 0.00 | 1.00 | — | 0.001 |
| 6 h | model | 30 | 270 | 1.00 | — | 0.001 | 1.00 | 0.00 | 1.00 | — | 0.001 |
| 6 h | open_meteo | 30 | 270 | 1.00 | — | 0.000 | 1.00 | 0.00 | 1.00 | — | 0.001 |
| 6 h | persistence | 30 | 270 | 1.00 | — | 0.040 | 0.95 | 0.00 | 0.95 | — | 0.001 |

### Threshold 1.0 mm

| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | climatology | 30 | 270 | 0.61 | 0.57 | 0.345 | 0.02 | 0.40 | 0.02 | 0.87 | 0.159 |
| 1 h | model | 30 | 270 | 0.61 | 0.87 | 0.159 | 0.87 | 0.23 | 0.69 | 0.87 | 0.159 |
| 1 h | persistence | 30 | 270 | 0.61 | 0.93 | 0.093 | 0.95 | 0.13 | 0.83 | 0.87 | 0.159 |
| 3 h | climatology | 30 | 270 | 0.71 | 0.59 | 0.284 | 0.32 | 0.23 | 0.29 | 0.88 | 0.138 |
| 3 h | model | 30 | 270 | 0.71 | 0.88 | 0.138 | 0.99 | 0.18 | 0.82 | 0.88 | 0.138 |
| 3 h | persistence | 30 | 270 | 0.71 | 0.81 | 0.165 | 0.88 | 0.14 | 0.77 | 0.88 | 0.138 |
| 6 h | climatology | 30 | 270 | 0.82 | 0.54 | 0.214 | 0.76 | 0.18 | 0.65 | 0.79 | 0.126 |
| 6 h | model | 30 | 270 | 0.82 | 0.79 | 0.126 | 0.96 | 0.17 | 0.81 | 0.79 | 0.126 |
| 6 h | persistence | 30 | 270 | 0.82 | 0.52 | 0.248 | 0.86 | 0.18 | 0.72 | 0.79 | 0.126 |

## Label: area-average rain (IMERG cell-mean, closer to a point forecast)

### Threshold 0.1 mm

| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | climatology | 30 | 270 | 0.73 | 0.49 | 0.216 | 0.93 | 0.26 | 0.70 | 0.87 | 0.195 |
| 1 h | model | 30 | 270 | 0.73 | 0.87 | 0.195 | 1.00 | 0.27 | 0.73 | 0.87 | 0.195 |
| 1 h | open_meteo | 30 | 270 | 0.73 | 0.63 | 0.214 | 1.00 | 0.25 | 0.75 | 0.87 | 0.195 |
| 1 h | persistence | 30 | 270 | 0.73 | 0.81 | 0.129 | 0.99 | 0.16 | 0.83 | 0.87 | 0.195 |
| 3 h | climatology | 30 | 270 | 0.86 | 0.49 | 0.129 | 1.00 | 0.14 | 0.86 | 0.82 | 0.121 |
| 3 h | model | 30 | 270 | 0.86 | 0.82 | 0.121 | 1.00 | 0.14 | 0.86 | 0.82 | 0.121 |
| 3 h | open_meteo | 30 | 270 | 0.86 | 0.57 | 0.144 | 1.00 | 0.14 | 0.86 | 0.82 | 0.121 |
| 3 h | persistence | 30 | 270 | 0.86 | 0.63 | 0.147 | 0.92 | 0.11 | 0.83 | 0.82 | 0.121 |
| 6 h | climatology | 30 | 270 | 0.97 | 0.46 | 0.029 | 1.00 | 0.03 | 0.97 | 0.60 | 0.026 |
| 6 h | model | 30 | 270 | 0.97 | 0.60 | 0.026 | 1.00 | 0.03 | 0.97 | 0.60 | 0.026 |
| 6 h | open_meteo | 30 | 270 | 0.97 | 0.48 | 0.026 | 1.00 | 0.03 | 0.97 | 0.60 | 0.026 |
| 6 h | persistence | 30 | 270 | 0.97 | 0.46 | 0.066 | 0.95 | 0.03 | 0.92 | 0.60 | 0.026 |

### Threshold 1.0 mm

| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | climatology | 30 | 270 | 0.43 | 0.61 | 0.259 | 0.02 | 0.60 | 0.02 | 0.78 | 0.218 |
| 1 h | model | 30 | 270 | 0.43 | 0.78 | 0.218 | 0.87 | 0.46 | 0.51 | 0.78 | 0.218 |
| 1 h | persistence | 30 | 270 | 0.43 | 0.85 | 0.218 | 0.99 | 0.36 | 0.64 | 0.78 | 0.218 |
| 3 h | climatology | 30 | 270 | 0.58 | 0.62 | 0.257 | 0.36 | 0.30 | 0.32 | 0.81 | 0.192 |
| 3 h | model | 30 | 270 | 0.58 | 0.81 | 0.192 | 0.99 | 0.33 | 0.66 | 0.81 | 0.192 |
| 3 h | persistence | 30 | 270 | 0.58 | 0.78 | 0.219 | 0.92 | 0.27 | 0.68 | 0.81 | 0.192 |
| 6 h | climatology | 30 | 270 | 0.68 | 0.55 | 0.232 | 0.78 | 0.31 | 0.57 | 0.77 | 0.192 |
| 6 h | model | 30 | 270 | 0.68 | 0.77 | 0.192 | 0.99 | 0.29 | 0.70 | 0.77 | 0.192 |
| 6 h | persistence | 30 | 270 | 0.68 | 0.59 | 0.312 | 0.89 | 0.30 | 0.64 | 0.77 | 0.192 |
