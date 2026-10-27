# Bangkok: live model vs public weather services (labels_v2)

Generated 2026-10-27 21:50. Nine 25 km cells over the Bangkok metro area, graded against IMERG. Each provider row is paired with the model on exactly the same issue hours and cells; 'model' columns are the model on that provider's sample. Flags use p >= 0.5 for every forecaster. Providers are graded at 0.1 mm only (their probability means 'measurable rain at the point'). Labels: wettest IMERG pixel's hourly total over the window [issue, issue+h); provider hour k refers to the preceding hour, so hours 1..h cover that same window. The 1 h rows are the primary comparison. Multi-hour provider rows are BOUNDS on P(any rain in the window), not probabilities: [max] = max of the hourly values (lower bound), [1-prod] = the independence product (upper bound under positive hour-to-hour correlation).

## Label: rain anywhere in the cell (project label, IMERG cell-max)

### Threshold 0.1 mm

| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | climatology | 46 | 414 | 0.70 | 0.48 | 0.228 | 0.95 | 0.29 | 0.68 | 0.91 | 0.175 |
| 1 h | model | 46 | 414 | 0.70 | 0.91 | 0.175 | 1.00 | 0.29 | 0.70 | 0.91 | 0.175 |
| 1 h | open_meteo | 46 | 414 | 0.70 | 0.82 | 0.155 | 0.99 | 0.20 | 0.79 | 0.91 | 0.175 |
| 1 h | persistence | 46 | 414 | 0.70 | 0.91 | 0.087 | 0.96 | 0.10 | 0.87 | 0.91 | 0.175 |
| 3 h | climatology | 46 | 414 | 0.79 | 0.45 | 0.178 | 1.00 | 0.20 | 0.80 | 0.96 | 0.124 |
| 3 h | model | 46 | 414 | 0.79 | 0.96 | 0.124 | 1.00 | 0.20 | 0.80 | 0.96 | 0.124 |
| 3 h | open_meteo[1-prod] | 46 | 414 | 0.79 | 0.88 | 0.131 | 1.00 | 0.17 | 0.83 | 0.96 | 0.124 |
| 3 h | open_meteo[max] | 46 | 414 | 0.79 | 0.85 | 0.097 | 0.99 | 0.12 | 0.87 | 0.96 | 0.124 |
| 3 h | persistence | 46 | 414 | 0.79 | 0.82 | 0.115 | 0.92 | 0.09 | 0.84 | 0.96 | 0.124 |
| 6 h | climatology | 46 | 414 | 0.87 | 0.44 | 0.120 | 1.00 | 0.13 | 0.87 | 0.98 | 0.092 |
| 6 h | model | 46 | 414 | 0.87 | 0.98 | 0.092 | 1.00 | 0.13 | 0.87 | 0.98 | 0.092 |
| 6 h | open_meteo[1-prod] | 46 | 414 | 0.87 | 0.98 | 0.091 | 0.99 | 0.11 | 0.89 | 0.98 | 0.092 |
| 6 h | open_meteo[max] | 46 | 414 | 0.87 | 0.96 | 0.054 | 0.97 | 0.06 | 0.91 | 0.98 | 0.092 |
| 6 h | persistence | 45 | 405 | 0.87 | 0.63 | 0.127 | 0.95 | 0.10 | 0.86 | 0.98 | 0.094 |

### Threshold 1.0 mm

| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | climatology | 46 | 414 | 0.39 | 0.42 | 0.273 | 0.01 | 0.78 | 0.01 | 0.90 | 0.155 |
| 1 h | model | 46 | 414 | 0.39 | 0.90 | 0.155 | 0.88 | 0.36 | 0.59 | 0.90 | 0.155 |
| 1 h | persistence | 46 | 414 | 0.39 | 0.96 | 0.114 | 0.96 | 0.27 | 0.71 | 0.90 | 0.155 |
| 3 h | climatology | 46 | 414 | 0.48 | 0.47 | 0.266 | 0.37 | 0.54 | 0.26 | 0.88 | 0.169 |
| 3 h | model | 46 | 414 | 0.48 | 0.88 | 0.169 | 0.97 | 0.35 | 0.64 | 0.88 | 0.169 |
| 3 h | persistence | 46 | 414 | 0.48 | 0.85 | 0.204 | 0.88 | 0.32 | 0.62 | 0.88 | 0.169 |
| 6 h | climatology | 46 | 414 | 0.59 | 0.53 | 0.246 | 0.81 | 0.41 | 0.52 | 0.89 | 0.159 |
| 6 h | model | 46 | 414 | 0.59 | 0.89 | 0.159 | 0.99 | 0.29 | 0.70 | 0.89 | 0.159 |
| 6 h | persistence | 45 | 405 | 0.58 | 0.68 | 0.328 | 0.86 | 0.37 | 0.57 | 0.89 | 0.161 |

## Label: area-average rain (IMERG cell-mean, closer to a point forecast)

### Threshold 0.1 mm

| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | climatology | 46 | 414 | 0.59 | 0.43 | 0.286 | 0.94 | 0.41 | 0.57 | 0.89 | 0.255 |
| 1 h | model | 46 | 414 | 0.59 | 0.89 | 0.255 | 1.00 | 0.40 | 0.60 | 0.89 | 0.255 |
| 1 h | open_meteo | 46 | 414 | 0.59 | 0.76 | 0.240 | 1.00 | 0.32 | 0.68 | 0.89 | 0.255 |
| 1 h | persistence | 46 | 414 | 0.59 | 0.84 | 0.154 | 0.99 | 0.21 | 0.78 | 0.89 | 0.255 |
| 3 h | climatology | 46 | 414 | 0.72 | 0.44 | 0.232 | 1.00 | 0.28 | 0.72 | 0.94 | 0.182 |
| 3 h | model | 46 | 414 | 0.72 | 0.94 | 0.182 | 1.00 | 0.28 | 0.72 | 0.94 | 0.182 |
| 3 h | open_meteo[1-prod] | 46 | 414 | 0.72 | 0.79 | 0.199 | 1.00 | 0.25 | 0.75 | 0.94 | 0.182 |
| 3 h | open_meteo[max] | 46 | 414 | 0.72 | 0.76 | 0.157 | 1.00 | 0.20 | 0.80 | 0.94 | 0.182 |
| 3 h | persistence | 46 | 414 | 0.72 | 0.77 | 0.154 | 0.93 | 0.15 | 0.80 | 0.94 | 0.182 |
| 6 h | climatology | 46 | 414 | 0.83 | 0.46 | 0.156 | 1.00 | 0.17 | 0.83 | 0.95 | 0.127 |
| 6 h | model | 46 | 414 | 0.83 | 0.95 | 0.127 | 1.00 | 0.17 | 0.83 | 0.95 | 0.127 |
| 6 h | open_meteo[1-prod] | 46 | 414 | 0.83 | 0.90 | 0.123 | 1.00 | 0.14 | 0.86 | 0.95 | 0.127 |
| 6 h | open_meteo[max] | 46 | 414 | 0.83 | 0.88 | 0.076 | 0.98 | 0.10 | 0.89 | 0.95 | 0.127 |
| 6 h | persistence | 45 | 405 | 0.83 | 0.63 | 0.154 | 0.96 | 0.14 | 0.83 | 0.96 | 0.130 |

### Threshold 1.0 mm

| Horizon | Forecaster | Issue hours | n | Wet rate | ROC-AUC | Brier | POD | FAR | CSI | Model ROC-AUC (same rows) | Model Brier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 h | climatology | 46 | 414 | 0.28 | 0.46 | 0.222 | 0.02 | 0.78 | 0.02 | 0.86 | 0.187 |
| 1 h | model | 46 | 414 | 0.28 | 0.86 | 0.187 | 0.90 | 0.52 | 0.45 | 0.86 | 0.187 |
| 1 h | persistence | 46 | 414 | 0.28 | 0.92 | 0.193 | 1.00 | 0.45 | 0.55 | 0.86 | 0.187 |
| 3 h | climatology | 46 | 414 | 0.37 | 0.46 | 0.256 | 0.34 | 0.67 | 0.20 | 0.88 | 0.202 |
| 3 h | model | 46 | 414 | 0.37 | 0.88 | 0.202 | 0.99 | 0.48 | 0.51 | 0.88 | 0.202 |
| 3 h | persistence | 46 | 414 | 0.37 | 0.87 | 0.234 | 0.95 | 0.43 | 0.55 | 0.88 | 0.202 |
| 6 h | climatology | 46 | 414 | 0.47 | 0.48 | 0.273 | 0.79 | 0.54 | 0.41 | 0.87 | 0.214 |
| 6 h | model | 46 | 414 | 0.47 | 0.87 | 0.214 | 1.00 | 0.42 | 0.57 | 0.87 | 0.214 |
| 6 h | persistence | 45 | 405 | 0.46 | 0.72 | 0.370 | 0.90 | 0.47 | 0.50 | 0.87 | 0.217 |
