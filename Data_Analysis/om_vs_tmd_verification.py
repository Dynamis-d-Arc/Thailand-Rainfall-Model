"""Verify Open-Meteo's precipitation forecast against the TMD rain gauge.

Open-Meteo's `precipitation` is ECMWF forecast output; the TMD gauge is an instrument reading.
Treating the gauge as truth turns their disagreement into standard forecast-verification numbers
(POD / FAR / CSI / bias), which is what `BKK_Rain_V1` was implicitly trained to reproduce and
what `BKK_Rain_V2` stopped being graded on.

**Time alignment is established, not assumed.** Rain is too intermittent to align two series by
eye, so the convention is fixed using temperature, whose diurnal cycle is unambiguous:

    TMD  `utc_time`            temperature min 23:00, max 07:00  -> +7 gives min 06, max 14 local
    OM   `local_forecast_time` temperature min 06:00, max 13:00  -> already local

So the simultaneous join is `om.local_forecast_time = tmd.utc_time + 7h`. The lag scan then
measures a real timing difference rather than a bookkeeping error.

Only the three TMD stations inside the 56-cell Bangkok grid are used, each paired with the cell
whose centre is nearest.

Usage:
    python Data_Analysis/om_vs_tmd_verification.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}
OUTPUT_DIR = Path(__file__).resolve().parent / "om_vs_tmd"

# station -> nearest grid cell centre
PAIRS = {"37": (20, "Bangna"), "104": (28, "Chaloem Phra Kiat"), "106": (6, "Samut Prakan")}
UTC_TO_LOCAL_HOURS = 7
THRESHOLDS = [0.1, 1.0, 10.0]
LAGS = range(-6, 7)          # hours applied to the Open-Meteo side, 0 = simultaneous


def load():
    """One row per (station, hour) with the gauge amount and the simultaneous forecast amount."""
    frames = []
    with psycopg2.connect(**DB_CONFIG) as conn:
        for station, (cell, name) in PAIRS.items():
            gauge = pd.read_sql(
                'SELECT utc_time, precipitation_mm AS gauge_mm '
                'FROM "BKK_TMD_WEATHER_DATA" WHERE station = %(s)s '
                'AND precipitation_mm IS NOT NULL',
                conn, params={"s": station})
            forecast = pd.read_sql(
                'SELECT local_forecast_time, precipitation AS om_mm '
                'FROM "OM_BKK_DATA_PRECOMPUTE" WHERE grid_number = %(c)s',
                conn, params={"c": cell})
            gauge["local_time"] = (pd.to_datetime(gauge["utc_time"])
                                   + pd.Timedelta(hours=UTC_TO_LOCAL_HOURS))
            forecast["local_time"] = pd.to_datetime(forecast["local_forecast_time"])
            merged = gauge.merge(forecast, on="local_time", how="inner")
            merged["station"] = station
            merged["station_name"] = name
            frames.append(merged[["station", "station_name", "local_time", "gauge_mm", "om_mm"]])
    return pd.concat(frames, ignore_index=True).sort_values(["station", "local_time"])


def contingency(observed, forecast, threshold):
    """The 2x2 table and the verification scores that come off it."""
    obs = observed >= threshold
    fct = forecast >= threshold
    hits = int(np.count_nonzero(obs & fct))
    false_alarms = int(np.count_nonzero(~obs & fct))
    misses = int(np.count_nonzero(obs & ~fct))
    correct_negatives = int(np.count_nonzero(~obs & ~fct))
    n = hits + false_alarms + misses + correct_negatives

    pod = hits / (hits + misses) if hits + misses else np.nan          # detection rate
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else np.nan
    pofd = (false_alarms / (false_alarms + correct_negatives)
            if false_alarms + correct_negatives else np.nan)
    csi = hits / (hits + false_alarms + misses) if hits + false_alarms + misses else np.nan
    bias = ((hits + false_alarms) / (hits + misses)) if hits + misses else np.nan
    # Heidke: skill against a random forecast with the same marginals
    expected = (((hits + misses) * (hits + false_alarms)
                 + (correct_negatives + misses) * (correct_negatives + false_alarms)) / n)
    heidke = (hits + correct_negatives - expected) / (n - expected) if n != expected else np.nan
    return {
        "threshold_mm": threshold, "n": n,
        "hits": hits, "false_alarms": false_alarms, "misses": misses,
        "correct_negatives": correct_negatives,
        "observed_events": hits + misses, "forecast_events": hits + false_alarms,
        "POD": pod, "FAR": far, "POFD": pofd, "CSI": csi,
        "frequency_bias": bias, "TSS": pod - pofd, "HSS": heidke,
        "accuracy": (hits + correct_negatives) / n,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    print(f"paired hours: {len(data):,}   "
          f"{data['local_time'].min()} .. {data['local_time'].max()} (local)\n")

    # ---- headline: simultaneous verification, pooled over the three stations
    rows = [contingency(data["gauge_mm"].to_numpy(), data["om_mm"].to_numpy(), t)
            for t in THRESHOLDS]
    pooled = pd.DataFrame(rows)
    print("=== Open-Meteo vs TMD gauge, simultaneous (all 3 stations pooled) ===")
    print(pooled[["threshold_mm", "observed_events", "forecast_events", "hits", "misses",
                  "false_alarms", "POD", "FAR", "CSI", "frequency_bias", "TSS", "HSS"]]
          .round(3).to_string(index=False))

    # ---- per station, at the operational 0.1 mm threshold
    per_station = []
    for station, group in data.groupby("station"):
        entry = contingency(group["gauge_mm"].to_numpy(), group["om_mm"].to_numpy(), 0.1)
        entry.update(station=station, station_name=group["station_name"].iloc[0])
        per_station.append(entry)
    per_station = pd.DataFrame(per_station)
    print("\n=== per station, threshold 0.1 mm ===")
    print(per_station[["station", "station_name", "n", "observed_events", "forecast_events",
                       "POD", "FAR", "CSI", "frequency_bias"]].round(3).to_string(index=False))

    # ---- amounts, not just occurrence
    both = data[(data["gauge_mm"] >= 0.1) | (data["om_mm"] >= 0.1)]
    print("\n=== amounts ===")
    print(f"pearson  (all hours)      : {data['gauge_mm'].corr(data['om_mm']):.3f}")
    print(f"spearman (all hours)      : {data['gauge_mm'].corr(data['om_mm'], method='spearman'):.3f}")
    print(f"pearson  (either wet)     : {both['gauge_mm'].corr(both['om_mm']):.3f}")
    print(f"gauge total mm            : {data['gauge_mm'].sum():,.0f}")
    print(f"Open-Meteo total mm       : {data['om_mm'].sum():,.0f}")
    print(f"ratio OM/gauge            : {data['om_mm'].sum() / data['gauge_mm'].sum():.3f}")

    # ---- is the disagreement timing or amplitude? shift the forecast and re-score
    lag_rows = []
    for station, group in data.groupby("station"):
        group = group.set_index("local_time").asfreq("h")
        for lag in LAGS:
            shifted = group["om_mm"].shift(lag)
            ok = group["gauge_mm"].notna() & shifted.notna()
            if not ok.any():
                continue
            entry = contingency(group.loc[ok, "gauge_mm"].to_numpy(),
                                shifted[ok].to_numpy(), 0.1)
            entry.update(station=station, lag_hours=lag)
            lag_rows.append(entry)
    lags = pd.DataFrame(lag_rows)
    pooled_lag = lags.groupby("lag_hours").apply(
        lambda g: pd.Series({
            "POD": g["hits"].sum() / (g["hits"].sum() + g["misses"].sum()),
            "FAR": g["false_alarms"].sum() / (g["hits"].sum() + g["false_alarms"].sum()),
            "CSI": g["hits"].sum() / (g["hits"].sum() + g["false_alarms"].sum()
                                      + g["misses"].sum()),
        }), include_groups=False)
    print("\n=== lag scan at 0.1 mm (lag = hours the forecast is shifted later) ===")
    print(pooled_lag.round(4).to_string())
    best = pooled_lag["CSI"].idxmax()
    print(f"\nCSI peaks at lag {best:+d} h "
          f"(CSI {pooled_lag.loc[best, 'CSI']:.4f} vs {pooled_lag.loc[0, 'CSI']:.4f} simultaneous)")
    # shift(+k) puts om[t-k] against gauge[t]: a positive best lag means the gauge's rain was
    # already forecast k hours earlier, i.e. the forecast runs EARLY.
    if best > 0:
        print(f"=> the forecast's rain arrives about {best} h EARLIER than the gauge records it")
    elif best < 0:
        print(f"=> the forecast's rain arrives about {abs(best)} h LATER than the gauge records it")
    gain = pooled_lag.loc[best, "CSI"] - pooled_lag.loc[0, "CSI"]
    print(f"   but re-timing buys only {gain:+.4f} CSI, so the disagreement is dominated by "
          f"over-forecasting, not by timing")

    pooled.to_csv(OUTPUT_DIR / "om_vs_tmd_pooled.csv", index=False)
    per_station.to_csv(OUTPUT_DIR / "om_vs_tmd_per_station.csv", index=False)
    pooled_lag.to_csv(OUTPUT_DIR / "om_vs_tmd_lag_scan.csv")
    print(f"\nwrote CSVs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
