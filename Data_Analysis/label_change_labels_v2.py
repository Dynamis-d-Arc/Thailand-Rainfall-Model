"""How much the 2026-10-27 label fix changes the six targets, on the full IMERG panel.

Two corrections are measured separately and together, per calendar month and pooled:

  window      [t+1, t+1+h)  ->  [t, t+h)          (IMERG rows are start-stamped)
  aggregation precipitation_max_mm (0.5*max(a) + 0.5*max(b), maxima may be on
              different pixels)  ->  precipitation_pixelmax_mm (per-pixel total, then max)

For every complete hour t with the full old and new windows present (same guard as the
panel loader), each target is labelled four ways and the disagreement counted.
Writes Data_Analysis/report_evidence/label_change_labels_v2.csv (by month) and
label_change_labels_v2_summary.csv (pooled). No model involved.

    python Data_Analysis/label_change_labels_v2.py
"""
import os
import time
from pathlib import Path

import pandas as pd
import psycopg2

OUT = Path(__file__).resolve().parent / "report_evidence"
TABLE = '"IMERG_THAILAND_DATA"'
MAX_LEAD = 6
TARGETS = [(1, 0.1), (3, 0.1), (6, 0.1), (1, 1.0), (3, 1.0), (6, 1.0)]
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}


def build_sql():
    leads = ",\n        ".join(
        f"LEAD(precipitation_max_mm, {k}) OVER w AS old{k}, "
        f"LEAD(precipitation_pixelmax_mm, {k}) OVER w AS new{k}"
        for k in range(MAX_LEAD + 1))
    guard = (f"t{MAX_LEAD} = t + interval '{MAX_LEAD} hour' AND "
             + " AND ".join(f"new{k} IS NOT NULL" for k in range(MAX_LEAD + 1)))

    def pos(prefix, ks, thr):
        return f"GREATEST({', '.join(f'{prefix}{k}' for k in ks)}) >= {thr}"

    counts = []
    for h, thr in TARGETS:
        name = f"h{h}_{thr}mm"
        old = pos("old", range(1, h + 1), thr)            # old window, old aggregation
        new = pos("new", range(0, h), thr)                # new window, new aggregation
        win = pos("old", range(0, h), thr)                # new window, old aggregation
        agg = pos("new", range(1, h + 1), thr)            # old window, new aggregation
        counts += [
            f"COUNT(*) FILTER (WHERE {old}) AS \"{name}|old_pos\"",
            f"COUNT(*) FILTER (WHERE {new}) AS \"{name}|new_pos\"",
            f"COUNT(*) FILTER (WHERE ({old}) <> ({new})) AS \"{name}|disagree\"",
            f"COUNT(*) FILTER (WHERE ({old}) <> ({win})) AS \"{name}|window_only_disagree\"",
            f"COUNT(*) FILTER (WHERE ({old}) AND NOT ({agg})) AS \"{name}|aggregation_only_flip\"",
        ]
    return f"""
    WITH s AS (
        SELECT local_observation_time AS t, run_type,
            LEAD(local_observation_time, {MAX_LEAD}) OVER w AS t{MAX_LEAD},
            {leads}
        FROM {TABLE}
        WHERE is_complete_hour
        WINDOW w AS (PARTITION BY grid_number ORDER BY local_observation_time)
    )
    SELECT date_trunc('month', t)::date AS month, COUNT(*) AS rows,
        {", ".join(counts)}
    FROM s
    WHERE {guard}
    GROUP BY 1 ORDER BY 1
    """


def main():
    t0 = time.time()
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        wide = pd.read_sql(build_sql(), conn)
    finally:
        conn.close()
    print(f"query {time.time() - t0:.0f}s, {len(wide)} months, {wide['rows'].sum():,} cell-hours")

    rows = []
    for _, r in wide.iterrows():
        for h, thr in TARGETS:
            name = f"h{h}_{thr}mm"
            rows.append({"month": r["month"], "target": name, "rows": int(r["rows"]),
                         "old_positives": int(r[f"{name}|old_pos"]),
                         "new_positives": int(r[f"{name}|new_pos"]),
                         "labels_changed": int(r[f"{name}|disagree"]),
                         "changed_by_window_shift_alone": int(r[f"{name}|window_only_disagree"]),
                         "positives_lost_to_pixel_max_alone": int(r[f"{name}|aggregation_only_flip"])})
    long = pd.DataFrame(rows)
    long["old_base_rate"] = long.old_positives / long.rows
    long["new_base_rate"] = long.new_positives / long.rows
    long["pct_labels_changed"] = 100 * long.labels_changed / long.rows
    long.to_csv(OUT / "label_change_labels_v2.csv", index=False, float_format="%.4f")

    g = long.groupby("target", sort=False)[["rows", "old_positives", "new_positives", "labels_changed",
                                            "changed_by_window_shift_alone",
                                            "positives_lost_to_pixel_max_alone"]].sum()
    g["old_base_rate"] = g.old_positives / g.rows
    g["new_base_rate"] = g.new_positives / g.rows
    g["pct_labels_changed"] = 100 * g.labels_changed / g.rows
    g["pct_changed_by_window_shift_alone"] = 100 * g.changed_by_window_shift_alone / g.rows
    g["pct_old_positives_lost_to_pixel_max_alone"] = 100 * g.positives_lost_to_pixel_max_alone / g.old_positives
    g.reset_index().to_csv(OUT / "label_change_labels_v2_summary.csv", index=False, float_format="%.4f")
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
