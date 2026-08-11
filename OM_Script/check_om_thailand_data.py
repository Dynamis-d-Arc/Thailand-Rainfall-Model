import os

import psycopg2


DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "Pass1234"),
}


def main():
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*),
                    COUNT(DISTINCT grid_number),
                    MIN(local_forecast_time),
                    MAX(local_forecast_time)
                FROM "OM_Thailand_Data";
                """
            )
            print(cur.fetchone())

            cur.execute(
                """
                SELECT local_forecast_time::date, COUNT(*)
                FROM "OM_Thailand_Data"
                GROUP BY local_forecast_time::date
                ORDER BY local_forecast_time::date DESC
                LIMIT 5;
                """
            )
            for row in cur.fetchall():
                print(row)


if __name__ == "__main__":
    main()
