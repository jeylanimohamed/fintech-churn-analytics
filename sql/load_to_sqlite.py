"""
Build a SQLite database from the generated CSVs so the queries in
sql/analysis_queries.sql can be run standalone.

Usage:
    python data/generate_data.py            # if the CSVs don't exist yet
    python sql/load_to_sqlite.py            # builds data/fintech.db
    sqlite3 data/fintech.db < sql/analysis_queries.sql
"""

import os
import sqlite3
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
DB = os.path.join(DATA, "fintech.db")

TABLES = ["users", "transactions", "ab_test"]


def main():
    missing = [t for t in TABLES if not os.path.exists(os.path.join(DATA, f"{t}.csv"))]
    if missing:
        raise SystemExit(
            f"Missing CSVs: {missing}. Run `python data/generate_data.py` first."
        )

    if os.path.exists(DB):
        os.remove(DB)

    conn = sqlite3.connect(DB)
    for t in TABLES:
        df = pd.read_csv(os.path.join(DATA, f"{t}.csv"))
        df.to_sql(t, conn, index=False, if_exists="replace")
        print(f"Loaded {len(df):,} rows into '{t}'")
    conn.close()
    print(f"Done. Database written to {DB}")


if __name__ == "__main__":
    main()
