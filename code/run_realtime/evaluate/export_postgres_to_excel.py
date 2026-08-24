import os
from datetime import datetime

import pandas as pd
import psycopg2


# ============================================================
# CONFIG
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "fraud_detection",
    "user": "fraud",
    "password": "fraud123",
}

TABLE_NAME = "fraud_predictions"

OUTPUT_DIR = r"D:\ThucTap_VinSmartFuture\run_realtime\exports"


# ============================================================
# MAIN
# ============================================================

def export_postgres_to_excel():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = os.path.join(
        OUTPUT_DIR,
        f"fraud_predictions_DataFromPostgreSQLSink_{timestamp}.xlsx"
    )

    print("=" * 70)
    print("EXPORT POSTGRESQL -> EXCEL")
    print("=" * 70)

    print()
    print("Connecting to PostgreSQL...")

    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )

    print("Connected successfully.")
    print()

    query = f"""
        SELECT *
        FROM {TABLE_NAME}
    """

    print(f"Reading table: {TABLE_NAME}")

    df = pd.read_sql_query(query, conn)

    conn.close()

    print(f"Rows exported: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()

    if df.empty:
        print("WARNING: PostgreSQL table is empty.")
        return

    # ========================================================
    # WRITE EXCEL
    # ========================================================

    print("Writing Excel file...")

    with pd.ExcelWriter(
        output_file,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="fraud_predictions",
            index=False
        )

    print()
    print("=" * 70)
    print("EXPORT COMPLETED")
    print("=" * 70)

    print()
    print(f"Excel file:")
    print(output_file)

    print()
    print(f"Total rows : {len(df)}")
    print(f"Total cols : {len(df.columns)}")


if __name__ == "__main__":
    export_postgres_to_excel()