# ============================================================
# TRANSACTION REPLAY PRODUCER
# ============================================================
#
# PURPOSE
# -------
# Read historical transaction data from:
#
# D:\ThucTap_VinSmartFuture\run_realtime\data
#
# Files:
#
#     2018-10-01.pkl
#     2018-10-02.pkl
#     ...
#     2020-05-22.pkl
#
# Replay transactions to Kafka as real-time transactions.
#
# Kafka:
#     localhost:9092
#
# Topic:
#     transactions
#
# Supported rates:
#     10 TPS
#     50 TPS
#     100 TPS
#     500 TPS
#
#
# INPUT COLUMNS
# -------------
# TRANSACTION_ID
# TX_DATETIME
# CUSTOMER_ID
# TERMINAL_ID
# TX_AMOUNT
# TX_TIME_SECONDS
# TX_TIME_DAYS
# TX_FRAUD
#
#
# OUTPUT COLUMNS
# --------------
# TRANSACTION_ID
# TX_DATETIME
# CUSTOMER_ID
# TERMINAL_ID
# TX_AMOUNT
# TX_TIME_SECONDS
# TX_TIME_DAYS
# TX_FRAUD
# PRODUCER_TIMESTAMP
#
#
# NO DATA LEAKAGE
# ---------------
# TX_FRAUD is ONLY ground truth.
#
# TX_FRAUD must NOT be used to:
#
#     - create features
#     - create customer history
#     - create terminal history
#     - create LSTM input
#     - calculate fraud probability
#     - calculate FRAUD_PREDICTION
#
# Evaluation later compares:
#
#     TX_FRAUD
#          vs
#     FRAUD_PREDICTION
#
#
# PRODUCER_TIMESTAMP
# ------------------
# PRODUCER_TIMESTAMP represents the REAL SYSTEM TIME
# when the transaction is sent by this replay producer.
#
# It is NOT the historical TX_DATETIME.
#
# Therefore:
#
# TX_DATETIME
#     = historical transaction time
#
# PRODUCER_TIMESTAMP
#     = real replay/send time
#
# This allows downstream components to calculate:
#
#     END_TO_END_LATENCY_MS
#
# ============================================================

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from kafka import KafkaProducer


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"D:\ThucTap_VinSmartFuture\run_realtime\data"

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

KAFKA_TOPIC = "transactions"

SUPPORTED_RATES = [
    10,
    50,
    100,
    500,
]

REQUIRED_COLUMNS = [
    "TRANSACTION_ID",
    "TX_DATETIME",
    "CUSTOMER_ID",
    "TERMINAL_ID",
    "TX_AMOUNT",
    "TX_TIME_SECONDS",
    "TX_TIME_DAYS",
    "TX_FRAUD",
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Replay historical .pkl transaction files "
            "to Kafka in real time."
        )
    )

    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=(
            "Directory containing daily .pkl files. "
            f"Default: {DATA_DIR}"
        ),
    )

    parser.add_argument(
        "--rate",
        type=int,
        required=True,
        choices=SUPPORTED_RATES,
        help=(
            "Replay rate: "
            "10, 50, 100 or 500 transactions/sec."
        ),
    )

    parser.add_argument(
        "--bootstrap",
        default=KAFKA_BOOTSTRAP_SERVERS,
        help=(
            "Kafka bootstrap server. "
            f"Default: {KAFKA_BOOTSTRAP_SERVERS}"
        ),
    )

    parser.add_argument(
        "--topic",
        default=KAFKA_TOPIC,
        help=(
            f"Kafka topic. Default: {KAFKA_TOPIC}"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum total number of transactions "
            "to replay."
        ),
    )

    parser.add_argument(
        "--start-date",
        default="2018-10-01",
        help=(
            "First dataset date. "
            "Default: 2018-10-01"
        ),
    )

    parser.add_argument(
        "--end-date",
        default="2020-05-22",
        help=(
            "Last dataset date. "
            "Default: 2020-05-22"
        ),
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Replay all files repeatedly."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Read and validate data without "
            "sending to Kafka."
        ),
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help=(
            "Print every N transactions. "
            "Default: 1 = print every transaction."
        ),
    )

    return parser.parse_args()


# ============================================================
# DATE
# ============================================================

def parse_date(date_string):

    try:

        return datetime.strptime(
            date_string,
            "%Y-%m-%d",
        ).date()

    except ValueError as exc:

        raise ValueError(
            f"Invalid date: {date_string}. "
            f"Expected YYYY-MM-DD."
        ) from exc


# ============================================================
# FIND PKL FILES
# ============================================================

def get_pkl_files(
    data_dir,
    start_date,
    end_date,
):

    if not os.path.isdir(data_dir):

        raise FileNotFoundError(
            f"Data directory does not exist:\n"
            f"{data_dir}"
        )

    files = []

    for filename in os.listdir(data_dir):

        if not filename.lower().endswith(".pkl"):
            continue

        date_part = filename[:-4]

        try:

            file_date = datetime.strptime(
                date_part,
                "%Y-%m-%d",
            ).date()

        except ValueError:

            continue

        if (
            start_date
            <= file_date
            <= end_date
        ):

            files.append(
                (
                    file_date,
                    os.path.join(
                        data_dir,
                        filename,
                    ),
                )
            )

    files.sort(
        key=lambda item: item[0]
    )

    return files


# ============================================================
# VALIDATE DATAFRAME
# ============================================================

def validate_dataframe(
    df,
    file_path,
):

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("=" * 80)
        print("ERROR: INVALID DATASET SCHEMA")
        print("=" * 80)

        print(
            f"File: {file_path}"
        )

        print()
        print("Required columns:")

        for column in REQUIRED_COLUMNS:

            print(
                f"  - {column}"
            )

        print()
        print("Missing columns:")

        for column in missing_columns:

            print(
                f"  - {column}"
            )

        print()
        print("Detected columns:")

        for column in df.columns:

            print(
                f"  - {column}"
            )

        raise ValueError(
            "Dataset schema is not compatible."
        )


# ============================================================
# LOAD ONE PKL
# ============================================================

def load_pkl_file(file_path):

    print()
    print(
        f"[LOAD] {os.path.basename(file_path)}"
    )

    df = pd.read_pickle(
        file_path
    )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            f"{file_path} does not contain "
            f"a pandas DataFrame."
        )

    validate_dataframe(
        df,
        file_path,
    )

    # ========================================================
    # SELECT ONLY ORIGINAL TRANSACTION COLUMNS
    # ========================================================

    df = df[
        REQUIRED_COLUMNS
    ].copy()

    # ========================================================
    # CONVERT TX_DATETIME
    # ========================================================

    df["TX_DATETIME"] = pd.to_datetime(
        df["TX_DATETIME"],
        errors="coerce",
    )

    invalid_datetime = (
        df["TX_DATETIME"].isna()
    )

    if invalid_datetime.any():

        count = int(
            invalid_datetime.sum()
        )

        print(
            f"[WARNING] "
            f"{count} rows have invalid TX_DATETIME."
        )

        df = df[
            ~invalid_datetime
        ]

    # ========================================================
    # SORT CHRONOLOGICALLY
    # ========================================================

    df = df.sort_values(
        by=[
            "TX_DATETIME",
            "TRANSACTION_ID",
        ],
        kind="stable",
    )

    df = df.reset_index(
        drop=True
    )

    print(
        f"[LOAD] rows={len(df):,}"
    )

    if len(df) > 0:

        print(
            f"[LOAD] first="
            f"{df.iloc[0]['TX_DATETIME']}"
        )

        print(
            f"[LOAD] last ="
            f"{df.iloc[-1]['TX_DATETIME']}"
        )

    return df


# ============================================================
# CONVERT VALUE TO JSON SAFE VALUE
# ============================================================

def json_safe(value):

    if value is None:

        return None

    try:

        if pd.isna(value):

            return None

    except Exception:

        pass

    if isinstance(
        value,
        pd.Timestamp,
    ):

        return value.isoformat()

    if isinstance(
        value,
        datetime,
    ):

        return value.isoformat()

    if hasattr(
        value,
        "item",
    ):

        try:

            return value.item()

        except Exception:

            pass

    return value


# ============================================================
# PRODUCER TIMESTAMP
# ============================================================

def get_producer_timestamp():

    """
    Return the REAL SYSTEM TIME when the producer
    is about to send the transaction.

    UTC ISO-8601 format.

    Example:

        2026-08-19T04:25:31.123456+00:00
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# ROW -> KAFKA MESSAGE
# ============================================================

def row_to_message(
    row,
    producer_timestamp,
):

    # ========================================================
    # IMPORTANT DATA-LEAKAGE RULE
    #
    # TX_FRAUD is copied from the original dataset ONLY.
    #
    # It is NOT used to calculate:
    #
    #     - TX_DURING_WEEKEND
    #     - TX_DURING_NIGHT
    #     - customer features
    #     - terminal features
    #     - rolling features
    #     - customer history
    #     - LSTM input
    #     - fraud probability
    #     - fraud prediction
    #
    # Spark downstream MUST ignore TX_FRAUD when
    # creating model features.
    # ========================================================

    message = {

        # ----------------------------------------------------
        # Original transaction data
        # ----------------------------------------------------

        "TRANSACTION_ID":
            json_safe(
                row["TRANSACTION_ID"]
            ),

        "TX_DATETIME":
            json_safe(
                row["TX_DATETIME"]
            ),

        "CUSTOMER_ID":
            json_safe(
                row["CUSTOMER_ID"]
            ),

        "TERMINAL_ID":
            json_safe(
                row["TERMINAL_ID"]
            ),

        "TX_AMOUNT":
            json_safe(
                row["TX_AMOUNT"]
            ),

        "TX_TIME_SECONDS":
            json_safe(
                row["TX_TIME_SECONDS"]
            ),

        "TX_TIME_DAYS":
            json_safe(
                row["TX_TIME_DAYS"]
            ),

        # ----------------------------------------------------
        # Ground truth ONLY
        # ----------------------------------------------------

        "TX_FRAUD":
            json_safe(
                row["TX_FRAUD"]
            ),

        # ----------------------------------------------------
        # REAL-TIME PRODUCER TIMESTAMP
        #
        # IMPORTANT:
        # This is NOT TX_DATETIME.
        #
        # TX_DATETIME:
        #     Historical dataset timestamp
        #
        # PRODUCER_TIMESTAMP:
        #     Actual replay time
        # ----------------------------------------------------

        "PRODUCER_TIMESTAMP":
            producer_timestamp,
    }

    return message


# ============================================================
# KAFKA PRODUCER
# ============================================================

def create_producer(
    bootstrap_servers,
):

    print()
    print("=" * 80)
    print("KAFKA CONNECTION")
    print("=" * 80)

    print(
        f"Bootstrap server : "
        f"{bootstrap_servers}"
    )

    producer = KafkaProducer(

        bootstrap_servers=
            bootstrap_servers,

        value_serializer=
            lambda value:
                json.dumps(
                    value,
                    ensure_ascii=False,
                ).encode("utf-8"),

        acks="all",

        retries=5,

        linger_ms=2,

        batch_size=16384,

        compression_type="gzip",
    )

    print(
        "Kafka connection : OK"
    )

    return producer


# ============================================================
# PRINT ONE TRANSACTION
# ============================================================

def print_transaction(
    message,
    sent,
):

    print()
    print(
        "-" * 100
    )

    print(
        f"[TRANSACTION #{sent:,}]"
    )

    print(
        f"TRANSACTION_ID      : "
        f"{message['TRANSACTION_ID']}"
    )

    print(
        f"TX_DATETIME         : "
        f"{message['TX_DATETIME']}"
    )

    print(
        f"PRODUCER_TIMESTAMP  : "
        f"{message['PRODUCER_TIMESTAMP']}"
    )

    print(
        f"CUSTOMER_ID         : "
        f"{message['CUSTOMER_ID']}"
    )

    print(
        f"TERMINAL_ID         : "
        f"{message['TERMINAL_ID']}"
    )

    print(
        f"TX_AMOUNT           : "
        f"{message['TX_AMOUNT']}"
    )

    print(
        f"TX_TIME_SECONDS     : "
        f"{message['TX_TIME_SECONDS']}"
    )

    print(
        f"TX_TIME_DAYS        : "
        f"{message['TX_TIME_DAYS']}"
    )

    print(
        f"TX_FRAUD            : "
        f"{message['TX_FRAUD']}"
        f"  <-- GROUND TRUTH ONLY"
    )

    print(
        "-" * 100
    )


# ============================================================
# REPLAY
# ============================================================

def replay_once(
    producer,
    files,
    topic,
    rate,
    limit=None,
    dry_run=False,
    print_every=1,
):

    interval = 1.0 / rate

    sent = 0

    start_time = time.perf_counter()

    next_send_time = start_time

    # ========================================================
    # STATISTICS
    # ========================================================

    total_fraud = 0

    total_legitimate = 0

    print()
    print("=" * 80)
    print("REAL-TIME REPLAY START")
    print("=" * 80)

    print(
        f"Files           : {len(files)}"
    )

    print(
        f"Target rate     : "
        f"{rate} transaction/s"
    )

    print(
        f"Interval        : "
        f"{interval:.6f} seconds"
    )

    print(
        f"Topic           : "
        f"{topic}"
    )

    print(
        f"Limit           : "
        f"{limit}"
    )

    print(
        f"Print every     : "
        f"{print_every} transaction(s)"
    )

    print()

    # ========================================================
    # FILE LOOP
    # ========================================================

    for file_date, file_path in files:

        if (
            limit is not None
            and sent >= limit
        ):

            break

        try:

            df = load_pkl_file(
                file_path
            )

        except Exception as exc:

            print()
            print("=" * 80)
            print(
                f"[ERROR] Cannot load "
                f"{file_path}"
            )

            print(
                f"[ERROR] {exc}"
            )

            raise

        # ====================================================
        # TRANSACTION LOOP
        # ====================================================

        for _, row in df.iterrows():

            if (
                limit is not None
                and sent >= limit
            ):

                break

            # =================================================
            # RATE CONTROL
            # =================================================

            now = time.perf_counter()

            if next_send_time > now:

                time.sleep(
                    next_send_time - now
                )

            # =================================================
            # PRODUCER TIMESTAMP
            #
            # IMPORTANT:
            # Generate this immediately before send.
            # =================================================

            producer_timestamp = (
                get_producer_timestamp()
            )

            # =================================================
            # CREATE MESSAGE
            # =================================================

            message = row_to_message(
                row=row,
                producer_timestamp=
                    producer_timestamp,
            )

            # =================================================
            # GROUND TRUTH STATISTICS
            #
            # TX_FRAUD is ONLY used for evaluation statistics.
            # It is NOT used for prediction.
            # =================================================

            fraud_value = int(
                message["TX_FRAUD"]
            )

            if fraud_value == 1:

                total_fraud += 1

            else:

                total_legitimate += 1

            # =================================================
            # SEND TO KAFKA
            # =================================================

            if not dry_run:

                future = producer.send(
                    topic,
                    value=message,
                )

            sent += 1

            next_send_time += interval

            # =================================================
            # PRINT EVERY TRANSACTION
            # =================================================

            if (
                print_every > 0
                and (
                    sent == 1
                    or sent % print_every == 0
                )
            ):

                print_transaction(
                    message=message,
                    sent=sent,
                )

            # =================================================
            # FLUSH
            # =================================================

            if (
                not dry_run
                and sent % 100 == 0
            ):

                producer.flush()

            # =================================================
            # PROGRESS
            # =================================================

            if (
                sent == 1
                or sent % max(rate, 1000) == 0
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                actual_rate = (
                    sent / elapsed
                    if elapsed > 0
                    else 0
                )

                print()
                print(
                    f"[REPLAY] "
                    f"sent={sent:,} | "
                    f"fraud={total_fraud:,} | "
                    f"legitimate="
                    f"{total_legitimate:,} | "
                    f"actual="
                    f"{actual_rate:.2f} tx/s | "
                    f"target={rate} tx/s"
                )

    # ========================================================
    # FINAL FLUSH
    # ========================================================

    if not dry_run:

        producer.flush()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    actual_rate = (
        sent / elapsed
        if elapsed > 0
        else 0
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("REPLAY SUMMARY")
    print("=" * 80)

    print(
        f"Transactions       : "
        f"{sent:,}"
    )

    print(
        f"Fraud ground truth : "
        f"{total_fraud:,}"
    )

    print(
        f"Legitimate         : "
        f"{total_legitimate:,}"
    )

    print(
        f"Elapsed            : "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Target rate        : "
        f"{rate} tx/s"
    )

    print(
        f"Actual rate        : "
        f"{actual_rate:.2f} tx/s"
    )

    if sent > 0:

        fraud_ratio = (
            total_fraud
            / sent
            * 100
        )

        print(
            f"Fraud ratio        : "
            f"{fraud_ratio:.4f}%"
        )

    print()

    return sent


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # VALIDATE PRINT INTERVAL
    # ========================================================

    if args.print_every < 1:

        print(
            "ERROR: --print-every must "
            "be >= 1."
        )

        sys.exit(1)

    # ========================================================
    # DATE
    # ========================================================

    start_date = parse_date(
        args.start_date
    )

    end_date = parse_date(
        args.end_date
    )

    if start_date > end_date:

        print(
            "ERROR: start-date must "
            "be <= end-date."
        )

        sys.exit(1)

    # ========================================================
    # FIND FILES
    # ========================================================

    files = get_pkl_files(
        args.data_dir,
        start_date,
        end_date,
    )

    if not files:

        print()
        print("=" * 80)
        print("ERROR: NO PKL FILES FOUND")
        print("=" * 80)

        print(
            f"Directory: "
            f"{args.data_dir}"
        )

        print(
            f"Date range: "
            f"{args.start_date} -> "
            f"{args.end_date}"
        )

        sys.exit(1)

    # ========================================================
    # CONFIG
    # ========================================================

    print()
    print("=" * 80)
    print("REAL-TIME FRAUD DETECTION")
    print("REPLAY PRODUCER")
    print("=" * 80)

    print(
        f"Data directory : "
        f"{args.data_dir}"
    )

    print(
        f"Date range     : "
        f"{args.start_date} -> "
        f"{args.end_date}"
    )

    print(
        f"PKL files      : "
        f"{len(files)}"
    )

    print(
        f"Kafka          : "
        f"{args.bootstrap}"
    )

    print(
        f"Topic          : "
        f"{args.topic}"
    )

    print(
        f"Rate           : "
        f"{args.rate} tx/s"
    )

    print(
        f"Limit          : "
        f"{args.limit}"
    )

    print(
        f"Loop           : "
        f"{args.loop}"
    )

    print(
        f"Print every    : "
        f"{args.print_every}"
    )

    print()

    print("First files:")

    for file_date, file_path in files[:5]:

        print(
            f"  {file_date} -> "
            f"{os.path.basename(file_path)}"
        )

    if len(files) > 5:

        print(
            f"  ... "
            f"{len(files) - 5} more files"
        )

    print()

    print("Kafka schema:")

    for column in REQUIRED_COLUMNS:

        print(
            f"  - {column}"
        )

    print(
        "  - PRODUCER_TIMESTAMP"
    )

    print()

    print("=" * 80)
    print("DATA LEAKAGE POLICY")
    print("=" * 80)

    print(
        "  TX_FRAUD = ground truth ONLY"
    )

    print(
        "  TX_FRAUD is NOT used for:"
    )

    print(
        "    - feature engineering"
    )

    print(
        "    - customer history"
    )

    print(
        "    - terminal history"
    )

    print(
        "    - LSTM input"
    )

    print(
        "    - fraud probability"
    )

    print(
        "    - fraud prediction"
    )

    print()

    print(
        "  PRODUCER_TIMESTAMP = real replay time"
    )

    print(
        "  TX_DATETIME        = historical dataset time"
    )

    print()

    # ========================================================
    # CREATE KAFKA PRODUCER
    # ========================================================

    producer = None

    try:

        if not args.dry_run:

            producer = create_producer(
                args.bootstrap
            )

        # ====================================================
        # REPLAY
        # ====================================================

        cycle = 0

        while True:

            cycle += 1

            print()
            print(
                f"================ "
                f"CYCLE {cycle} "
                f"================"
            )

            sent = replay_once(
                producer=producer,
                files=files,
                topic=args.topic,
                rate=args.rate,
                limit=args.limit,
                dry_run=args.dry_run,
                print_every=args.print_every,
            )

            if not args.loop:

                break

            if sent == 0:

                print(
                    "No transactions sent."
                )

                break

            if args.dry_run:

                break

            print()
            print(
                "Dataset completed."
            )

            print(
                "Restarting from "
                "2018-10-01..."
            )

    except KeyboardInterrupt:

        print()
        print("=" * 80)
        print("PRODUCER STOPPED BY USER")
        print("=" * 80)

    except Exception as exc:

        print()
        print("=" * 80)
        print("PRODUCER ERROR")
        print("=" * 80)

        print(
            str(exc)
        )

        print()

        sys.exit(1)

    finally:

        if producer is not None:

            try:

                producer.flush()

                producer.close()

            except Exception:

                pass

        print()
        print(
            "Replay Producer stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()