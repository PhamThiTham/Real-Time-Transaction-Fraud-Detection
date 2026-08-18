# ============================================================
# CUSTOMER HISTORY - REAL-TIME FRAUD DETECTION
# ============================================================
#
# Input Kafka topic:
#     transactions_features
#
# This script receives the 15 already-computed features from
# feature_engineering.py and builds:
#
#     X = 5 PREVIOUS transactions
#     y = CURRENT transaction TX_FRAUD
#
# Therefore:
#
#     T1 T2 T3 T4 T5 -> X
#     T6             -> y
#
# The current transaction is NOT inserted into X.
# This avoids target / current-row leakage.
#
# Expected LSTM input:
#
#     (5, 15)
#
# ============================================================

from collections import defaultdict

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    TimestampType,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"

KAFKA_TOPIC = "transactions_features"

SEQ_LEN = 5

OUTPUT_FEATURE = "TX_FRAUD"

CHECKPOINT_LOCATION = "/tmp/checkpoint/customer_history"


# ============================================================
# 2. 15 LSTM INPUT FEATURES
# ============================================================

input_features = [

    # Transaction features
    "TX_AMOUNT",
    "TX_DURING_WEEKEND",
    "TX_DURING_NIGHT",

    # Customer features
    "CUSTOMER_ID_NB_TX_1DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW",

    "CUSTOMER_ID_NB_TX_7DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW",

    "CUSTOMER_ID_NB_TX_30DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW",

    # Terminal features
    "TERMINAL_ID_NB_TX_1DAY_WINDOW",
    "TERMINAL_ID_RISK_1DAY_WINDOW",

    "TERMINAL_ID_NB_TX_7DAY_WINDOW",
    "TERMINAL_ID_RISK_7DAY_WINDOW",

    "TERMINAL_ID_NB_TX_30DAY_WINDOW",
    "TERMINAL_ID_RISK_30DAY_WINDOW",
]


# ============================================================
# 3. PRINT CONFIGURATION
# ============================================================

print("=" * 100)
print("CUSTOMER HISTORY")
print("=" * 100)
print(f"Kafka                : {KAFKA_BOOTSTRAP_SERVERS}")
print(f"Topic                : {KAFKA_TOPIC}")
print(f"Sequence length      : {SEQ_LEN}")
print(f"Number input features: {len(input_features)}")
print(f"Output feature       : {OUTPUT_FEATURE}")
print()
print("INPUT FEATURES:")

for i, feature in enumerate(input_features, start=1):
    print(f"  {i:02d}. {feature}")

print("=" * 100)


# ============================================================
# 4. SPARK SESSION
# ============================================================

spark = (
    SparkSession
    .builder
    .appName("CustomerHistory")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 5. INPUT SCHEMA
# ============================================================

schema = StructType([

    StructField(
        "TRANSACTION_ID",
        IntegerType(),
        True
    ),

    StructField(
        "TX_DATETIME",
        TimestampType(),
        True
    ),

    StructField(
        "PRODUCER_TIMESTAMP",
        TimestampType(),
        True
    ),

    StructField(
        "CUSTOMER_ID",
        IntegerType(),
        True
    ),

    StructField(
        "TERMINAL_ID",
        IntegerType(),
        True
    ),

    StructField(
        "TX_AMOUNT",
        DoubleType(),
        True
    ),

    StructField(
        "TX_DURING_WEEKEND",
        IntegerType(),
        True
    ),

    StructField(
        "TX_DURING_NIGHT",
        IntegerType(),
        True
    ),

    StructField(
        "CUSTOMER_ID_NB_TX_1DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "CUSTOMER_ID_NB_TX_7DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "CUSTOMER_ID_NB_TX_30DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "TERMINAL_ID_NB_TX_1DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "TERMINAL_ID_RISK_1DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "TERMINAL_ID_NB_TX_7DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "TERMINAL_ID_RISK_7DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "TERMINAL_ID_NB_TX_30DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "TERMINAL_ID_RISK_30DAY_WINDOW",
        DoubleType(),
        True
    ),

    StructField(
        "TX_FRAUD",
        IntegerType(),
        True
    ),
])


# ============================================================
# 6. CUSTOMER HISTORY
# ============================================================
#
# customer_history[customer_id] = latest 5 transactions
#
# Each transaction stores:
#   metadata
#   15 features
#   TX_FRAUD
# ============================================================

customer_history = defaultdict(list)


# ============================================================
# 7. READ FROM KAFKA
# ============================================================

raw_stream = (
    spark
    .readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        KAFKA_BOOTSTRAP_SERVERS
    )
    .option(
        "subscribe",
        KAFKA_TOPIC
    )
    .option(
        "startingOffsets",
        "earliest"
    )
    .load()
)


# ============================================================
# 8. KAFKA VALUE -> JSON
# ============================================================

json_stream = (
    raw_stream
    .select(
        col("value")
        .cast("string")
        .alias("json")
    )
)


# ============================================================
# 9. PARSE JSON
# ============================================================

transactions = (
    json_stream
    .select(
        from_json(
            col("json"),
            schema
        ).alias("data")
    )
    .select("data.*")
)


# ============================================================
# 10. PROCESS EACH MICRO-BATCH
# ============================================================

def process_batch(batch_df, batch_id):

    print()
    print("=" * 100)
    print(f"BATCH ID = {batch_id}")
    print("=" * 100)

    print(f"NUMBER OF ROWS = {batch_df.count()}")

    if batch_df.isEmpty():
        print("NO TRANSACTIONS IN THIS BATCH")
        return

    batch_df.select(
        "TRANSACTION_ID",
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TX_AMOUNT",
        "TX_FRAUD"
    ).show(n=20,truncate=False)


    rows = batch_df.collect()

    # --------------------------------------------------------
    # Always process transactions chronologically.
    # --------------------------------------------------------

    rows = sorted(
        rows,
        key=lambda row: (
            row["TX_DATETIME"]
            if row["TX_DATETIME"] is not None
            else 0
        )
    )

    for row in rows:

        transaction_id = row["TRANSACTION_ID"]
        tx_datetime = row["TX_DATETIME"]
        producer_timestamp = row["PRODUCER_TIMESTAMP"]
        customer_id = row["CUSTOMER_ID"]
        terminal_id = row["TERMINAL_ID"]
        current_label = row["TX_FRAUD"]
        amount = row["TX_AMOUNT"]

        if customer_id is None:
            print("SKIP: CUSTOMER_ID is NULL")
            continue

        if current_label is None:
            current_label = 0

        if amount is None:
            amount = 0.0

        print()
        print("-" * 100)

        print(
            f"CUSTOMER_ID       : "
            f"{customer_id}"
        )

        print(
            f"CURRENT TRANSACTION: "
            f"{transaction_id}"
        )

        print(
            f"TX_DATETIME       : "
            f"{tx_datetime}"
        )

        print(
            f"PRODUCER_TIMESTAMP: "
            f"{producer_timestamp}"
        )

        print(
            f"TX_AMOUNT         : "
            f"{amount:.2f}"
        )

        print(
            f"TX_FRAUD          : "
            f"{current_label}"
        )

        # ====================================================
        # CREATE CURRENT TRANSACTION OBJECT
        # ====================================================

        transaction = {
            "TRANSACTION_ID": transaction_id,
            "TX_DATETIME": tx_datetime,
            "PRODUCER_TIMESTAMP": producer_timestamp,
            "CUSTOMER_ID": customer_id,
            "TERMINAL_ID": terminal_id,
            "TX_FRAUD": current_label,
        }

        # ----------------------------------------------------
        # Store all 15 features exactly as received from
        # feature_engineering.py.
        # ----------------------------------------------------

        for feature in input_features:

            value = row[feature]

            if value is None:
                value = 0.0

            transaction[feature] = float(value)

        # ====================================================
        # GET PREVIOUS CUSTOMER HISTORY
        # ====================================================

        history = customer_history[customer_id]

        print(
            f"HISTORY LENGTH    : "
            f"{len(history)}"
        )

        # ====================================================
        # PRINT HISTORY
        # ====================================================

        print()
        print("SEQUENCE:")

        if len(history) == 0:

            print(
                "  No previous transaction"
            )

        else:

            for i, tx in enumerate(
                history,
                start=1
            ):

                print(
                    f"  [{i}] "
                    f"id={tx['TRANSACTION_ID']} "
                    f"| time={tx['TX_DATETIME']} "
                    f"| amount={tx['TX_AMOUNT']:.2f} "
                    f"| fraud={tx['TX_FRAUD']}"
                )

        # ====================================================
        # WAIT UNTIL 5 PREVIOUS TRANSACTIONS EXIST
        # ====================================================

        if len(history) < SEQ_LEN:

            remaining = SEQ_LEN - len(history)

            print()
            print(
                "STATUS            : "
                f"WAITING FOR {remaining} "
                f"MORE TRANSACTION(S)"
            )

            # Add current transaction AFTER checking.
            customer_history[customer_id].append(
                transaction
            )

            customer_history[customer_id] = (
                customer_history[customer_id][-SEQ_LEN:]
            )

            print("=" * 100)

            continue

        # ====================================================
        # BUILD LSTM SEQUENCE
        # ====================================================
        #
        # IMPORTANT:
        #
        # history contains ONLY previous transactions.
        #
        # current transaction is y.
        #
        # X = history[0:5]
        # y = current_label
        # ====================================================

        sequence = []

        for tx in history:

            feature_vector = []

            for feature in input_features:

                value = tx[feature]

                if value is None:
                    value = 0.0

                feature_vector.append(
                    float(value)
                )

            sequence.append(feature_vector)

        sequence_length = len(sequence)
        number_features = len(input_features)

        # ====================================================
        # READY FOR LSTM
        # ====================================================

        print()
        print("=" * 100)
        print("READY FOR LSTM")
        print("=" * 100)

        print(
            f"CUSTOMER_ID      : "
            f"{customer_id}"
        )

        print(
            f"SEQUENCE LENGTH   : "
            f"{sequence_length}"
        )

        print(
            f"NUMBER OF FEATURES: "
            f"{number_features}"
        )

        print(
            f"INPUT SHAPE       : "
            f"({sequence_length}, "
            f"{number_features})"
        )

        # ====================================================
        # TRANSACTION IDS
        # ====================================================

        print()
        print("TRANSACTION IDS:")

        print([
            tx["TRANSACTION_ID"]
            for tx in history
        ])

        # ====================================================
        # FEATURE SEQUENCE
        # ====================================================

        print()
        print("FEATURE SEQUENCE:")

        for i, feature_vector in enumerate(
            sequence,
            start=1
        ):

            print(
                f"  T{i}: "
                f"{feature_vector}"
            )

        # ====================================================
        # TARGET
        # ====================================================

        print()
        print("TARGET TRANSACTION:")

        print(
            f"  TRANSACTION_ID : "
            f"{transaction_id}"
        )

        print(
            f"  OUTPUT FEATURE : "
            f"{OUTPUT_FEATURE}"
        )

        print(
            f"  OUTPUT VALUE   : "
            f"{current_label}"
        )

        # ====================================================
        # FEATURE SANITY CHECK
        # ====================================================
        #
        # This is useful to verify that the feature engineering
        # stage is actually supplying non-zero history features.
        # ====================================================

        non_zero_count = sum(
            1
            for feature_vector in sequence
            for value in feature_vector
            if float(value) != 0.0
        )

        total_values = (
            sequence_length * number_features
        )

        print()
        print(
            "FEATURE SANITY CHECK:"
        )

        print(
            f"  Non-zero values : "
            f"{non_zero_count}/{total_values}"
        )

        if non_zero_count == 0:

            print(
                "  WARNING: ALL LSTM INPUT VALUES ARE 0.0"
            )

            print(
                "  Check feature_engineering.py and "
                "Kafka topic transactions_features."
            )

        else:

            print(
                "  OK: FEATURE HISTORY CONTAINS "
                "NON-ZERO VALUES."
            )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        print()
        print("-" * 100)
        print("LSTM INPUT / TARGET")

        print(
            f"X shape : "
            f"({sequence_length}, "
            f"{number_features})"
        )

        print(
            f"y      : "
            f"{current_label}"
        )

        print("-" * 100)

        print(
            "READY FOR LSTM INFERENCE"
        )

        print("=" * 100)

        # ====================================================
        # ADD CURRENT TRANSACTION TO HISTORY
        # ====================================================
        #
        # This happens AFTER X is built.
        #
        # Therefore the current transaction cannot leak into
        # its own input sequence.
        # ====================================================

        customer_history[customer_id].append(
            transaction
        )

        customer_history[customer_id] = (
            customer_history[customer_id][-SEQ_LEN:]
        )


# ============================================================
# 11. START STREAMING
# ============================================================

query = (
    transactions
    .writeStream
    .foreachBatch(process_batch)
    .outputMode("update")
    .option(
        "checkpointLocation",
        CHECKPOINT_LOCATION
    )
    .start()
)


# ============================================================
# 12. WAIT
# ============================================================

query.awaitTermination()
