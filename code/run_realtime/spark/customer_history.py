# ============================================================
# CUSTOMER HISTORY - REAL-TIME FRAUD DETECTION
# ============================================================
#
# Input Kafka topic:
#     transactions_features
#
# Builds:
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
# This avoids target/current-row leakage.
#
# Expected LSTM input:
#
#     (5, 15)
#
# ============================================================

from collections import defaultdict

import time


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
    StringType,
    TimestampType,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"

KAFKA_TOPIC = "transactions_features"

SEQ_LEN = 5

OUTPUT_FEATURE = "TX_FRAUD"

CHECKPOINT_LOCATION = "/opt/spark-data/checkpoint/customer_history"


# ============================================================
# 1.5 RUN CONTROL
# ============================================================
#
# VERBOSE      : if False, per-transaction debug prints are hidden.
# END_MARKER   : special Kafka message that signals completion.
# END_RECEIVED : set to True when this stage receives the marker.
#
# ============================================================

VERBOSE = False

END_MARKER = "__END_OF_STREAM__"

END_RECEIVED = False


def log(
    *args,
    **kwargs
):

    if VERBOSE:

        log(
            *args,
            **kwargs
        )


# ============================================================
# 2. 15 LSTM INPUT FEATURES
# ============================================================

input_features = [

    # --------------------------------------------------------
    # Transaction features
    # --------------------------------------------------------

    "TX_AMOUNT",
    "TX_DURING_WEEKEND",
    "TX_DURING_NIGHT",

    # --------------------------------------------------------
    # Customer features
    # --------------------------------------------------------

    "CUSTOMER_ID_NB_TX_1DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW",

    "CUSTOMER_ID_NB_TX_7DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW",

    "CUSTOMER_ID_NB_TX_30DAY_WINDOW",
    "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW",

    # --------------------------------------------------------
    # Terminal features
    # --------------------------------------------------------

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
print(f"Checkpoint           : {CHECKPOINT_LOCATION}")

print()

print("INPUT FEATURES:")

for i, feature in enumerate(input_features, start=1):
    print(f"  {i:02d}. {feature}")

print("=" * 100)


# ============================================================
# 4. VALIDATE NUMBER OF FEATURES
# ============================================================

if len(input_features) != 15:

    raise ValueError(
        f"Expected 15 input features, "
        f"but got {len(input_features)}"
    )


# ============================================================
# 5. SPARK SESSION
# ============================================================

spark = (
    SparkSession
    .builder
    .appName("CustomerHistory")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 6. INPUT SCHEMA
# ============================================================
#
# IMPORTANT:
#
# feature_engineering.py writes:
#
#     PRODUCER_TIMESTAMP -> StringType
#
# Therefore this file MUST read it as StringType.
#
# TX_DATETIME remains TimestampType because it is converted
# to timestamp in feature_engineering.py before being written
# to Kafka.
#
# ============================================================

schema = StructType([

    StructField(
        "TRANSACTION_ID",
        IntegerType(),
        True
    ),

    StructField(
        "PRODUCER_TIMESTAMP",
        StringType(),
        True
    ),

    StructField(
        "TX_DATETIME",
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
# 7. CUSTOMER HISTORY
# ============================================================
#
# customer_history[customer_id] =
#
#     latest 5 transactions
#
# Each transaction contains:
#
#     metadata
#     15 features
#     TX_FRAUD
#
# ============================================================

customer_history = defaultdict(list)


# ============================================================
# 8. READ FROM KAFKA
# ============================================================
#
# IMPORTANT:
#
# We use "earliest" because we are resetting the pipeline
# and want to process all transactions currently available
# in transactions_features.
#
# The checkpoint determines where Spark actually resumes.
#
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
    .option(
        "maxOffsetsPerTrigger",
        5000
    )
    .option(
        "failOnDataLoss",
        "true"
    )
    .load()
)


# ============================================================
# 9. KAFKA VALUE -> JSON
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
# 10. PARSE JSON
# ============================================================

transactions = (
    json_stream
    .select(
        col("json").alias("_RAW_VALUE"),
        from_json(
            col("json"),
            schema
        ).alias("data")
    )
    .select("_RAW_VALUE", "data.*")
)


# ============================================================
# 11. PROCESS EACH MICRO-BATCH
# ============================================================

def process_batch(batch_df, batch_id):

    global END_RECEIVED

    print()
    print("=" * 100)
    print(f"CUSTOMER HISTORY - BATCH ID = {batch_id}")
    print("=" * 100)

    # --------------------------------------------------------
    # Check whether batch is empty
    # --------------------------------------------------------

    if batch_df.isEmpty():

        log("NO TRANSACTIONS IN THIS BATCH")
        log("=" * 100)

        return


    # --------------------------------------------------------
    # Collect rows
    # --------------------------------------------------------

    rows = batch_df.collect()

    # --------------------------------------------------------
    # END MARKER DETECTION
    # --------------------------------------------------------

    end_marker_present = any(
        row["_RAW_VALUE"] == END_MARKER
        for row in rows
    )

    rows = [
        row
        for row in rows
        if row["_RAW_VALUE"] != END_MARKER
    ]

    if not rows:

        if end_marker_present:

            print()
            print("=" * 100)
            print(
                "[ALL DONE] "
                "CUSTOMER_HISTORY"
            )
            print("=" * 100)

            END_RECEIVED = True

        return

    print(
        f"NUMBER OF ROWS = {len(rows)}"
    )


    # --------------------------------------------------------
    # Process chronologically
    # --------------------------------------------------------

    rows = sorted(
        rows,
        key=lambda row: (
            row["TX_DATETIME"]
            if row["TX_DATETIME"] is not None
            else 0
        )
    )


    # ========================================================
    # PROCESS EACH TRANSACTION
    # ========================================================

    for row in rows:

        transaction_id = row["TRANSACTION_ID"]

        tx_datetime = row["TX_DATETIME"]

        producer_timestamp = row["PRODUCER_TIMESTAMP"]

        customer_id = row["CUSTOMER_ID"]

        terminal_id = row["TERMINAL_ID"]

        current_label = row["TX_FRAUD"]

        amount = row["TX_AMOUNT"]


        # ====================================================
        # VALIDATION
        # ====================================================

        if transaction_id is None:

            log(
                "SKIP: TRANSACTION_ID is NULL"
            )

            continue


        if tx_datetime is None:

            log(
                f"SKIP transaction {transaction_id}: "
                f"TX_DATETIME is NULL"
            )

            continue


        if customer_id is None:

            log(
                f"SKIP transaction {transaction_id}: "
                f"CUSTOMER_ID is NULL"
            )

            continue


        if current_label is None:

            current_label = 0


        if amount is None:

            amount = 0.0


        if terminal_id is None:

            terminal_id = 0


        # ====================================================
        # PRINT CURRENT TRANSACTION
        # ====================================================

        log()
        log("-" * 100)

        log(
            f"CUSTOMER_ID        : "
            f"{customer_id}"
        )

        log(
            f"CURRENT TRANSACTION : "
            f"{transaction_id}"
        )

        log(
            f"TX_DATETIME         : "
            f"{tx_datetime}"
        )

        log(
            f"PRODUCER_TIMESTAMP  : "
            f"{producer_timestamp}"
        )

        log(
            f"TX_AMOUNT           : "
            f"{amount:.2f}"
        )

        log(
            f"TX_FRAUD            : "
            f"{current_label}"
        )


        # ====================================================
        # CREATE CURRENT TRANSACTION OBJECT
        # ====================================================

        transaction = {

            "TRANSACTION_ID":
                int(transaction_id),

            "TX_DATETIME":
                tx_datetime,

            "PRODUCER_TIMESTAMP":
                producer_timestamp,

            "CUSTOMER_ID":
                int(customer_id),

            "TERMINAL_ID":
                int(terminal_id),

            "TX_FRAUD":
                int(current_label),
        }


        # ====================================================
        # STORE 15 FEATURES
        # ====================================================

        for feature in input_features:

            value = row[feature]

            if value is None:

                value = 0.0

            transaction[feature] = float(value)


        # ====================================================
        # GET PREVIOUS CUSTOMER HISTORY
        # ====================================================

        history = customer_history[customer_id]

        log()
        log(
            f"HISTORY LENGTH     : "
            f"{len(history)}"
        )


        # ====================================================
        # PRINT HISTORY
        # ====================================================

        log()
        log("PREVIOUS CUSTOMER HISTORY:")

        if len(history) == 0:

            log(
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

            remaining = (
                SEQ_LEN - len(history)
            )

            log()
            log(
                "STATUS              : "
                f"WAITING FOR {remaining} "
                f"MORE TRANSACTION(S)"
            )


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Add CURRENT transaction only AFTER checking
            # whether 5 previous transactions exist.
            #
            # ------------------------------------------------

            customer_history[
                customer_id
            ].append(
                transaction
            )


            customer_history[
                customer_id
            ] = (
                customer_history[
                    customer_id
                ][-SEQ_LEN:]
            )


            log("=" * 100)

            continue


        # ====================================================
        # BUILD LSTM SEQUENCE
        # ====================================================
        #
        # history contains:
        #
        #     T1 T2 T3 T4 T5
        #
        # current transaction:
        #
        #     T6
        #
        # Therefore:
        #
        # X = T1 T2 T3 T4 T5
        #
        # y = T6 TX_FRAUD
        #
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


            sequence.append(
                feature_vector
            )


        sequence_length = len(sequence)

        number_features = len(input_features)


        # ====================================================
        # VERIFY INPUT SHAPE
        # ====================================================

        if sequence_length != SEQ_LEN:

            log(
                "ERROR: INVALID SEQUENCE LENGTH"
            )

            log(
                f"Expected: {SEQ_LEN}"
            )

            log(
                f"Actual  : {sequence_length}"
            )

            continue


        if number_features != 15:

            log(
                "ERROR: INVALID FEATURE COUNT"
            )

            log(
                f"Expected: 15"
            )

            log(
                f"Actual  : {number_features}"
            )

            continue


        # ====================================================
        # READY FOR LSTM
        # ====================================================

        log()
        log("=" * 100)
        log("READY FOR LSTM")
        log("=" * 100)

        log(
            f"CUSTOMER_ID       : "
            f"{customer_id}"
        )

        log(
            f"SEQUENCE LENGTH   : "
            f"{sequence_length}"
        )

        log(
            f"NUMBER OF FEATURES: "
            f"{number_features}"
        )

        log(
            f"INPUT SHAPE       : "
            f"({sequence_length}, "
            f"{number_features})"
        )


        # ====================================================
        # TRANSACTION IDS
        # ====================================================

        log()
        log("TRANSACTION IDS:")

        log(
            [
                tx["TRANSACTION_ID"]
                for tx in history
            ]
        )


        # ====================================================
        # FEATURE SEQUENCE
        # ====================================================

        log()
        log("FEATURE SEQUENCE:")

        for i, feature_vector in enumerate(
            sequence,
            start=1
        ):

            log(
                f"  T{i}: "
                f"{feature_vector}"
            )


        # ====================================================
        # TARGET
        # ====================================================

        log()
        log("TARGET TRANSACTION:")

        log(
            f"  TRANSACTION_ID : "
            f"{transaction_id}"
        )

        log(
            f"  OUTPUT FEATURE : "
            f"{OUTPUT_FEATURE}"
        )

        log(
            f"  OUTPUT VALUE   : "
            f"{current_label}"
        )


        # ====================================================
        # FEATURE SANITY CHECK
        # ====================================================

        non_zero_count = sum(

            1

            for feature_vector in sequence

            for value in feature_vector

            if float(value) != 0.0
        )


        total_values = (
            sequence_length *
            number_features
        )


        log()
        log(
            "FEATURE SANITY CHECK:"
        )

        log(
            f"  Non-zero values : "
            f"{non_zero_count}/"
            f"{total_values}"
        )


        if non_zero_count == 0:

            log(
                "  WARNING: ALL LSTM INPUT "
                "VALUES ARE 0.0"
            )

            log(
                "  Check:"
            )

            log(
                "    1. feature_engineering.py"
            )

            log(
                "    2. Kafka topic "
                "transactions_features"
            )

        else:

            log(
                "  OK: FEATURE HISTORY "
                "CONTAINS NON-ZERO VALUES."
            )


        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        log()
        log("-" * 100)
        log("LSTM INPUT / TARGET")

        log(
            f"X shape : "
            f"({sequence_length}, "
            f"{number_features})"
        )

        log(
            f"y      : "
            f"{current_label}"
        )

        log("-" * 100)

        log(
            "READY FOR LSTM INFERENCE"
        )

        log("=" * 100)


        # ====================================================
        # ADD CURRENT TRANSACTION TO HISTORY
        # ====================================================
        #
        # VERY IMPORTANT:
        #
        # This happens AFTER X is built.
        #
        # Therefore:
        #
        # current transaction T6
        #
        # cannot leak into:
        #
        # X = T1 T2 T3 T4 T5
        #
        # ====================================================

        customer_history[
            customer_id
        ].append(
            transaction
        )


        customer_history[
            customer_id
        ] = (
            customer_history[
                customer_id
            ][-SEQ_LEN:]
        )


# ============================================================
# 12. START STREAMING
# ============================================================

query = (

    transactions

    .writeStream

    .foreachBatch(
        process_batch
    )

    .outputMode(
        "append"
    )

    .option(
        "checkpointLocation",
        CHECKPOINT_LOCATION
    )

    .start()
)


# ============================================================
# 13. WAIT
# ============================================================

# ============================================================
# WAIT FOR END MARKER / COMPLETION
# ============================================================

print()
print("=" * 100)

print(
    "[READY] CUSTOMER_HISTORY - "
    f"listening to topic '{KAFKA_TOPIC}'"
)

print("=" * 100)


while (
    query.isActive
    and not END_RECEIVED
):

    time.sleep(1)


query.stop()


print()
print("=" * 100)

print(
    "[STOPPED] CUSTOMER_HISTORY"
)

print("=" * 100)