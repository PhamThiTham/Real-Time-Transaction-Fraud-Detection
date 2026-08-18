# ============================================================
# FEATURE ENGINEERING - REAL-TIME FRAUD DETECTION
# ============================================================
#
# Input Kafka topic:
#     transactions
#
# Output Kafka topic:
#     transactions_features
#
# Creates the 15 features required by the LSTM:
#
#  1  TX_AMOUNT
#  2  TX_DURING_WEEKEND
#  3  TX_DURING_NIGHT
#  4  CUSTOMER_ID_NB_TX_1DAY_WINDOW
#  5  CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW
#  6  CUSTOMER_ID_NB_TX_7DAY_WINDOW
#  7  CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW
#  8  CUSTOMER_ID_NB_TX_30DAY_WINDOW
#  9  CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW
# 10  TERMINAL_ID_NB_TX_1DAY_WINDOW
# 11  TERMINAL_ID_RISK_1DAY_WINDOW
# 12  TERMINAL_ID_NB_TX_7DAY_WINDOW
# 13  TERMINAL_ID_RISK_7DAY_WINDOW
# 14  TERMINAL_ID_NB_TX_30DAY_WINDOW
# 15  TERMINAL_ID_RISK_30DAY_WINDOW
#
# Important:
# - No amount / amount_log / tx_hour / tx_weekday columns are created.
# - History features are calculated from transactions BEFORE the
#   current transaction, avoiding current-row leakage.
# - A stateful Python dictionary is used inside foreachBatch.
#   This is appropriate for a prototype with a bounded number of
#   customers / terminals.
# ============================================================

from collections import defaultdict, deque

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    from_json,
    to_json,
    struct,
    to_timestamp,
    when,
    dayofweek,
    hour,
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

INPUT_TOPIC = "transactions"
OUTPUT_TOPIC = "transactions_features"

CHECKPOINT_LOCATION = "/tmp/checkpoint/feature_engineering"

# Keep enough state for the largest 30-day window.
HISTORY_DAYS = 30


# ============================================================
# 2. INPUT SCHEMA
# ============================================================
#
# The producer may send TX_DATETIME as an ISO string.
# Therefore TX_DATETIME is parsed as StringType first and
# converted to TimestampType after JSON parsing.
#
# TX_FRAUD is kept because customer_history needs it as the
# target label during testing / evaluation.
# ============================================================

schema = StructType([
    StructField("TRANSACTION_ID", IntegerType(), True),
    StructField("PRODUCER_TIMESTAMP", StringType(), True),
    StructField("TX_DATETIME", StringType(), True),
    StructField("CUSTOMER_ID", IntegerType(), True),
    StructField("TERMINAL_ID", IntegerType(), True),
    StructField("TX_AMOUNT", DoubleType(), True),
    StructField("TX_FRAUD", IntegerType(), True),
])


# ============================================================
# 3. SPARK SESSION
# ============================================================

spark = (
    SparkSession
    .builder
    .appName("FeatureEngineering")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("=" * 100)
print("FEATURE ENGINEERING")
print("=" * 100)
print(f"Kafka bootstrap      : {KAFKA_BOOTSTRAP_SERVERS}")
print(f"Input topic          : {INPUT_TOPIC}")
print(f"Output topic         : {OUTPUT_TOPIC}")
print(f"History window       : {HISTORY_DAYS} days")
print("Number input features: 15")
print("=" * 100)


# ============================================================
# 4. STATE
# ============================================================
#
# customer_state[customer_id] = deque of:
#     (timestamp, amount)
#
# terminal_state[terminal_id] = deque of:
#     (timestamp, fraud)
#
# Only transactions from the previous 30 days are retained.
# ============================================================

customer_state = defaultdict(deque)
terminal_state = defaultdict(deque)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def remove_old_customer_transactions(customer_id, current_time):
    history = customer_state[customer_id]

    while history:
        age_seconds = (current_time - history[0][0]).total_seconds()

        if age_seconds > HISTORY_DAYS * 86400:
            history.popleft()
        else:
            break


def remove_old_terminal_transactions(terminal_id, current_time):
    history = terminal_state[terminal_id]

    while history:
        age_seconds = (current_time - history[0][0]).total_seconds()

        if age_seconds > HISTORY_DAYS * 86400:
            history.popleft()
        else:
            break


def customer_features(customer_id, current_time):
    """
    Calculate customer features using only previous transactions.
    """

    remove_old_customer_transactions(customer_id, current_time)

    history = customer_state[customer_id]

    result = {}

    for days in (1, 7, 30):
        cutoff_seconds = days * 86400

        values = [
            amount
            for timestamp, amount in history
            if (current_time - timestamp).total_seconds() <= cutoff_seconds
        ]

        count = len(values)

        if count > 0:
            avg_amount = sum(values) / count
        else:
            avg_amount = 0.0

        result[f"CUSTOMER_ID_NB_TX_{days}DAY_WINDOW"] = float(count)
        result[f"CUSTOMER_ID_AVG_AMOUNT_{days}DAY_WINDOW"] = float(avg_amount)

    return result


def terminal_features(terminal_id, current_time):
    """
    Calculate terminal features using only previous transactions.
    """

    remove_old_terminal_transactions(terminal_id, current_time)

    history = terminal_state[terminal_id]

    result = {}

    for days in (1, 7, 30):
        cutoff_seconds = days * 86400

        values = [
            fraud
            for timestamp, fraud in history
            if (current_time - timestamp).total_seconds() <= cutoff_seconds
        ]

        count = len(values)

        if count > 0:
            risk = sum(values) / count
        else:
            risk = 0.0

        result[f"TERMINAL_ID_NB_TX_{days}DAY_WINDOW"] = float(count)
        result[f"TERMINAL_ID_RISK_{days}DAY_WINDOW"] = float(risk)

    return result


# ============================================================
# 6. READ TRANSACTIONS FROM KAFKA
# ============================================================

raw_stream = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        KAFKA_BOOTSTRAP_SERVERS
    )
    .option(
        "subscribe",
        INPUT_TOPIC
    )
    .option(
        "startingOffsets",
        "latest"
    )
    .load()
)


# ============================================================
# 7. KAFKA VALUE -> JSON -> DATAFRAME
# ============================================================

json_stream = (
    raw_stream
    .select(
        col("value")
        .cast("string")
        .alias("json")
    )
)

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
# 8. CONVERT TX_DATETIME
# ============================================================

transactions = (
    transactions
    .withColumn(
        "TX_DATETIME",
        to_timestamp(col("TX_DATETIME"))
    )
)


# ============================================================
# 9. BASIC FEATURES
# ============================================================
#
# Only:
#   TX_DURING_WEEKEND
#   TX_DURING_NIGHT
#
# No amount / amount_log / tx_hour / tx_weekday.
# ============================================================

features = (
    transactions

    .withColumn(
        "TX_DURING_WEEKEND",
        when(
            dayofweek(col("TX_DATETIME")).isin(1, 7),
            1
        ).otherwise(0)
    )

    .withColumn(
        "TX_DURING_NIGHT",
        when(
            hour(col("TX_DATETIME")) <= 6,
            1
        ).otherwise(0)
    )
)


# ============================================================
# 10. PROCESS MICRO-BATCH
# ============================================================

def process_batch(batch_df, batch_id):

    print()
    print("=" * 100)
    print(f"FEATURE ENGINEERING - BATCH ID = {batch_id}")
    print("=" * 100)

    if batch_df.isEmpty():
        print("NO TRANSACTIONS IN THIS BATCH")
        return

    rows = batch_df.collect()

    # Process chronologically.
    rows = sorted(
        rows,
        key=lambda row: (
            row["TX_DATETIME"]
            if row["TX_DATETIME"] is not None
            else 0
        )
    )

    output_rows = []

    for row in rows:

        transaction_id = row["TRANSACTION_ID"]
        tx_datetime = row["TX_DATETIME"]
        producer_timestamp = row["PRODUCER_TIMESTAMP"]
        customer_id = row["CUSTOMER_ID"]
        terminal_id = row["TERMINAL_ID"]
        amount = row["TX_AMOUNT"]
        fraud = row["TX_FRAUD"]

        if tx_datetime is None:
            print(
                f"SKIP transaction {transaction_id}: "
                f"TX_DATETIME is NULL"
            )
            continue

        if customer_id is None:
            print(
                f"SKIP transaction {transaction_id}: "
                f"CUSTOMER_ID is NULL"
            )
            continue

        if terminal_id is None:
            print(
                f"SKIP transaction {transaction_id}: "
                f"TERMINAL_ID is NULL"
            )
            continue

        if amount is None:
            amount = 0.0

        if fraud is None:
            fraud = 0

        # ----------------------------------------------------
        # IMPORTANT:
        # Calculate history features BEFORE adding current tx.
        # This prevents current transaction leakage.
        # ----------------------------------------------------

        c_features = customer_features(
            customer_id,
            tx_datetime
        )

        t_features = terminal_features(
            terminal_id,
            tx_datetime
        )

        output_row = {
            "TRANSACTION_ID": int(transaction_id),
            "PRODUCER_TIMESTAMP": producer_timestamp,
            "TX_DATETIME": tx_datetime,
            "CUSTOMER_ID": int(customer_id),
            "TERMINAL_ID": int(terminal_id),

            "TX_AMOUNT": float(amount),

            "TX_DURING_WEEKEND": int(
                row["TX_DURING_WEEKEND"]
                if row["TX_DURING_WEEKEND"] is not None
                else 0
            ),

            "TX_DURING_NIGHT": int(
                row["TX_DURING_NIGHT"]
                if row["TX_DURING_NIGHT"] is not None
                else 0
            ),

            **c_features,
            **t_features,

            "TX_FRAUD": int(fraud),
        }

        output_rows.append(output_row)

        # ----------------------------------------------------
        # Print features for verification.
        # ----------------------------------------------------

        print()
        print("-" * 100)
        print(f"TRANSACTION_ID : {transaction_id}")
        print(f"CUSTOMER_ID    : {customer_id}")
        print(f"TERMINAL_ID    : {terminal_id}")
        print(f"TX_DATETIME    : {tx_datetime}")
        print(f"TX_AMOUNT      : {amount:.2f}")
        print(f"TX_FRAUD       : {fraud}")
        print()
        print("CUSTOMER FEATURES:")
        for key in c_features:
            print(f"  {key:<42}: {c_features[key]:.4f}")
        print()
        print("TERMINAL FEATURES:")
        for key in t_features:
            print(f"  {key:<42}: {t_features[key]:.4f}")

        # ----------------------------------------------------
        # Add CURRENT transaction to history only AFTER
        # feature calculation.
        # ----------------------------------------------------

        customer_state[customer_id].append(
            (tx_datetime, float(amount))
        )

        terminal_state[terminal_id].append(
            (tx_datetime, int(fraud))
        )

        remove_old_customer_transactions(
            customer_id,
            tx_datetime
        )

        remove_old_terminal_transactions(
            terminal_id,
            tx_datetime
        )

    if not output_rows:
        return

    # ========================================================
    # CREATE OUTPUT DATAFRAME
    # ========================================================

    output_schema = StructType([
        StructField("TRANSACTION_ID", IntegerType(), False),
        StructField("PRODUCER_TIMESTAMP", StringType(), True),
        StructField("TX_DATETIME", TimestampType(), True),
        StructField("CUSTOMER_ID", IntegerType(), True),
        StructField("TERMINAL_ID", IntegerType(), True),

        StructField("TX_AMOUNT", DoubleType(), True),
        StructField("TX_DURING_WEEKEND", IntegerType(), True),
        StructField("TX_DURING_NIGHT", IntegerType(), True),

        StructField("CUSTOMER_ID_NB_TX_1DAY_WINDOW", DoubleType(), True),
        StructField("CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW", DoubleType(), True),
        StructField("CUSTOMER_ID_NB_TX_7DAY_WINDOW", DoubleType(), True),
        StructField("CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW", DoubleType(), True),
        StructField("CUSTOMER_ID_NB_TX_30DAY_WINDOW", DoubleType(), True),
        StructField("CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW", DoubleType(), True),

        StructField("TERMINAL_ID_NB_TX_1DAY_WINDOW", DoubleType(), True),
        StructField("TERMINAL_ID_RISK_1DAY_WINDOW", DoubleType(), True),
        StructField("TERMINAL_ID_NB_TX_7DAY_WINDOW", DoubleType(), True),
        StructField("TERMINAL_ID_RISK_7DAY_WINDOW", DoubleType(), True),
        StructField("TERMINAL_ID_NB_TX_30DAY_WINDOW", DoubleType(), True),
        StructField("TERMINAL_ID_RISK_30DAY_WINDOW", DoubleType(), True),

        StructField("TX_FRAUD", IntegerType(), True),
    ])

    output_df = spark.createDataFrame(
        output_rows,
        schema=output_schema
    )

    # ========================================================
    # WRITE TO KAFKA
    # ========================================================

    kafka_output = (
        output_df
        .select(
            col("TRANSACTION_ID").cast("string").alias("key"),
            to_json(
                struct(
                    *[
                        col(c)
                        for c in output_df.columns
                    ]
                )
            ).alias("value")
        )
    )

    (
        kafka_output
        .write
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            KAFKA_BOOTSTRAP_SERVERS
        )
        .option(
            "topic",
            OUTPUT_TOPIC
        )
        .save()
    )

    print()
    print("=" * 100)
    print(
        f"WROTE {len(output_rows)} FEATURED TRANSACTION(S) "
        f"TO KAFKA TOPIC: {OUTPUT_TOPIC}"
    )
    print("=" * 100)


# ============================================================
# 11. START STREAMING
# ============================================================

query = (
    features
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
