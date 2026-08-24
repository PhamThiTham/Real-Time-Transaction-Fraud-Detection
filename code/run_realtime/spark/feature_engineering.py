# ============================================================
# FEATURE ENGINEERING - REAL-TIME FRAUD DETECTION
# ============================================================
#
# INPUT:
#     Kafka topic:
#         transactions
#
# OUTPUT:
#     Kafka topic:
#         transactions_features
#
# FEATURES:
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
# IMPORTANT:
#
# 1. TRANSACTION_ID is preserved from Producer/Kafka.
#
# 2. PRODUCER_TIMESTAMP is preserved from Producer/Kafka.
#
# 3. TX_DATETIME is the original dataset transaction time.
#
# 4. History features are calculated BEFORE the current
#    transaction is inserted into state.
#
#    Therefore current transaction leakage is avoided.
#
# 5. Kafka starts from EARLIEST when no checkpoint exists.
#
# 6. maxOffsetsPerTrigger limits each micro-batch.
#
# 7. Checkpoint is stored in /opt/spark-data so it survives
#    Spark application restart when the directory is mounted.
#
# 8. Invalid transactions are counted and reported.
#
# ============================================================


from collections import defaultdict, deque

import time


from pyspark.sql import SparkSession


from pyspark.sql.functions import (
    col,
    from_json,
    lit,
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


# IMPORTANT:
# Persistent checkpoint location.
#
# DO NOT use /tmp for production/realtime experiments.
#
CHECKPOINT_LOCATION = (
    "/opt/spark-data/checkpoint/feature_engineering"
)


# Maximum history required by the feature engineering.
HISTORY_DAYS = 30


# Maximum number of Kafka records read per micro-batch.
#
# This prevents a very large batch from being collected
# completely into Python memory.
#
MAX_OFFSETS_PER_TRIGGER = 20000


# ============================================================
# RUN CONTROL
# ============================================================
#
# VERBOSE      : if False, per-transaction debug prints are hidden.
# END_MARKER   : special Kafka message that travels through the
#                whole pipeline to signal completion.
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

        print(
            *args,
            **kwargs
        )


def write_end_marker():

    marker_df = (
        spark
        .range(1)
        .select(
            lit(END_MARKER).alias("value")
        )
    )

    (
        marker_df
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

    print(
        "END MARKER WROTE TO "
        f"TOPIC {OUTPUT_TOPIC}"
    )


# ============================================================
# 2. INPUT SCHEMA
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
        StringType(),
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
        "TX_FRAUD",
        IntegerType(),
        True
    ),
])


# ============================================================
# 3. OUTPUT SCHEMA
# ============================================================

output_schema = StructType([

    StructField(
        "TRANSACTION_ID",
        IntegerType(),
        False
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

    # --------------------------------------------------------
    # BASIC FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CUSTOMER FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TERMINAL FEATURES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    StructField(
        "TX_FRAUD",
        IntegerType(),
        True
    ),
])


# ============================================================
# 4. SPARK SESSION
# ============================================================

spark = (
    SparkSession
    .builder
    .appName(
        "FeatureEngineering"
    )
    .getOrCreate()
)


spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 5. START INFORMATION
# ============================================================

print()
print("=" * 100)
print("FEATURE ENGINEERING - REAL-TIME FRAUD DETECTION")
print("=" * 100)

print()
print(
    f"Kafka bootstrap       : "
    f"{KAFKA_BOOTSTRAP_SERVERS}"
)

print(
    f"Input topic           : "
    f"{INPUT_TOPIC}"
)

print(
    f"Output topic          : "
    f"{OUTPUT_TOPIC}"
)

print(
    f"Checkpoint            : "
    f"{CHECKPOINT_LOCATION}"
)

print(
    f"History window        : "
    f"{HISTORY_DAYS} days"
)

print(
    f"Max offsets/trigger   : "
    f"{MAX_OFFSETS_PER_TRIGGER}"
)

print(
    "Starting offsets      : earliest "
    "(when checkpoint does not exist)"
)

print(
    "Output mode           : append"
)

print()
print(
    "Number of LSTM features: 15"
)

print("=" * 100)
print()


# ============================================================
# 6. STATE
# ============================================================
#
# customer_state:
#
#     customer_id
#         |
#         +-- (timestamp, amount)
#         +-- (timestamp, amount)
#         +-- ...
#
#
# terminal_state:
#
#     terminal_id
#         |
#         +-- (timestamp, fraud)
#         +-- (timestamp, fraud)
#         +-- ...
#
# Only the latest 30 days are retained.
#
# ============================================================

customer_state = defaultdict(deque)

terminal_state = defaultdict(deque)


# ============================================================
# 7. REMOVE OLD CUSTOMER TRANSACTIONS
# ============================================================

def remove_old_customer_transactions(
    customer_id,
    current_time
):

    history = customer_state[customer_id]

    while history:

        age_seconds = (
            current_time
            - history[0][0]
        ).total_seconds()

        if age_seconds > HISTORY_DAYS * 86400:

            history.popleft()

        else:

            break


# ============================================================
# 8. REMOVE OLD TERMINAL TRANSACTIONS
# ============================================================

def remove_old_terminal_transactions(
    terminal_id,
    current_time
):

    history = terminal_state[terminal_id]

    while history:

        age_seconds = (
            current_time
            - history[0][0]
        ).total_seconds()

        if age_seconds > HISTORY_DAYS * 86400:

            history.popleft()

        else:

            break


# ============================================================
# 9. CUSTOMER FEATURES
# ============================================================

def customer_features(
    customer_id,
    current_time,
    current_amount
):

    # Remove transactions older than 30 days.

    remove_old_customer_transactions(
        customer_id,
        current_time
    )

    history = customer_state[customer_id]

    result = {}

    for days in (
        1,
        7,
        30
    ):

        cutoff_seconds = (
            days * 86400
        )

        values = [

            amount

            for timestamp, amount in history

            if (
                current_time
                - timestamp
            ).total_seconds()
            <= cutoff_seconds

        ]

        # IMPORTANT:
        #   Match the training notebook's rolling window, which
        #   INCLUDES the current transaction:
        #
        #       NB_TX_WINDOW = rolling(...).count()   (includes current)
        #       AVG_AMOUNT   = SUM_AMOUNT / NB_TX     (includes current)
        #
        #   count = past_count_in_window + 1
        #   sum   = past_sum_in_window + current_amount

        count = (
            len(values)
            + 1
        )

        avg_amount = (
            (
                sum(values)
                + float(current_amount)
            )
            / count
        )

        result[
            f"CUSTOMER_ID_NB_TX_{days}DAY_WINDOW"
        ] = float(count)

        result[
            f"CUSTOMER_ID_AVG_AMOUNT_{days}DAY_WINDOW"
        ] = float(avg_amount)

    return result


# ============================================================
# 10. TERMINAL FEATURES
# ============================================================

def terminal_features(
    terminal_id,
    current_time
):

    remove_old_terminal_transactions(
        terminal_id,
        current_time
    )

    history = terminal_state[terminal_id]

    result = {}

    for days in (
        1,
        7,
        30
    ):

        cutoff_seconds = (
            days * 86400
        )

        values = [

            fraud

            for timestamp, fraud in history

            if (
                current_time
                - timestamp
            ).total_seconds()
            <= cutoff_seconds

        ]

        count = len(values)

        if count > 0:

            risk = (
                sum(values)
                / count
            )

        else:

            risk = 0.0

        result[
            f"TERMINAL_ID_NB_TX_{days}DAY_WINDOW"
        ] = float(count)

        result[
            f"TERMINAL_ID_RISK_{days}DAY_WINDOW"
        ] = float(risk)

    return result


# ============================================================
# 11. READ FROM KAFKA
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
        INPUT_TOPIC
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # If there is NO checkpoint:
    #
    #     earliest
    #
    # means Spark reads existing Kafka records.
    #
    # If checkpoint already exists:
    #
    #     checkpoint offsets take precedence.
    #
    # --------------------------------------------------------

    .option(
        "startingOffsets",
        "earliest"
    )

    # --------------------------------------------------------
    # Prevent very large micro-batches.
    # --------------------------------------------------------

    .option(
        "maxOffsetsPerTrigger",
        str(
            MAX_OFFSETS_PER_TRIGGER
        )
    )

    .option(
        "failOnDataLoss",
        "true"
    )

    .load()
)


# ============================================================
# 12. KAFKA VALUE -> JSON
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
# 13. PARSE JSON
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

    .select(
        "_RAW_VALUE",
        "data.*"
    )
)


# ============================================================
# 14. CONVERT TX_DATETIME
# ============================================================

transactions = (

    transactions

    .withColumn(

        "TX_DATETIME",

        to_timestamp(
            col("TX_DATETIME")
        )

    )
)


# ============================================================
# 15. BASIC FEATURES
# ============================================================

features = (

    transactions

    # --------------------------------------------------------
    # Weekend
    # --------------------------------------------------------

    .withColumn(

        "TX_DURING_WEEKEND",

        when(

            dayofweek(
                col("TX_DATETIME")
            ).isin(
                1,
                7
            ),

            1

        ).otherwise(0)

    )

    # --------------------------------------------------------
    # Night
    # --------------------------------------------------------

    .withColumn(

        "TX_DURING_NIGHT",

        when(

            hour(
                col("TX_DATETIME")
            ) <= 6,

            1

        ).otherwise(0)

    )
)


# ============================================================
# 16. PROCESS MICRO-BATCH
# ============================================================

def process_batch(
    batch_df,
    batch_id
):

    global END_RECEIVED

    print()
    print("=" * 100)

    print(
        f"FEATURE ENGINEERING - "
        f"BATCH ID = {batch_id}"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # Count input rows
    # --------------------------------------------------------

    input_count = batch_df.count()

    print()
    print(
        f"INPUT ROWS       : "
        f"{input_count}"
    )

    # --------------------------------------------------------
    # Empty batch
    # --------------------------------------------------------

    if input_count == 0:

        print(
            "NO TRANSACTIONS "
            "IN THIS BATCH"
        )

        return

    # --------------------------------------------------------
    # Collect only this bounded micro-batch.
    #
    # maxOffsetsPerTrigger prevents this from becoming huge.
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

            write_end_marker()

            print()
            print("=" * 100)
            print(
                "[ALL DONE] "
                "FEATURE_ENGINEERING"
            )
            print("=" * 100)

            END_RECEIVED = True

        return

    # --------------------------------------------------------
    # Sort by TX_DATETIME.
    #
    # This keeps the original history-feature logic.
    #
    # IMPORTANT:
    # TX_DATETIME is dataset transaction time,
    # not realtime latency timestamp.
    # --------------------------------------------------------

    rows = sorted(

        rows,

        key=lambda row: (

            row["TX_DATETIME"]

            if row["TX_DATETIME"] is not None

            else 0

        )

    )

    output_rows = []

    skipped_count = 0

    skipped_ids = []


    # ========================================================
    # PROCESS EACH TRANSACTION
    # ========================================================

    for row in rows:

        # ----------------------------------------------------
        # IMPORTANT:
        # Preserve original TRANSACTION_ID.
        # ----------------------------------------------------

        transaction_id = (
            row["TRANSACTION_ID"]
        )

        producer_timestamp = (
            row["PRODUCER_TIMESTAMP"]
        )

        tx_datetime = (
            row["TX_DATETIME"]
        )

        customer_id = (
            row["CUSTOMER_ID"]
        )

        terminal_id = (
            row["TERMINAL_ID"]
        )

        amount = (
            row["TX_AMOUNT"]
        )

        fraud = (
            row["TX_FRAUD"]
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if transaction_id is None:

            skipped_count += 1

            skipped_ids.append(
                "NULL_ID"
            )

            continue


        if tx_datetime is None:

            skipped_count += 1

            skipped_ids.append(
                transaction_id
            )

            log(
                f"SKIP TRANSACTION "
                f"{transaction_id}: "
                f"TX_DATETIME IS NULL"
            )

            continue


        if customer_id is None:

            skipped_count += 1

            skipped_ids.append(
                transaction_id
            )

            log(
                f"SKIP TRANSACTION "
                f"{transaction_id}: "
                f"CUSTOMER_ID IS NULL"
            )

            continue


        if terminal_id is None:

            skipped_count += 1

            skipped_ids.append(
                transaction_id
            )

            log(
                f"SKIP TRANSACTION "
                f"{transaction_id}: "
                f"TERMINAL_ID IS NULL"
            )

            continue


        # ----------------------------------------------------
        # Default values
        # ----------------------------------------------------

        if amount is None:

            amount = 0.0


        if fraud is None:

            fraud = 0


        # ====================================================
        # CALCULATE HISTORY FEATURES
        # BEFORE CURRENT TRANSACTION
        # ====================================================

        c_features = customer_features(

            customer_id,

            tx_datetime,

            amount

        )


        t_features = terminal_features(

            terminal_id,

            tx_datetime

        )


        # ====================================================
        # CREATE OUTPUT ROW
        # ====================================================

        output_row = {

            # ------------------------------------------------
            # IMPORTANT:
            # ORIGINAL ID FROM PRODUCER
            # ------------------------------------------------

            "TRANSACTION_ID":
                int(transaction_id),

            "PRODUCER_TIMESTAMP":
                producer_timestamp,

            "TX_DATETIME":
                tx_datetime,

            "CUSTOMER_ID":
                int(customer_id),

            "TERMINAL_ID":
                int(terminal_id),

            "TX_AMOUNT":
                float(amount),

            # ------------------------------------------------
            # BASIC FEATURES
            # ------------------------------------------------

            "TX_DURING_WEEKEND":
                int(

                    row[
                        "TX_DURING_WEEKEND"
                    ]

                    if row[
                        "TX_DURING_WEEKEND"
                    ] is not None

                    else 0

                ),

            "TX_DURING_NIGHT":
                int(

                    row[
                        "TX_DURING_NIGHT"
                    ]

                    if row[
                        "TX_DURING_NIGHT"
                    ] is not None

                    else 0

                ),

            # ------------------------------------------------
            # CUSTOMER FEATURES
            # ------------------------------------------------

            **c_features,

            # ------------------------------------------------
            # TERMINAL FEATURES
            # ------------------------------------------------

            **t_features,

            # ------------------------------------------------
            # GROUND TRUTH
            # ------------------------------------------------

            "TX_FRAUD":
                int(fraud),

        }


        output_rows.append(
            output_row
        )


        # ====================================================
        # UPDATE STATE
        #
        # VERY IMPORTANT:
        # Current transaction is inserted AFTER calculating
        # features.
        #
        # This prevents target/current-row leakage.
        # ====================================================

        customer_state[
            customer_id
        ].append(

            (
                tx_datetime,
                float(amount)
            )

        )


        terminal_state[
            terminal_id
        ].append(

            (
                tx_datetime,
                int(fraud)
            )

        )


        # ----------------------------------------------------
        # Remove old history.
        # ----------------------------------------------------

        remove_old_customer_transactions(

            customer_id,

            tx_datetime

        )


        remove_old_terminal_transactions(

            terminal_id,

            tx_datetime

        )


    # ========================================================
    # EMPTY OUTPUT
    # ========================================================

    if not output_rows:

        print()
        print(
            "NO VALID TRANSACTIONS "
            "TO WRITE"
        )

        print(
            f"SKIPPED ROWS : "
            f"{skipped_count}"
        )

        if end_marker_present:

            write_end_marker()

            print()
            print("=" * 100)
            print(
                "[ALL DONE] "
                "FEATURE_ENGINEERING"
            )
            print("=" * 100)

            END_RECEIVED = True

        return


    # ========================================================
    # CREATE OUTPUT DATAFRAME
    # ========================================================

    output_df = (

        spark.createDataFrame(

            output_rows,

            schema=output_schema

        )

    )


    # ========================================================
    # DUPLICATE CHECK INSIDE BATCH
    # ========================================================

    output_count = (
        output_df.count()
    )


    distinct_id_count = (

        output_df

        .select(
            "TRANSACTION_ID"
        )

        .distinct()

        .count()

    )


    duplicate_count = (
        output_count
        - distinct_id_count
    )


    # ========================================================
    # WRITE TO KAFKA
    # ========================================================

    kafka_output = (

        output_df

        .select(

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Kafka KEY = original TRANSACTION_ID
            #
            # ------------------------------------------------

            col(
                "TRANSACTION_ID"
            )
            .cast("string")
            .alias("key"),

            # ------------------------------------------------
            # JSON VALUE
            # ------------------------------------------------

            to_json(

                struct(

                    *[
                        col(c)
                        for c
                        in output_df.columns
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


    # ========================================================
    # FINAL BATCH STATISTICS
    # ========================================================

    print()
    print("-" * 100)

    print(
        f"BATCH ID             : "
        f"{batch_id}"
    )

    print(
        f"INPUT ROWS           : "
        f"{input_count}"
    )

    print(
        f"VALID OUTPUT ROWS    : "
        f"{output_count}"
    )

    print(
        f"SKIPPED ROWS         : "
        f"{skipped_count}"
    )

    print(
        f"DISTINCT IDs         : "
        f"{distinct_id_count}"
    )

    print(
        f"DUPLICATE IDs        : "
        f"{duplicate_count}"
    )

    print(
        f"OUTPUT TOPIC         : "
        f"{OUTPUT_TOPIC}"
    )

    print("-" * 100)

    if skipped_count > 0:

        print(
            "WARNING: "
            f"{skipped_count} TRANSACTION(S) "
            "WERE SKIPPED."
        )

        print(
            "Skipped IDs:"
        )

        print(
            skipped_ids[:50]
        )

        if len(skipped_ids) > 50:

            print(
                f"... and "
                f"{len(skipped_ids) - 50} "
                f"more."
            )

    print()
    print("=" * 100)

    print(
        f"WROTE {output_count} "
        f"FEATURED TRANSACTION(S) "
        f"TO KAFKA: {OUTPUT_TOPIC}"
    )

    print()
    print("=" * 100)
    print(
        f"[BATCH DONE] "
        f"FEATURE_ENGINEERING - batch {batch_id}: "
        f"processed {output_count} transactions -> "
        f"{OUTPUT_TOPIC}"
    )
    print("=" * 100)

    if end_marker_present:

        write_end_marker()

        print()
        print("=" * 100)
        print(
            "[ALL DONE] "
            "FEATURE_ENGINEERING"
        )
        print("=" * 100)

        END_RECEIVED = True
    print()


# ============================================================
# 17. START STREAMING QUERY
# ============================================================

query = (

    features

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
# 18. WAIT FOR TERMINATION
# ============================================================

print()
print("=" * 100)

print(
    "FEATURE ENGINEERING STREAMING STARTED"
)

print("=" * 100)

print()
print(
    f"Input topic       : "
    f"{INPUT_TOPIC}"
)

print(
    f"Output topic      : "
    f"{OUTPUT_TOPIC}"
)

print(
    f"Checkpoint        : "
    f"{CHECKPOINT_LOCATION}"
)

print(
    f"Max offsets/batch : "
    f"{MAX_OFFSETS_PER_TRIGGER}"
)

print()


# ============================================================
# WAIT FOR END MARKER / COMPLETION
# ============================================================

print()
print("=" * 100)

print(
    "[READY] FEATURE_ENGINEERING - "
    f"listening to topic '{INPUT_TOPIC}'"
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
    "[STOPPED] FEATURE_ENGINEERING"
)

print("=" * 100)