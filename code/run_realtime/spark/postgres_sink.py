# ============================================================
# POSTGRES SINK - REAL-TIME FRAUD DETECTION
# ============================================================
#
# INPUT:
#     Kafka topic:
#         fraud_predictions
#
# OUTPUT:
#     PostgreSQL:
#         fraud_predictions
#
# REAL-TIME LATENCY:
#
#     PRODUCER_TIMESTAMP
#             |
#             v
#          Kafka
#             |
#             v
#          Spark
#             |
#             v
#           LSTM
#             |
#             v
#      PostgreSQL Sink
#             |
#             v
#       SINK_TIMESTAMP
#
# END_TO_END_LATENCY_MS =
#
#     SINK_TIMESTAMP - PRODUCER_TIMESTAMP
#
# IMPORTANT:
#     TX_DATETIME is the original transaction timestamp
#     from the dataset. It is NOT used for real-time
#     pipeline latency.
#
# ============================================================


from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    from_json,
    current_timestamp,
    unix_micros,
    round as spark_round,
)

from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
)


# ============================================================
# CONFIG
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"

KAFKA_TOPIC = "fraud_predictions"

CHECKPOINT_LOCATION = (
    "/opt/spark-data/checkpoint/postgres_sink"
)

POSTGRES_URL = (
    "jdbc:postgresql://postgres:5432/fraud_detection"
)

POSTGRES_TABLE = "fraud_predictions"

POSTGRES_USER = "fraud"

POSTGRES_PASSWORD = "fraud123"

POSTGRES_DRIVER = "org.postgresql.Driver"


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession
    .builder
    .appName(
        "PostgreSQLFraudPredictionSink"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# KAFKA MESSAGE SCHEMA
# ============================================================

schema = StructType([

    StructField(
        "TRANSACTION_ID",
        IntegerType(),
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
        "FRAUD_PROBABILITY",
        DoubleType(),
        True
    ),

    StructField(
        "FRAUD_PREDICTION",
        IntegerType(),
        True
    ),

    StructField(
        "THRESHOLD",
        DoubleType(),
        True
    ),

    StructField(
        "TX_FRAUD",
        IntegerType(),
        True
    ),

    StructField(
        "PRODUCER_TIMESTAMP",
        StringType(),
        True
    ),

    StructField(
        "PREDICTION_START_TIMESTAMP",
        StringType(),
        True
    ),

    StructField(
        "PREDICTION_END_TIMESTAMP",
        StringType(),
        True
    ),

    StructField(
        "PREDICTION_LATENCY_MS",
        DoubleType(),
        True
    ),

    StructField(
        "END_TO_END_LATENCY_MS",
        DoubleType(),
        True
    ),
])


# ============================================================
# START MESSAGE
# ============================================================

print()
print("=" * 100)

print(
    "POSTGRESQL SINK - REAL-TIME FRAUD DETECTION"
)

print("=" * 100)

print()
print(
    f"Kafka bootstrap : "
    f"{KAFKA_BOOTSTRAP_SERVERS}"
)

print(
    f"Kafka topic     : "
    f"{KAFKA_TOPIC}"
)

print(
    f"PostgreSQL URL  : "
    f"{POSTGRES_URL}"
)

print(
    f"PostgreSQL table: "
    f"{POSTGRES_TABLE}"
)

print(
    f"Checkpoint      : "
    f"{CHECKPOINT_LOCATION}"
)

print()


# ============================================================
# READ FROM KAFKA
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
        "latest"
    )

    .option(
        "failOnDataLoss",
        "false"
    )

    .load()
)


# ============================================================
# PARSE JSON
# ============================================================

parsed_stream = (

    raw_stream

    .select(

        from_json(

            col("value").cast("string"),

            schema

        ).alias("data")

    )

    .select(
        "data.*"
    )
)


# ============================================================
# ADD SINK TIMESTAMP
# ============================================================
#
# current_timestamp() represents the processing time of
# the current Spark micro-batch.
#
# This timestamp is used as the endpoint of the
# real-time end-to-end latency measurement.
#
# ============================================================

final_stream = (

    parsed_stream

    .withColumn(

        "SINK_TIMESTAMP",

        current_timestamp()

    )
)


# ============================================================
# WRITE BATCH
# ============================================================

def write_batch(
    batch_df,
    batch_id
):

    print()
    print("=" * 100)

    print(
        f"POSTGRES SINK - BATCH ID = {batch_id}"
    )

    print("=" * 100)

    # --------------------------------------------------------
    # COUNT
    # --------------------------------------------------------

    row_count = batch_df.count()

    print()

    print(
        f"ROWS TO WRITE : {row_count}"
    )

    # --------------------------------------------------------
    # EMPTY BATCH
    # --------------------------------------------------------

    if row_count == 0:

        print(
            "EMPTY BATCH"
        )

        return

    # --------------------------------------------------------
    # CONVERT TIMESTAMP COLUMNS
    # --------------------------------------------------------
    #
    # PostgreSQL table expects:
    #
    #     timestamp without time zone
    #
    # Therefore all timestamp string fields are explicitly
    # converted to Spark TimestampType before JDBC writing.
    #
    # --------------------------------------------------------

    df = (

        batch_df

        # ----------------------------------------------------
        # ORIGINAL TRANSACTION TIMESTAMP
        # ----------------------------------------------------

        .withColumn(

            "TX_DATETIME",

            col(
                "TX_DATETIME"
            ).cast("timestamp")

        )

        # ----------------------------------------------------
        # PRODUCER TIMESTAMP
        # ----------------------------------------------------

        .withColumn(

            "PRODUCER_TIMESTAMP",

            col(
                "PRODUCER_TIMESTAMP"
            ).cast("timestamp")

        )

        # ----------------------------------------------------
        # PREDICTION START
        # ----------------------------------------------------

        .withColumn(

            "PREDICTION_START_TIMESTAMP",

            col(
                "PREDICTION_START_TIMESTAMP"
            ).cast("timestamp")

        )

        # ----------------------------------------------------
        # PREDICTION END
        # ----------------------------------------------------

        .withColumn(

            "PREDICTION_END_TIMESTAMP",

            col(
                "PREDICTION_END_TIMESTAMP"
            ).cast("timestamp")

        )

        # ----------------------------------------------------
        # SINK TIMESTAMP
        # ----------------------------------------------------

        .withColumn(

            "SINK_TIMESTAMP",

            col(
                "SINK_TIMESTAMP"
            ).cast("timestamp")

        )

        # ----------------------------------------------------
        # REAL-TIME END-TO-END LATENCY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # We do NOT use TX_DATETIME.
        #
        # Instead:
        #
        #     PRODUCER_TIMESTAMP
        #             |
        #             v
        #          Kafka
        #             |
        #             v
        #          Spark
        #             |
        #             v
        #           LSTM
        #             |
        #             v
        #      PostgreSQL Sink
        #             |
        #             v
        #       SINK_TIMESTAMP
        #
        # Using unix_micros() preserves sub-second precision.
        #
        # ----------------------------------------------------

        .withColumn(

            "END_TO_END_LATENCY_MS",

            spark_round(

                (

                    unix_micros(
                        col(
                            "SINK_TIMESTAMP"
                        )
                    )

                    -

                    unix_micros(
                        col(
                            "PRODUCER_TIMESTAMP"
                        )
                    )

                ) / 1000.0,

                3

            )

        )

    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()

    print(
        "PREDICTION DATA:"
    )

    (

        df

        .select(

            "TRANSACTION_ID",

            "TX_DATETIME",

            "CUSTOMER_ID",

            "TERMINAL_ID",

            "TX_AMOUNT",

            "FRAUD_PROBABILITY",

            "FRAUD_PREDICTION",

            "THRESHOLD",

            "TX_FRAUD",

            "PRODUCER_TIMESTAMP",

            "PREDICTION_START_TIMESTAMP",

            "PREDICTION_END_TIMESTAMP",

            "PREDICTION_LATENCY_MS",

            "END_TO_END_LATENCY_MS",

            "SINK_TIMESTAMP",

        )

        .show(

            20,

            truncate=False

        )

    )


    # ========================================================
    # WRITE TO POSTGRESQL
    # ========================================================

    (

        df

        .select(

            "TRANSACTION_ID",

            "TX_DATETIME",

            "CUSTOMER_ID",

            "TERMINAL_ID",

            "TX_AMOUNT",

            "FRAUD_PROBABILITY",

            "FRAUD_PREDICTION",

            "THRESHOLD",

            "TX_FRAUD",

            "PRODUCER_TIMESTAMP",

            "PREDICTION_START_TIMESTAMP",

            "PREDICTION_END_TIMESTAMP",

            "PREDICTION_LATENCY_MS",

            "END_TO_END_LATENCY_MS",

            "SINK_TIMESTAMP",

        )

        .write

        .format("jdbc")

        .option(
            "url",
            POSTGRES_URL
        )

        .option(
            "dbtable",
            POSTGRES_TABLE
        )

        .option(
            "user",
            POSTGRES_USER
        )

        .option(
            "password",
            POSTGRES_PASSWORD
        )

        .option(
            "driver",
            POSTGRES_DRIVER
        )

        .option(
            "batchsize",
            "1000"
        )

        .option(
            "stringtype",
            "unspecified"
        )

        .mode(
            "append"
        )

        .save()

    )


    # ========================================================
    # SUCCESS
    # ========================================================

    print()

    print(
        f"WROTE {row_count} ROW(S) "
        f"TO POSTGRESQL"
    )

    print("=" * 100)


# ============================================================
# START STREAMING QUERY
# ============================================================

query = (

    final_stream

    .writeStream

    .foreachBatch(
        write_batch
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
# WAIT
# ============================================================

query.awaitTermination()