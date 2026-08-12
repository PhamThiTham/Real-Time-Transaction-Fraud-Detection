from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType
)


# ==========================================
# 1. Create Spark Session
# ==========================================

spark = (
    SparkSession.builder
    .appName("RealTimeFraudDetection")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ==========================================
# 2. Define transaction schema
# ==========================================

transaction_schema = StructType([
    StructField("CUSTOMER_ID", IntegerType(), True),
    StructField("TX_AMOUNT", DoubleType(), True),
    StructField("TX_FRAUD", IntegerType(), True),
    StructField("TX_DATETIME", StringType(), True)
])


# ==========================================
# 3. Read Kafka stream
# ==========================================

raw_stream = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:29092")
    .option("subscribe", "transactions")
    .option("startingOffsets", "latest")
    .load()
)


# ==========================================
# 4. Convert Kafka value from binary to string
# ==========================================

json_stream = (
    raw_stream
    .selectExpr("CAST(value AS STRING) AS json")
)


# ==========================================
# 5. Parse JSON
# ==========================================

transactions = (
    json_stream
    .select(
        from_json(
            col("json"),
            transaction_schema
        ).alias("data")
    )
    .select("data.*")
)


# ==========================================
# 6. Display streaming data
# ==========================================

query = (
    transactions
    .writeStream
    .format("console")
    .outputMode("append")
    .option("truncate", False)
    .option("numRows", 20)
    .start()
)


# ==========================================
# 7. Keep streaming query alive
# ==========================================

query.awaitTermination()