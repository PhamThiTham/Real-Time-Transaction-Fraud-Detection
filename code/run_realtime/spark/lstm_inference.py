# ============================================================
# LSTM INFERENCE - REAL-TIME FRAUD DETECTION
# ============================================================
#
# INPUT:
#     Kafka topic:
#         transactions_features
#
# OUTPUT:
#     Kafka topic:
#         fraud_predictions
#
# MODEL:
#     LSTM + Attention
#
# SEQUENCE:
#     5 transactions x 15 features
#
# IMPORTANT:
#
#     Current transaction is NOT used for prediction.
#
#     Prediction Tn:
#
#         history:
#             T1 T2 T3 T4
#
#         sequence:
#             PAD T1 T2 T3 T4
#
#         predict Tn
#
#         AFTER prediction:
#             add Tn to history
#
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

from collections import defaultdict, deque

import time

from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    col,
    from_json,
    struct,
    to_json,
    to_timestamp,
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

INPUT_TOPIC = "transactions_features"

OUTPUT_TOPIC = "fraud_predictions"

CHECKPOINT_LOCATION = (
    "/tmp/checkpoint/lstm_inference"
)

MODEL_PATH = (
    "/opt/spark-models/"
    "Tham_lstm_attn_checkpoint.pth"
)

SEQ_LEN = 5

NUM_FEATURES = 15

THRESHOLD = 0.5

PAD_VALUE = 0.0

DEVICE = torch.device("cpu")

MAX_LATENCY_SAMPLES = 10000


# ============================================================
# 2. LATENCY HISTORY
# ============================================================

prediction_latency_history = deque(
    maxlen=MAX_LATENCY_SAMPLES
)

end_to_end_latency_history = deque(
    maxlen=MAX_LATENCY_SAMPLES
)


# ============================================================
# 3. FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [

    "TX_AMOUNT",

    "TX_DURING_WEEKEND",

    "TX_DURING_NIGHT",

    "CUSTOMER_ID_NB_TX_1DAY_WINDOW",

    "CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW",

    "CUSTOMER_ID_NB_TX_7DAY_WINDOW",

    "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW",

    "CUSTOMER_ID_NB_TX_30DAY_WINDOW",

    "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW",

    "TERMINAL_ID_NB_TX_1DAY_WINDOW",

    "TERMINAL_ID_RISK_1DAY_WINDOW",

    "TERMINAL_ID_NB_TX_7DAY_WINDOW",

    "TERMINAL_ID_RISK_7DAY_WINDOW",

    "TERMINAL_ID_NB_TX_30DAY_WINDOW",

    "TERMINAL_ID_RISK_30DAY_WINDOW",

]


if len(FEATURE_COLUMNS) != NUM_FEATURES:

    raise ValueError(
        f"Expected {NUM_FEATURES} features, "
        f"got {len(FEATURE_COLUMNS)}"
    )


# ============================================================
# 4. LATENCY FUNCTIONS
# ============================================================

def percentile(values, percentile_value):

    if not values:
        return None

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (
        percentile_value / 100.0
    ) * (len(sorted_values) - 1)

    lower_index = int(rank)

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1
    )

    fraction = rank - lower_index

    lower_value = sorted_values[
        lower_index
    ]

    upper_value = sorted_values[
        upper_index
    ]

    return (
        lower_value
        + (
            upper_value
            - lower_value
        )
        * fraction
    )


def calculate_latency_statistics(values):

    if not values:

        return {
            "count": 0,
            "average": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }

    values = list(values)

    return {

        "count": len(values),

        "average": (
            sum(values) / len(values)
        ),

        "p50": percentile(
            values,
            50
        ),

        "p95": percentile(
            values,
            95
        ),

        "p99": percentile(
            values,
            99
        ),
    }


def print_latency_statistics(
    title,
    values
):

    stats = calculate_latency_statistics(
        values
    )

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if stats["count"] == 0:

        print("No latency samples.")
        print("=" * 100)

        return

    print(
        f"Samples : {stats['count']}"
    )

    print(
        f"Average : "
        f"{stats['average']:.3f} ms"
    )

    print(
        f"P50     : "
        f"{stats['p50']:.3f} ms"
    )

    print(
        f"P95     : "
        f"{stats['p95']:.3f} ms"
    )

    print(
        f"P99     : "
        f"{stats['p99']:.3f} ms"
    )

    print("=" * 100)


# ============================================================
# 5. PRODUCER TIMESTAMP
# ============================================================

def parse_producer_timestamp(
    producer_timestamp
):

    if producer_timestamp is None:
        return None

    try:

        value = str(
            producer_timestamp
        ).strip()

        if not value:
            return None

        if value.endswith("Z"):
            value = (
                value[:-1]
                + "+00:00"
            )

        dt = datetime.fromisoformat(
            value
        )

        if dt.tzinfo is None:

            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return dt

    except Exception as e:

        print(
            f"WARNING: Cannot parse "
            f"PRODUCER_TIMESTAMP={producer_timestamp}"
        )

        print(
            f"ERROR: {e}"
        )

        return None


# ============================================================
# 6. ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.linear_out = nn.Linear(
            dim * 2,
            dim
        )

        self.mask = None


    def set_mask(self, mask):

        self.mask = mask


    def forward(
        self,
        output,
        context
    ):

        batch_size = output.size(0)

        hidden_size = output.size(2)

        input_size = context.size(1)

        attn = torch.bmm(
            output,
            context.transpose(1, 2)
        )

        if self.mask is not None:

            attn.data.masked_fill_(
                self.mask,
                -float("inf")
            )

        attn = F.softmax(
            attn.view(
                -1,
                input_size
            ),
            dim=1
        ).view(
            batch_size,
            -1,
            input_size
        )

        mix = torch.bmm(
            attn,
            context
        )

        combined = torch.cat(
            (
                mix,
                output
            ),
            dim=2
        )

        output = F.tanh(
            self.linear_out(
                combined.view(
                    -1,
                    2 * hidden_size
                )
            )
        ).view(
            batch_size,
            -1,
            hidden_size
        )

        return output, attn


# ============================================================
# 7. MODEL
# ============================================================

class FraudLSTMWithAttention(nn.Module):

    def __init__(
        self,
        num_features,
        hidden_size=100,
        hidden_size_lstm=100,
        num_layers_lstm=1,
        dropout_lstm=0,
        attention_out_dim=100
    ):

        super().__init__()

        self.num_features = num_features

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers_lstm,
            batch_first=True,
            dropout=dropout_lstm
        )

        self.ff = nn.Linear(
            num_features,
            hidden_size_lstm
        )

        self.attention = Attention(
            attention_out_dim
        )

        self.fc1 = nn.Linear(
            hidden_size_lstm,
            hidden_size
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            hidden_size,
            1
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        hidden_states, _ = self.lstm(x)

        last_transaction = x[:, -1, :]

        context_vector = self.ff(
            last_transaction
        )

        context_vector = (
            context_vector.unsqueeze(1)
        )

        combined_state, _ = (
            self.attention(
                context_vector,
                hidden_states
            )
        )

        combined_state = (
            combined_state[:, 0, :]
        )

        hidden = self.fc1(
            combined_state
        )

        hidden = self.relu(hidden)

        output = self.fc2(
            hidden
        )

        output = self.sigmoid(
            output
        )

        return output


# ============================================================
# 8. LOAD MODEL
# ============================================================

print()
print("=" * 100)
print("LOADING LSTM + ATTENTION MODEL")
print("=" * 100)

print(
    f"Model path : {MODEL_PATH}"
)

print(
    f"Device     : {DEVICE}"
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


model = FraudLSTMWithAttention(
    num_features=NUM_FEATURES,
    hidden_size=100,
    hidden_size_lstm=100,
    num_layers_lstm=1,
    dropout_lstm=0,
    attention_out_dim=100
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()


print(
    "Model loaded successfully."
)

print(
    f"Features  : {NUM_FEATURES}"
)

print(
    f"Sequence  : {SEQ_LEN}"
)

print(
    f"Threshold : {THRESHOLD}"
)

print("=" * 100)


# ============================================================
# 9. INPUT SCHEMA
# ============================================================

input_schema = StructType([

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
        "PRODUCER_TIMESTAMP",
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
# 10. SPARK
# ============================================================

spark = (
    SparkSession
    .builder
    .appName(
        "LSTMRealTimeFraudInference"
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel(
    "WARN"
)


# ============================================================
# 11. CUSTOMER HISTORY
# ============================================================

customer_history = defaultdict(
    lambda: deque(
        maxlen=SEQ_LEN
    )
)


# ============================================================
# 12. READ KAFKA
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
# 13. PARSE
# ============================================================

transactions_features = (
    raw_stream
    .select(
        from_json(
            col("value").cast("string"),
            input_schema
        ).alias("data")
    )
    .select("data.*")
    .withColumn(
        "TX_DATETIME",
        to_timestamp(
            col("TX_DATETIME")
        )
    )
)


# ============================================================
# 14. PAD
# ============================================================

def create_pad_vector():

    return [
        float(PAD_VALUE)
        for _ in range(NUM_FEATURES)
    ]


def build_padded_sequence(history):

    sequence = [
        item[2]
        for item in history
    ]

    sequence = sequence[-SEQ_LEN:]

    number_of_padding = (
        SEQ_LEN - len(sequence)
    )

    padding = [
        create_pad_vector()
        for _ in range(number_of_padding)
    ]

    result = (
        padding
        + sequence
    )

    if len(result) != SEQ_LEN:

        raise ValueError(
            f"Invalid sequence length "
            f"{len(result)}"
        )

    return result


# ============================================================
# 15. PROCESS BATCH
# ============================================================

def process_batch(
    batch_df,
    batch_id
):

    print()
    print("=" * 100)
    print(
        f"LSTM INFERENCE - BATCH ID = {batch_id}"
    )
    print("=" * 100)

    if batch_df.isEmpty():

        print("EMPTY BATCH")
        return

    rows = batch_df.collect()

    rows = sorted(
        rows,
        key=lambda r: (
            r["TX_DATETIME"]
            if r["TX_DATETIME"] is not None
            else datetime.min
        )
    )

    print(
        f"ROWS = {len(rows)}"
    )

    prediction_rows = []

    batch_prediction_latencies = []

    batch_end_to_end_latencies = []


    # ========================================================
    # EACH TRANSACTION
    # ========================================================

    for row in rows:

        transaction_id = row[
            "TRANSACTION_ID"
        ]

        customer_id = row[
            "CUSTOMER_ID"
        ]

        tx_datetime = row[
            "TX_DATETIME"
        ]

        producer_timestamp = row[
            "PRODUCER_TIMESTAMP"
        ]

        terminal_id = row[
            "TERMINAL_ID"
        ]


        if transaction_id is None:
            continue

        if customer_id is None:
            continue

        if tx_datetime is None:
            continue


        # ====================================================
        # CURRENT FEATURES
        # ====================================================

        current_features = []

        for feature_name in FEATURE_COLUMNS:

            value = row[
                feature_name
            ]

            if value is None:
                value = 0.0

            current_features.append(
                float(value)
            )


        # ====================================================
        # HISTORY BEFORE CURRENT TX
        # ====================================================

        history = customer_history[
            customer_id
        ]


        print()
        print("-" * 100)

        print(
            f"TRANSACTION_ID : {transaction_id}"
        )

        print(
            f"CUSTOMER_ID    : {customer_id}"
        )

        print(
            f"TX_AMOUNT      : "
            f"{float(row['TX_AMOUNT'] or 0):.2f}"
        )

        print(
            f"HISTORY LENGTH : {len(history)}"
        )


        # ====================================================
        # BUILD SEQUENCE
        # ====================================================

        padded_sequence = (
            build_padded_sequence(
                history
            )
        )


        x = torch.tensor(
            padded_sequence,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)


        if tuple(x.shape) != (
            1,
            SEQ_LEN,
            NUM_FEATURES
        ):

            raise ValueError(
                f"Invalid model input "
                f"shape={tuple(x.shape)}"
            )


        print(
            f"MODEL INPUT SHAPE : "
            f"{tuple(x.shape)}"
        )


        # ====================================================
        # PREDICTION START
        # ====================================================

        prediction_start_timestamp = (
            datetime.now(timezone.utc)
        )

        prediction_start = (
            time.perf_counter()
        )


        with torch.no_grad():

            probability = (
                model(x)
                .item()
            )


        prediction_end = (
            time.perf_counter()
        )

        prediction_end_timestamp = (
            datetime.now(timezone.utc)
        )


        prediction_latency_ms = (
            prediction_end
            - prediction_start
        ) * 1000.0


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = int(
            probability >= THRESHOLD
        )


        # ====================================================
        # END TO END
        # ====================================================

        end_to_end_latency_ms = None

        producer_dt = (
            parse_producer_timestamp(
                producer_timestamp
            )
        )

        if producer_dt is not None:

            end_to_end_latency_ms = (

                (
                    prediction_end_timestamp
                    - producer_dt
                ).total_seconds()

                * 1000.0
            )

            if end_to_end_latency_ms >= 0:

                end_to_end_latency_history.append(
                    end_to_end_latency_ms
                )

                batch_end_to_end_latencies.append(
                    end_to_end_latency_ms
                )

            else:

                print(
                    "WARNING: Negative E2E latency."
                )

                end_to_end_latency_ms = None


        prediction_latency_history.append(
            prediction_latency_ms
        )

        batch_prediction_latencies.append(
            prediction_latency_ms
        )


        # ====================================================
        # LOG
        # ====================================================

        print(
            f"FRAUD PROBABILITY : "
            f"{probability:.6f}"
        )

        print(
            f"PREDICTION        : "
            f"{prediction}"
        )

        print(
            f"PREDICTION LATENCY: "
            f"{prediction_latency_ms:.3f} ms"
        )

        if end_to_end_latency_ms is not None:

            print(
                f"END-TO-END LATENCY: "
                f"{end_to_end_latency_ms:.3f} ms"
            )

        else:

            print(
                "END-TO-END LATENCY: N/A"
            )


        # ====================================================
        # OUTPUT
        # ====================================================

        prediction_rows.append({

            "TRANSACTION_ID":
                int(transaction_id),

            "TX_DATETIME":
                tx_datetime,

            "PRODUCER_TIMESTAMP":
                producer_timestamp,

            "CUSTOMER_ID":
                int(customer_id),

            "TERMINAL_ID":
                (
                    int(terminal_id)
                    if terminal_id is not None
                    else None
                ),

            "TX_AMOUNT":
                float(
                    row["TX_AMOUNT"]
                    if row["TX_AMOUNT"] is not None
                    else 0.0
                ),

            "FRAUD_PROBABILITY":
                float(probability),

            "FRAUD_PREDICTION":
                int(prediction),

            "THRESHOLD":
                float(THRESHOLD),

            "TX_FRAUD":
                int(
                    row["TX_FRAUD"]
                    if row["TX_FRAUD"] is not None
                    else 0
                ),

            "PREDICTION_START_TIMESTAMP":
                prediction_start_timestamp,

            "PREDICTION_END_TIMESTAMP":
                prediction_end_timestamp,

            "PREDICTION_LATENCY_MS":
                float(prediction_latency_ms),

            "END_TO_END_LATENCY_MS":
                (
                    float(end_to_end_latency_ms)
                    if end_to_end_latency_ms is not None
                    else None
                ),
        })


        # ====================================================
        # IMPORTANT:
        #
        # CURRENT TRANSACTION ADDED AFTER PREDICTION
        # ====================================================

        history.append(

            (
                tx_datetime,
                int(transaction_id),
                current_features
            )

        )


    # ========================================================
    # NO OUTPUT
    # ========================================================

    if not prediction_rows:

        print(
            "NO PREDICTIONS GENERATED."
        )

        return


    # ========================================================
    # LATENCY METRICS
    # ========================================================

    print_latency_statistics(
        "PREDICTION LATENCY - CURRENT BATCH",
        batch_prediction_latencies
    )

    print_latency_statistics(
        "END-TO-END LATENCY - CURRENT BATCH",
        batch_end_to_end_latencies
    )

    print_latency_statistics(
        "PREDICTION LATENCY - CUMULATIVE",
        prediction_latency_history
    )

    print_latency_statistics(
        "END-TO-END LATENCY - CUMULATIVE",
        end_to_end_latency_history
    )


    # ========================================================
    # OUTPUT SCHEMA
    # ========================================================

    output_schema = StructType([

        StructField(
            "TRANSACTION_ID",
            IntegerType(),
            False
        ),

        StructField(
            "TX_DATETIME",
            TimestampType(),
            True
        ),

        StructField(
            "PRODUCER_TIMESTAMP",
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
            "PREDICTION_START_TIMESTAMP",
            TimestampType(),
            True
        ),

        StructField(
            "PREDICTION_END_TIMESTAMP",
            TimestampType(),
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


    output_df = spark.createDataFrame(
        prediction_rows,
        schema=output_schema
    )


    print()
    print(
        "PREDICTION DATA:"
    )

    output_df.select(
        "TRANSACTION_ID",
        "CUSTOMER_ID",
        "TX_AMOUNT",
        "FRAUD_PROBABILITY",
        "FRAUD_PREDICTION",
        "TX_FRAUD",
        "PREDICTION_LATENCY_MS",
        "END_TO_END_LATENCY_MS"
    ).show(
        len(prediction_rows),
        truncate=False
    )


    # ========================================================
    # KAFKA OUTPUT
    # ========================================================

    kafka_output = (

        output_df

        .select(

            col(
                "TRANSACTION_ID"
            )
            .cast("string")
            .alias("key"),

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
    print(
        f"WROTE {len(prediction_rows)} "
        f"PREDICTION(S) TO "
        f"{OUTPUT_TOPIC}"
    )

    print("=" * 100)


# ============================================================
# 16. START
# ============================================================

print()
print("=" * 100)
print("STARTING LSTM REAL-TIME INFERENCE")
print("=" * 100)

print(
    f"Input topic      : {INPUT_TOPIC}"
)

print(
    f"Output topic     : {OUTPUT_TOPIC}"
)

print(
    f"Sequence         : {SEQ_LEN}"
)

print(
    f"Features         : {NUM_FEATURES}"
)

print(
    f"Threshold        : {THRESHOLD}"
)

print(
    "Padding          : LEFT ZERO PADDING"
)

print(
    "Latency metrics  : "
    "Prediction + End-to-End"
)

print(
    "Statistics       : "
    "Average / P50 / P95 / P99"
)

print("=" * 100)


query = (

    transactions_features

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


query.awaitTermination()