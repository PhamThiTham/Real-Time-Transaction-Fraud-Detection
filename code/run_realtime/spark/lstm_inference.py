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
#     5 PREVIOUS transactions x 15 features
#
# IMPORTANT:
#
#     Current transaction is NOT used for prediction.
#
#     Prediction T6:
#
#         history:
#             T1 T2 T3 T4 T5
#
#         sequence:
#             T1 T2 T3 T4 T5
#
#         predict:
#             T6
#
#         AFTER prediction:
#             add T6 to history
#
# Therefore:
#
#     T1 -> X=[0,0,0,0,0]     (left zero-padded)
#     T2 -> X=[0,0,0,0,T1]
#     T3 -> X=[0,0,0,T1,T2]
#     T4 -> X=[0,0,T1,T2,T3]
#     T5 -> X=[0,T1,T2,T3,T4]
#     T6 -> X=[T1,T2,T3,T4,T5]
#
# LEFT ZERO-PADDING.
#
# When a customer has fewer than 5 previous transactions,
# the sequence is LEFT-padded with zero feature vectors so
# that a prediction is produced for EVERY transaction.
#
# The current transaction is NEVER part of its own X
# (this still avoids current-row and target leakage).
#
# Expected LSTM input:
#
#     (1, 5, 15)
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
    lit,
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
    "/opt/spark-data/checkpoint/lstm_inference"
)

MODEL_PATH = (
    "/opt/spark-models/"
    "Tham_lstm_attn_checkpoint.pth"
)

SEQ_LEN = 5

NUM_FEATURES = 15

THRESHOLD = 0.7

DEVICE = torch.device("cpu")

MAX_LATENCY_SAMPLES = 10000


# ============================================================
# FEATURE STANDARDIZATION (StandardScaler from training)
# ============================================================
#
# IMPORTANT:
#   The LSTM model was trained on features standardized with a
#   sklearn StandardScaler fitted on the training set
#   (notebook Tham_DuBao_GianLan_GiaoDich.ipynb,
#    train window = 2018-07-11 -> 2018-07-18).
#
#   Real-time features are RAW (TX_AMOUNT, counts, averages...).
#   They MUST be standardized with the SAME mean/std before
#   being fed to the model; otherwise the LSTM/sigmoid output
#   saturates and predictions become meaningless.
#
#   Order matches FEATURE_COLUMNS exactly.
#
# ============================================================

FEATURE_MEAN = [
    53.76273204040162,      # 1  TX_AMOUNT
    0.2845445852259144,     # 2  TX_DURING_WEEKEND
    0.17373894334209897,    # 3  TX_DURING_NIGHT
    3.549575663399474,      # 4  CUSTOMER_ID_NB_TX_1DAY_WINDOW
    53.73393176675504,      # 5  CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW
    18.88403956490557,      # 6  CUSTOMER_ID_NB_TX_7DAY_WINDOW
    53.66266882583177,      # 7  CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW
    77.90946987807793,      # 8  CUSTOMER_ID_NB_TX_30DAY_WINDOW
    53.5233331667343,       # 9  CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW
    0.9971910112359551,     # 10 TERMINAL_ID_NB_TX_1DAY_WINDOW
    0.0059967427683480755,  # 11 TERMINAL_ID_RISK_1DAY_WINDOW
    7.047020678938561,      # 12 TERMINAL_ID_NB_TX_7DAY_WINDOW
    0.009511224112498533,   # 13 TERMINAL_ID_RISK_7DAY_WINDOW
    30.067729500358595,     # 14 TERMINAL_ID_NB_TX_30DAY_WINDOW
    0.008949904198249757,   # 15 TERMINAL_ID_RISK_30DAY_WINDOW
]

FEATURE_STD = [
    42.43813058744915,
    0.45119725646830644,
    0.3788848412228575,
    1.8302544305676758,
    34.97844326499479,
    7.616160602877247,
    30.45348722840948,
    28.808621200612603,
    29.12568463008163,
    1.0165487018827732,
    0.07346656643709902,
    3.039536045798072,
    0.07725363358380612,
    8.30088569344886,
    0.06232344757511963,
]


def standardize_features(
    features
):

    if len(features) != NUM_FEATURES:

        raise ValueError(
            f"Invalid feature length for "
            f"standardization: "
            f"len={len(features)}, "
            f"expected={NUM_FEATURES}"
        )

    return [
        (
            float(features[i])
            - FEATURE_MEAN[i]
        )
        / FEATURE_STD[i]
        for i in range(NUM_FEATURES)
    ]


# ============================================================
# 1.5 RUN CONTROL
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
#
# IMPORTANT:
#
# This order MUST be exactly the same as the order used
# during model training.
#
# ============================================================

FEATURE_COLUMNS = [

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
# 4. FEATURE COUNT CHECK
# ============================================================

if len(FEATURE_COLUMNS) != NUM_FEATURES:

    raise ValueError(
        f"Expected {NUM_FEATURES} features, "
        f"got {len(FEATURE_COLUMNS)}"
    )


# ============================================================
# 5. PRINT CONFIGURATION
# ============================================================

print()
print("=" * 100)
print("LSTM REAL-TIME FRAUD INFERENCE")
print("=" * 100)

print(
    f"Input topic       : {INPUT_TOPIC}"
)

print(
    f"Output topic      : {OUTPUT_TOPIC}"
)

print(
    f"Sequence length   : {SEQ_LEN}"
)

print(
    f"Number features   : {NUM_FEATURES}"
)

print(
    f"Threshold         : {THRESHOLD}"
)

print(
    "Padding           : LEFT-ZERO-PAD"
)

print(
    "Current TX in X   : NO"
)

print(
    "Current TX added  : AFTER prediction"
)

print()
print("FEATURE ORDER:")

for i, feature in enumerate(
    FEATURE_COLUMNS,
    start=1
):

    print(
        f"  {i:02d}. {feature}"
    )

print("=" * 100)


# ============================================================
# 6. LATENCY FUNCTIONS
# ============================================================

def percentile(
    values,
    percentile_value
):

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

    fraction = (
        rank - lower_index
    )

    lower_value = (
        sorted_values[lower_index]
    )

    upper_value = (
        sorted_values[upper_index]
    )

    return (
        lower_value
        + (
            upper_value
            - lower_value
        )
        * fraction
    )


def calculate_latency_statistics(
    values
):

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

        "average":
            sum(values) / len(values),

        "p50":
            percentile(
                values,
                50
            ),

        "p95":
            percentile(
                values,
                95
            ),

        "p99":
            percentile(
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

        print(
            "No latency samples."
        )

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
# 7. PRODUCER TIMESTAMP
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
            "WARNING: Cannot parse "
            f"PRODUCER_TIMESTAMP="
            f"{producer_timestamp}"
        )

        print(
            f"ERROR: {e}"
        )

        return None


# ============================================================
# 8. ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.linear_out = nn.Linear(
            dim * 2,
            dim
        )

        self.mask = None


    def set_mask(
        self,
        mask
    ):

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
# 9. MODEL
# ============================================================

class FraudLSTMWithAttention(
    nn.Module
):

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

        self.num_features = (
            num_features
        )

        self.hidden_size = (
            hidden_size
        )

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


    def forward(
        self,
        x
    ):

        hidden_states, _ = (
            self.lstm(x)
        )

        last_transaction = (
            x[:, -1, :]
        )

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

        hidden = self.relu(
            hidden
        )

        output = self.fc2(
            hidden
        )

        output = self.sigmoid(
            output
        )

        return output


# ============================================================
# 10. LOAD MODEL
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
    checkpoint[
        "model_state_dict"
    ]
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
# 11. INPUT SCHEMA
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
# 12. SPARK
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
# 13. CUSTOMER HISTORY
# ============================================================
#
# customer_history[customer_id]
#     =
# deque containing ONLY previous transactions
#
# Maximum:
#
#     5 transactions
#
# ============================================================

customer_history = defaultdict(
    lambda: deque(
        maxlen=SEQ_LEN
    )
)


# ============================================================
# 14. READ KAFKA
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
        "earliest"
    )
    .option(
        "failOnDataLoss",
        "true"
    )
    .load()
)


# ============================================================
# 15. PARSE JSON
# ============================================================

transactions_features = (
    raw_stream
    .select(
        col("value").cast("string").alias("_RAW_VALUE"),
        from_json(
            col("value").cast("string"),
            input_schema
        ).alias("data")
    )
    .select("_RAW_VALUE", "data.*")
    .withColumn(
        "TX_DATETIME",
        to_timestamp(
            col("TX_DATETIME")
        )
    )
)


# ============================================================
# 16. BUILD 5-SLOT SEQUENCE (LEFT ZERO-PADDING)
# ============================================================
#
# IMPORTANT:
#
# LEFT ZERO-PADDING.
#
# If history has:
#
#     0, 1, 2, 3, 4 transactions
#
# the sequence is LEFT-padded with zero feature vectors so
# that the most recent transaction stays at the last position
# (the model uses x[:, -1, :] as its attention context).
#
# This produces a prediction for EVERY transaction.
#
# ============================================================

def build_sequence(
    history
):

    sequence = []

    for item in history:

        feature_vector = item[2]

        if len(feature_vector) != NUM_FEATURES:

            raise ValueError(
                "Invalid feature vector "
                f"length={len(feature_vector)}, "
                f"expected={NUM_FEATURES}"
            )

        sequence.append(
            feature_vector
        )

    sequence = sequence[-SEQ_LEN:]

    if len(sequence) < SEQ_LEN:

        pad_count = SEQ_LEN - len(sequence)

        sequence = (
            [[0.0] * NUM_FEATURES for _ in range(pad_count)]
            + sequence
        )

    if len(sequence) != SEQ_LEN:

        raise ValueError(
            f"Invalid sequence length "
            f"{len(sequence)}, "
            f"expected={SEQ_LEN}"
        )

    return sequence


# ============================================================
# 17. PROCESS BATCH
# ============================================================

def process_batch(
    batch_df,
    batch_id
):

    global END_RECEIVED

    print()
    print("=" * 100)

    print(
        f"LSTM INFERENCE - "
        f"BATCH ID = {batch_id}"
    )

    print("=" * 100)


    if batch_df.isEmpty():

        print(
            "EMPTY BATCH"
        )

        return


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
                "LSTM_INFERENCE"
            )
            print("=" * 100)

            END_RECEIVED = True

        return


    # --------------------------------------------------------
    # Process transactions chronologically.
    # --------------------------------------------------------

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
    # EACH TRANSACTION - build X + update history
    # ========================================================

    pending = []


    for row in rows:


        transaction_id = row["TRANSACTION_ID"]

        customer_id = row["CUSTOMER_ID"]

        tx_datetime = row["TX_DATETIME"]

        producer_timestamp = row["PRODUCER_TIMESTAMP"]


        # ====================================================
        # BASIC VALIDATION
        # ====================================================

        if transaction_id is None:

            log("SKIP: TRANSACTION_ID is NULL")

            continue


        if customer_id is None:

            log("SKIP: CUSTOMER_ID is NULL")

            continue


        if tx_datetime is None:

            log(
                f"SKIP: TX_DATETIME is NULL "
                f"for transaction {transaction_id}"
            )

            continue


        # ====================================================
        # CURRENT FEATURES
        # ====================================================

        current_features = []


        for feature_name in FEATURE_COLUMNS:

            value = row[feature_name]

            if value is None:

                value = 0.0

            current_features.append(
                float(value)
            )


        if len(current_features) != NUM_FEATURES:

            raise ValueError(
                f"Invalid current feature "
                f"length={len(current_features)}"
            )


        # ====================================================
        # STANDARDIZE FEATURES
        # ====================================================
        #
        # IMPORTANT:
        #   Apply the SAME StandardScaler used during training
        #   so the model receives features with the same
        #   distribution as in training.
        #
        # ====================================================

        current_features = standardize_features(
            current_features
        )


        # ====================================================
        # HISTORY BEFORE CURRENT TRANSACTION
        # ====================================================

        history = customer_history[customer_id]


        log()
        log("-" * 100)
        log(f"TRANSACTION_ID : {transaction_id}")
        log(f"CUSTOMER_ID    : {customer_id}")
        log(f"TX_DATETIME    : {tx_datetime}")
        log(
            f"TX_AMOUNT      : "
            f"{float(row['TX_AMOUNT'] or 0):.2f}"
        )
        log(f"HISTORY LENGTH : {len(history)}")


        # ====================================================
        # BUILD X (WITH LEFT ZERO-PADDING)
        # ====================================================

        # IMPORTANT:
        #   The sequence must END with the CURRENT transaction
        #   (last element), exactly like the training
        #   FraudSequenceDataset.
        #
        #   Only PAST transactions are read from history; the
        #   current transaction is appended to history only
        #   AFTER X is built (no self-leakage).
        #
        #   (Fixed: previously only the past transactions were
        #   passed to the model, so the current transaction -
        #   the most predictive signal, e.g. TX_AMOUNT - was
        #   never seen by the model.)

        sequence = build_sequence(
            list(history) + [
                (
                    tx_datetime,
                    int(transaction_id),
                    current_features
                )
            ]
        )


        if sequence is None:

            raise RuntimeError(
                "Sequence should contain "
                f"{SEQ_LEN} transactions."
            )


        x = torch.tensor(
            sequence,
            dtype=torch.float32,
            device=DEVICE
        ).unsqueeze(0)


        expected_shape = (
            1,
            SEQ_LEN,
            NUM_FEATURES
        )


        if tuple(x.shape) != expected_shape:

            raise ValueError(
                f"Invalid model input "
                f"shape={tuple(x.shape)}, "
                f"expected={expected_shape}"
            )


        pending.append(
            {
                "x": x,
                "row": row,
                "transaction_id": int(transaction_id),
                "tx_datetime": tx_datetime,
                "producer_timestamp": producer_timestamp,
            }
        )


        # ------------------------------------------------
        # ADD CURRENT TRANSACTION TO HISTORY
        # AFTER X IS BUILT (no self-leakage).
        # ------------------------------------------------

        history.append(
            (
                tx_datetime,
                int(transaction_id),
                current_features
            )
        )


    # ========================================================
    # NO PREDICTIONS
    # ========================================================

    if not pending:

        print()
        print(
            "NO PREDICTIONS GENERATED "
            "IN THIS BATCH."
        )

        if end_marker_present:

            write_end_marker()

            print()
            print("=" * 100)
            print(
                "[ALL DONE] "
                "LSTM_INFERENCE"
            )
            print("=" * 100)

            END_RECEIVED = True

        return


    # ========================================================
    # BATCHED MODEL INFERENCE
    # ========================================================

    x_batch = torch.cat(
        [item["x"] for item in pending],
        dim=0
    )


    prediction_start_timestamp = (
        datetime.now(timezone.utc)
    )


    prediction_start = time.perf_counter()


    with torch.no_grad():

        probability_batch = model(x_batch)


    prediction_end = time.perf_counter()


    prediction_end_timestamp = (
        datetime.now(timezone.utc)
    )


    batch_latency_ms = (
        prediction_end - prediction_start
    ) * 1000.0


    probabilities = (
        probability_batch.squeeze(1).tolist()
    )


    print()

    print(
        f"BATCHED INFERENCE : "
        f"{len(pending)} sample(s) -> "
        f"{batch_latency_ms:.3f} ms "
        f"({batch_latency_ms / len(pending):.3f} ms/tx)"
    )


    # ========================================================
    # BUILD PREDICTION ROWS
    # ========================================================

    for item, probability in zip(
        pending,
        probabilities
    ):

        row = item["row"]

        transaction_id = item["transaction_id"]

        tx_datetime = item["tx_datetime"]

        producer_timestamp = item["producer_timestamp"]


        prediction = int(
            probability >= THRESHOLD
        )


        # ------------------------------------------------
        # END-TO-END LATENCY
        # ------------------------------------------------

        end_to_end_latency_ms = None

        producer_dt = parse_producer_timestamp(
            producer_timestamp
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


        # ------------------------------------------------
        # PREDICTION LATENCY HISTORY
        # ------------------------------------------------

        prediction_latency_history.append(
            batch_latency_ms
        )

        batch_prediction_latencies.append(
            batch_latency_ms
        )


        prediction_rows.append({

            "TRANSACTION_ID":
                transaction_id,

            "TX_DATETIME":
                tx_datetime,

            "PRODUCER_TIMESTAMP":
                (
                    str(producer_timestamp)
                    if producer_timestamp is not None
                    else None
                ),

            "CUSTOMER_ID":
                row["CUSTOMER_ID"],

            "TERMINAL_ID":
                row["TERMINAL_ID"],

            "TX_AMOUNT":
                float(row["TX_AMOUNT"] or 0.0),

            "FRAUD_PROBABILITY":
                float(probability),

            "FRAUD_PREDICTION":
                prediction,

            "THRESHOLD":
                THRESHOLD,

            "TX_FRAUD":
                int(row["TX_FRAUD"] or 0),

            "PREDICTION_START_TIMESTAMP":
                prediction_start_timestamp,

            "PREDICTION_END_TIMESTAMP":
                prediction_end_timestamp,

            "PREDICTION_LATENCY_MS":
                float(batch_latency_ms),

            "END_TO_END_LATENCY_MS":
                (
                    float(end_to_end_latency_ms)
                    if end_to_end_latency_ms is not None
                    else None
                ),
        })


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

        StructField("TRANSACTION_ID", IntegerType(), False),

        StructField("TX_DATETIME", TimestampType(), True),

        StructField("PRODUCER_TIMESTAMP", StringType(), True),

        StructField("CUSTOMER_ID", IntegerType(), True),

        StructField("TERMINAL_ID", IntegerType(), True),

        StructField("TX_AMOUNT", DoubleType(), True),

        StructField("FRAUD_PROBABILITY", DoubleType(), True),

        StructField("FRAUD_PREDICTION", IntegerType(), True),

        StructField("THRESHOLD", DoubleType(), True),

        StructField("TX_FRAUD", IntegerType(), True),

        StructField("PREDICTION_START_TIMESTAMP", TimestampType(), True),

        StructField("PREDICTION_END_TIMESTAMP", TimestampType(), True),

        StructField("PREDICTION_LATENCY_MS", DoubleType(), True),

        StructField("END_TO_END_LATENCY_MS", DoubleType(), True),

    ])


    output_df = spark.createDataFrame(
        prediction_rows,
        schema=output_schema
    )


    # ========================================================
    # DISPLAY OUTPUT (only when VERBOSE)
    # ========================================================

    if VERBOSE:

        print()
        print("PREDICTION DATA:")

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

            col("TRANSACTION_ID")
            .cast("string")
            .alias("key"),

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

        .coalesce(1)

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

    print()
    print("=" * 100)
    print(
        f"[BATCH DONE] "
        f"LSTM_INFERENCE - batch {batch_id}: "
        f"processed {len(prediction_rows)} transactions -> "
        f"{OUTPUT_TOPIC}"
    )
    print("=" * 100)


    # ========================================================
    # END MARKER - propagate downstream
    # ========================================================

    if end_marker_present:

        write_end_marker()

        print()
        print("=" * 100)
        print(
            "[ALL DONE] "
            "LSTM_INFERENCE"
        )
        print("=" * 100)

        END_RECEIVED = True



# ============================================================
# 18. START STREAMING
# ============================================================

print()
print("=" * 100)

print(
    "STARTING LSTM REAL-TIME INFERENCE"
)

print("=" * 100)

print(
    f"Input topic      : "
    f"{INPUT_TOPIC}"
)

print(
    f"Output topic     : "
    f"{OUTPUT_TOPIC}"
)

print(
    f"Sequence         : "
    f"{SEQ_LEN} previous transactions"
)

print(
    f"Features         : "
    f"{NUM_FEATURES}"
)

print(
    f"Threshold        : "
    f"{THRESHOLD}"
)

print(
    "Padding           : LEFT-ZERO-PAD"
)

print(
    "Current TX in X  : NO"
)

print(
    "Prediction       : "
    "EVERY TX (left zero-padding)"
)

print(
    "History update   : "
    "AFTER prediction"
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


# ============================================================
# WAIT FOR END MARKER / COMPLETION
# ============================================================

print()
print("=" * 100)

print(
    "[READY] LSTM_INFERENCE - "
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
    "[STOPPED] LSTM_INFERENCE"
)

print("=" * 100)