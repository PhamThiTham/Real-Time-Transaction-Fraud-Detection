# ============================================================
# REAL-TIME FRAUD DETECTION EVALUATION
# ============================================================
#
# Purpose:
#   Evaluate the real-time fraud detection pipeline:
#
#       Kafka
#          |
#          v
#       Spark
#          |
#          v
#       LSTM
#          |
#          v
#       PostgreSQL
#
# IMPORTANT:
#
#   TX_DATETIME
#       = original dataset transaction time
#
#   PRODUCER_TIMESTAMP
#       = real wall-clock time when transaction was produced
#
#   PREDICTION_START_TIMESTAMP
#       = time LSTM inference started
#
#   PREDICTION_END_TIMESTAMP
#       = time LSTM inference finished
#
#   SINK_TIMESTAMP
#       = time PostgreSQL sink received/wrote transaction
#
# Therefore:
#
#   Prediction latency:
#
#       PREDICTION_END_TIMESTAMP
#       -
#       PREDICTION_START_TIMESTAMP
#
#   End-to-end latency:
#
#       SINK_TIMESTAMP
#       -
#       PRODUCER_TIMESTAMP
#
#   Throughput:
#
#       number of processed transactions
#       /
#       real pipeline duration
#
#   DO NOT use TX_DATETIME for real-time throughput.
#
# ============================================================

import csv
import json
import math
import os
import statistics

from datetime import datetime, timezone

import psycopg2

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)


# ============================================================
# CONFIG
# ============================================================

DB_HOST = "localhost"

DB_PORT = 5432

DB_NAME = "fraud_detection"

DB_USER = "fraud"

DB_PASSWORD = "fraud123"

TABLE_NAME = "fraud_predictions"


# ============================================================
# RESULT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "evaluation_results"
)

HISTORY_CSV = os.path.join(
    RESULT_DIR,
    "realtime_evaluation_history.csv"
)

LATEST_JSON = os.path.join(
    RESULT_DIR,
    "realtime_evaluation_latest.json"
)

PR_CURVE_PNG = os.path.join(
    RESULT_DIR,
    "precision_recall_curve.png"
)


# ============================================================
# PRINT
# ============================================================

def separator():

    print("=" * 78)


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value):

    if value is None:

        return None

    try:

        value = float(value)

        if math.isfinite(value):

            return value

    except Exception:

        pass

    return None


# ============================================================
# COMPUTE AVERAGE PRECISION
# ============================================================
#
# Formula from the Fraud Detection Handbook
# (Chapter 4, section "Precision-Recall curve"):
#
#     AP = sum(
#         (recall[i] - recall[i-1])
#         * precision[i]
#     )
#
# i.e. the weighted mean of precisions achieved at each
# threshold, with the increase in recall from the previous
# threshold used as the weight.
#
# IMPORTANT (input ordering):
#
#     recall[i] - recall[i-1] >= 0
#
# requires the arrays to be ordered in increasing recall
# (recall from 0 to 1). sklearn's precision_recall_curve()
# returns decreasing recall, so the arrays MUST be reversed
# with [::-1] before calling this function (see the call
# site below, mirroring the handbook:
#
#     precision = precision[::-1]
#     recall    = recall[::-1]
#
# )
#
# ============================================================

def compute_AP(precision, recall):

    AP = 0

    n_thresholds = len(precision)

    for i in range(1, n_thresholds):

        if recall[i] - recall[i-1] >= 0:

            AP = AP + (
                recall[i] - recall[i-1]
            ) * precision[i]

    return AP


# ============================================================
# COMPUTE PRECISION TOP-K
# ============================================================
#
# Precision top-k metric from the Fraud Detection Handbook
# (Chapter 4, section "Precision top-k metrics"):
#
#     P@k = |A^Fraud| / |A| = |A^Fraud| / k
#
# where A is the set of alerts, i.e. the top-k transactions
# with the highest fraud probability, and A^Fraud is the
# subset of A that is actually fraudulent.
#
# Implementation follows the handbook's precision_top_k_day():
#
#     1. rank transactions by decreasing fraud probability
#     2. keep the top-k most suspicious transactions
#     3. P@k = (# fraudulent transactions in top-k) / k
#
# Returns None when fewer than k transactions are available.
#
# Input: pairs = list of (fraud_probability, actual) tuples.
#
# ============================================================

def compute_precision_top_k(pairs, k):

    if len(pairs) < k:

        return None

    # Rank transactions by decreasing fraud probability
    ranked = sorted(
        pairs,
        key=lambda item: item[0],
        reverse=True
    )

    # Top-k most suspicious transactions
    top_k = ranked[:k]

    # P@k = (# fraudulent transactions in top-k) / k
    fraud_in_top_k = sum(
        1
        for item in top_k
        if item[1] == 1
    )

    return (
        fraud_in_top_k
        / float(k)
    )


# ============================================================
# PARSE TIMESTAMP
# ============================================================

def parse_timestamp(value):

    """
    Convert PostgreSQL timestamp / datetime / string
    into timezone-aware UTC datetime.

    Supported examples:

        2026-08-18 04:48:14.727592
        2026-08-18 04:48:34.879
        2026-08-18T04:48:14.727592+00:00
        2026-08-18T04:48:28.676Z
    """

    if value is None:

        return None

    # --------------------------------------------------------
    # Already datetime
    # --------------------------------------------------------

    if isinstance(value, datetime):

        result = value

        if result.tzinfo is None:

            result = result.replace(
                tzinfo=timezone.utc
            )

        return result.astimezone(
            timezone.utc
        )

    # --------------------------------------------------------
    # String
    # --------------------------------------------------------

    try:

        text = str(value).strip()

        if not text:

            return None

        if text.endswith("Z"):

            text = (
                text[:-1]
                + "+00:00"
            )

        result = datetime.fromisoformat(
            text
        )

        if result.tzinfo is None:

            result = result.replace(
                tzinfo=timezone.utc
            )

        return result.astimezone(
            timezone.utc
        )

    except Exception:

        return None


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    values,
    p
):

    if not values:

        return None

    values = sorted(
        float(x)
        for x in values
    )

    if len(values) == 1:

        return values[0]

    index = (
        (len(values) - 1)
        * p
        / 100.0
    )

    lower = math.floor(index)

    upper = math.ceil(index)

    if lower == upper:

        return values[lower]

    fraction = index - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        )
        * fraction
    )


# ============================================================
# LATENCY STATISTICS
# ============================================================

def latency_statistics(values):

    if not values:

        return {
            "samples": 0,
            "average_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    return {

        "samples": len(values),

        "average_ms": statistics.mean(
            values
        ),

        "p50_ms": percentile(
            values,
            50
        ),

        "p95_ms": percentile(
            values,
            95
        ),

        "p99_ms": percentile(
            values,
            99
        ),

        "min_ms": min(
            values
        ),

        "max_ms": max(
            values
        ),

    }


# ============================================================
# PRINT LATENCY
# ============================================================

def print_latency(
    title,
    values
):

    separator()

    print(title)

    separator()

    stats = latency_statistics(
        values
    )

    if stats["samples"] == 0:

        print(
            "No latency samples."
        )

        return

    print(
        f"Samples : "
        f"{stats['samples']}"
    )

    print(
        f"Average : "
        f"{stats['average_ms']:.3f} ms"
    )

    print(
        f"P50     : "
        f"{stats['p50_ms']:.3f} ms"
    )

    print(
        f"P95     : "
        f"{stats['p95_ms']:.3f} ms"
    )

    print(
        f"P99     : "
        f"{stats['p99_ms']:.3f} ms"
    )

    print(
        f"Min     : "
        f"{stats['min_ms']:.3f} ms"
    )

    print(
        f"Max     : "
        f"{stats['max_ms']:.3f} ms"
    )


# ============================================================
# PLOT PRECISION-RECALL CURVE
# ============================================================

def plot_precision_recall_curve(
    precision,
    recall,
    average_precision
):

    if (
        precision is None
        or recall is None
        or average_precision is None
    ):

        return

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    plt.figure(
        figsize=(8, 6)
    )

    # Step-wise function (not linear interpolation), as
    # recommended by the Fraud Detection Handbook:
    #
    #     "Linear interpolation ... should not be used for
    #      plotting PR curves, nor for assessing their AUC.
    #      The use of the step-wise function for plotting,
    #      and AP as a measure of AUC are well established."
    #
    plt.step(
        recall,
        precision,
        linewidth=2,
        label=(
            f"AP = "
            f"{average_precision:.6f}"
        )
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        PR_CURVE_PNG,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# DATABASE
# ============================================================

def connect_db():

    return psycopg2.connect(

        host=DB_HOST,

        port=DB_PORT,

        dbname=DB_NAME,

        user=DB_USER,

        password=DB_PASSWORD,

    )


# ============================================================
# GET COLUMNS
# ============================================================

def get_columns(
    connection
):

    cursor = connection.cursor()

    cursor.execute(

        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
        ORDER BY ordinal_position
        """,

        (
            TABLE_NAME,
        )

    )

    columns = [

        row[0].lower()

        for row in cursor.fetchall()

    ]

    cursor.close()

    return columns


# ============================================================
# SAVE JSON
# ============================================================

def make_json_serializable(value):

    if isinstance(
        value,
        datetime
    ):

        return value.isoformat()

    if isinstance(
        value,
        dict
    ):

        return {
            str(k):
            make_json_serializable(v)

            for k, v in value.items()
        }

    if isinstance(
        value,
        list
    ):

        return [
            make_json_serializable(v)
            for v in value
        ]

    return value


def save_latest_json(
    result
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    result = make_json_serializable(
        result
    )

    with open(
        LATEST_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# SAVE HISTORY CSV
# ============================================================

def save_history_csv(
    result
):

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    summary = result["summary"]

    confusion = result["confusion_matrix"]

    classification = result[
        "classification_metrics"
    ]

    probability = result[
        "probability_metrics"
    ]

    prediction_latency = result[
        "prediction_latency"
    ]

    e2e_latency = result[
        "end_to_end_latency"
    ]

    throughput = result[
        "throughput"
    ]

    precision_top_k = result[
        "precision_top_k"
    ]

    row = {

        "evaluation_timestamp":
            result[
                "evaluation_timestamp"
            ],

        "total_transactions":
            summary[
                "total_transactions"
            ],

        "actual_fraud":
            summary[
                "actual_fraud"
            ],

        "actual_legitimate":
            summary[
                "actual_legitimate"
            ],

        "predicted_fraud":
            summary[
                "predicted_fraud"
            ],

        "predicted_legitimate":
            summary[
                "predicted_legitimate"
            ],

        "tn":
            confusion["tn"],

        "fp":
            confusion["fp"],

        "fn":
            confusion["fn"],

        "tp":
            confusion["tp"],

        "accuracy":
            classification["accuracy"],

        "precision":
            classification["precision"],

        "recall":
            classification["recall"],

        "f1_score":
            classification["f1_score"],

        "fpr":
            classification["fpr"],

        "roc_auc":
            probability["roc_auc"],

        "average_precision":
            probability[
                "average_precision"
            ],

        "pr_curve_points":
            probability[
                "pr_curve_points"
            ],

        "prediction_latency_samples":
            prediction_latency[
                "samples"
            ],

        "prediction_latency_average_ms":
            prediction_latency[
                "average_ms"
            ],

        "prediction_latency_p50_ms":
            prediction_latency[
                "p50_ms"
            ],

        "prediction_latency_p95_ms":
            prediction_latency[
                "p95_ms"
            ],

        "prediction_latency_p99_ms":
            prediction_latency[
                "p99_ms"
            ],

        "prediction_latency_min_ms":
            prediction_latency[
                "min_ms"
            ],

        "prediction_latency_max_ms":
            prediction_latency[
                "max_ms"
            ],

        "e2e_latency_samples":
            e2e_latency[
                "samples"
            ],

        "e2e_latency_average_ms":
            e2e_latency[
                "average_ms"
            ],

        "e2e_latency_p50_ms":
            e2e_latency[
                "p50_ms"
            ],

        "e2e_latency_p95_ms":
            e2e_latency[
                "p95_ms"
            ],

        "e2e_latency_p99_ms":
            e2e_latency[
                "p99_ms"
            ],

        "e2e_latency_min_ms":
            e2e_latency[
                "min_ms"
            ],

        "e2e_latency_max_ms":
            e2e_latency[
                "max_ms"
            ],

        "throughput_transactions":
            throughput[
                "transactions"
            ],

        "throughput_duration_seconds":
            throughput[
                "duration_seconds"
            ],

        "throughput_transactions_per_sec":
            throughput[
                "transactions_per_sec"
            ],

        "precision_top_k_50":
            precision_top_k[
                "precision_at_50"
            ],

        "precision_top_k_100":
            precision_top_k[
                "precision_at_100"
            ],

        "precision_top_k_200":
            precision_top_k[
                "precision_at_200"
            ],

    }

    file_exists = os.path.exists(
        HISTORY_CSV
    )

    with open(
        HISTORY_CSV,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            )
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    separator()

    print(
        "REAL-TIME FRAUD DETECTION EVALUATION"
    )

    separator()

    print(
        f"Database : {DB_NAME}"
    )

    print(
        f"Table    : {TABLE_NAME}"
    )

    print()

    # ========================================================
    # CONNECT
    # ========================================================

    connection = connect_db()

    # ========================================================
    # COLUMNS
    # ========================================================

    columns = get_columns(
        connection
    )

    print(
        "Detected columns:"
    )

    for column in columns:

        print(
            f"  - {column}"
        )

    # ========================================================
    # REQUIRED
    # ========================================================

    required_columns = [

        "transaction_id",

        "fraud_probability",

        "fraud_prediction",

        "tx_fraud",

        "producer_timestamp",

        "prediction_start_timestamp",

        "prediction_end_timestamp",

        "sink_timestamp",

    ]

    missing = [

        column

        for column in required_columns

        if column not in columns

    ]

    if missing:

        connection.close()

        raise RuntimeError(

            "Missing required columns: "

            + ", ".join(missing)

        )

    # ========================================================
    # QUERY
    # ========================================================

    cursor = connection.cursor()

    query = f"""
        SELECT
            transaction_id,
            tx_datetime,
            customer_id,
            terminal_id,
            tx_amount,
            fraud_probability,
            fraud_prediction,
            threshold,
            tx_fraud,
            producer_timestamp,
            prediction_start_timestamp,
            prediction_end_timestamp,
            prediction_latency_ms,
            end_to_end_latency_ms,
            sink_timestamp
        FROM "{TABLE_NAME}"
        ORDER BY transaction_id
    """

    cursor.execute(
        query
    )

    rows = cursor.fetchall()

    print()

    print(
        f"Rows read from PostgreSQL : "
        f"{len(rows)}"
    )

    if not rows:

        cursor.close()

        connection.close()

        raise RuntimeError(
            "No rows found in fraud_predictions."
        )

    # ========================================================
    # COLUMN INDEX
    # ========================================================

    index = {

        name: position

        for position, name

        in enumerate(

            [

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

            ]

        )

    }

    # ========================================================
    # ARRAYS
    # ========================================================

    y_true = []

    y_pred = []

    y_probability = []

    prediction_latencies = []

    e2e_latencies = []

    # ========================================================
    # REAL-TIME TIMESTAMPS
    # ========================================================

    producer_timestamps = []

    sink_timestamps = []

    # ========================================================
    # PROCESS ROWS
    # ========================================================

    for row in rows:

        # ----------------------------------------------------
        # Actual
        # ----------------------------------------------------

        actual = row[
            index["TX_FRAUD"]
        ]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = row[
            index["FRAUD_PREDICTION"]
        ]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = row[
            index["FRAUD_PROBABILITY"]
        ]

        if (

            actual is None

            or prediction is None

            or probability is None

        ):

            continue

        try:

            actual = int(
                actual
            )

            prediction = int(
                prediction
            )

            probability = float(
                probability
            )

        except Exception:

            continue

        if actual not in (
            0,
            1
        ):

            continue

        if prediction not in (
            0,
            1
        ):

            continue

        if not math.isfinite(
            probability
        ):

            continue

        # ----------------------------------------------------
        # Evaluation arrays
        # ----------------------------------------------------

        y_true.append(
            actual
        )

        y_pred.append(
            prediction
        )

        y_probability.append(
            probability
        )

        # ====================================================
        # PREDICTION LATENCY
        # ====================================================

        prediction_start = parse_timestamp(

            row[
                index[
                    "PREDICTION_START_TIMESTAMP"
                ]
            ]

        )

        prediction_end = parse_timestamp(

            row[
                index[
                    "PREDICTION_END_TIMESTAMP"
                ]
            ]

        )

        if (

            prediction_start is not None

            and prediction_end is not None

        ):

            latency_ms = (

                prediction_end
                - prediction_start

            ).total_seconds() * 1000.0

            if (

                math.isfinite(
                    latency_ms
                )

                and latency_ms >= 0

            ):

                prediction_latencies.append(
                    latency_ms
                )

        # ====================================================
        # END-TO-END LATENCY
        # ====================================================

        producer_timestamp = parse_timestamp(

            row[
                index[
                    "PRODUCER_TIMESTAMP"
                ]
            ]

        )

        sink_timestamp = parse_timestamp(

            row[
                index[
                    "SINK_TIMESTAMP"
                ]
            ]

        )

        if (

            producer_timestamp is not None

            and sink_timestamp is not None

        ):

            e2e_ms = (

                sink_timestamp
                - producer_timestamp

            ).total_seconds() * 1000.0

            if (

                math.isfinite(
                    e2e_ms
                )

                and e2e_ms >= 0

            ):

                e2e_latencies.append(
                    e2e_ms
                )

                producer_timestamps.append(
                    producer_timestamp
                )

                sink_timestamps.append(
                    sink_timestamp
                )

    cursor.close()

    # ========================================================
    # USABLE ROWS
    # ========================================================

    print(
        f"Usable evaluation rows : "
        f"{len(y_true)}"
    )

    if not y_true:

        connection.close()

        raise RuntimeError(
            "No usable rows were found "
            "in fraud_predictions."
        )

    # ========================================================
    # BASIC COUNTS
    # ========================================================

    total = len(
        y_true
    )

    actual_fraud = sum(
        y_true
    )

    actual_legitimate = (
        total
        - actual_fraud
    )

    predicted_fraud = sum(
        y_pred
    )

    predicted_legitimate = (
        total
        - predicted_fraud
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    tn, fp, fn, tp = (

        confusion_matrix(

            y_true,

            y_pred,

            labels=[
                0,
                1
            ]

        ).ravel()

    )

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # ========================================================
    # FPR
    # ========================================================

    if (
        tn + fp
    ) > 0:

        fpr = (
            fp
            / (
                fp + tn
            )
        )

    else:

        fpr = 0.0

    # ========================================================
    # ROC AUC
    # ========================================================

    roc_auc = None

    if len(
        set(y_true)
    ) == 2:

        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )

    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    pr_precision = None

    pr_recall = None

    pr_thresholds = None

    average_precision = None

    if len(
        set(y_true)
    ) == 2:

        (
            pr_precision,
            pr_recall,
            pr_thresholds
        ) = precision_recall_curve(

            y_true,

            y_probability

        )

        # ====================================================
        # IMPORTANT (input ordering):
        #
        # sklearn's precision_recall_curve() returns the
        # arrays in DECREASING-recall order
        # (recall starts at 1 and ends at 0).
        #
        # The handbook formula:
        #
        #     AP = sum(
        #         (recall[i] - recall[i-1])
        #         * precision[i]
        #     )
        #
        # requires INCREASING-recall order, exactly as the
        # handbook does before plotting / computing AP:
        #
        #     precision = precision[::-1]
        #     recall    = recall[::-1]
        #
        # Without this reversal all recall differences are
        # negative and compute_AP() returns 0.
        #
        # ====================================================

        pr_precision = pr_precision[::-1]

        pr_recall = pr_recall[::-1]

        pr_thresholds = pr_thresholds[::-1]

        # ====================================================
        # IMPORTANT:
        #
        # AP is calculated using the exact custom formula
        # provided by the user (same as the handbook).
        #
        # Do NOT replace this with:
        #
        #     average_precision_score()
        #
        # ====================================================

        average_precision = compute_AP(
            pr_precision,
            pr_recall
        )

        # ----------------------------------------------------
        # Plot PR curve
        # ----------------------------------------------------

        plot_precision_recall_curve(

            pr_precision,

            pr_recall,

            average_precision

        )

    # ========================================================
    # THROUGHPUT
    # ========================================================
    #
    # Do NOT use TX_DATETIME.
    #
    # Real-time duration:
    #
    #   earliest PRODUCER_TIMESTAMP
    #       ->
    #   latest SINK_TIMESTAMP
    #
    # ========================================================

    throughput_duration = None

    throughput = None

    measurement_start = None

    measurement_end = None

    throughput_transactions = 0

    if (

        producer_timestamps

        and sink_timestamps

    ):

        measurement_start = min(
            producer_timestamps
        )

        measurement_end = max(
            sink_timestamps
        )

        throughput_duration = (

            measurement_end
            - measurement_start

        ).total_seconds()

        throughput_transactions = len(
            producer_timestamps
        )

        if throughput_duration > 0:

            throughput = (

                throughput_transactions
                / throughput_duration

            )

    # ========================================================
    # PRECISION TOP-K
    # ========================================================
    #
    # P@k = (# fraudulent transactions in the top-k alerts) / k
    #
    # The top-k alerts are the k transactions with the highest
    # fraud probability (Fraud Detection Handbook, Chapter 4,
    # section "Precision top-k metrics").
    #
    # Computed over the whole evaluation batch, mirroring the
    # handbook's precision_top_k_day() applied to the batch.
    #
    # ========================================================

    transaction_pairs = []

    for row in rows:

        probability = row[
            index["FRAUD_PROBABILITY"]
        ]

        actual = row[
            index["TX_FRAUD"]
        ]

        if (
            probability is None
            or actual is None
        ):

            continue

        try:

            probability = float(
                probability
            )

            actual = int(
                actual
            )

        except Exception:

            continue

        transaction_pairs.append(
            (
                probability,
                actual
            )
        )

    precision_top_k_50 = compute_precision_top_k(
        transaction_pairs,
        50
    )

    precision_top_k_100 = compute_precision_top_k(
        transaction_pairs,
        100
    )

    precision_top_k_200 = compute_precision_top_k(
        transaction_pairs,
        200
    )

    # ========================================================
    # LATENCY STATISTICS
    # ========================================================

    prediction_latency_stats = (
        latency_statistics(
            prediction_latencies
        )
    )

    e2e_latency_stats = (
        latency_statistics(
            e2e_latencies
        )
    )

    # ========================================================
    # OUTPUT - DATASET SUMMARY
    # ========================================================

    print()

    separator()

    print(
        "DATASET SUMMARY"
    )

    separator()

    print(
        f"Total transactions : "
        f"{total}"
    )

    print(
        f"Actual fraud       : "
        f"{actual_fraud}"
    )

    print(
        f"Actual legitimate  : "
        f"{actual_legitimate}"
    )

    print(
        f"Predicted fraud    : "
        f"{predicted_fraud}"
    )

    print(
        f"Predicted legit    : "
        f"{predicted_legitimate}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print()

    separator()

    print(
        "CONFUSION MATRIX"
    )

    separator()

    print(
        f"TN = {tn}"
    )

    print(
        f"FP = {fp}"
    )

    print(
        f"FN = {fn}"
    )

    print(
        f"TP = {tp}"
    )

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    print()

    separator()

    print(
        "CLASSIFICATION METRICS"
    )

    separator()

    print(
        f"Accuracy  : "
        f"{accuracy:.6f}"
    )

    print(
        f"Precision : "
        f"{precision:.6f}"
    )

    print(
        f"Recall    : "
        f"{recall:.6f}"
    )

    print(
        f"F1-score  : "
        f"{f1:.6f}"
    )

    print(
        f"FPR       : "
        f"{fpr:.6f}"
    )

    # ========================================================
    # PROBABILITY METRICS
    # ========================================================

    print()

    separator()

    print(
        "PROBABILITY METRICS"
    )

    separator()

    if roc_auc is None:

        print(
            "ROC-AUC : N/A "
            "(only one class present)"
        )

    else:

        print(
            f"ROC-AUC : "
            f"{roc_auc:.6f}"
        )

    if average_precision is None:

        print(
            "Average Precision : N/A "
            "(only one class present)"
        )

    else:

        print(
            f"Average Precision : "
            f"{average_precision:.6f}"
        )

        print(
            f"PR curve points   : "
            f"{len(pr_precision)}"
        )

        print(
            f"PR thresholds     : "
            f"{len(pr_thresholds)}"
        )

        print(
            f"PR curve saved    : "
            f"{PR_CURVE_PNG}"
        )

    # ========================================================
    # PREDICTION LATENCY
    # ========================================================

    print()

    print_latency(

        "PREDICTION LATENCY",

        prediction_latencies

    )

    # ========================================================
    # END-TO-END LATENCY
    # ========================================================

    print()

    print_latency(

        "END-TO-END LATENCY",

        e2e_latencies

    )

    # ========================================================
    # THROUGHPUT
    # ========================================================

    print()

    separator()

    print(
        "THROUGHPUT"
    )

    separator()

    if throughput is None:

        print(
            "Throughput : N/A"
        )

    else:

        print(
            "Measurement start : "
            f"{measurement_start.isoformat()}"
        )

        print(
            "Measurement end   : "
            f"{measurement_end.isoformat()}"
        )

        print(
            f"Duration          : "
            f"{throughput_duration:.3f} seconds"
        )

        print(
            f"Transactions      : "
            f"{throughput_transactions}"
        )

        print(
            f"Throughput        : "
            f"{throughput:.6f} "
            f"transactions/sec"
        )

    # ========================================================
    # PRECISION TOP-K
    # ========================================================

    print()

    separator()

    print(
        "PRECISION TOP-K"
    )

    separator()

    if precision_top_k_50 is None:

        print(
            "Precision@50 : N/A "
            "(fewer than 50 transactions)"
        )

    else:

        print(
            f"Precision@50 : "
            f"{precision_top_k_50:.6f}"
        )

    if precision_top_k_100 is None:

        print(
            "Precision@100 : N/A "
            "(fewer than 100 transactions)"
        )

    else:

        print(
            f"Precision@100 : "
            f"{precision_top_k_100:.6f}"
        )

    if precision_top_k_200 is None:

        print(
            "Precision@200 : N/A "
            "(fewer than 200 transactions)"
        )

    else:

        print(
            f"Precision@200 : "
            f"{precision_top_k_200:.6f}"
        )

    # ========================================================
    # BUILD RESULT OBJECT
    # ========================================================

    evaluation_timestamp = (
        datetime.now(
            timezone.utc
        )
    )

    result = {

        "evaluation_timestamp":
            evaluation_timestamp,

        "database":
            DB_NAME,

        "table":
            TABLE_NAME,

        "summary": {

            "total_transactions":
                total,

            "actual_fraud":
                actual_fraud,

            "actual_legitimate":
                actual_legitimate,

            "predicted_fraud":
                predicted_fraud,

            "predicted_legitimate":
                predicted_legitimate,

        },

        "confusion_matrix": {

            "tn":
                int(tn),

            "fp":
                int(fp),

            "fn":
                int(fn),

            "tp":
                int(tp),

        },

        "classification_metrics": {

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1_score":
                float(f1),

            "fpr":
                float(fpr),

        },

        "probability_metrics": {

            "roc_auc":
                (
                    float(roc_auc)
                    if roc_auc is not None
                    else None
                ),

            "average_precision":
                (
                    float(
                        average_precision
                    )

                    if average_precision is not None

                    else None
                ),

            "pr_curve_points":
                (
                    int(
                        len(pr_precision)
                    )

                    if pr_precision is not None

                    else 0
                ),

            "pr_curve_file":
                (
                    PR_CURVE_PNG
                    if average_precision is not None
                    else None
                ),

        },

        "prediction_latency":
            prediction_latency_stats,

        "end_to_end_latency":
            e2e_latency_stats,

        "throughput": {

            "measurement_start":
                measurement_start,

            "measurement_end":
                measurement_end,

            "duration_seconds":
                (
                    float(
                        throughput_duration
                    )

                    if throughput_duration is not None

                    else None
                ),

            "transactions":
                int(
                    throughput_transactions
                ),

            "transactions_per_sec":
                (
                    float(throughput)

                    if throughput is not None

                    else None
                ),

        },

        "precision_top_k": {

            "total_transactions":
                len(transaction_pairs),

            "precision_at_50":
                (
                    float(
                        precision_top_k_50
                    )

                    if precision_top_k_50 is not None

                    else None
                ),

            "precision_at_100":
                (
                    float(
                        precision_top_k_100
                    )

                    if precision_top_k_100 is not None

                    else None
                ),

            "precision_at_200":
                (
                    float(
                        precision_top_k_200
                    )

                    if precision_top_k_200 is not None

                    else None
                ),

        },

    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    save_latest_json(
        result
    )

    save_history_csv(
        result
    )

    # ========================================================
    # SAVE MESSAGE
    # ========================================================

    print()

    separator()

    print(
        "RESULTS SAVED"
    )

    separator()

    print(
        f"History CSV : "
        f"{HISTORY_CSV}"
    )

    print(
        f"Latest JSON : "
        f"{LATEST_JSON}"
    )

    if average_precision is not None:

        print(
            f"PR Curve    : "
            f"{PR_CURVE_PNG}"
        )

    # ========================================================
    # CLOSE DATABASE
    # ========================================================

    connection.close()

    # ========================================================
    # FINISH
    # ========================================================

    print()

    separator()

    print(
        "EVALUATION COMPLETED"
    )

    separator()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()