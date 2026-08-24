# ============================================================
# REAL-TIME FRAUD DETECTION DASHBOARD
# ============================================================
#
# DATA SOURCE:
#
#   PostgreSQL
#       |
#       v
#   fraud_detection.fraud_predictions
#
# NO DEPENDENCY ON:
#
#   realtime_evaluation_latest.json
#   realtime_evaluation_history.csv
#   resource_metrics.csv
#
# Dashboard calculates metrics directly from PostgreSQL.
#
# ============================================================

import math
import os
import re
import subprocess
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    average_precision_score,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Real-Time Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "fraud_detection"
DB_USER = "fraud"
DB_PASSWORD = "fraud123"

TABLE_NAME = "fraud_predictions"


# ============================================================
# REFRESH CONFIG
# ============================================================

DEFAULT_REFRESH_SECONDS = 5


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .sub-title {
        font-size: 15px;
        color: #666;
        margin-bottom: 20px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #fafafa;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Real-Time Fraud Detection Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
    PostgreSQL → Spark → LSTM → Fraud Detection Evaluation
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Dashboard Settings")

auto_refresh = st.sidebar.checkbox(
    "Auto refresh",
    value=False
)

refresh_seconds = st.sidebar.number_input(
    "Refresh interval (seconds)",
    min_value=1,
    max_value=60,
    value=5
)

if auto_refresh:

    time.sleep(
        refresh_seconds
    )

    st.rerun()

if st.sidebar.button("🔄 Refresh now"):
    st.rerun()

# ============================================================
# DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_connection():

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(ttl=2)
def load_data():

    connection = get_connection()

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

    df = pd.read_sql_query(
        query,
        connection,
    )

    return df


# ============================================================
# SAFE NUMERIC
# ============================================================

def clean_numeric(df):

    numeric_columns = [
        "transaction_id",
        "customer_id",
        "terminal_id",
        "tx_amount",
        "fraud_probability",
        "fraud_prediction",
        "threshold",
        "tx_fraud",
        "prediction_latency_ms",
        "end_to_end_latency_ms",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# TIMESTAMP
# ============================================================

def prepare_timestamps(df):

    timestamp_columns = [
        "tx_datetime",
        "producer_timestamp",
        "prediction_start_timestamp",
        "prediction_end_timestamp",
        "sink_timestamp",
    ]

    for column in timestamp_columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                utc=True,
            )

    return df


# ============================================================
# LATENCY CALCULATION
# ============================================================

def calculate_latencies(df):

    # --------------------------------------------------------
    # Prediction latency
    # --------------------------------------------------------

    prediction_calculated = (
        (
            df["prediction_end_timestamp"]
            -
            df["prediction_start_timestamp"]
        )
        .dt.total_seconds()
        * 1000.0
    )

    # --------------------------------------------------------
    # E2E latency
    # --------------------------------------------------------

    e2e_calculated = (
        (
            df["sink_timestamp"]
            -
            df["producer_timestamp"]
        )
        .dt.total_seconds()
        * 1000.0
    )

    # --------------------------------------------------------
    # Prefer timestamp-calculated values.
    #
    # If unavailable, use PostgreSQL stored latency.
    # --------------------------------------------------------

    df["prediction_latency_calculated_ms"] = (
        prediction_calculated
    )

    df["e2e_latency_calculated_ms"] = (
        e2e_calculated
    )

    df["prediction_latency_used_ms"] = (
        prediction_calculated
        .where(
            prediction_calculated.notna()
        )
        .fillna(
            df["prediction_latency_ms"]
        )
    )

    df["e2e_latency_used_ms"] = (
        e2e_calculated
        .where(
            e2e_calculated.notna()
        )
        .fillna(
            df["end_to_end_latency_ms"]
        )
    )

    # Remove impossible negative values

    df.loc[
        df["prediction_latency_used_ms"] < 0,
        "prediction_latency_used_ms",
    ] = None

    df.loc[
        df["e2e_latency_used_ms"] < 0,
        "e2e_latency_used_ms",
    ] = None

    return df


# ============================================================
# PERCENTILE
# ============================================================

def percentile(
    series,
    p,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    values = values[
        values >= 0
    ]

    if len(values) == 0:
        return None

    return float(
        values.quantile(
            p / 100.0,
            interpolation="linear",
        )
    )


# ============================================================
# LATENCY STATISTICS
# ============================================================

def latency_statistics(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    values = values[
        values >= 0
    ]

    if len(values) == 0:

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

        "samples": int(len(values)),

        "average_ms":
            float(values.mean()),

        "p50_ms":
            percentile(values, 50),

        "p95_ms":
            percentile(values, 95),

        "p99_ms":
            percentile(values, 99),

        "min_ms":
            float(values.min()),

        "max_ms":
            float(values.max()),
    }


# ============================================================
# PRECISION TOP-K
# ============================================================
#
# Transaction-level Precision top-k, per the Fraud Detection
# Handbook (Chapter 4 - Precision top-k metrics):
#
#   Sort all transactions by fraud_probability DESC.
#
#   P@k(d) = |fraud transactions in top K| / K
#
# Reference:
#   https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_4_PerformanceMetrics/TopKBased.html
#
# ============================================================

def precision_top_k_day(
    df,
    top_k=100,
):

    ranking = df[
        [
            "fraud_probability",
            "tx_fraud",
        ]
    ].dropna()

    # Stable sort: matches evaluate_realtime.compute_precision_top_k,
    # which uses Python's sorted(..., reverse=True) (stable sort).
    ranking = ranking.sort_values(
        "fraud_probability",
        ascending=False,
        kind="stable",
    ).reset_index(
        drop=False
    )

    # Match evaluate_realtime.compute_precision_top_k: return None
    # when fewer than top_k transactions are available.
    if len(ranking) < top_k:
        return 0, None

    top_k_rows = ranking.head(
        top_k
    )

    nb_frauds = int(
        (
            top_k_rows["tx_fraud"]
            .astype(int)
            == 1
        ).sum()
    )

    # Note: the denominator is always top_k, exactly as in the
    # handbook's precision_top_k_day function (P@k = |fraud in top K| / K).
    precision_top_k = (
        nb_frauds
        / top_k
    )

    return nb_frauds, float(precision_top_k)


# ============================================================
# THROUGHPUT
# ============================================================

def calculate_throughput(df):

    timestamps = []

    if "producer_timestamp" in df.columns:

        timestamps.extend(
            df["producer_timestamp"]
            .dropna()
            .tolist()
        )

    sink_timestamps = []

    if "sink_timestamp" in df.columns:

        sink_timestamps.extend(
            df["sink_timestamp"]
            .dropna()
            .tolist()
        )

    if not timestamps or not sink_timestamps:

        return {
            "transactions": 0,
            "duration_seconds": None,
            "transactions_per_sec": None,
            "start": None,
            "end": None,
        }

    start = min(
        timestamps
    )

    end = max(
        sink_timestamps
    )

    duration = (
        end - start
    ).total_seconds()

    transactions = min(
        len(timestamps),
        len(sink_timestamps),
    )

    if duration <= 0:

        throughput = None

    else:

        throughput = (
            transactions
            / duration
        )

    return {

        "transactions":
            int(transactions),

        "duration_seconds":
            float(duration),

        "transactions_per_sec":
            (
                float(throughput)
                if throughput is not None
                else None
            ),

        "start":
            start,

        "end":
            end,
    }


# ============================================================
# DOCKER RESOURCE MONITOR
# ============================================================

def parse_cpu(value):

    if value is None:
        return None

    match = re.search(
        r"[-+]?[0-9]*\.?[0-9]+",
        str(value),
    )

    if not match:
        return None

    try:
        return float(
            match.group()
        )
    except Exception:
        return None


def parse_memory_mb(value):

    if value is None:
        return None

    text = str(value)

    # Docker format example:
    #
    # 123.4MiB / 7.8GiB

    first = text.split("/")[0].strip()

    match = re.search(
        r"([-+]?[0-9]*\.?[0-9]+)\s*([KMGTP]?i?B)",
        first,
        re.IGNORECASE,
    )

    if not match:
        return None

    number = float(
        match.group(1)
    )

    unit = match.group(2).lower()

    factors = {

        "b": 1 / (1024 * 1024),

        "kb": 1 / 1024,

        "kib": 1 / 1024,

        "mb": 1,

        "mib": 1,

        "gb": 1024,

        "gib": 1024,

        "tb": 1024 * 1024,

        "tib": 1024 * 1024,
    }

    factor = factors.get(
        unit,
        1,
    )

    return number * factor


def get_docker_resources():

    try:

        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:

            return pd.DataFrame(
                columns=[
                    "container",
                    "cpu_percent",
                    "memory_usage",
                    "memory_percent",
                    "memory_mb",
                ]
            )

        rows = []

        for line in result.stdout.splitlines():

            parts = line.split("|")

            if len(parts) != 4:
                continue

            container = parts[0]
            cpu = parts[1]
            memory = parts[2]
            memory_percent = parts[3]

            rows.append({

                "container":
                    container,

                "cpu_percent":
                    parse_cpu(cpu),

                "memory_usage":
                    memory,

                "memory_percent":
                    parse_cpu(
                        memory_percent
                    ),

                "memory_mb":
                    parse_memory_mb(
                        memory
                    ),
            })

        return pd.DataFrame(rows)

    except Exception:

        return pd.DataFrame(
            columns=[
                "container",
                "cpu_percent",
                "memory_usage",
                "memory_percent",
                "memory_mb",
            ]
        )


# ============================================================
# LOAD
# ============================================================

try:

    df = load_data()

except Exception as exc:

    st.error(
        "Không thể kết nối PostgreSQL."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ============================================================
# PREPARE
# ============================================================

df = clean_numeric(df)

df = prepare_timestamps(df)

df = calculate_latencies(df)


# ============================================================
# EMPTY CHECK
# ============================================================

if df.empty:

    st.warning(
        "Bảng fraud_predictions hiện chưa có dữ liệu."
    )

    st.stop()


# ============================================================
# USABLE DATA
# ============================================================

usable = df[
    df["tx_fraud"].notna()
    &
    df["fraud_prediction"].notna()
    &
    df["fraud_probability"].notna()
].copy()


usable["tx_fraud"] = (
    usable["tx_fraud"]
    .astype(int)
)

usable["fraud_prediction"] = (
    usable["fraud_prediction"]
    .astype(int)
)


# ============================================================
# BASIC COUNTS
# ============================================================

total = len(
    usable
)

actual_fraud = int(
    usable["tx_fraud"].sum()
)

actual_legitimate = (
    total
    - actual_fraud
)

predicted_fraud = int(
    usable["fraud_prediction"].sum()
)

predicted_legitimate = (
    total
    - predicted_fraud
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = (
    confusion_matrix(
        usable["tx_fraud"],
        usable["fraud_prediction"],
        labels=[0, 1],
    )
    .ravel()
)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    usable["tx_fraud"],
    usable["fraud_prediction"],
)

precision = precision_score(
    usable["tx_fraud"],
    usable["fraud_prediction"],
    zero_division=0,
)

recall = recall_score(
    usable["tx_fraud"],
    usable["fraud_prediction"],
    zero_division=0,
)

f1 = f1_score(
    usable["tx_fraud"],
    usable["fraud_prediction"],
    zero_division=0,
)

if (
    tn + fp
) > 0:

    fpr = (
        fp
        /
        (tn + fp)
    )

else:

    fpr = 0.0


# ============================================================
# PROBABILITY METRICS
# ============================================================

roc_auc = None

average_precision = None

if (
    usable["tx_fraud"]
    .nunique()
    == 2
):

    roc_fpr, roc_tpr, _ = roc_curve(
        usable["tx_fraud"],
        usable["fraud_probability"],
    )

    roc_auc = auc(
        roc_fpr,
        roc_tpr,
    )

    average_precision = (
        average_precision_score(
            usable["tx_fraud"],
            usable["fraud_probability"],
        )
    )


# ============================================================
# LATENCY
# ============================================================

prediction_latency = latency_statistics(
    usable[
        "prediction_latency_used_ms"
    ]
)

e2e_latency = latency_statistics(
    usable[
        "e2e_latency_used_ms"
    ]
)


# ============================================================
# THROUGHPUT
# ============================================================

throughput = calculate_throughput(
    usable
)


# ============================================================
# PRECISION TOP-K
# ============================================================

_, precision_50 = precision_top_k_day(
    df,
    top_k=50,
)

_, precision_100 = precision_top_k_day(
    df,
    top_k=100,
)

_, precision_200 = precision_top_k_day(
    df,
    top_k=200,
)


# ============================================================
# LAST TRANSACTION
# ============================================================

last_producer = (
    usable["producer_timestamp"]
    .dropna()
    .max()
)

last_sink = (
    usable["sink_timestamp"]
    .dropna()
    .max()
)


# ============================================================
# TOP HEADER
# ============================================================

st.markdown(
    '<div class="section-title">📊 Overall Performance</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Transactions",
    f"{total:,}",
)

c2.metric(
    "Actual Fraud",
    f"{actual_fraud:,}",
)

c3.metric(
    "Predicted Fraud",
    f"{predicted_fraud:,}",
)

c4.metric(
    "Accuracy",
    f"{accuracy:.4f}",
)

c5.metric(
    "Precision",
    f"{precision:.4f}",
)

c6.metric(
    "Recall",
    f"{recall:.4f}",
)


# ============================================================
# CLASSIFICATION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Classification Metrics</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Accuracy",
    f"{accuracy:.6f}",
)

c2.metric(
    "Precision",
    f"{precision:.6f}",
)

c3.metric(
    "Recall",
    f"{recall:.6f}",
)

c4.metric(
    "F1-score",
    f"{f1:.6f}",
)

c5.metric(
    "FPR",
    f"{fpr:.6f}",
)


# ============================================================
# PROBABILITY
# ============================================================

st.markdown(
    '<div class="section-title">📈 Probability Metrics</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

if roc_auc is None:

    c1.metric(
        "ROC-AUC",
        "N/A",
    )

else:

    c1.metric(
        "ROC-AUC",
        f"{roc_auc:.6f}",
    )

if average_precision is None:

    c2.metric(
        "Average Precision",
        "N/A",
    )

else:

    c2.metric(
        "Average Precision",
        f"{average_precision:.6f}",
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">🔲 Confusion Matrix</div>',
    unsafe_allow_html=True,
)

cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)

cm_col1.metric(
    "TN",
    f"{int(tn):,}",
)

cm_col2.metric(
    "FP",
    f"{int(fp):,}",
)

cm_col3.metric(
    "FN",
    f"{int(fn):,}",
)

cm_col4.metric(
    "TP",
    f"{int(tp):,}",
)


# ============================================================
# CONFUSION MATRIX TABLE
# ============================================================

cm_df = pd.DataFrame(
    [
        [
            int(tn),
            int(fp),
        ],
        [
            int(fn),
            int(tp),
        ],
    ],
    index=[
        "Actual Legitimate",
        "Actual Fraud",
    ],
    columns=[
        "Predicted Legitimate",
        "Predicted Fraud",
    ],
)

st.dataframe(
    cm_df,
    use_container_width=True,
)


# ============================================================
# PRECISION TOP-K
# ============================================================

st.markdown(
    '<div class="section-title">🏆 Precision@K</div>',
    unsafe_allow_html=True,
)

p1, p2, p3 = st.columns(3)

p1.metric(
    "Precision@50",
    (
        f"{precision_50:.6f}"
        if precision_50 is not None
        else "N/A"
    ),
)

p2.metric(
    "Precision@100",
    (
        f"{precision_100:.6f}"
        if precision_100 is not None
        else "N/A"
    ),
)

p3.metric(
    "Precision@200",
    (
        f"{precision_200:.6f}"
        if precision_200 is not None
        else "N/A"
    ),
)

st.caption(
    "Precision top-k theo Fraud Detection Handbook: "
    "P@k = số giao dịch fraud thực tế trong K giao dịch "
    "có fraud_probability cao nhất / K."
)


# ============================================================
# LATENCY
# ============================================================

st.markdown(
    '<div class="section-title">⚡ Prediction Latency</div>',
    unsafe_allow_html=True,
)

p = prediction_latency

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

c1.metric(
    "Samples",
    f"{p['samples']:,}",
)

c2.metric(
    "Average",
    (
        f"{p['average_ms']:.3f} ms"
        if p["average_ms"] is not None
        else "N/A"
    ),
)

c3.metric(
    "P50",
    (
        f"{p['p50_ms']:.3f} ms"
        if p["p50_ms"] is not None
        else "N/A"
    ),
)

c4.metric(
    "P95",
    (
        f"{p['p95_ms']:.3f} ms"
        if p["p95_ms"] is not None
        else "N/A"
    ),
)

c5.metric(
    "P99",
    (
        f"{p['p99_ms']:.3f} ms"
        if p["p99_ms"] is not None
        else "N/A"
    ),
)

c6.metric(
    "Min",
    (
        f"{p['min_ms']:.3f} ms"
        if p["min_ms"] is not None
        else "N/A"
    ),
)

c7.metric(
    "Max",
    (
        f"{p['max_ms']:.3f} ms"
        if p["max_ms"] is not None
        else "N/A"
    ),
)


# ============================================================
# E2E LATENCY
# ============================================================

st.markdown(
    '<div class="section-title">🌐 End-to-End Latency</div>',
    unsafe_allow_html=True,
)

p = e2e_latency

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

c1.metric(
    "Samples",
    f"{p['samples']:,}",
)

c2.metric(
    "Average",
    (
        f"{p['average_ms']:.3f} ms"
        if p["average_ms"] is not None
        else "N/A"
    ),
)

c3.metric(
    "P50",
    (
        f"{p['p50_ms']:.3f} ms"
        if p["p50_ms"] is not None
        else "N/A"
    ),
)

c4.metric(
    "P95",
    (
        f"{p['p95_ms']:.3f} ms"
        if p["p95_ms"] is not None
        else "N/A"
    ),
)

c5.metric(
    "P99",
    (
        f"{p['p99_ms']:.3f} ms"
        if p["p99_ms"] is not None
        else "N/A"
    ),
)

c6.metric(
    "Min",
    (
        f"{p['min_ms']:.3f} ms"
        if p["min_ms"] is not None
        else "N/A"
    ),
)

c7.metric(
    "Max",
    (
        f"{p['max_ms']:.3f} ms"
        if p["max_ms"] is not None
        else "N/A"
    ),
)


# ============================================================
# THROUGHPUT
# ============================================================

st.markdown(
    '<div class="section-title">🚀 Throughput</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Transactions",
    f"{throughput['transactions']:,}",
)

c2.metric(
    "Duration",
    (
        f"{throughput['duration_seconds']:.3f} s"
        if throughput["duration_seconds"] is not None
        else "N/A"
    ),
)

c3.metric(
    "Transactions/sec",
    (
        f"{throughput['transactions_per_sec']:.6f}"
        if throughput["transactions_per_sec"] is not None
        else "N/A"
    ),
)

c4.metric(
    "Database Rows",
    f"{len(df):,}",
)


# ============================================================
# THROUGHPUT TIMELINE
# ============================================================

if (
    throughput["start"] is not None
    and throughput["end"] is not None
):

    st.write(
        f"**Start:** "
        f"{throughput['start'].isoformat()}"
    )

    st.write(
        f"**End:** "
        f"{throughput['end'].isoformat()}"
    )


# ============================================================
# RESOURCE MONITOR
# ============================================================

st.markdown(
    '<div class="section-title">💻 Docker Resource Usage</div>',
    unsafe_allow_html=True,
)

resource_df = get_docker_resources()

if resource_df.empty:

    st.warning(
        "Không lấy được Docker stats. "
        "Hãy chạy Streamlit trên máy có quyền truy cập Docker."
    )

else:

    resource_display = (
        resource_df[
            [
                "container",
                "cpu_percent",
                "memory_usage",
                "memory_percent",
                "memory_mb",
            ]
        ]
        .copy()
    )

    resource_display[
        "cpu_percent"
    ] = resource_display[
        "cpu_percent"
    ].map(
        lambda x:
            f"{x:.2f}%"
            if pd.notna(x)
            else "N/A"
    )

    resource_display[
        "memory_percent"
    ] = resource_display[
        "memory_percent"
    ].map(
        lambda x:
            f"{x:.2f}%"
            if pd.notna(x)
            else "N/A"
    )

    resource_display[
        "memory_mb"
    ] = resource_display[
        "memory_mb"
    ].map(
        lambda x:
            f"{x:.2f} MB"
            if pd.notna(x)
            else "N/A"
    )

    resource_display.columns = [
        "Container",
        "CPU",
        "Memory Usage",
        "Memory %",
        "Memory MB",
    ]

    st.dataframe(
        resource_display,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CPU CHART
# ============================================================

if not resource_df.empty:

    cpu_chart = (
        resource_df[
            [
                "container",
                "cpu_percent",
            ]
        ]
        .dropna()
        .set_index("container")
    )

    if not cpu_chart.empty:

        st.write("### CPU usage")

        st.bar_chart(
            cpu_chart[
                "cpu_percent"
            ]
        )


# ============================================================
# MEMORY CHART
# ============================================================

if not resource_df.empty:

    memory_chart = (
        resource_df[
            [
                "container",
                "memory_mb",
            ]
        ]
        .dropna()
        .set_index("container")
    )

    if not memory_chart.empty:

        st.write("### Memory usage")

        st.bar_chart(
            memory_chart[
                "memory_mb"
            ]
        )


# ============================================================
# TRANSACTION DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">📦 Transaction Distribution</div>',
    unsafe_allow_html=True,
)

distribution_df = pd.DataFrame(
    {
        "Category": [
            "Actual Fraud",
            "Actual Legitimate",
            "Predicted Fraud",
            "Predicted Legitimate",
        ],
        "Transactions": [
            actual_fraud,
            actual_legitimate,
            predicted_fraud,
            predicted_legitimate,
        ],
    }
)

st.bar_chart(
    distribution_df.set_index(
        "Category"
    )
)


# ============================================================
# FRAUD PROBABILITY DISTRIBUTION
# ============================================================

st.markdown(
    '<div class="section-title">📊 Fraud Probability Distribution</div>',
    unsafe_allow_html=True,
)

probability_chart = (
    usable[
        [
            "fraud_probability"
        ]
    ]
    .dropna()
)

if not probability_chart.empty:

    st.line_chart(
        probability_chart.reset_index(
            drop=True
        )
    )


# ============================================================
# RECENT TRANSACTIONS
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Recent Transactions</div>',
    unsafe_allow_html=True,
)

recent_columns = [
    "transaction_id",
    "tx_datetime",
    "customer_id",
    "terminal_id",
    "tx_amount",
    "fraud_probability",
    "fraud_prediction",
    "threshold",
    "tx_fraud",
    "prediction_latency_used_ms",
    "e2e_latency_used_ms",
]

recent_columns = [
    column
    for column in recent_columns
    if column in usable.columns
]

recent_df = (
    usable[
        recent_columns
    ]
    .sort_values(
        "transaction_id",
        ascending=False,
    )
    .head(50)
    .copy()
)

st.dataframe(
    recent_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SYSTEM STATUS
# ============================================================

st.markdown(
    '<div class="section-title">🟢 System Status</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "PostgreSQL",
    "CONNECTED",
)

c2.metric(
    "Fraud Predictions",
    f"{len(df):,}",
)

if last_producer is not None:

    c3.metric(
        "Last Producer",
        last_producer.strftime(
            "%H:%M:%S"
        ),
    )

else:

    c3.metric(
        "Last Producer",
        "N/A",
    )

if last_sink is not None:

    c4.metric(
        "Last Sink",
        last_sink.strftime(
            "%H:%M:%S"
        ),
    )

else:

    c4.metric(
        "Last Sink",
        "N/A",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

now = datetime.now(
    timezone.utc
)

st.caption(
    "Dashboard reads metrics directly from PostgreSQL. "
    "No evaluation JSON/CSV files are required."
)

st.caption(
    f"Last dashboard update: {now.isoformat()}"
)
