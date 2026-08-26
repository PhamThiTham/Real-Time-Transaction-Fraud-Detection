# ============================================================
# DASHBOARD SO SÁNH CÁC ĐỘ ĐO — REAL-TIME FRAUD DETECTION
# ------------------------------------------------------------
# So sánh metrics của 3 cấu hình producer:
#     50 tx-s, 100 tx-s, 500 tx-s
#
# Chạy:
#     streamlit run dashboard_compare.py
#
# Dữ liệu đọc từ:
#     exports/50_100_500tx-s_evaluate/{variant}_realtime_evaluation_latest.json
#     exports/50_100_500tx-s_evaluate/{variant}_resource_metrics.csv
#     exports/50_100_500tx-s_evaluate/{variant}_precision_recall_curve.png
# ============================================================

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="So sánh 50/100/500 tx-s",
    page_icon="📊",
    layout="wide",
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "exports", "50_100_500tx-s_evaluate")

VARIANTS = ["50tx-s", "100tx-s", "500tx-s"]

COLORS = {
    "50tx-s": "#1f77b4",
    "100tx-s": "#ff7f0e",
    "500tx-s": "#2ca02c",
}

CONTAINERS = ["kafka", "spark", "postgres"]

# Nhãn hiển thị đẹp cho các cột metrics
LABELS = {
    "summary_total_transactions": "Total transactions",
    "summary_actual_fraud": "Actual fraud",
    "summary_actual_legitimate": "Actual legit",
    "summary_predicted_fraud": "Predicted fraud",
    "summary_predicted_legitimate": "Predicted legit",
    "confusion_matrix_tn": "TN",
    "confusion_matrix_fp": "FP",
    "confusion_matrix_fn": "FN",
    "confusion_matrix_tp": "TP",
    "classification_metrics_accuracy": "Accuracy",
    "classification_metrics_precision": "Precision",
    "classification_metrics_recall": "Recall",
    "classification_metrics_f1_score": "F1-score",
    "classification_metrics_fpr": "FPR",
    "probability_metrics_roc_auc": "ROC-AUC",
    "probability_metrics_average_precision": "Avg Precision",
    "prediction_latency_average_ms": "Avg",
    "prediction_latency_p50_ms": "P50",
    "prediction_latency_p95_ms": "P95",
    "prediction_latency_p99_ms": "P99",
    "end_to_end_latency_average_ms": "Avg",
    "end_to_end_latency_p50_ms": "P50",
    "end_to_end_latency_p95_ms": "P95",
    "end_to_end_latency_p99_ms": "P99",
    "throughput_transactions_per_sec": "Throughput (tx/s)",
    "throughput_duration_seconds": "Duration (s)",
    "precision_top_k_precision_at_50": "Top-50",
    "precision_top_k_precision_at_100": "Top-100",
    "precision_top_k_precision_at_200": "Top-200",
}

# Các nhóm cột metrics (theo prefix trong JSON đã flatten)
SUMMARY_COLS = [
    "summary_total_transactions",
    "summary_actual_fraud",
    "summary_actual_legitimate",
    "summary_predicted_fraud",
    "summary_predicted_legitimate",
]

CONFUSION_COLS = [
    "confusion_matrix_tn",
    "confusion_matrix_fp",
    "confusion_matrix_fn",
    "confusion_matrix_tp",
]

CLASSIFICATION_COLS = [
    "classification_metrics_accuracy",
    "classification_metrics_precision",
    "classification_metrics_recall",
    "classification_metrics_f1_score",
    "classification_metrics_fpr",
]

PROBABILITY_COLS = [
    "probability_metrics_roc_auc",
    "probability_metrics_average_precision",
]

LATENCY_PERCENTILES = ["average_ms", "p50_ms", "p95_ms", "p99_ms"]
PRED_LATENCY_COLS = [f"prediction_latency_{c}" for c in LATENCY_PERCENTILES]
E2E_LATENCY_COLS = [f"end_to_end_latency_{c}" for c in LATENCY_PERCENTILES]

THROUGHPUT_TPS_COLS = ["throughput_transactions_per_sec"]
THROUGHPUT_DURATION_COLS = ["throughput_duration_seconds"]

TOP_K_COLS = [
    "precision_top_k_precision_at_50",
    "precision_top_k_precision_at_100",
    "precision_top_k_precision_at_200",
]


# ============================================================
# LOAD DATA
# ============================================================

def _variant_file(variant, suffix):
    return os.path.join(DATA_DIR, f"{variant}{suffix}")


def _flatten_metrics(variant, data):
    """Chuyển JSON lồng nhau thành 1 dòng phẳng."""
    row = {"variant": variant}

    def walk(node, prefix):
        for key, value in node.items():
            key_ = f"{prefix}{key}"
            if isinstance(value, dict):
                walk(value, f"{key_}_")
            else:
                row[key_] = value

    walk(data, "")
    return row


@st.cache_data
def load_comparison_df():
    """Đọc tất cả file latest.json -> DataFrame 1 dòng / variant."""
    rows = []
    for variant in VARIANTS:
        path = _variant_file(variant, "_realtime_evaluation_latest.json")
        with open(path, encoding="utf-8") as fh:
            rows.append(_flatten_metrics(variant, json.load(fh)))
    return pd.DataFrame(rows)


def _parse_percent(value):
    """'400.28%' -> 400.28"""
    return float(str(value).replace("%", "").strip())


def _parse_mib(value):
    """'877.1MiB', '7.678GiB', '512KiB' -> MiB (float)."""
    value = value.strip()
    number = float(value[:-3].strip())
    unit = value[-3:]
    multiplier = {"KiB": 1.0 / 1024.0, "MiB": 1.0, "GiB": 1024.0}
    return number * multiplier[unit]


@st.cache_data
def load_resource_df(variant):
    """Đọc file resource_metrics.csv và parse các cột dạng string."""
    path = _variant_file(variant, "_resource_metrics.csv")
    df = pd.read_csv(path)
    df["variant"] = variant
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["cpu_percent"] = df["cpu_percent"].map(_parse_percent)
    df["memory_percent"] = df["memory_percent"].map(_parse_percent)

    parts = df["memory_usage"].str.split("/", expand=True)
    df["mem_used_mib"] = parts[0].map(_parse_mib)
    df["mem_total_mib"] = parts[1].map(_parse_mib)
    df["mem_used_gib"] = df["mem_used_mib"] / 1024.0

    df["elapsed_min"] = (
        (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 60.0
    )
    return df


# ============================================================
# CHART / TABLE HELPERS
# ============================================================

def _label(metric):
    return LABELS.get(metric, metric.replace("_", " ").title())


def _melt(comp_df, variants, value_cols):
    """Lọc theo variant rồi melt các cột value thành 1 cột 'metric'/'value'."""
    sub = comp_df[comp_df["variant"].isin(variants)][["variant"] + value_cols]
    melted = sub.melt(id_vars="variant", var_name="metric", value_name="value")
    melted["metric"] = melted["metric"].map(_label)
    return melted


def _bar(melted, title, y_title, height=460, log_scale=False):
    if melted["metric"].nunique() == 1:
        fig = px.bar(
            melted,
            x="variant",
            y="value",
            color="variant",
            color_discrete_map=COLORS,
            title=title,
        )
    else:
        fig = px.bar(
            melted,
            x="metric",
            y="value",
            color="variant",
            barmode="group",
            color_discrete_map=COLORS,
            title=title,
        )
    fig.update_layout(
        yaxis_title=y_title,
        xaxis_title=None,
        height=height,
        legend_title_text="Variant",
        hovermode="x unified",
    )
    if log_scale:
        fig.update_yaxes(type="log")
    return fig


def _format_value(value, spec=".4f"):
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:{spec}}"


def _metric_table(comp_df, variants, value_cols, spec=".4f"):
    """Bảng so sánh: index = tên metric, cột = variant."""
    sub = comp_df[comp_df["variant"].isin(variants)][["variant"] + value_cols]
    display = sub.set_index("variant").T
    display.index = display.index.map(_label)
    display.columns.name = "Variant"
    display = display.map(lambda v: _format_value(v, spec))
    st.dataframe(display, width="stretch")


def _compute_resource_summary(variants, containers):
    """Bảng tổng hợp CPU/RAM trung bình & max theo variant × container."""
    frames = [load_resource_df(v) for v in variants]
    res_df = pd.concat(frames, ignore_index=True)

    summary_rows = []
    for variant in variants:
        sub = res_df[res_df["variant"] == variant]
        for container in containers:
            part = sub[sub["container"] == container]
            if part.empty:
                continue
            summary_rows.append(
                {
                    "variant": variant,
                    "container": container,
                    "cpu_mean_%": round(part["cpu_percent"].mean(), 2),
                    "cpu_max_%": round(part["cpu_percent"].max(), 2),
                    "mem_mean_GiB": round(part["mem_used_gib"].mean(), 3),
                    "mem_max_GiB": round(part["mem_used_gib"].max(), 3),
                    "mem_mean_%": round(part["memory_percent"].mean(), 2),
                    "mem_max_%": round(part["memory_percent"].max(), 2),
                }
            )
    return pd.DataFrame(summary_rows)


def _compute_ratios(comp_df, variants):
    """Tính tỷ lệ (×) của từng variant so với baseline (throughput thấp nhất)."""
    sub = comp_df[comp_df["variant"].isin(variants)].copy()
    baseline = sub.loc[sub["throughput_transactions_per_sec"].idxmin(), "variant"]

    metric_cols = [
        "throughput_transactions_per_sec",
        "prediction_latency_average_ms",
        "prediction_latency_p99_ms",
        "end_to_end_latency_average_ms",
        "end_to_end_latency_p99_ms",
    ]
    display_names = {
        "throughput_transactions_per_sec": "Throughput (tx/s)",
        "prediction_latency_average_ms": "Pred latency avg (ms)",
        "prediction_latency_p99_ms": "Pred latency p99 (ms)",
        "end_to_end_latency_average_ms": "E2E latency avg (ms)",
        "end_to_end_latency_p99_ms": "E2E latency p99 (ms)",
    }
    ratio_names = {
        "throughput_transactions_per_sec": "Throughput ×",
        "prediction_latency_average_ms": "Pred avg ×",
        "prediction_latency_p99_ms": "Pred p99 ×",
        "end_to_end_latency_average_ms": "E2E avg ×",
        "end_to_end_latency_p99_ms": "E2E p99 ×",
    }

    base_vals = sub.loc[sub["variant"] == baseline, metric_cols]

    out = pd.DataFrame({"Variant": sub["variant"].values})
    for col in metric_cols:
        out[display_names[col]] = sub[col].values
        out[ratio_names[col]] = (sub[col].values / base_vals[col].values[0]).round(2)
    return out, baseline


def _tradeoff_bubble(comp_df, variants):
    """Bubble chart: throughput (X) vs E2E latency (Y, log) — size = prediction latency."""
    sub = comp_df[comp_df["variant"].isin(variants)].copy()
    fig = px.scatter(
        sub,
        x="throughput_transactions_per_sec",
        y="end_to_end_latency_average_ms",
        size="prediction_latency_average_ms",
        size_max=60,
        color="variant",
        color_discrete_map=COLORS,
        text="variant",
        labels={
            "throughput_transactions_per_sec": "Throughput (tx/s)",
            "end_to_end_latency_average_ms": "E2E latency trung bình (ms)",
            "prediction_latency_average_ms": "Prediction latency (ms)",
        },
        title="Trade-off: Throughput vs End-to-end Latency",
    )
    fig.update_yaxes(type="log")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=460)
    return fig


# ============================================================
# SECTIONS
# ============================================================

def section_overview(comp_df, variants):
    st.header("1. Tổng quan & Throughput")

    c1, c2 = st.columns([3, 2])
    with c1:
        melted = _melt(comp_df, variants, SUMMARY_COLS)
        st.plotly_chart(
            _bar(melted, "Tổng quan dữ liệu (summary)", "Số lượng"),
            width="stretch",
        )
    with c2:
        st.subheader("Bảng tổng quan")
        _metric_table(comp_df, variants, SUMMARY_COLS, spec=".0f")

    st.markdown("**Throughput** — số giao dịch xử lý được mỗi giây và tổng thời gian chạy:")
    c1, c2 = st.columns(2)
    with c1:
        melted = _melt(comp_df, variants, THROUGHPUT_TPS_COLS)
        st.plotly_chart(
            _bar(melted, "Throughput (tx/giây)", "Transactions / sec"),
            width="stretch",
        )
    with c2:
        melted = _melt(comp_df, variants, THROUGHPUT_DURATION_COLS)
        st.plotly_chart(
            _bar(melted, "Tổng thời gian chạy (giây)", "Seconds"),
            width="stretch",
        )


def section_classification(comp_df, variants):
    st.header("2. Classification Metrics")
    st.caption("Accuracy · Precision · Recall · F1-score · FPR (False Positive Rate)")
    melted = _melt(comp_df, variants, CLASSIFICATION_COLS)
    st.plotly_chart(
        _bar(melted, "So sánh classification metrics", "Giá trị"),
        width="stretch",
    )
    _metric_table(comp_df, variants, CLASSIFICATION_COLS)


def section_confusion(comp_df, variants):
    st.header("3. Confusion Matrix")
    melted = _melt(comp_df, variants, CONFUSION_COLS)
    st.plotly_chart(
        _bar(melted, "So sánh confusion matrix (TN / FP / FN / TP)", "Số lượng"),
        width="stretch",
    )
    _metric_table(comp_df, variants, CONFUSION_COLS, spec=".0f")


def section_probability(comp_df, variants):
    st.header("4. Probability Metrics")
    st.caption("ROC-AUC và Average Precision (diện tích dưới đường PR)")
    melted = _melt(comp_df, variants, PROBABILITY_COLS)
    st.plotly_chart(
        _bar(melted, "So sánh probability metrics", "Giá trị"),
        width="stretch",
    )
    _metric_table(comp_df, variants, PROBABILITY_COLS)


def section_latency(comp_df, variants):
    st.header("5. Latency (ms)")
    st.caption(
        "Prediction latency = thời gian LSTM inference.\n"
        "End-to-end latency = thời gian từ producer đến PostgreSQL sink. "
        "Trục log để dễ so sánh vì 500 tx-s lớn hơn nhiều."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prediction latency")
        melted = _melt(comp_df, variants, PRED_LATENCY_COLS)
        st.plotly_chart(
            _bar(melted, "Prediction latency (avg / p50 / p95 / p99)", "ms", log_scale=True),
            width="stretch",
        )
        _metric_table(comp_df, variants, PRED_LATENCY_COLS, spec=".1f")
    with c2:
        st.subheader("End-to-end latency")
        melted = _melt(comp_df, variants, E2E_LATENCY_COLS)
        st.plotly_chart(
            _bar(melted, "End-to-end latency (avg / p50 / p95 / p99)", "ms", log_scale=True),
            width="stretch",
        )
        _metric_table(comp_df, variants, E2E_LATENCY_COLS, spec=".1f")


def section_topk(comp_df, variants):
    st.header("6. Precision@Top-K")
    st.caption("Precision khi lấy top 50 / 100 / 200 giao dịch có xác suất gian lận cao nhất")
    melted = _melt(comp_df, variants, TOP_K_COLS)
    st.plotly_chart(
        _bar(melted, "Precision tại top 50 / 100 / 200", "Precision"),
        width="stretch",
    )
    _metric_table(comp_df, variants, TOP_K_COLS)


def section_resources(variants, containers):
    st.header("7. Tài nguyên hệ thống (Docker)")
    st.caption("CPU % và bộ nhớ (GiB) của các container theo thời gian, đối chiếu 3 cấu hình.")

    if not containers:
        st.info("Chọn ít nhất 1 container ở sidebar để xem tài nguyên.")
        return

    frames = [load_resource_df(v) for v in variants]
    res_df = pd.concat(frames, ignore_index=True)

    for container in containers:
        sub = res_df[res_df["container"] == container]
        if sub.empty:
            st.warning(f"Không có dữ liệu tài nguyên cho container `{container}`.")
            continue

        st.subheader(f"Container: `{container}`")
        c1, c2 = st.columns(2)

        with c1:
            fig = px.line(
                sub,
                x="elapsed_min",
                y="cpu_percent",
                color="variant",
                color_discrete_map=COLORS,
                title=f"{container} — CPU (%)",
                labels={"elapsed_min": "Thời gian (phút)", "cpu_percent": "CPU %"},
            )
            st.plotly_chart(fig, width="stretch")

        with c2:
            fig = px.line(
                sub,
                x="elapsed_min",
                y="mem_used_gib",
                color="variant",
                color_discrete_map=COLORS,
                title=f"{container} — Memory (GiB)",
                labels={"elapsed_min": "Thời gian (phút)", "mem_used_gib": "Memory (GiB)"},
            )
            st.plotly_chart(fig, width="stretch")

    st.subheader("Bảng tổng hợp tài nguyên (trung bình / max)")
    st.dataframe(
        _compute_resource_summary(variants, containers),
        width="stretch",
    )


def section_pr_curves(variants):
    st.header("8. Precision-Recall Curves")
    st.caption("Đường Precision-Recall của từng cấu hình (lưu dưới dạng PNG).")

    cols = st.columns(len(variants))
    for col, variant in zip(cols, variants):
        path = _variant_file(variant, "_precision_recall_curve.png")
        if os.path.exists(path):
            col.image(path, caption=f"{variant}", width="stretch")
        else:
            col.info(f"Không tìm thấy `{os.path.basename(path)}`")


# ============================================================
# PHÂN TÍCH & ĐÁNH GIÁ
# ============================================================

def _analysis_paragraphs(comp_df, variants):
    """Sinh các đoạn giải thích chênh lệch, số liệu tính tự động từ dữ liệu."""
    sub = comp_df[comp_df["variant"].isin(variants)]
    f1_delta = (
        sub["classification_metrics_f1_score"].max()
        - sub["classification_metrics_f1_score"].min()
    )
    auc_delta = (
        sub["probability_metrics_roc_auc"].max()
        - sub["probability_metrics_roc_auc"].min()
    )

    min_v = sub.loc[sub["throughput_transactions_per_sec"].idxmin()]
    max_v = sub.loc[sub["throughput_transactions_per_sec"].idxmax()]

    tps_min = min_v["throughput_transactions_per_sec"]
    tps_max = max_v["throughput_transactions_per_sec"]
    pred_min = min_v["prediction_latency_average_ms"]
    pred_max = max_v["prediction_latency_average_ms"]
    e2e_min = min_v["end_to_end_latency_average_ms"]
    e2e_max = max_v["end_to_end_latency_average_ms"]
    p99_min = min_v["prediction_latency_p99_ms"]
    p99_max = max_v["prediction_latency_p99_ms"]

    tps_ratio = tps_max / tps_min
    pred_ratio = pred_max / pred_min
    e2e_ratio = e2e_max / e2e_min

    paragraphs = [
        f"**① Chất lượng mô hình gần như không đổi.** "
        f"Cả {len(variants)} cấu hình dùng **cùng một mô hình LSTM** và cùng xử lý 500 000 giao dịch. "
        f"Chênh lệch F1-score giữa các cấu hình chỉ khoảng **{f1_delta:.4f}** "
        f"và ROC-AUC khoảng **{auc_delta:.4f}**. "
        f"Vì vậy, tăng tốc độ producer **không làm suy giảm đáng kể chất lượng phát hiện gian lận**; "
        f"khác biệt nhỏ (ví dụ 500 tx-s phát hiện ít hơn vài TP) chủ yếu do thứ tự và cách gộp "
        f"micro-batch trong pipeline streaming.",
        f"**② Latency tăng phi tuyến khi tăng throughput.** "
        f"So **{min_v['variant']}** (throughput thấp nhất) với **{max_v['variant']}** (cao nhất): "
        f"throughput tăng **×{tps_ratio:.2f}** ({tps_min:.1f} → {tps_max:.1f} tx/s), "
        f"nhưng prediction latency trung bình tăng **×{pred_ratio:.2f}** "
        f"({pred_min:.0f} → {pred_max:.0f} ms) và E2E latency tăng **×{e2e_ratio:.2f}** "
        f"({e2e_min/1000:.1f} → {e2e_max/1000:.0f} s).",
    ]

    ordered = sub.sort_values("throughput_transactions_per_sec")
    if len(ordered) >= 3:
        second, first = ordered.iloc[-2], ordered.iloc[-1]
        hi_tps = first["throughput_transactions_per_sec"] / second["throughput_transactions_per_sec"]
        hi_pred = first["prediction_latency_average_ms"] / second["prediction_latency_average_ms"]
        hi_e2e = first["end_to_end_latency_average_ms"] / second["end_to_end_latency_average_ms"]
        paragraphs.append(
            f"Đặc biệt, giữa **{second['variant']}** và **{first['variant']}**, throughput chỉ tăng "
            f"**×{hi_tps:.2f}** nhưng prediction latency tăng **×{hi_pred:.1f}** và E2E tăng "
            f"**×{hi_e2e:.1f}** — tăng **phi tuyến**, dấu hiệu điển hình của **hàng đợi bão hòa "
            f"(queue saturation)**: producer đẩy nhanh hơn năng lực xử lý của Spark."
        )

    res = _compute_resource_summary(variants, CONTAINERS)
    spark = res[res["container"] == "spark"]
    if not spark.empty:
        scpu_min = spark["cpu_mean_%"].min()
        scpu_max = spark["cpu_mean_%"].max()
        kafka = res[res["container"] == "kafka"]
        kafka_cpu = ", ".join(
            f"{r['variant']} ≈ {r['cpu_mean_%']:.0f}%" for _, r in kafka.iterrows()
        )
        paragraphs.append(
            f"**③ Điểm nghẽn nằm ở Spark (LSTM).** CPU trung bình của Spark gần như **không đổi** "
            f"(~{scpu_min:.0f}–{scpu_max:.0f}%) ở mọi cấu hình, nghĩa là LSTM inference đã chạy "
            f"sát giới hạn (~3.5 nhân CPU). Trong khi đó CPU Kafka **tăng theo tốc độ producer** "
            f"({kafka_cpu}) vì phải chuyển nhiều message hơn. Khi producer đẩy nhanh hơn năng lực "
            f"xử lý của Spark, giao dịch bị **xếp hàng chờ** và thời gian chờ này chiếm phần lớn "
            f"E2E latency."
        )

    paragraphs.append(
        f"**④ Bằng chứng bão hòa.** Prediction latency p99 của **{max_v['variant']}** chạm trần "
        f"~**{p99_max:.0f} ms** (giới hạn window xử lý), trong khi **{min_v['variant']}** chỉ "
        f"~**{p99_min:.0f} ms** — cho thấy hệ thống đang hoạt động sát giới hạn thay vì "
        f"chỉ chậm hơn đồng đều."
    )

    return paragraphs


def _recommendation_lines(comp_df, variants):
    """Sinh nhận định theo từng variant, số liệu tính tự động."""
    sub = comp_df[comp_df["variant"].isin(variants)].set_index("variant")
    notes = {
        "50tx-s": "phù hợp khi ưu tiên **độ trễ thấp nhất** (real-time nghiêm ngặt)",
        "100tx-s": "điểm **cân bằng tốt nhất** giữa throughput và độ trễ",
        "500tx-s": "chỉ phù hợp xử lý **gần-batch / offline**, không nên dùng để chặn gian lận thời gian thực",
    }
    lines = []
    for variant in variants:
        row = sub.loc[variant]
        lines.append(
            f"- **{variant}**: throughput ≈ **{row['throughput_transactions_per_sec']:.0f} tx/s**, "
            f"E2E latency trung bình ≈ **{row['end_to_end_latency_average_ms']/1000:.0f} s** — "
            f"{notes.get(variant, '')}."
        )
    return lines


def section_analysis(comp_df, variants):
    st.header("9. Phân tích & Đánh giá")

    if len(variants) < 2:
        st.info("Chọn ít nhất 2 variant ở sidebar để xem phân tích so sánh.")
        return

    st.subheader("9.1. Đánh đổi giữa Throughput và Latency")
    st.plotly_chart(_tradeoff_bubble(comp_df, variants), width="stretch")
    st.caption(
        "Trục Y theo thang log. Kích thước bong bóng = prediction latency trung bình (ms). "
        "Điểm lý tưởng là góc dưới bên phải (throughput cao, latency thấp)."
    )

    st.subheader("9.2. So sánh tương đối (so với baseline)")
    ratio_df, baseline = _compute_ratios(comp_df, variants)
    st.dataframe(ratio_df, width="stretch")
    st.caption(
        f"Các cột kết thúc bằng \"×\" = gấp bao nhiêu lần giá trị tương ứng "
        f"của **{baseline}** (cấu hình throughput thấp nhất)."
    )

    st.subheader("9.3. Giải thích sự chênh lệch")
    for para in _analysis_paragraphs(comp_df, variants):
        st.markdown(para)

    st.subheader("9.4. Kết luận & Khuyến nghị")
    for line in _recommendation_lines(comp_df, variants):
        st.markdown(line)

    min_e2e_s = (
        comp_df[comp_df["variant"].isin(variants)]["end_to_end_latency_average_ms"].min() / 1000.0
    )
    st.success(
        "**Khuyến nghị:** nếu cần cân bằng giữa throughput và độ trễ thời gian thực, hãy chọn "
        "**100 tx-s**. Nếu ưu tiên độ trễ tuyệt đối → **50 tx-s**. Chỉ dùng **500 tx-s** khi "
        "chấp nhận độ trễ lớn để đạt throughput tối đa."
    )
    st.warning(
        f"Lưu ý: ngay cả cấu hình nhanh nhất, E2E latency trung bình vẫn ~**{min_e2e_s:.0f} s**. "
        "Nếu hệ thống yêu cầu độ trễ dưới 1–2 giây (chặn gian lận ngay tại điểm bán), nên tối ưu "
        "pipeline (giảm micro-batch interval, tăng tài nguyên Spark, tăng partition Kafka) "
        "thay vì chỉ giảm tốc độ producer."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    st.sidebar.header("⚙️ Bộ lọc")
    selected_variants = st.sidebar.multiselect(
        "Variant so sánh",
        VARIANTS,
        default=VARIANTS,
    )
    selected_containers = st.sidebar.multiselect(
        "Container theo dõi tài nguyên",
        CONTAINERS,
        default=CONTAINERS,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dữ liệu đọc từ thư mục:\n`{DATA_DIR}`")

    st.title("📊 Dashboard so sánh các độ đo — 50 / 100 / 500 tx-s")
    st.caption(
        "Real-time Fraud Detection Pipeline — Kafka → Spark (LSTM) → PostgreSQL. "
        "So sánh metrics giữa các cấu hình producer 50 tx-s, 100 tx-s và 500 tx-s."
    )

    if not selected_variants:
        st.warning("Chọn ít nhất 1 variant ở sidebar để xem biểu đồ.")
        return

    comp_df = load_comparison_df()

    section_overview(comp_df, selected_variants)
    st.divider()
    section_classification(comp_df, selected_variants)
    st.divider()
    section_confusion(comp_df, selected_variants)
    st.divider()
    section_probability(comp_df, selected_variants)
    st.divider()
    section_latency(comp_df, selected_variants)
    st.divider()
    section_topk(comp_df, selected_variants)
    st.divider()
    section_resources(selected_variants, selected_containers)
    st.divider()
    section_pr_curves(selected_variants)
    st.divider()
    section_analysis(comp_df, selected_variants)

    st.caption("Dashboard được sinh tự động từ các file export trong `50_100_500tx-s_evaluate`.")


if __name__ == "__main__":
    main()
