"""
XAI OULAD Risk Warning System - Streamlit App
Phase 2: Cấu trúc UI cơ bản
"""
import sys
import json
from pathlib import Path

import streamlit as st

# Đường dẫn deployment package
DEPLOY_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(DEPLOY_DIR))

# ============================================================
# CẤU HÌNH PAGE
# ============================================================
st.set_page_config(
    page_title="XAI OULAD - Cảnh báo rủi ro học tập",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# HEADER
# ============================================================
st.title("Hệ thống cảnh báo sớm rủi ro học tập")
st.caption("Powered by XGBoost + SHAP/LIME | Dataset: OULAD")

# ============================================================
# SIDEBAR - Model info
# ============================================================
with st.sidebar:
    st.header("Thông tin Model")

    # Load metrics từ metadata
    try:
        with open(DEPLOY_DIR / "metadata" / "performance_metrics.json") as f:
            perf = json.load(f)
        metrics = perf["test_set"]["metrics"]
        n_test = perf["test_set"]["n_samples"]

        st.subheader("Hiệu suất trên Test set")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("F1 (Risk)", f"{metrics['f1_0']:.3f}")
        with col2:
            st.metric("Recall (Risk)", f"{metrics['recall_0']:.3f}")
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        st.caption(f"Test set: {n_test:,} sinh viên")
    except Exception as e:
        st.error(f"Không load được metadata: {e}")

    st.divider()

    # Load thresholds
    try:
        with open(DEPLOY_DIR / "config" / "thresholds.json") as f:
            th = json.load(f)

        st.subheader("Ngưỡng cảnh báo")
        st.text(f"Decision: {th['decision_threshold']:.3f}")
        st.text(f"EXTREME: >= {th['warning_levels']['extreme']:.2f}")
        st.text(f"HIGH:    >= {th['warning_levels']['high']:.2f}")
        st.text(f"MEDIUM:  >= {th['warning_levels']['medium']:.2f}")
    except Exception as e:
        st.warning(f"Thresholds không load được: {e}")

# ============================================================
# 3 TABS (khung sườn - sẽ điền nội dung ở phase sau)
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "Dự đoán 1 sinh viên",
    "Upload CSV (batch)",
    "Về hệ thống",
])

with tab1:
    st.header("Dự đoán cho 1 sinh viên")
    st.info("Phase 3 sẽ thêm form nhập features ở đây.")

with tab2:
    st.header("Upload CSV cho nhiều sinh viên")
    st.info("Phase 6 sẽ thêm chức năng upload file ở đây.")

with tab3:
    st.header("Về hệ thống")
    st.markdown("""
    Hệ thống cảnh báo sớm rủi ro học tập sử dụng dữ liệu LMS (clickstream).

    **Pipeline:**
    - 22+ blocks code Python
    - 7 mô hình so sánh (5 baselines + Naive Bayes + GMM)
    - Production model: XGBoost + SMOTE + Isotonic Calibration

    **Tính năng:**
    - Dự đoán xác suất rủi ro
    - 4 mức cảnh báo (EXTREME/HIGH/MEDIUM/SAFE)
    - Diễn giải bằng SHAP và LIME
    - Gợi ý can thiệp dựa trên knowledge base
    """)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("XAI OULAD v1.0 | Học máy nâng cao - Cao học | 2026")
