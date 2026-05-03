"""
XAI OULAD Risk Warning System - Streamlit App
Phase 3: Form predict 1 sinh viên
"""
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

DEPLOY_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(DEPLOY_DIR))

from inference import predict_new_student, _FEATURE_COLS, _THRESHOLDS

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="XAI OULAD - Cảnh báo rủi ro học tập",
    layout="wide",
    initial_sidebar_state="expanded",
)

WARNING_COLORS = {
    "EXTREME": "#C0392B",
    "HIGH":    "#E67E22",
    "MEDIUM":  "#F39C12",
    "SAFE":    "#27AE60",
}

WARNING_LABELS_VN = {
    "EXTREME": "Cực cao",
    "HIGH":    "Cao",
    "MEDIUM":  "Trung bình",
    "SAFE":    "An toàn",
}

# ============================================================
# HEADER
# ============================================================
st.title("Hệ thống cảnh báo sớm rủi ro học tập")
st.caption("Powered by XGBoost + SHAP/LIME | Dataset: OULAD")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Thông tin Model")
    
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
    
    st.subheader("Ngưỡng cảnh báo")
    st.text(f"Decision: {_THRESHOLDS['decision_threshold']:.3f}")
    st.text(f"EXTREME: >= {_THRESHOLDS['warning_levels']['extreme']:.2f}")
    st.text(f"HIGH:    >= {_THRESHOLDS['warning_levels']['high']:.2f}")
    st.text(f"MEDIUM:  >= {_THRESHOLDS['warning_levels']['medium']:.2f}")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "Dự đoán 1 sinh viên",
    "Upload CSV (batch)",
    "Về hệ thống",
])

# ============================================================
# TAB 1: PREDICT 1 STUDENT - PHASE 3
# ============================================================
with tab1:
    st.header("Dự đoán cho 1 sinh viên")
    st.caption("Nhập đặc trưng của sinh viên để dự đoán mức rủi ro học tập.")
    
    # Phân loại 43 features
    behavioral_features = [
        f for f in _FEATURE_COLS if f.startswith("click_") or f in [
            "total_clicks", "active_days", "last_active_day",
            "first_active_day", "pre_start_clicks", "recent_30d_clicks"
        ]
    ]
    demographic_features = [
        f for f in _FEATURE_COLS if f in [
            "gender", "age_band", "highest_education", "imd_band",
            "disability", "num_of_prev_attempts", "studied_credits"
        ]
    ]
    module_features = [f for f in _FEATURE_COLS if f.startswith("mod_")]
    region_features = [f for f in _FEATURE_COLS if f.startswith("region_")]
    
    # Default values
    default_values = {
        "total_clicks": 500, "active_days": 30, "last_active_day": 80,
        "first_active_day": 1, "pre_start_clicks": 0, "recent_30d_clicks": 100,
        "highest_education": 2, "age_band": 0, "imd_band": 5,
        "disability": 0, "num_of_prev_attempts": 0,
        "studied_credits": 60, "gender": 0,
    }
    
    # FORM
    with st.expander("Nhập đặc trưng (43 features)", expanded=True):
        col1, col2, col3 = st.columns(3)
        user_inputs = {}
        
        with col1:
            st.markdown("**Hành vi (Behavioral)**")
            for feat in behavioral_features:
                user_inputs[feat] = st.number_input(
                    feat,
                    value=int(default_values.get(feat, 0)),
                    min_value=0,
                    key=f"in_{feat}",
                )
        
        with col2:
            st.markdown("**Nhân khẩu (Demographic)**")
            for feat in demographic_features:
                user_inputs[feat] = st.number_input(
                    feat,
                    value=int(default_values.get(feat, 0)),
                    min_value=0,
                    key=f"in_{feat}",
                )
            
            st.markdown("**Module (chọn 1)**")
            mod_selected = st.selectbox(
                "Khóa học",
                options=module_features,
                index=0,
                key="mod_select",
            )
            for m in module_features:
                user_inputs[m] = 1 if m == mod_selected else 0
        
        with col3:
            st.markdown("**Region (chọn 1)**")
            region_selected = st.selectbox(
                "Vùng",
                options=region_features,
                index=0,
                key="region_select",
            )
            for r in region_features:
                user_inputs[r] = 1 if r == region_selected else 0
            
            st.divider()
            st.caption(f"Tổng features: {len(user_inputs)}/43")
    
    # PREDICT BUTTON
    predict_btn = st.button(
        "Dự đoán",
        type="primary",
        use_container_width=True,
        key="predict_btn",
    )
    
    # SHOW RESULT
    if predict_btn:
        missing = set(_FEATURE_COLS) - set(user_inputs.keys())
        if missing:
            st.error(f"Thiếu features: {missing}")
        else:
            student_df = pd.DataFrame([user_inputs])[_FEATURE_COLS]
            
            with st.spinner("Đang phân tích..."):
                report = predict_new_student(student_df, include_shap=True)
            
            st.divider()
            
            level = report["prediction"]["warning_level"]
            color = WARNING_COLORS[level]
            label_vn = WARNING_LABELS_VN[level]
            p_risk = report["prediction"]["p_risk"]
            
            banner_html = (
                f'<div style="background-color:' + color + ';padding:25px;'
                f'border-radius:10px;text-align:center;color:white;">'
                f'<h2 style="margin:0;color:white;">'
                f'Mức cảnh báo: ' + label_vn + ' (' + level + ')</h2>'
                f'<p style="margin:10px 0 0;font-size:20px;color:white;">'
                f'P(Rủi ro) = ' + f"{p_risk:.4f}" + '</p>'
                f'<p style="margin:5px 0 0;font-size:14px;color:white;opacity:0.9;">'
                f'Decision: ' + report["prediction"]["decision"]
                + ' (threshold = ' + f"{_THRESHOLDS['decision_threshold']:.3f}"
                + ')</p></div>'
            )
            st.markdown(banner_html, unsafe_allow_html=True)
            
            st.write("")
            
            st.subheader("Top 5 đặc trưng ảnh hưởng nhất")
            shap_data = report["explanations"]["shap_factors"]
            shap_df = pd.DataFrame(shap_data)
            shap_df["direction_emoji"] = shap_df["direction"].map({
                "Risk": "⬇️ Risk", "Safe": "⬆️ Safe"
            })
            st.dataframe(
                shap_df[["feature", "value", "shap_value", "direction_emoji"]],
                use_container_width=True,
                hide_index=True,
            )
            
            st.caption(
                f"Thời gian xử lý: "
                f"{report['metadata']['processing_time_seconds']}s"
            )

# ============================================================
# TAB 2 + TAB 3 (placeholders)
# ============================================================
with tab2:
    st.header("Upload CSV cho nhiều sinh viên")
    st.info("Phase 6 sẽ thêm chức năng upload file ở đây.")

with tab3:
    st.header("Về hệ thống")
    st.markdown("""
    Hệ thống cảnh báo sớm rủi ro học tập sử dụng dữ liệu LMS.
    
    **Pipeline:**
    - 22+ blocks code Python
    - 7 mô hình so sánh (5 baselines + Naive Bayes + GMM)
    - Production model: XGBoost + SMOTE + Isotonic Calibration
    """)

# FOOTER
st.divider()
st.caption("XAI OULAD v1.0 | Học máy nâng cao - Cao học | 2026")
