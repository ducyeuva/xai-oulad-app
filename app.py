"""
XAI OULAD Risk Warning System - Streamlit App
Phase 6 (Cach C): Form predict + Dashboard OULAD 32k
"""
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

DEPLOY_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(DEPLOY_DIR))

from inference import predict_new_student, _FEATURE_COLS, _THRESHOLDS

st.set_page_config(
    page_title="XAI OULAD - Cảnh báo rủi ro học tập",
    layout="wide",
    initial_sidebar_state="expanded",
)

WARNING_COLORS = {
    "EXTREME": "#C0392B", "HIGH": "#E67E22",
    "MEDIUM": "#F39C12",  "SAFE": "#27AE60",
}
WARNING_LABELS_VN = {
    "EXTREME": "Cực cao", "HIGH": "Cao",
    "MEDIUM": "Trung bình", "SAFE": "An toàn",
}
PRIORITY_COLORS = {
    "critical": "#C0392B", "high": "#E67E22",
    "medium": "#F39C12", "low": "#27AE60", "none": "#95A5A6",
}
PRIORITY_ICONS = {
    "critical": "🔴", "high": "🟠", "medium": "🟡",
    "low": "🟢", "none": "ℹ️",
}
PRIORITY_LABELS_VN = {
    "critical": "Cấp bách", "high": "Cao",
    "medium": "Trung bình", "low": "Thấp", "none": "Bình thường",
}


def make_gauge(p_risk, warning_level):
    color = WARNING_COLORS[warning_level]
    th_extreme = _THRESHOLDS["warning_levels"]["extreme"]
    th_high = _THRESHOLDS["warning_levels"]["high"]
    th_medium = _THRESHOLDS["warning_levels"]["medium"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p_risk,
        number={"valueformat": ".3f", "font": {"size": 40}},
        title={"text": "P(Rủi ro)", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "gray"},
            "bar": {"color": color, "thickness": 0.8},
            "steps": [
                {"range": [0, th_medium], "color": "#D5F5E3"},
                {"range": [th_medium, th_high], "color": "#FCF3CF"},
                {"range": [th_high, th_extreme], "color": "#F5CBA7"},
                {"range": [th_extreme, 1], "color": "#F5B7B1"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": _THRESHOLDS["decision_threshold"],
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_shap_bar_chart(shap_factors):
    df = pd.DataFrame(shap_factors).sort_values("shap_value")
    colors = ["#27AE60" if v > 0 else "#C0392B" for v in df["shap_value"]]
    fig = go.Figure(go.Bar(
        x=df["shap_value"], y=df["feature"], orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in df["shap_value"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 5 đặc trưng ảnh hưởng (SHAP)",
        xaxis_title="SHAP value (← Risk | Safe →)",
        yaxis_title="", height=350,
        margin=dict(l=20, r=80, t=50, b=40),
        showlegend=False, plot_bgcolor="white",
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    return fig


@st.cache_data
def load_dashboard_data():
    """Load 32k predictions từ CSV (cached)."""
    csv_path = DEPLOY_DIR / "oulad_32k_predictions.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


st.title("Hệ thống cảnh báo sớm rủi ro học tập")
st.caption("Powered by XGBoost + SHAP/LIME | Dataset: OULAD")

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

tab1, tab2, tab3 = st.tabs([
    "Dự đoán 1 sinh viên",
    "Dashboard OULAD 32K",
    "Về hệ thống",
])

# ===== TAB 1 (Phase 4 - giữ nguyên) =====
with tab1:
    st.header("Dự đoán cho 1 sinh viên")
    st.caption("Nhập đặc trưng của sinh viên để dự đoán mức rủi ro học tập.")
    
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
    
    default_values = {
        "total_clicks": 500, "active_days": 30, "last_active_day": 80,
        "first_active_day": 1, "pre_start_clicks": 0, "recent_30d_clicks": 100,
        "highest_education": 2, "age_band": 0, "imd_band": 5,
        "disability": 0, "num_of_prev_attempts": 0,
        "studied_credits": 60, "gender": 0,
    }
    
    with st.expander("Nhập đặc trưng (43 features)", expanded=True):
        col1, col2, col3 = st.columns(3)
        user_inputs = {}
        with col1:
            st.markdown("**Hành vi (Behavioral)**")
            for feat in behavioral_features:
                user_inputs[feat] = st.number_input(
                    feat, value=int(default_values.get(feat, 0)),
                    min_value=0, key=f"in_{feat}",
                )
        with col2:
            st.markdown("**Nhân khẩu (Demographic)**")
            for feat in demographic_features:
                user_inputs[feat] = st.number_input(
                    feat, value=int(default_values.get(feat, 0)),
                    min_value=0, key=f"in_{feat}",
                )
            st.markdown("**Module (chọn 1)**")
            mod_selected = st.selectbox(
                "Khóa học", options=module_features, index=0, key="mod_select",
            )
            for m in module_features:
                user_inputs[m] = 1 if m == mod_selected else 0
        with col3:
            st.markdown("**Region (chọn 1)**")
            region_selected = st.selectbox(
                "Vùng", options=region_features, index=0, key="region_select",
            )
            for r in region_features:
                user_inputs[r] = 1 if r == region_selected else 0
            st.divider()
            st.caption(f"Tổng features: {len(user_inputs)}/43")
    
    predict_btn = st.button(
        "Dự đoán", type="primary",
        use_container_width=True, key="predict_btn",
    )
    
    if predict_btn:
        student_df = pd.DataFrame([user_inputs])[_FEATURE_COLS]
        with st.spinner("Đang phân tích..."):
            report = predict_new_student(student_df, include_shap=True)
        
        st.divider()
        level = report["prediction"]["warning_level"]
        color = WARNING_COLORS[level]
        label_vn = WARNING_LABELS_VN[level]
        p_risk = report["prediction"]["p_risk"]
        
        banner_html = (
            '<div style="background-color:' + color + ';padding:25px;'
            'border-radius:10px;text-align:center;color:white;margin-bottom:20px;">'
            '<h2 style="margin:0;color:white;">'
            'Mức cảnh báo: ' + label_vn + ' (' + level + ')</h2>'
            '<p style="margin:5px 0 0;font-size:14px;color:white;opacity:0.9;">'
            'Decision: ' + report["prediction"]["decision"]
            + ' (threshold = ' + f"{_THRESHOLDS['decision_threshold']:.3f}"
            + ')</p></div>'
        )
        st.markdown(banner_html, unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.plotly_chart(make_gauge(p_risk, level), use_container_width=True)
        with col_right:
            st.plotly_chart(
                make_shap_bar_chart(report["explanations"]["shap_factors"]),
                use_container_width=True,
            )
        
        with st.expander("Xem chi tiết SHAP values", expanded=False):
            shap_df = pd.DataFrame(report["explanations"]["shap_factors"])
            shap_df["direction_emoji"] = shap_df["direction"].map({
                "Risk": "⬇️ Risk", "Safe": "⬆️ Safe"
            })
            st.dataframe(
                shap_df[["feature", "value", "shap_value", "direction_emoji"]],
                use_container_width=True, hide_index=True,
            )
        
        st.divider()
        st.subheader("🚨 Gợi ý can thiệp")
        interventions = report.get("interventions", [])
        
        if not interventions:
            st.info("Không có gợi ý can thiệp.")
        elif interventions[0].get("priority") == "none":
            st.success("✅ **SV trong vùng an toàn** - không cần can thiệp đặc biệt.")
        else:
            st.caption(
                f"Dựa trên {len(interventions)} đặc trưng đẩy về Risk, "
                "đây là các gợi ý can thiệp theo thứ tự ưu tiên:"
            )
            for idx, intv in enumerate(interventions, 1):
                priority = intv.get("priority", "medium")
                p_color = PRIORITY_COLORS.get(priority, "#95A5A6")
                p_icon = PRIORITY_ICONS.get(priority, "ℹ️")
                p_label = PRIORITY_LABELS_VN.get(priority, priority)
                feature = intv.get("feature", "N/A")
                interpretation = intv.get("interpretation", "")
                intervention_text = intv.get("intervention", "")
                shap_contrib = intv.get("shap_contribution", 0)
                
                card_html = (
                    '<div style="background-color:#F8F9FA;'
                    'border-left:5px solid ' + p_color + ';'
                    'padding:15px;margin-bottom:10px;border-radius:5px;">'
                    '<div style="display:flex;justify-content:space-between;'
                    'align-items:center;margin-bottom:8px;">'
                    '<span style="font-weight:bold;font-size:16px;">'
                    + p_icon + ' Mức độ: ' + p_label
                    + ' (Ưu tiên ' + str(idx) + ')</span>'
                    '<span style="background-color:' + p_color
                    + ';color:white;padding:3px 10px;'
                    'border-radius:12px;font-size:12px;">'
                    + feature + '</span></div>'
                    '<p style="margin:5px 0;color:#555;font-size:13px;">'
                    '<i>📊 ' + interpretation + '</i></p>'
                    '<p style="margin:8px 0 0 0;font-size:14px;">'
                    '<b>💡 Hành động:</b> ' + intervention_text + '</p>'
                    '<p style="margin:5px 0 0 0;color:#888;font-size:11px;">'
                    'SHAP contribution: ' + f"{shap_contrib:.4f}"
                    + '</p></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
        
        st.caption(f"Thời gian xử lý: {report['metadata']['processing_time_seconds']}s")

# ===== TAB 2: DASHBOARD OULAD 32K =====
with tab2:
    st.header("📊 Dashboard tổng quan dataset OULAD")
    st.caption(
        "Phân tích trực quan kết quả dự đoán trên toàn bộ dataset OULAD. "
        "Kết quả được tính sẵn từ pipeline trong Google Colab."
    )
    
    df = load_dashboard_data()
    
    if df is None:
        st.error(
            "❌ Không tìm thấy file `oulad_32k_predictions.csv`. "
            "Vui lòng upload file lên repo trước."
        )
    else:
        # ===== METRICS =====
        st.subheader("Tổng quan")
        col1, col2, col3, col4, col5 = st.columns(5)
        counts = df["warning_level"].value_counts().to_dict()
        total = len(df)
        
        with col1:
            st.metric("Tổng SV", f"{total:,}")
        with col2:
            n = counts.get("EXTREME", 0)
            st.metric("🔴 EXTREME", f"{n:,}", f"{n/total*100:.1f}%")
        with col3:
            n = counts.get("HIGH", 0)
            st.metric("🟠 HIGH", f"{n:,}", f"{n/total*100:.1f}%")
        with col4:
            n = counts.get("MEDIUM", 0)
            st.metric("🟡 MEDIUM", f"{n:,}", f"{n/total*100:.1f}%")
        with col5:
            n = counts.get("SAFE", 0)
            st.metric("🟢 SAFE", f"{n:,}", f"{n/total*100:.1f}%")
        
        st.divider()
        
        # ===== HÀNG 1: Bar + Pie =====
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Bar chart phân bố
            levels_order = ["EXTREME", "HIGH", "MEDIUM", "SAFE"]
            counts_ordered = pd.Series(counts).reindex(levels_order, fill_value=0)
            
            fig_bar = go.Figure(go.Bar(
                x=[WARNING_LABELS_VN[l] for l in counts_ordered.index],
                y=counts_ordered.values,
                marker=dict(color=[WARNING_COLORS[l] for l in counts_ordered.index]),
                text=[f"{v:,}" for v in counts_ordered.values],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Phân bố mức cảnh báo",
                xaxis_title="Mức cảnh báo",
                yaxis_title="Số SV",
                height=400, plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_b:
            # Pie chart
            fig_pie = go.Figure(go.Pie(
                labels=[WARNING_LABELS_VN[l] for l in counts_ordered.index],
                values=counts_ordered.values,
                marker=dict(colors=[WARNING_COLORS[l] for l in counts_ordered.index]),
                hole=0.4,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                title="Tỷ lệ phần trăm",
                height=400, showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.divider()
        
        # ===== HÀNG 2: Phân bố theo Module =====
        st.subheader("Phân bố theo khóa học (Module)")
        
        if "module" in df.columns:
            mod_dist = df.groupby(["module", "warning_level"]).size().unstack(fill_value=0)
            mod_dist = mod_dist.reindex(columns=levels_order, fill_value=0)
            
            fig_mod = go.Figure()
            for level in levels_order:
                fig_mod.add_trace(go.Bar(
                    name=WARNING_LABELS_VN[level],
                    x=mod_dist.index,
                    y=mod_dist[level],
                    marker_color=WARNING_COLORS[level],
                ))
            fig_mod.update_layout(
                barmode="stack",
                title="Số SV theo Module và Mức cảnh báo",
                xaxis_title="Module",
                yaxis_title="Số SV",
                height=450, plot_bgcolor="white",
            )
            st.plotly_chart(fig_mod, use_container_width=True)
        
        st.divider()
        
        # ===== HÀNG 3: Histogram P(Risk) =====
        st.subheader("Phân bố xác suất P(Risk) trên 32K SV")
        
        fig_hist = px.histogram(
            df, x="p_risk", nbins=50,
            color="warning_level",
            color_discrete_map=WARNING_COLORS,
            category_orders={"warning_level": levels_order},
        )
        fig_hist.update_layout(
            title="Histogram P(Risk)",
            xaxis_title="P(Risk)",
            yaxis_title="Số SV",
            height=400, plot_bgcolor="white",
        )
        # Thêm vertical lines cho thresholds
        for th_name, th_val in [
            ("Decision (0.325)", _THRESHOLDS["decision_threshold"]),
            ("HIGH (0.45)", _THRESHOLDS["warning_levels"]["high"]),
            ("EXTREME (0.70)", _THRESHOLDS["warning_levels"]["extreme"]),
        ]:
            fig_hist.add_vline(
                x=th_val, line_dash="dash", line_color="gray",
                annotation_text=th_name, annotation_position="top right",
            )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.divider()
        
        # ===== HÀNG 4: Top SV rủi ro =====
        st.subheader("Top 50 sinh viên rủi ro cao nhất")
        
        top_risky = df.sort_values("p_risk", ascending=False).head(50)
        
        # Format columns
        display_cols = ["id_student", "p_risk", "warning_level", "module"]
        if "highest_education" in top_risky.columns:
            display_cols += ["highest_education", "studied_credits"]
        if "total_clicks" in top_risky.columns:
            display_cols += ["total_clicks", "active_days"]
        
        # Style với color
        def highlight_warning(row):
            color = WARNING_COLORS.get(row["warning_level"], "#FFFFFF")
            return [f"background-color: {color}33" for _ in row]
        
        styled = top_risky[display_cols].style.apply(highlight_warning, axis=1).format({
            "p_risk": "{:.4f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)
        
        # Download button
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download toàn bộ kết quả 32K SV (CSV)",
            data=csv_bytes,
            file_name="oulad_32k_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ===== TAB 3 =====
with tab3:
    st.header("Về hệ thống")
    st.markdown("""
    Hệ thống cảnh báo sớm rủi ro học tập sử dụng dữ liệu LMS.
    
    **Pipeline:**
    - 22+ blocks code Python
    - 7 mô hình so sánh (5 baselines + Naive Bayes + GMM)
    - Production model: XGBoost + SMOTE + Isotonic Calibration
    """)

st.divider()
st.caption("XAI OULAD v1.0 | Học máy nâng cao - Cao học | 2026")
