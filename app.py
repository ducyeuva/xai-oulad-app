"""
XAI OULAD Risk Warning System - Streamlit App
Phase 7: Form predict + Dashboard 32K + SHAP analysis + Footer
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
            "axis": {"range": [0, 1]},
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


def make_shap_bar_chart(shap_factors, title="Top 5 đặc trưng ảnh hưởng (SHAP)"):
    df = pd.DataFrame(shap_factors).sort_values("shap_value")
    colors = ["#27AE60" if v > 0 else "#C0392B" for v in df["shap_value"]]
    fig = go.Figure(go.Bar(
        x=df["shap_value"], y=df["feature"], orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in df["shap_value"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="SHAP value (← Risk | Safe →)",
        yaxis_title="", height=350,
        margin=dict(l=20, r=80, t=50, b=40),
        showlegend=False, plot_bgcolor="white",
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    return fig


def make_shap_global_chart(shap_global):
    df = pd.DataFrame(shap_global)
    df = df.sort_values("mean_abs_shap", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"], y=df["feature"], orientation="h",
        marker=dict(color="#3498DB"),
        text=[f"{v:.3f}" for v in df["mean_abs_shap"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 15 đặc trưng quan trọng nhất (SHAP Global)",
        xaxis_title="Mean |SHAP value| (trên 1000 SV mẫu)",
        yaxis_title="", height=550,
        margin=dict(l=20, r=80, t=50, b=40),
        showlegend=False, plot_bgcolor="white",
    )
    return fig
def _randomize_t4_id():
    """Callback để chọn random ID cho Tab 4."""
    df_temp = load_dashboard_data()
    if df_temp is None:
        return
    df_f = df_temp.copy()
    sm = st.session_state.get("filt_mod_t4", "Tất cả")
    sl = st.session_state.get("filt_lvl_t4", "Tất cả")
    if sm != "Tất cả" and "module" in df_f.columns:
        df_f = df_f[df_f["module"] == sm]
    if sl != "Tất cả":
        df_f = df_f[df_f["warning_level"] == sl]
    if len(df_f) > 0:
        st.session_state["id_input_t4"] = str(
            df_f["id_student"].sample(1).iloc[0]
        )


@st.cache_data
def load_dashboard_data():
    csv_path = DEPLOY_DIR / "oulad_32k_predictions.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


@st.cache_data
def load_shap_data():
    json_path = DEPLOY_DIR / "oulad_32k_shap_data.json"
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


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

tab1, tab2, tab3, tab4 = st.tabs([
    "Dự đoán 1 sinh viên",
    "Dashboard OULAD 32K + SHAP",
    "Về hệ thống",
    "Tra cứu theo ID",
])

# ===== TAB 1 =====
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
    
    predict_btn = st.button("Dự đoán", type="primary", use_container_width=True, key="predict_btn")
    
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

# ===== TAB 2: Dashboard + SHAP =====
with tab2:
    st.header("📊 Dashboard tổng quan dataset OULAD + Phân tích SHAP")
    st.caption(
        "Phân tích trực quan kết quả dự đoán trên toàn bộ dataset OULAD "
        "kèm phân tích Explainable AI (SHAP)."
    )
    
    df = load_dashboard_data()
    shap_data = load_shap_data()
    
    if df is None:
        st.error("❌ Không tìm thấy file `oulad_32k_predictions.csv`.")
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
        levels_order = ["EXTREME", "HIGH", "MEDIUM", "SAFE"]
        col_a, col_b = st.columns(2)
        
        with col_a:
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
                xaxis_title="Mức cảnh báo", yaxis_title="Số SV",
                height=400, plot_bgcolor="white", showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_b:
            fig_pie = go.Figure(go.Pie(
                labels=[WARNING_LABELS_VN[l] for l in counts_ordered.index],
                values=counts_ordered.values,
                marker=dict(colors=[WARNING_COLORS[l] for l in counts_ordered.index]),
                hole=0.4,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(title="Tỷ lệ phần trăm", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.divider()
        
        # ===== Phân bố theo Module =====
        st.subheader("Phân bố theo khóa học (Module)")
        if "module" in df.columns:
            mod_dist = df.groupby(["module", "warning_level"]).size().unstack(fill_value=0)
            mod_dist = mod_dist.reindex(columns=levels_order, fill_value=0)
            
            fig_mod = go.Figure()
            for level in levels_order:
                fig_mod.add_trace(go.Bar(
                    name=WARNING_LABELS_VN[level],
                    x=mod_dist.index, y=mod_dist[level],
                    marker_color=WARNING_COLORS[level],
                ))
            fig_mod.update_layout(
                barmode="stack",
                title="Số SV theo Module và Mức cảnh báo",
                xaxis_title="Module", yaxis_title="Số SV",
                height=450, plot_bgcolor="white",
            )
            st.plotly_chart(fig_mod, use_container_width=True)
        
        st.divider()
        
        # ===== Histogram P(Risk) =====
        st.subheader("Phân bố xác suất P(Risk) trên 32K SV")
        fig_hist = px.histogram(
            df, x="p_risk", nbins=50, color="warning_level",
            color_discrete_map=WARNING_COLORS,
            category_orders={"warning_level": levels_order},
        )
        fig_hist.update_layout(
            title="Histogram P(Risk)",
            xaxis_title="P(Risk)", yaxis_title="Số SV",
            height=400, plot_bgcolor="white",
        )
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
        
        # ===== SHAP GLOBAL =====
        if shap_data is not None:
            st.subheader("🔬 Phân tích SHAP Global")
            st.caption(
                "Đặc trưng quan trọng nhất trên toàn bộ model "
                f"(tính từ {shap_data['metadata']['n_sample_global']:,} SV mẫu)"
            )
            st.plotly_chart(
                make_shap_global_chart(shap_data["shap_global_top15"]),
                use_container_width=True,
            )
            
            st.divider()
            
            # ===== SHAP LOCAL - 4 CASE STUDIES =====
            st.subheader("🔍 Phân tích SHAP Local - 4 Case Studies")
            st.caption("Phân tích SHAP cho 4 sinh viên đại diện 4 mức cảnh báo:")
            
            case_tabs = st.tabs([
                f"🔴 EXTREME (P={shap_data['shap_local_cases']['EXTREME']['p_risk']:.3f})",
                f"🟠 HIGH (P={shap_data['shap_local_cases']['HIGH']['p_risk']:.3f})",
                f"🟡 MEDIUM (P={shap_data['shap_local_cases']['MEDIUM']['p_risk']:.3f})",
                f"🟢 SAFE (P={shap_data['shap_local_cases']['SAFE']['p_risk']:.3f})",
            ])
            
            for i, level in enumerate(levels_order):
                with case_tabs[i]:
                    case = shap_data["shap_local_cases"][level]
                    st.markdown(
                        f"**SV index:** {case['sv_index']}  | "
                        f"**P(Risk):** {case['p_risk']:.4f}  | "
                        f"**Mức:** {WARNING_LABELS_VN[level]} ({level})"
                    )
                    
                    col_g, col_s = st.columns([1, 2])
                    with col_g:
                        st.plotly_chart(
                            make_gauge(case['p_risk'], level),
                            use_container_width=True,
                        )
                    with col_s:
                        st.plotly_chart(
                            make_shap_bar_chart(
                                case["top5_factors"],
                                title=f"SHAP cho SV {level}",
                            ),
                            use_container_width=True,
                        )
                    
                    # Bảng chi tiết
                    factors_df = pd.DataFrame(case["top5_factors"])
                    factors_df["direction_emoji"] = factors_df["direction"].map({
                        "Risk": "⬇️ Risk", "Safe": "⬆️ Safe"
                    })
                    st.dataframe(
                        factors_df[["feature", "value", "shap_value", "direction_emoji"]],
                        use_container_width=True, hide_index=True,
                    )
        else:
            st.info("ℹ️ Chưa có dữ liệu SHAP, vui lòng upload `oulad_32k_shap_data.json`.")
        
        st.divider()
        
        # ===== Top 50 SV rủi ro =====
        st.subheader("⚠️ Top 50 sinh viên rủi ro cao nhất")
        top_risky = df.sort_values("p_risk", ascending=False).head(50)
        
        display_cols = ["id_student", "p_risk", "warning_level", "module"]
        if "highest_education" in top_risky.columns:
            display_cols += ["highest_education", "studied_credits"]
        if "total_clicks" in top_risky.columns:
            display_cols += ["total_clicks", "active_days"]
        
        def highlight_warning(row):
            color = WARNING_COLORS.get(row["warning_level"], "#FFFFFF")
            return [f"background-color: {color}33" for _ in row]
        
        styled = top_risky[display_cols].style.apply(highlight_warning, axis=1).format({
            "p_risk": "{:.4f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)
        
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


# ===== TAB 4: Tra cứu theo ID =====
with tab4:
    st.header("🔍 Tra cứu sinh viên theo ID")
    st.caption(
        "Nhập mã sinh viên (id_student) trong dataset OULAD 32K để xem "
        "mức cảnh báo rủi ro của sinh viên đó."
    )
    
    df_32k = load_dashboard_data()
    if df_32k is None:
        st.error("❌ Không tìm thấy file `oulad_32k_predictions.csv`.")
    else:
        # Bộ lọc danh sách
        with st.expander("🔧 Bộ lọc danh sách", expanded=False):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                if "module" in df_32k.columns:
                    modules_avail = ["Tất cả"] + sorted(df_32k["module"].unique().tolist())
                    sel_mod = st.selectbox("Khóa học", modules_avail, key="filt_mod_t4")
                else:
                    sel_mod = "Tất cả"
            with fcol2:
                levels_avail = ["Tất cả", "EXTREME", "HIGH", "MEDIUM", "SAFE"]
                sel_lvl = st.selectbox("Mức cảnh báo", levels_avail, key="filt_lvl_t4")
            
            df_filt = df_32k.copy()
            if sel_mod != "Tất cả" and "module" in df_filt.columns:
                df_filt = df_filt[df_filt["module"] == sel_mod]
            if sel_lvl != "Tất cả":
                df_filt = df_filt[df_filt["warning_level"] == sel_lvl]
            st.caption(f"Số SV phù hợp: **{len(df_filt):,}** / {len(df_32k):,}")
        
        # Input ID
        if "id_input_t4" not in st.session_state:
            st.session_state["id_input_t4"] = ""
        
        col_in1, col_in2, col_in3 = st.columns([3, 1, 1])
        with col_in1:
            id_str = st.text_input(
                "Mã sinh viên (id_student)",
                placeholder="VD: 11391",
                key="id_input_t4",
            )
        
        with col_in2:
            st.button(
                "🎲 Random",
                on_click=_randomize_t4_id,
                use_container_width=True,
                key="btn_random_t4",
            )
        with col_in3:
            predict_id_btn = st.button(
                "✓ Tra cứu",
                type="primary",
                use_container_width=True,
                key="btn_predict_id_t4",
            )
        
        # Predict & display
        if predict_id_btn and id_str.strip():
            id_clean = id_str.strip()
            student_row = df_32k[df_32k["id_student"].astype(str) == id_clean]
            
            if len(student_row) == 0:
                st.error(f"❌ Không tìm thấy SV ID = **{id_clean}** trong dataset 32K.")
            else:
                student = student_row.iloc[0]
                level = student["warning_level"]
                p_risk = float(student["p_risk"])
                color = WARNING_COLORS[level]
                label_vn = WARNING_LABELS_VN[level]
                
                # Tính rank
                rank = int((df_32k["p_risk"] >= p_risk).sum())
                percentile_top = rank / len(df_32k) * 100
                
                st.divider()
                
                # Banner cảnh báo
                banner_html = (
                    '<div style="background-color:' + color + ';padding:25px;'
                    'border-radius:10px;text-align:center;color:white;margin-bottom:20px;">'
                    '<h2 style="margin:0;color:white;">'
                    'Sinh viên #' + str(student["id_student"]) + ' — Mức cảnh báo: '
                    + label_vn + ' (' + level + ')</h2>'
                    '<p style="margin:5px 0 0;font-size:14px;color:white;opacity:0.9;">'
                    f'P(Rủi ro) = {p_risk:.3f} | threshold = {_THRESHOLDS["decision_threshold"]:.3f}'
                    '</p></div>'
                )
                st.markdown(banner_html, unsafe_allow_html=True)
                
                col_g, col_info = st.columns([1, 2])
                
                with col_g:
                    st.plotly_chart(make_gauge(p_risk, level), use_container_width=True)
                
                with col_info:
                    st.subheader("📊 Vị trí trong dataset")
                    rank_cols = st.columns(2)
                    rank_cols[0].metric(
                        "Xếp hạng theo P(Risk)",
                        f"{rank:,} / {len(df_32k):,}",
                    )
                    rank_cols[1].metric(
                        "Top % rủi ro nhất",
                        f"{percentile_top:.1f}%",
                    )
                    
                    st.subheader("👤 Thông tin sinh viên")
                    info_show = student.drop(
                        ["id_student", "p_risk", "warning_level"],
                        errors="ignore"
                    )
                    st.dataframe(
                        pd.DataFrame({
                            "Đặc trưng": info_show.index,
                            "Giá trị": info_show.values,
                        }),
                        use_container_width=True,
                        hide_index=True,
                        height=250,
                    )
                
                st.info(
                    "💡 **Để xem SHAP analysis chi tiết và gợi ý can thiệp cho SV này:** "
                    "Chuyển sang **Tab 1 - Dự đoán cho 1 sinh viên** và nhập các đặc trưng "
                    "từ bảng thông tin trên."
                )


# ===== FOOTER =====
st.divider()
footer_html = """
<div style="background:linear-gradient(135deg,#F8F9FA 0%,#E9ECEF 100%);
            padding:30px 40px;border-radius:12px;margin-top:30px;
            border-top:4px solid #2C3E50;">
    <div style="display:grid;grid-template-columns:2fr 1.5fr 1.5fr;gap:30px;
                margin-bottom:20px;">
        <div>
            <h4 style="margin:0 0 10px;color:#2C3E50;font-size:16px;">
                🎓 XAI OULAD Risk Warning System
            </h4>
            <p style="margin:0;color:#555;font-size:13px;line-height:1.6;">
                Hệ thống cảnh báo sớm rủi ro học tập sử dụng 
                Explainable AI trên dataset OULAD.
            </p>
            <p style="margin:8px 0 0;color:#888;font-size:11px;">
                Powered by XGBoost + SMOTE + Isotonic Calibration + SHAP/LIME
            </p>
        </div>
        <div style="border-left:2px solid #DEE2E6;padding-left:25px;">
            <h4 style="margin:0 0 10px;color:#2C3E50;font-size:14px;">
                📚 Học phần
            </h4>
            <p style="margin:0;color:#555;font-size:13px;line-height:1.8;">
                <b>Học máy nâng cao</b><br>
                <span style="color:#888;">GVHD:</span> TS. Ngô Quốc Việt<br>
                <span style="color:#888;">Trường:</span> ĐHSP TP.HCM
            </p>
        </div>
        <div style="border-left:2px solid #DEE2E6;padding-left:25px;">
            <h4 style="margin:0 0 10px;color:#2C3E50;font-size:14px;">
                👥 Nhóm thực hiện
            </h4>
            <p style="margin:0;color:#555;font-size:13px;line-height:1.8;">
                Hoàng Châu Ngọc Phương<br>
                <span style="color:#888;font-size:11px;">KHMT836028</span><br>
                Đoàn Huỳnh Thanh Tú<br>
                <span style="color:#888;font-size:11px;">KHMT836034</span>
            </p>
        </div>
    </div>
    <div style="text-align:center;padding-top:15px;
                border-top:1px solid #DEE2E6;color:#999;font-size:11px;">
        XAI OULAD v1.0 © 2026 | All rights reserved
    </div>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
