"""Inference module cho XAI OULAD Risk Warning System."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

_MODULE_DIR = Path(__file__).parent.absolute()

_MODEL = XGBClassifier()
_MODEL.load_model(str(_MODULE_DIR / "model" / "xgb_model.json"))

with open(_MODULE_DIR / "model" / "feature_cols.json") as f:
    _FEATURE_COLS = json.load(f)["feature_cols"]
with open(_MODULE_DIR / "config" / "thresholds.json") as f:
    _THRESHOLDS = json.load(f)
with open(_MODULE_DIR / "config" / "intervention_kb.json", encoding="utf-8") as f:
    _INTERVENTION_KB = json.load(f)

_TH_DECISION = _THRESHOLDS["decision_threshold"]
_TH_EXTREME = _THRESHOLDS["warning_levels"]["extreme"]
_TH_HIGH = _THRESHOLDS["warning_levels"]["high"]
_TH_MEDIUM = _THRESHOLDS["warning_levels"]["medium"]


def classify_warning_level(p_risk):
    if p_risk >= _TH_EXTREME: return "EXTREME"
    if p_risk >= _TH_HIGH: return "HIGH"
    if p_risk >= _TH_MEDIUM: return "MEDIUM"
    return "SAFE"


def get_intervention_suggestions(risk_signals, warning_level, max_n=3):
    if warning_level == "SAFE":
        return [{"feature": None, "interpretation": "SV trong vùng an toàn",
                 "intervention": "Theo dõi định kỳ.", "priority": "none"}]
    suggestions = []
    for feature, shap_val in risk_signals[:max_n]:
        if feature in _INTERVENTION_KB:
            entry = dict(_INTERVENTION_KB[feature])
            entry["feature"] = feature
            entry["shap_contribution"] = float(shap_val)
            suggestions.append(entry)
    pri_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
    suggestions.sort(key=lambda s: (pri_order[s["priority"]],
                                     -abs(s["shap_contribution"])))
    return suggestions


def predict_new_student(student_features, include_shap=True):
    start = time.time()
    if not isinstance(student_features, pd.DataFrame):
        raise TypeError("student_features must be pd.DataFrame")
    if len(student_features) != 1:
        raise ValueError("Predict 1 student at a time")
    student_features = student_features[_FEATURE_COLS]

    proba = _MODEL.predict_proba(student_features.values)[0]
    p_risk = float(proba[0])
    warning_level = classify_warning_level(p_risk)

    shap_factors = []
    risk_signals = []
    if include_shap:
        import shap
        explainer = shap.TreeExplainer(_MODEL)
        shap_vals = explainer.shap_values(student_features.values)[0]
        shap_top = pd.DataFrame({
            "feature": _FEATURE_COLS,
            "value": student_features.values[0],
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals),
        }).sort_values("abs_shap", ascending=False).head(5)
        for _, row in shap_top.iterrows():
            shap_factors.append({
                "feature": row["feature"],
                "value": float(row["value"]),
                "shap_value": float(row["shap_value"]),
                "direction": "Safe" if row["shap_value"] > 0 else "Risk",
            })
            if row["shap_value"] < 0:
                risk_signals.append((row["feature"], row["shap_value"]))

    interventions = get_intervention_suggestions(risk_signals, warning_level)
    return {
        "prediction": {
            "p_risk": p_risk, "p_safe": float(proba[1]),
            "decision": "RISK" if p_risk >= _TH_DECISION else "SAFE",
            "warning_level": warning_level,
        },
        "explanations": {"shap_factors": shap_factors},
        "interventions": interventions,
        "metadata": {
            "processing_time_seconds": round(time.time() - start, 3),
            "model_threshold": _TH_DECISION,
        },
    }


def predict_batch(students_df, include_shap=False):
    proba = _MODEL.predict_proba(students_df[_FEATURE_COLS].values)
    p_risk = proba[:, 0]
    return pd.DataFrame({
        "p_risk": p_risk, "p_safe": proba[:, 1],
        "warning_level": [classify_warning_level(p) for p in p_risk],
        "decision": ["RISK" if p >= _TH_DECISION else "SAFE" for p in p_risk],
    }, index=students_df.index)
