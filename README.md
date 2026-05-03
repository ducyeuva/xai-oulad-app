# XAI OULAD Risk Warning System

Hệ thống cảnh báo sớm rủi ro học tập sử dụng XGBoost + SHAP/LIME.

## Performance

| Metric | Test set |
|---|---|
| Accuracy | 0.7953 |
| Recall (Risk) | 0.8687 |
| Precision (Risk) | 0.7720 |
| F1 (Risk) | 0.8175 |
| ROC-AUC | 0.9004 |

Test set: 4889 sinh viên.

## Usage

```python
from inference import predict_new_student
import pandas as pd

student_data = pd.DataFrame([{"total_clicks": 250, "active_days": 15}])
report = predict_new_student(student_data, include_shap=True)
print(report["prediction"]["warning_level"])
```

## 4 mức cảnh báo

| Level | P(Risk) range |
|---|---|
| EXTREME | >= 0.70 |
| HIGH | >= 0.45 |
| MEDIUM | >= 0.33 |
| SAFE | < 0.33 |

Generated: 2026-05-03
