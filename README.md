# Credit Card Fraud Detection - End-to-End BFSI Project

End-to-end fraud detection system: EDA -> feature engineering -> class-imbalance handling -> model training -> out-of-time validation -> deployed scoring API.

## Dataset
- Source: Hugging Face [`alenc123/credit-card-fraud`](https://huggingface.co/datasets/alenc123/credit-card-fraud) (Sparkov synthetic, ~1.3M transactions, Jan 2019 - Jun 2020, fraud rate 0.579%)
- Download the parquet into this folder (109 MB, excluded from git):
  `https://huggingface.co/datasets/alenc123/credit-card-fraud/resolve/main/credit_card_transactions.parquet`

## Results (out-of-time test: 2020)
| Model | PR-AUC | ROC | KS | precision@500 |
|---|---|---|---|---|
| Logistic Regression | 0.294 | 0.960 | 0.794 | 0.212 |
| **XGBoost (champion)** | **0.844** | 0.994 | 0.918 | **1.000** |

- Threshold tuned on validation only: 0.675 -> precision 0.957 / recall 0.639
- Bias audit, PSI stability, bootstrap CI (+0.550 uplift, CI [+0.533, +0.567]) included
- Report: `model_summary.txt` | Charts: `model_curves.png`, `eda_fraud_patterns.png`, `feature_importance.png`, `confusion_matrix.png`

## Pipeline (CREDIT_PROJECT.py)
1. Load + reality check
2. EDA - fraud fingerprint (night fraud 11.6x riskier)
3. Cleaning + PII/leakage removal
4. Feature engineering (Haversine distance, card-velocity, behavior ratios)
5. Screening: Information Value / WOE + VIF
6. Temporal split + 2 models (class-weighted)
7. Metrics: PR-AUC (primary), Gini, KS, precision@k, dollar-weighted recall
8. Threshold tuning on validation + business view
9. Bias audit (gender / age groups)
10. PSI population stability
11. Model documentation + limitations
12. Model persistence -> `fraud_model.joblib`

## Run training
```bash
py CREDIT_PROJECT.py
```

## Run the scoring API (deployment)
```bash
py -m uvicorn serve:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```

Example request:
```json
{
  "amt": 850.0, "category": "shopping_net", "gender": "M",
  "trans_time": "2021-03-15 02:30:00", "dob": "1990-05-10",
  "lat": 40.7, "long": -74.0, "merch_lat": 33.5, "merch_long": -112.0,
  "city_pop": 5000, "time_since_prev_h": 0.4, "tx_seq": 14,
  "card_prev_avg_amt": 55.0
}
```
Response: fraud probability, decision band (ALLOW / MANUAL REVIEW / DECLINE), top-3 SHAP reasons.

## Limitations (disclosed)
1. Synthetic Sparkov data - real-world metrics will be lower.
2. Labels are instant here; production fraud labels arrive after chargebacks (label latency).
3. Velocity features are cumulative-history proxies; live systems need identical rolling windows at serving time.

## Requirements
`pandas, numpy, scikit-learn, pyarrow, xgboost, fastapi, uvicorn, joblib, matplotlib`
