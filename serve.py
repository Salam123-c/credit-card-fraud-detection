"""
=====================================================================
 FRAUD SCORING API - DEPLOYMENT LAYER
=====================================================================
 Loads   : fraud_model.joblib  (produced by CREDIT_PROJECT.py STEP 12)
 Run     : uvicorn serve:app --host 0.0.0.0 --port 8000
 Test    : POST /predict  (JSON body - see SAMPLE below)
           GET  /health

 Design notes (BFSI):
 - Feature engineering is replicated EXACTLY as at training time
   (same transforms, same column order) -> guards training-serving skew.
 - Card-history features (velocity) are supplied by the caller; in a
   live system these come from a streaming feature store.
 - Per-decision top-3 reasons via XGBoost native SHAP (pred_contribs).
=====================================================================
"""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, Field

BUNDLE_PATH = r"C:/Users/hp/Desktop/PRO_1_10/CREDIT/fraud_model.joblib"

bundle = joblib.load(BUNDLE_PATH)
MODEL = bundle["model"]
COLUMNS = bundle["feature_columns"]
CATEGORIES = bundle["categories"]
THRESHOLD = bundle["threshold"]
MODEL_NAME = bundle["model_name"]

app = FastAPI(title="Credit Card Fraud Scoring API", version="1.0")


class Transaction(BaseModel):
    amt: float = Field(..., gt=0, description="Transaction amount ($)")
    category: str = Field(..., description="e.g. shopping_net, grocery_pos")
    gender: str = Field(..., description="M or F")
    trans_time: str = Field(..., description="Transaction timestamp, e.g. 2021-03-15 02:30:00")
    dob: str = Field(..., description="Cardholder date of birth, e.g. 1990-05-10")
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    city_pop: int = 10000
    job_freq: int = 500          # in production: from reference tables
    state_freq: int = 20000
    # card-history / velocity features (from feature store in production)
    time_since_prev_h: float = 999.0
    tx_seq: int = 1
    card_prev_avg_amt: float = None


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return float(6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def build_features(t: Transaction) -> pd.DataFrame:
    ts = pd.to_datetime(t.trans_time)
    dob = pd.to_datetime(t.dob)
    prev_avg = t.card_prev_avg_amt if t.card_prev_avg_amt else t.amt
    row = {
        "amt_log1p": np.log1p(t.amt),
        "hour": ts.hour,
        "dayofweek": ts.dayofweek,
        "month": ts.month,
        "age": float(np.clip((ts - dob).days / 365.25, 14, 100)),
        "distance_km": haversine(t.lat, t.long, t.merch_lat, t.merch_long),
        "city_pop": t.city_pop,
        "time_since_prev_h": t.time_since_prev_h,
        "tx_seq": t.tx_seq,
        "card_prev_avg_amt": prev_avg,
        "amt_vs_card_avg": float(np.clip(t.amt / (prev_avg + 1e-9), 0, 100)),
        "job_freq": t.job_freq,
        "state_freq": t.state_freq,
        "is_night": int(ts.hour in [22, 23, 0, 1, 2, 3, 4, 5]),
        "gender_M": int(t.gender.upper() == "M"),
    }
    X = pd.DataFrame([row])
    # one-hot categories -> reindex to EXACT training column order
    for c in CATEGORIES:
        X[f"cat_{c}"] = int(t.category == c)
    return X.reindex(columns=COLUMNS, fill_value=0).astype("float32")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "threshold": THRESHOLD}


@app.post("/predict")
def predict(t: Transaction):
    X = build_features(t)
    prob = float(MODEL.predict_proba(X)[:, 1][0])

    # decision bands around the validation-tuned threshold
    if prob >= THRESHOLD:
        decision = "DECLINE / REVIEW"
    elif prob >= THRESHOLD * 0.5:
        decision = "MANUAL REVIEW"
    else:
        decision = "ALLOW"

    # top-3 reasons (XGBoost native SHAP contributions)
    reasons = []
    if MODEL_NAME == "XGBoost" and hasattr(MODEL, "get_booster"):
        import xgboost as xgb
        contribs = MODEL.get_booster().predict(
            xgb.DMatrix(X), pred_contribs=True)[0][:-1]  # drop bias term
        idx = np.argsort(-np.abs(contribs))[:3]
        reasons = [{"feature": COLUMNS[i],
                    "impact": float(contribs[i])} for i in idx]

    return {
        "fraud_probability": round(prob, 4),
        "threshold": THRESHOLD,
        "decision": decision,
        "top_reasons": reasons,
        "model": MODEL_NAME,
    }


SAMPLE = {
    "amt": 850.0, "category": "shopping_net", "gender": "M",
    "trans_time": "2021-03-15 02:30:00", "dob": "1990-05-10",
    "lat": 40.7, "long": -74.0, "merch_lat": 33.5, "merch_long": -112.0,
    "city_pop": 5000, "time_since_prev_h": 0.4, "tx_seq": 14,
    "card_prev_avg_amt": 55.0,
}

if __name__ == "__main__":
    import uvicorn
    print("Example request body:\n", SAMPLE)
    uvicorn.run(app, host="0.0.0.0", port=8000)
