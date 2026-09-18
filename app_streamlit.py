"""
=====================================================================
 FRAUD DETECTION - STREAMLIT LIVE DASHBOARD
=====================================================================
 Loads  : fraud_model.joblib (from CREDIT_PROJECT.py STEP 12)
 Run    : py -m streamlit run app_streamlit.py --server.port 8501
 Open   : http://localhost:8501
=====================================================================
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="wide")

BUNDLE_PATH = r"C:/Users/hp/Desktop/PRO_1_10/CREDIT/fraud_model.joblib"
bundle = joblib.load(BUNDLE_PATH)
MODEL, COLUMNS = bundle["model"], bundle["feature_columns"]
CATEGORIES, THRESHOLD = bundle["categories"], bundle["threshold"]

st.title("💳 Credit Card Fraud Detection - Live Scoring")
st.caption(f"Model: {bundle['model_name']} | PR-AUC 0.844 (out-of-time 2020) | "
           f"Threshold: {THRESHOLD:.2f} (tuned on validation)")


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return float(6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def build_features(amt, category, gender, ts, dob, lat, lon, mlat, mlon,
                   city_pop, job_freq, state_freq, t_prev, tx_seq, prev_avg):
    row = {
        "amt_log1p": np.log1p(amt),
        "hour": ts.hour, "dayofweek": ts.dayofweek, "month": ts.month,
        "age": float(np.clip((ts - dob).days / 365.25, 14, 100)),
        "distance_km": haversine(lat, lon, mlat, mlon),
        "city_pop": city_pop,
        "time_since_prev_h": t_prev, "tx_seq": tx_seq,
        "card_prev_avg_amt": prev_avg,
        "amt_vs_card_avg": float(np.clip(amt / (prev_avg + 1e-9), 0, 100)),
        "job_freq": job_freq, "state_freq": state_freq,
        "is_night": int(ts.hour in [22, 23, 0, 1, 2, 3, 4, 5]),
        "gender_M": int(gender == "M"),
    }
    X = pd.DataFrame([row])
    for c in CATEGORIES:
        X[f"cat_{c}"] = int(category == c)
    return X.reindex(columns=COLUMNS, fill_value=0).astype("float32")


# ---------------- sidebar: transaction input ----------------
st.sidebar.header("Transaction Details")
c1, c2 = st.sidebar.columns(2)
amt = c1.number_input("Amount ($)", 1.0, 50000.0, 120.0, step=5.0)
category = c2.selectbox("Category", CATEGORIES)
gender = c1.radio("Gender", ["M", "F"], horizontal=True)
d = c2.date_input("Date", value=pd.Timestamp("2020-06-15").date())
t = c2.time_input("Time", value=pd.Timestamp("14:30").time())
ts = pd.Timestamp(f"{d} {t}")

st.sidebar.subheader("Location")
c3, c4 = st.sidebar.columns(2)
lat, lon = c3.number_input("Card lat", 24.0, 50.0, 40.7), c4.number_input("Card long", -125.0, -66.0, -74.0)
mlat, mlon = c3.number_input("Merch lat", 24.0, 50.0, 40.75), c4.number_input("Merch long", -125.0, -66.0, -73.98)
city_pop = st.sidebar.number_input("City population", 10, 25_000_000, 100_000)

st.sidebar.subheader("Card History (velocity)")
c5, c6 = st.sidebar.columns(2)
t_prev = c5.number_input("Hrs since prev tx", 0.0, 500.0, 20.0)
tx_seq = c6.number_input("Tx count so far", 1, 5000, 300)
prev_avg = st.sidebar.number_input("Card avg amount ($)", 1.0, 10000.0, 50.0)
job_freq = st.sidebar.slider("Job frequency", 1, 5000, 500)
state_freq = st.sidebar.slider("State frequency", 100, 100_000, 20_000)
dob = st.sidebar.date_input("Cardholder DOB", value=pd.Timestamp("1990-05-10").date())

# ---------------- predict ----------------
if st.sidebar.button("🔍 Score Transaction", type="primary", use_container_width=True):
    X = build_features(amt, category, gender, ts, pd.Timestamp(dob),
                       lat, lon, mlat, mlon, city_pop, job_freq,
                       state_freq, t_prev, tx_seq, prev_avg)
    prob = float(MODEL.predict_proba(X)[:, 1][0])

    if prob >= THRESHOLD:
        decision, color = "🚫 DECLINE / REVIEW", "red"
    elif prob >= THRESHOLD * 0.5:
        decision, color = "👀 MANUAL REVIEW", "orange"
    else:
        decision, color = "✅ ALLOW", "green"

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Fraud Probability", f"{prob*100:.2f}%")
        st.markdown(f"### Decision: :{color}[{decision}]")
        st.progress(min(prob / max(THRESHOLD, 0.01), 1.0))
        st.caption(f"Flags at ≥ {THRESHOLD:.2f} (validation-tuned)")

    with col2:
        st.subheader("Top reasons (SHAP)")
        if hasattr(MODEL, "get_booster"):
            import xgboost as xgb
            contribs = MODEL.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
            idx = np.argsort(-np.abs(contribs))[:3]
            for i in idx:
                arrow = "⬆️ increases fraud risk" if contribs[i] > 0 else "⬇️ decreases fraud risk"
                st.write(f"**{COLUMNS[i]}** ({contribs[i]:+.2f}) — {arrow}")
        else:
            st.write("Reasons available with XGBoost model only.")

    st.info("⚠️ Synthetic Sparkov data par trained hai — production metrics lower honge.")
else:
    st.info("Sidebar mein transaction details bharo aur 'Score Transaction' dabao — "
            "ya raat 2 baje $900 ki shopping_net transaction try karo 😉")
