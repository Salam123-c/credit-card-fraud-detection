"""
=====================================================================
 CREDIT CARD FRAUD DETECTION - SCORING CONSOLE
=====================================================================
 Loads  : fraud_model.joblib (from CREDIT_PROJECT.py STEP 12)
 Run    : py -m streamlit run app_streamlit.py --server.port 8501
=====================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Scoring Console",
                   page_icon=None, layout="wide")

BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fraud_model.joblib")
bundle = joblib.load(BUNDLE_PATH)
MODEL, COLUMNS = bundle["model"], bundle["feature_columns"]
CATEGORIES, THRESHOLD = bundle["categories"], bundle["threshold"]

PRIMARY = "#0F4C81"
GREY = "#5A6472"


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


# ---------------- header ----------------
st.markdown(f"""
<div style="border-bottom:2px solid {PRIMARY}; padding-bottom:10px; margin-bottom:5px">
  <h2 style="color:{PRIMARY}; margin:0; font-weight:600;">
      Transaction Fraud Scoring Console</h2>
</div>
<p style="color:{GREY}; font-size:0.9rem; margin-top:8px;">
  Model: XGBoost &nbsp;|&nbsp; Validation: PR-AUC 0.844 (out-of-time, 2020)
  &nbsp;|&nbsp; Decision threshold: {THRESHOLD:.2f} (tuned on validation set)
  &nbsp;|&nbsp; Training data: synthetic (Sparkov)
</p>
""", unsafe_allow_html=True)

# ---------------- sidebar: transaction input ----------------
with st.sidebar.form("transaction_form"):
    st.subheader("Transaction Parameters")
    c1, c2 = st.columns(2)
    amt = c1.number_input("Amount (USD)", 1.0, 50000.0, 120.0, step=5.0)
    category = c2.selectbox("Merchant category", CATEGORIES)
    gender = c1.radio("Cardholder gender", ["M", "F"], horizontal=True)
    d = c2.date_input("Transaction date", value=pd.Timestamp("2020-06-15").date())
    t = c2.time_input("Transaction time", value=pd.Timestamp("14:30").time())
    ts = pd.Timestamp(f"{d} {t}")

    st.subheader("Geography")
    c3, c4 = st.columns(2)
    lat = c3.number_input("Cardholder latitude", 24.0, 50.0, 40.7)
    lon = c4.number_input("Cardholder longitude", -125.0, -66.0, -74.0)
    mlat = c3.number_input("Merchant latitude", 24.0, 50.0, 40.75)
    mlon = c4.number_input("Merchant longitude", -125.0, -66.0, -73.98)
    city_pop = st.number_input("Cardholder city population", 10, 25_000_000, 100_000)

    st.subheader("Card History")
    c5, c6 = st.columns(2)
    t_prev = c5.number_input("Hours since previous transaction", 0.0, 500.0, 20.0)
    tx_seq = c6.number_input("Transaction sequence on card", 1, 5000, 300)
    prev_avg = st.number_input("Card average amount (USD)", 1.0, 10000.0, 50.0)
    job_freq = st.slider("Occupation frequency encoding", 1, 5000, 500)
    state_freq = st.slider("State frequency encoding", 100, 100_000, 20_000)
    dob = st.date_input("Cardholder date of birth",
                        value=pd.Timestamp("1990-05-10").date())
    submitted = st.form_submit_button("Score Transaction", type="primary",
                                      use_container_width=True)

# ---------------- results ----------------
if submitted:
    X = build_features(amt, category, gender, ts, pd.Timestamp(dob),
                       lat, lon, mlat, mlon, city_pop, job_freq,
                       state_freq, t_prev, tx_seq, prev_avg)
    prob = float(MODEL.predict_proba(X)[:, 1][0])

    if prob >= THRESHOLD:
        decision, rationale = "DECLINE - ROUTE TO FRAUD REVIEW", (
            "Score exceeds the validated decision threshold.")
        banner_bg, banner_tx = "#8B1A1A", "#FFFFFF"
    elif prob >= THRESHOLD * 0.5:
        decision, rationale = "MANUAL REVIEW", (
            "Score falls in the grey zone; refer to analyst queue.")
        banner_bg, banner_tx = "#B8860B", "#FFFFFF"
    else:
        decision, rationale = "APPROVE", (
            "Score is comfortably below the decision threshold.")
        banner_bg, banner_tx = "#1A6B3C", "#FFFFFF"

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown(f"""
        <div style="background:{banner_bg}; color:{banner_tx}; padding:20px;
                    border-radius:4px; text-align:center;">
          <div style="font-size:0.75rem; letter-spacing:2px;">DECISION</div>
          <div style="font-size:1.4rem; font-weight:600; margin-top:6px;">{decision}</div>
          <div style="font-size:0.8rem; margin-top:8px; opacity:0.9;">{rationale}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Fraud Probability", f"{prob*100:.2f}%")
        st.caption(f"Decision boundary: {THRESHOLD:.2f} "
                   f"(grey zone from {THRESHOLD*0.5:.2f})")

    with col2:
        st.markdown("**Primary Risk Drivers** (SHAP contribution to log-odds)")
        if hasattr(MODEL, "get_booster"):
            import xgboost as xgb
            contribs = MODEL.get_booster().predict(
                xgb.DMatrix(X), pred_contribs=True)[0][:-1]
            idx = np.argsort(-np.abs(contribs))[:3]
            rows = []
            for i in idx:
                direction = ("Increases risk" if contribs[i] > 0
                             else "Decreases risk")
                rows.append({"Feature": COLUMNS[i],
                             "Contribution": f"{contribs[i]:+.2f}",
                             "Effect": direction})
            st.table(pd.DataFrame(rows))

    st.markdown("---")
    st.caption("Note: model trained on synthetic transaction data. "
               "Production performance will differ; scores should inform, "
               "not replace, analyst review.")
else:
    st.info("Enter transaction parameters in the left panel and submit "
            "to generate a risk assessment.")

st.sidebar.markdown("---")
st.sidebar.caption("Fraud Scoring Console v1.0")
