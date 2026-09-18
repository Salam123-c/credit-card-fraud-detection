"""
=====================================================================
 CREDIT CARD FRAUD DETECTION - END-TO-END BFSI PIPELINE
=====================================================================
 Data   : alenc123/credit-card-fraud (Sparkov synthetic, ~1.3M txns,
          Jan 2019 - Dec 2020)  -> local file: credit_card_transactions.parquet
 Run    : py CREDIT_PROJECT.py
 Deps   : pandas, numpy, scikit-learn, pyarrow, xgboost (all installed)

 Steps:
   1. Load + reality check          7. Metrics (PR-AUC, Gini, KS,
   2. EDA (fraud fingerprint)          precision@k, $-weighted recall,
   3. Cleaning + PII removal           bootstrap CI)
   4. Feature engineering           8. Threshold + business view
      (Haversine, velocity)         9. Bias audit (gender/age)
   5. Screening: IV + WOE + VIF    10. PSI (population stability)
   6. Temporal split + 2 models    11. Final summary -> model_summary.txt
=====================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
DATA_PATH = r"C:/Users/hp/Desktop/PRO_1_10/CREDIT/credit_card_transactions.parquet"
OUT_PATH = r"C:/Users/hp/Desktop/PRO_1_10/CREDIT/model_summary.txt"

REPORT = []  # every line also goes to the summary file (transparency)


def rep(text=""):
    print(text)
    REPORT.append(str(text))


def banner(no, title):
    rep("")
    rep("=" * 68)
    rep(f" STEP {no}: {title}")
    rep("=" * 68)


# ------------------------------------------------------------------
# STEP 1: LOAD + REALITY CHECK
# ------------------------------------------------------------------
banner(1, "LOAD + REALITY CHECK")

# Memory-conscious load: only the columns the pipeline actually uses
# (skips PII/ID strings upfront - saves ~400 MB on low-RAM machines)
USE_COLS = ["trans_date_trans_time", "cc_num", "category", "amt", "gender",
            "state", "city_pop", "lat", "long", "merch_lat", "merch_long",
            "dob", "unix_time", "job", "is_fraud"]
df = pd.read_parquet(DATA_PATH, columns=USE_COLS)
for c in ["amt", "lat", "long", "merch_lat", "merch_long"]:
    df[c] = df[c].astype("float32")
rep(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
rep(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")

df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
rep(f"Time range: {df['trans_dt'].min()}  ->  {df['trans_dt'].max()}")
rep(f"Fraud rate: {df['is_fraud'].mean()*100:.3f}%  "
    f"({df['is_fraud'].sum():,} fraud / {len(df):,} total)")
rep(f"Missing values (top 5): "
    f"{df.isna().sum().sort_values(ascending=False).head(5).to_dict()}")

# ------------------------------------------------------------------
# STEP 2: EDA - THE FRAUD FINGERPRINT
# ------------------------------------------------------------------
banner(2, "EDA - FRAUD FINGERPRINT")

df["hour"] = df["trans_dt"].dt.hour

hr = df.groupby("hour")["is_fraud"].agg(["mean", "sum"])
night = hr.loc[list(range(0, 7)) + [22, 23], "mean"].mean()
day = hr.loc[list(range(7, 22)), "mean"].mean()
rep(f"Fraud rate night (22-06h): {night*100:.2f}%  vs  day (07-21h): {day*100:.2f}%")
rep(f"  -> night is {night/day:.1f}x riskier (classic card-theft pattern)")

cat = df.groupby("category")["is_fraud"].agg(["mean", "sum"]).sort_values("mean", ascending=False)
rep("Top fraud categories:")
rep(cat.head(5).to_string())

rep(f"Amount median - fraud: ${df.loc[df.is_fraud==1,'amt'].median():.0f}"
    f"  |  legit: ${df.loc[df.is_fraud==0,'amt'].median():.0f}")

# ------------------------------------------------------------------
# STEP 3: CLEANING + PII REMOVAL (leakage & compliance control)
# ------------------------------------------------------------------
banner(3, "CLEANING + PII REMOVAL")

# PII / IDs / leakage: never go into the model (BFSI compliance)
drop_cols = ["Unnamed: 0", "trans_num", "first", "last", "street",
             "zip", "merch_zipcode", "city", "merchant"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# engineered time features
df["dayofweek"] = df["trans_dt"].dt.dayofweek
df["month"] = df["trans_dt"].dt.month
df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

# age at transaction time
dob = pd.to_datetime(df["dob"], errors="coerce")
df["age"] = ((df["trans_dt"] - dob).dt.days / 365.25).clip(14, 100)

# frequency encodings (high-cardinality -> counts), then drop raw strings
# (object columns are the heaviest in RAM - drop as soon as encoded)
for c in ["job", "state"]:
    df[c + "_freq"] = df.groupby(c)[c].transform("count")
df = df.drop(columns=["job", "state"])

rep(f"Dropped (PII/ID/leakage): {drop_cols}")
rep(f"Added: hour, dayofweek, month, is_night, age, job_freq, state_freq")

# ------------------------------------------------------------------
# STEP 4: FEATURE ENGINEERING (Haversine + card-velocity)
# ------------------------------------------------------------------
banner(4, "FEATURE ENGINEERING")

# 4a. Haversine distance cardholder <-> merchant (km)
lat1, lon1 = np.radians(df["lat"]), np.radians(df["long"])
lat2, lon2 = np.radians(df["merch_lat"]), np.radians(df["merch_long"])
a = (np.sin((lat2 - lat1) / 2) ** 2
     + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# 4b. Amount transforms
df["amt_log1p"] = np.log1p(df["amt"])

# 4c. Card-velocity (vectorised, no slow rolling windows):
#     card ki ab tak ki history -> next transaction kitni 'unusual' hai
# (in-place sort/reset - a deep copy here can OOM on low-RAM machines)
df.sort_values(["cc_num", "unix_time"], inplace=True)
df.reset_index(drop=True, inplace=True)
g = df.groupby("cc_num")
df["time_since_prev_h"] = g["unix_time"].diff() / 3600.0
df["tx_seq"] = g.cumcount() + 1
cum_amt = g["amt"].cumsum()
df["card_prev_avg_amt"] = (cum_amt - df["amt"]) / df["tx_seq"].replace(1, np.nan)
df["amt_vs_card_avg"] = df["amt"] / (df["card_prev_avg_amt"] + 1e-9)
df["time_since_prev_h"] = df["time_since_prev_h"].fillna(999)  # cold-start
df["card_prev_avg_amt"] = df["card_prev_avg_amt"].fillna(df["amt"])
df["amt_vs_card_avg"] = df["amt_vs_card_avg"].clip(0, 100)

rep("Added: distance_km, amt_log1p, time_since_prev_h, tx_seq,")
rep("       card_prev_avg_amt, amt_vs_card_avg (card-behavior velocity)")
rep(f"distance_km  fraud median: {df.loc[df.is_fraud==1,'distance_km'].median():.1f} km"
    f" | legit median: {df.loc[df.is_fraud==0,'distance_km'].median():.1f} km")

# ------------------------------------------------------------------
# STEP 5: SCREENING - IV / WOE + VIF
# ------------------------------------------------------------------
banner(5, "SCREENING: IV / WOE + VIF")

NUM_FEATURES = ["amt", "amt_log1p", "hour", "dayofweek", "month", "age",
                "distance_km", "city_pop", "time_since_prev_h", "tx_seq",
                "card_prev_avg_amt", "amt_vs_card_avg",
                "job_freq", "state_freq", "is_night"]

def iv_woe(data, feature, target, bins=10):
    """IV + WOE via quantile binning (numeric). IV bands:
       <0.02 useless | 0.02-0.1 weak | 0.1-0.3 medium | 0.3-0.5 strong | >0.5 SUSPICIOUS"""
    x = data[feature]
    if x.nunique() > bins:
        b = pd.qcut(x.rank(method="first"), bins, duplicates="drop")
    else:
        b = x
    t = pd.DataFrame({"bin": b, "y": data[target]})
    tab = t.groupby("bin")["y"].agg(["sum", "count"])
    tab["good"] = tab["count"] - tab["sum"]
    n_ev, n_gd = tab["sum"].sum(), tab["good"].sum()
    eps = 0.5  # Laplace smoothing
    tab["woe"] = np.log(((tab["good"] + eps) / n_gd) / ((tab["sum"] + eps) / n_ev))
    tab["iv"] = ((tab["good"] + eps) / n_gd - (tab["sum"] + eps) / n_ev) * tab["woe"]
    return tab["iv"].sum()

def label_iv(v):
    if v < 0.02: return "useless"
    if v < 0.1:  return "weak"
    if v < 0.3:  return "medium"
    if v < 0.5:  return "STRONG"
    return "SUSPICIOUS(leakage?)"

rep("Information Value per feature (predictive power):")
for f in NUM_FEATURES:
    v = iv_woe(df, f, "is_fraud")
    rep(f"  IV  {f:<22} {v:.3f}   [{label_iv(v)}]")

def vif_manual(X):
    """VIF = 1/(1-R^2), OLS via numpy (no statsmodels needed).
    Columns are z-standardised first - raw scale disparity (0-999 range
    mixes) makes SVD numerically unstable."""
    Xc = X.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    Xc = (Xc - Xc.mean()) / Xc.std().replace(0, 1)
    cols = list(Xc.columns)
    out = {}
    for c in cols:
        others = [x for x in cols if x != c]
        A = Xc[others].values
        yv = Xc[c].values
        try:
            coef, *_ = np.linalg.lstsq(A, yv, rcond=1e-6)
            r2 = 1 - np.sum((yv - A @ coef) ** 2) / np.sum((yv - yv.mean()) ** 2)
            out[c] = 1 / (1 - min(r2, 0.9999))
        except np.linalg.LinAlgError:
            out[c] = np.inf
    return pd.Series(out).sort_values(ascending=False)

# VIF on a representative numeric subset (dummies excluded - drop-first rule)
vif_set = df[["amt_log1p", "hour", "age", "distance_km",
              "time_since_prev_h", "amt_vs_card_avg"]].sample(100_000, random_state=SEED)
rep("\nVIF (redundancy; >10 = remove one of the pair) on numeric core:")
rep(vif_manual(vif_set).round(1).to_string())
rep("Note: amt vs amt_log1p are kept apart - scorecard uses log version only.")

# ------------------------------------------------------------------
# STEP 6: TEMPORAL SPLIT + 2 MODELS
# ------------------------------------------------------------------
banner(6, "TEMPORAL SPLIT + MODELS")

# Final model matrix
X_num = df[NUM_FEATURES].drop(columns=["amt"])          # log version retained
X_cat = pd.get_dummies(df["category"], prefix="cat", drop_first=True)
gender = df["gender"].map({"M": 1, "F": 0}).rename("gender_M")
X = pd.concat([X_num, gender, X_cat], axis=1)
y = df["is_fraud"]

# safety net: any residual NaN/inf (e.g. unparseable dob -> age) -> median
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0)
X = X.astype("float32")  # halves model-matrix memory, no accuracy cost here
rep(f"NaN/inf remaining in X: {int((X.isna().sum().sum()))}")
del df["dob"], df["merch_lat"], df["merch_long"], df["lat"], df["long"]
import gc; gc.collect()

train_mask = df["trans_dt"].dt.year.eq(2019) & df["month"].le(10)
valid_mask = df["trans_dt"].dt.year.eq(2019) & df["month"].ge(11)  # Nov-Dec 2019
test_mask = df["trans_dt"].dt.year.eq(2020)                        # OOT test

X_tr, y_tr = X[train_mask], y[train_mask]
X_va, y_va = X[valid_mask], y[valid_mask]
X_te, y_te = X[test_mask], y[test_mask]
rep(f"Train (Jan-Oct 2019):   {len(X_tr):,} rows, fraud {y_tr.mean()*100:.3f}%")
rep(f"Valid (Nov-Dec 2019):   {len(X_va):,} rows, fraud {y_va.mean()*100:.3f}%")
rep(f"TEST  (Jan-Dec 2020):   {len(X_te):,} rows, fraud {y_te.mean()*100:.3f}%")

# 6a. Logistic Regression baseline (scorecard-style, class-weighted)
scaler = StandardScaler().fit(X_tr)
X_tr_s, X_va_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te)
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
lr.fit(X_tr_s, y_tr)
prob_lr = lr.predict_proba(X_te_s)[:, 1]

# 6b. Gradient boosting (XGBoost, fallback HistGB)
try:
    from xgboost import XGBClassifier
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    gb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                       scale_pos_weight=spw, tree_method="hist",
                       eval_metric="aucpr", n_jobs=-1, random_state=SEED)
    gb_name = "XGBoost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    gb = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                        random_state=SEED)
    gb_name = "HistGradientBoosting"
gb.fit(X_tr, y_tr)
prob_gb = gb.predict_proba(X_te)[:, 1]

# validation probabilities (threshold tuning is done on VALID, never test)
prob_gb_va = gb.predict_proba(X_va)[:, 1]

# ------------------------------------------------------------------
# STEP 7: METRICS (with BFSI honesty: ROC flatters imbalance)
# ------------------------------------------------------------------
banner(7, "METRICS - PR-AUC / GINI / KS / PRECISION@K / $-RECALL")

def ks_stat(y_true, prob):
    order = np.argsort(-prob)
    yy = np.asarray(y_true)[order]
    n1, n0 = yy.sum(), (1 - yy).sum()
    cum_fraud = np.cumsum(yy) / max(n1, 1)
    cum_good = np.cumsum(1 - yy) / max(n0, 1)
    return np.max(np.abs(cum_fraud - cum_good))

def report_model(name, y_true, prob, k=500):
    pr = average_precision_score(y_true, prob)
    roc = roc_auc_score(y_true, prob)
    ks = ks_stat(y_true, prob)
    order = np.argsort(-prob)
    top_k = np.asarray(y_true)[order[:k]].mean()
    dollars = np.asarray(y_true, dtype=float) * np.asarray(
        df.loc[test_mask, "amt"].values, dtype=float)
    val_recall = dollars[np.argsort(-prob)][: max(k, int(y_true.sum()))].sum() / dollars.sum()
    rep(f"{name:<12} PR-AUC {pr:.3f} | ROC {roc:.3f} (Gini {2*roc-1:.3f}) "
        f"| KS {ks:.3f} | precision@{k} {top_k:.3f} | $-recall@top {val_recall*100:.1f}%")
    return pr

rep("(PR-AUC = primary metric; ROC/Gini/KS = reporting metrics)")
pr_lr = report_model("LogReg", y_te.values, prob_lr)
pr_gb = report_model(gb_name, y_te.values, prob_gb)

# Bootstrap CI on the PR-AUC gap (is the improvement real or noise?)
y_arr = y_te.values
rng = np.random.default_rng(SEED)
diffs = []
idx = np.arange(len(y_arr))
for _ in range(200):
    b = rng.choice(idx, len(idx), replace=True)
    diffs.append(average_precision_score(y_arr[b], prob_gb[b])
                 - average_precision_score(y_arr[b], prob_lr[b]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
rep(f"PR-AUC uplift ({gb_name} vs LogReg): {pr_gb-pr_lr:+.3f} "
    f"95% CI [{lo:+.3f}, {hi:+.3f}] -> "
    f"{'REAL' if lo > 0 else 'NOT significant'}")

# ------------------------------------------------------------------
# STEP 8: THRESHOLD (tuned on VALIDATION) + BUSINESS VIEW
# ------------------------------------------------------------------
banner(8, "THRESHOLD TUNING ON VALID + BUSINESS VIEW")

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.05, 0.96, 0.025):
    f1 = f1_score(y_va, prob_gb_va > t)
    if f1 > best_f1:
        best_f1, best_t = f1, t
rep(f"Best threshold on VALIDATION: {best_t:.3f} (F1 {best_f1:.3f})")

pred = prob_gb > best_t
tp = int(((pred == 1) & (y_arr == 1)).sum())
fp = int(((pred == 1) & (y_arr == 0)).sum())
fn = int(((pred == 0) & (y_arr == 1)).sum())
amt_te = df.loc[test_mask, "amt"].values
rep(f"Confusion @ threshold: TP {tp:,} | FP {fp:,} (false alarms) | FN {fn:,} (missed fraud)")
rep(f"Precision {precision_score(y_arr, pred):.3f} | Recall {recall_score(y_arr, pred):.3f}")
rep(f"Missed fraud value: ${amt_te[(pred==0) & (y_arr==1)].sum():,.0f}"
    f" | Blocked legit customers: {fp:,}")

# ------------------------------------------------------------------
# STEP 7.5: VISUALIZATIONS (saved as PNG in the project folder)
# ------------------------------------------------------------------
banner("7.5", "VISUALIZATIONS (PNG)")

import matplotlib
matplotlib.use("Agg")  # headless save (no display window)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

IMG_DIR = r"C:/Users/hp/Desktop/PRO_1_10/CREDIT/"

# 7.5a - PR curve + ROC curve (both models)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for y_true, prob, name in [(y_arr, prob_gb, gb_name), (y_arr, prob_lr, "LogReg")]:
    p, r, _ = precision_recall_curve(y_true, prob)
    axes[0].plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y_true, prob):.3f})")
    fpr, tpr, _ = roc_curve(y_true, prob)
    axes[1].plot(fpr, tpr, label=f"{name} (ROC {roc_auc_score(y_true, prob):.3f})")
axes[0].axhline(y_arr.mean(), ls="--", c="gray", label=f"baseline fraud rate {y_arr.mean():.3f}")
axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
axes[1].plot([0, 1], [0, 1], ls="--", c="gray")
axes[1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
for ax in axes: ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(IMG_DIR + "model_curves.png", dpi=130); plt.close(fig)

# 7.5b - EDA patterns: fraud rate by hour + amount distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
hr_all = df.groupby("hour")["is_fraud"].mean()
axes[0].bar(hr_all.index, hr_all * 100, color=np.where(hr_all.index.isin([22,23,0,1,2,3,4,5]), "crimson", "steelblue"))
axes[0].set(xlabel="Hour of day", ylabel="Fraud rate (%)", title="Fraud Rate by Hour (night = red)")
axes[1].hist(np.log1p(df.loc[df.is_fraud == 0, "amt"]), bins=60, alpha=0.6, label="Legit", density=True)
axes[1].hist(np.log1p(df.loc[df.is_fraud == 1, "amt"]), bins=60, alpha=0.6, label="Fraud", density=True)
axes[1].set(xlabel="log1p(amount)", ylabel="Density", title="Amount Distribution by Class")
axes[1].legend()
fig.tight_layout(); fig.savefig(IMG_DIR + "eda_fraud_patterns.png", dpi=130); plt.close(fig)

# 7.5c - Feature importance (top 15)
imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(imp.index[::-1], imp.values[::-1], color="teal")
ax.set(title=f"Top 15 Feature Importance ({gb_name})")
fig.tight_layout(); fig.savefig(IMG_DIR + "feature_importance.png", dpi=130); plt.close(fig)
rep("Top 5 features: " + ", ".join(f"{k} ({v:.3f})" for k, v in imp.head(5).items()))

# 7.5d - Confusion matrix @ tuned threshold
cm = confusion_matrix(y_arr, pred)
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center",
            color="white" if v > cm.max() / 2 else "black", fontsize=12)
ax.set(xticks=[0, 1], yticks=[0, 1],
       xticklabels=["Pred Legit", "Pred Fraud"], yticklabels=["Legit", "Fraud"],
       title=f"Confusion Matrix @ threshold {best_t:.3f}")
fig.tight_layout(); fig.savefig(IMG_DIR + "confusion_matrix.png", dpi=130); plt.close(fig)

rep("Saved: model_curves.png, eda_fraud_patterns.png, feature_importance.png, confusion_matrix.png")

# ------------------------------------------------------------------
# STEP 9: BIAS AUDIT (fairness across protected groups)
# ------------------------------------------------------------------
banner(9, "BIAS AUDIT")

aud = pd.DataFrame({"pred": pred, "y": y_arr,
                    "gender": df.loc[test_mask, "gender"].values,
                    "age": df.loc[test_mask, "age"].values})
aud["agegrp"] = pd.cut(aud["age"], [17, 30, 50, 101], labels=["18-30", "31-50", "51+"])
for col in ["gender", "agegrp"]:
    t = aud.groupby(col, observed=True).agg(flag_rate=("pred", "mean"),
                                            real_fraud=("y", "mean"))
    rep(t.round(4).to_string())
    rep("")

# ------------------------------------------------------------------
# STEP 10: PSI - POPULATION STABILITY (train vs OOT test)
# ------------------------------------------------------------------
banner(10, "PSI - TRAIN vs 2020 TEST")

def psi(train_s, test_s, bins=10):
    qs = np.quantile(train_s, np.linspace(0, 1, bins + 1)[1:-1])
    a = np.histogram(train_s, qs)[0] / len(train_s)
    b = np.histogram(test_s, qs)[0] / len(test_s)
    eps = 1e-4
    return np.sum((b - a) * np.log((b + eps) / (a + eps)))

for f in ["amt_log1p", "hour", "distance_km", "age"]:
    v = psi(X_tr[f].values, X_te[f].values)
    rep(f"  PSI {f:<14} {v:.4f}  [{'stable' if v < 0.1 else 'MODERATE' if v < 0.25 else 'DRIFT -> retrain'}]")

# ------------------------------------------------------------------
# STEP 11: FINAL SUMMARY
# ------------------------------------------------------------------
banner(11, "SUMMARY + LIMITATIONS (model documentation)")
rep(f"Champion model : {gb_name}")
rep(f"Primary metric : PR-AUC (test 2020, out-of-time)")
rep("Limitations to disclose:")
rep("  1. Synthetic Sparkov data - real-world metrics will be lower.")
rep("  2. Labels are instant here; production fraud labels arrive after")
rep("     chargebacks (weeks) - label latency is a known constraint.")
rep("  3. Velocity features are cumulative-history proxies; a live system")
rep("     needs true rolling windows computed identically at serving time")
rep("     (training-serving skew risk).")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(REPORT))
rep("")
rep(f"Full report saved -> {OUT_PATH}")

# ------------------------------------------------------------------
# STEP 12: MODEL PERSISTENCE (deployment artifacts)
# ------------------------------------------------------------------
banner(12, "MODEL PERSISTENCE (deployment artifacts)")

import joblib
ART_PATH = r"C:/Users/hp/Desktop/PRO_1_10/CREDIT/fraud_model.joblib"
joblib.dump({
    "model": gb,                      # champion model
    "feature_columns": list(X.columns),  # exact feature order at training time
    "categories": sorted(df["category"].unique()),
    "threshold": float(best_t),       # tuned on VALID, applied at serving
    "model_name": gb_name,
}, ART_PATH)
rep(f"Saved -> {ART_PATH}")
rep("Bundle contains: model + feature column order + threshold + categories")
rep("Deploy with: uvicorn serve:app --host 0.0.0.0 --port 8000")
rep("DONE.")
