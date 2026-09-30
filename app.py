import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ChurnGuard - Telecom Churn Prediction",
    page_icon="CG",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FILE = "logisticregression_model.pkl"
METRICS_FILE = "logisticregression_metrics.json"
COMPARISON_FILE = "model_comparison_metrics.csv"
FEEDBACK_FILE = "feedback_log.json"

FEATURE_COLUMNS = [
    "Gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Tenure",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]
NUMERIC_COLUMNS = ["SeniorCitizen", "Tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_VALUES = {
    "Gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "InternetService": ["No", "DSL", "Fiber optic"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;700&display=swap');
  :root {
    --bg: #07111f;
    --panel: #0d1c2f;
    --panel-2: #10243a;
    --line: rgba(125, 211, 252, .22);
    --text: #eaf6ff;
    --muted: #94a9bd;
    --cyan: #22d3ee;
    --teal: #2dd4bf;
    --indigo: #818cf8;
    --coral: #fb7185;
    --amber: #fbbf24;
    --green: #34d399;
  }
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text); }
  p, span, div, label { color: #dbeafe; }
  .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span { color: #cfe8ff !important; }
  div[data-testid="stMarkdownContainer"] p { color: #cfe8ff !important; }
  .stApp {
    background:
      linear-gradient(120deg, rgba(34,211,238,.10), transparent 28%),
      linear-gradient(210deg, rgba(129,140,248,.12), transparent 32%),
      var(--bg);
  }
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
  header[data-testid="stHeader"] { background: rgba(7,17,31,.92) !important; border-bottom: 1px solid rgba(125,211,252,.12); }
  div[data-testid="stToolbar"] { color: #d8ecff !important; }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081523 0%, #0d1f33 58%, #07111f 100%) !important;
    border-right: 1px solid rgba(125, 211, 252, .18);
    box-shadow: 12px 0 34px rgba(0,0,0,.20);
  }
  section[data-testid="stSidebar"] * { color: #d8ecff !important; }
  h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; letter-spacing: 0; }
  .brand-lockup {
    padding: 18px 4px 24px;
    border-bottom: 1px solid rgba(125, 211, 252, .16);
    margin-bottom: 18px;
  }
  .brand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #f8fdff;
    line-height: 1.05;
  }
  .brand-tag { color: #7dd3fc; font-size: .82rem; margin-top: 7px; }
  .hero-banner {
    position: relative;
    overflow: hidden;
    background:
      linear-gradient(135deg, rgba(13,28,47,.98) 0%, rgba(17,42,65,.96) 54%, rgba(19,32,54,.98) 100%);
    border: 1px solid rgba(125,211,252,.24);
    border-radius: 18px;
    padding: 34px 38px;
    margin-bottom: 24px;
    box-shadow: 0 22px 70px rgba(0,0,0,.26), inset 0 1px 0 rgba(255,255,255,.05);
  }
  .hero-kicker { color: var(--teal); font-size: .78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.8px; margin: 0 0 10px; }
  .hero-title { font-family:'Syne',sans-serif; font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.02; font-weight:800; color:#f8fdff; margin:0 0 12px; }
  .hero-sub { color:#a8c4d8; font-size:1.02rem; max-width: 760px; margin:0; line-height: 1.65; }
  .hero-pills { display:flex; flex-wrap:wrap; gap:10px; margin-top: 24px; }
  .hero-pill { border:1px solid rgba(125,211,252,.24); color:#dff7ff; background:rgba(8,21,35,.72); border-radius:999px; padding:8px 13px; font-size:.82rem; font-weight:700; }
  .section-card {
    background: linear-gradient(145deg, rgba(13,28,47,.96), rgba(16,36,58,.92));
    border: 1px solid rgba(125,211,252,.20);
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 22px;
    box-shadow: 0 18px 46px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.04);
  }
  .section-title {
    font-family:'Syne',sans-serif;
    font-size:.92rem;
    font-weight:800;
    color:#7dd3fc;
    text-transform:uppercase;
    letter-spacing:1.2px;
    margin-bottom:18px;
    padding-bottom:12px;
    border-bottom:1px solid rgba(125,211,252,.18);
  }
  .result-churn, .result-retain {
    border-radius:18px;
    padding:34px;
    text-align:center;
    box-shadow: 0 24px 70px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.05);
  }
  .result-churn { background:linear-gradient(135deg,rgba(62,18,32,.98),rgba(31,17,36,.96)); border:1px solid rgba(251,113,133,.42); }
  .result-retain { background:linear-gradient(135deg,rgba(9,50,47,.98),rgba(11,36,45,.96)); border:1px solid rgba(45,212,191,.42); }
  .result-label { font-family:'Syne',sans-serif; font-size:1.55rem; font-weight:800; margin:0 0 8px; }
  .result-prob { font-family:'Syne',sans-serif; font-size:3.35rem; font-weight:800; }
  .result-desc { font-size:.95rem; color:#b7c8d8; margin-top:8px; }
  .metric-chip { display:inline-block; background:rgba(34,211,238,.10); border:1px solid rgba(34,211,238,.24); border-radius:999px; padding:6px 14px; font-size:.78rem; color:#cffafe; margin:4px; }
  .rec-card { background:rgba(7,17,31,.58); border:1px solid rgba(125,211,252,.18); border-left:4px solid var(--teal); border-radius:10px; padding:17px 20px; margin-bottom:12px; }
  .rec-title { font-family:'Syne',sans-serif; font-size:.95rem; font-weight:800; color:#99f6e4; margin-bottom:5px; }
  .rec-body { font-size:.88rem; color:#b2c4d4; line-height:1.55; }
  .risk-high { color:var(--coral); font-weight:800; }
  .risk-medium { color:var(--amber); font-weight:800; }
  .risk-low { color:var(--green); font-weight:800; }
  label { color:#9bd7f2 !important; font-size:.84rem !important; font-weight: 700 !important; }
  div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(13,28,47,.98), rgba(13,43,57,.92));
    border: 1px solid rgba(125,211,252,.20);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,.18);
  }
  div[data-testid="stMetricLabel"] { color: #9bd7f2 !important; }
  div[data-testid="stMetricValue"] { color: #f8fdff !important; font-family:'Syne',sans-serif; }
  .stButton>button, .stDownloadButton>button, div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #2dd4bf 0%, #22d3ee 48%, #818cf8 100%) !important;
    color:#06111f !important;
    font-weight:800 !important;
    border:none !important;
    border-radius:10px !important;
    padding:.72rem 1.6rem !important;
    box-shadow: 0 14px 28px rgba(34,211,238,.22);
    transition: transform .18s ease, box-shadow .18s ease !important;
  }
  .stButton>button:hover, .stDownloadButton>button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 38px rgba(45,212,191,.28);
  }
  input[type="number"], textarea, div[data-baseweb="select"]>div {
    background: rgba(7,17,31,.88) !important;
    border-color: rgba(125,211,252,.24) !important;
    color:#eaf6ff !important;
    border-radius: 10px !important;
  }
  div[data-baseweb="radio"] { background: rgba(13,28,47,.54); border: 1px solid rgba(125,211,252,.16); border-radius: 12px; padding: 8px 12px; }
  div[data-testid="stRadio"] label p { color: #eaf6ff !important; font-weight: 700 !important; opacity: 1 !important; }
  div[data-testid="stRadio"] label:hover p { color: #99f6e4 !important; opacity: 1 !important; }
  div[data-testid="stRadio"] label[data-baseweb="radio"] p,
  div[role="radiogroup"] label p,
  div[role="radiogroup"] p { color: #eaf6ff !important; opacity: 1 !important; }
  div[role="radiogroup"] label:has(input:checked) p { color: #ffffff !important; }
  div[data-testid="stFileUploader"] section {
    background: rgba(7,17,31,.56) !important;
    border: 1px dashed rgba(45,212,191,.55) !important;
    border-radius: 14px !important;
  }
  div[data-testid="stDataFrame"] { border: 1px solid rgba(125,211,252,.18); border-radius: 12px; overflow: hidden; }
  code { color: #99f6e4 !important; background: rgba(7,17,31,.7) !important; border: 1px solid rgba(125,211,252,.14); border-radius: 8px; }
  hr { border-color: rgba(125,211,252,.16) !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_metrics():
    fallback = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0}
    if not os.path.exists(METRICS_FILE):
        return fallback
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {key: float(data.get(key, fallback[key])) for key in fallback}


def save_feedback(entry):
    data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append(entry)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def risk_label(probability):
    if probability >= 0.65:
        return "High", "#f87171", "Immediate retention action recommended"
    if probability >= 0.35:
        return "Medium", "#fbbf24", "Proactive outreach recommended"
    return "Low", "#34d399", "Customer looks stable"


def clean_prediction_frame(df):
    working = df.copy()
    if "CustomerID" not in working.columns:
        working.insert(0, "CustomerID", [f"ROW-{i + 1}" for i in range(len(working))])

    missing = [col for col in FEATURE_COLUMNS if col not in working.columns]
    if missing:
        return None, missing

    working = working[["CustomerID"] + FEATURE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    working["SeniorCitizen"] = working["SeniorCitizen"].fillna(0).astype(int).clip(0, 1)
    working["Tenure"] = working["Tenure"].fillna(0).clip(lower=0)
    working["MonthlyCharges"] = working["MonthlyCharges"].fillna(0).clip(lower=0)
    working["TotalCharges"] = working["TotalCharges"].fillna(working["Tenure"] * working["MonthlyCharges"]).clip(lower=0)

    for col, allowed in CATEGORICAL_VALUES.items():
        working[col] = working[col].astype(str).str.strip()
        working.loc[~working[col].isin(allowed), col] = allowed[0]

    return working, []


def predict_dataframe(model, df):
    clean_df, missing = clean_prediction_frame(df)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    features = clean_df[FEATURE_COLUMNS]
    churn_prob = model.predict_proba(features)[:, 1]
    prediction = (churn_prob >= 0.5).astype(int)

    result = clean_df.copy()
    result["ChurnProbability"] = np.round(churn_prob * 100, 2)
    result["Prediction"] = np.where(prediction == 1, "Churn", "Retain")
    result["RiskLevel"] = [risk_label(float(p))[0] for p in churn_prob]
    return result


def make_gauge(probability):
    level, color, _ = risk_label(probability)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={"suffix": "%", "font": {"size": 44, "color": color, "family": "Syne"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#1e3a5f", "tickfont": {"color": "#64b5d9", "size": 10}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#0d1a2e",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(52,211,153,.12)"},
                    {"range": [35, 65], "color": "rgba(251,191,36,.12)"},
                    {"range": [65, 100], "color": "rgba(248,113,113,.12)"},
                ],
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=20, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def make_donut(probability):
    fig = go.Figure(
        go.Pie(
            values=[probability * 100, (1 - probability) * 100],
            labels=["Churn Risk", "Retention"],
            hole=0.62,
            marker_colors=["#f87171", "#34d399"],
            textinfo="none",
        )
    )
    fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", legend=dict(font={"color": "#64b5d9"}))
    return fig


def make_risk_bar(row):
    factors = [
        f"Contract: {row['Contract']}",
        f"Internet: {row['InternetService']}",
        f"Tenure: {int(row['Tenure'])} months",
        f"Monthly charges: ${float(row['MonthlyCharges']):.0f}",
        f"Payment: {row['PaymentMethod']}",
    ]
    scores = [
        {"Month-to-month": 28, "One year": 14, "Two year": 4}.get(row["Contract"], 10),
        {"Fiber optic": 20, "DSL": 10, "No": 4}.get(row["InternetService"], 8),
        max(0, 30 - int(row["Tenure"])),
        24 if float(row["MonthlyCharges"]) > 80 else 12 if float(row["MonthlyCharges"]) > 50 else 4,
        14 if row["PaymentMethod"] == "Electronic check" else 3,
    ]
    colors = ["#f87171" if s >= 20 else "#fbbf24" if s >= 10 else "#34d399" for s in scores]
    fig = go.Figure(go.Bar(x=scores, y=factors, orientation="h", marker_color=colors, text=scores, textposition="outside"))
    fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=60), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(color="#c8d8f0"))
    return fig


def get_recommendations(row, probability):
    recs = []
    if row["Contract"] == "Month-to-month":
        recs.append(("Move to a yearly plan", "Offer a one-year or two-year contract with a price-lock benefit."))
    if row["PaymentMethod"] == "Electronic check":
        recs.append(("Encourage auto-pay", "Auto-pay or card payment can reduce payment friction and missed billing events."))
    if float(row["MonthlyCharges"]) > 70:
        recs.append(("Review monthly charges", "Offer a bundle, loyalty discount, or plan adjustment before the bill becomes a churn trigger."))
    if row["InternetService"] == "DSL":
        recs.append(("Improve internet value", "Suggest a faster plan or service-quality check for DSL customers."))
    if probability >= 0.65:
        recs.append(("Retention priority", "Route this customer to a retention agent with a personalized offer."))
    if not recs:
        recs.append(("Maintain relationship", "Customer is low risk. Continue regular engagement and service-quality monitoring."))
    return recs


def render_single_result(result_df):
    row = result_df.iloc[0]
    probability = float(row["ChurnProbability"]) / 100
    level, color, description = risk_label(probability)
    card_class = "result-churn" if row["Prediction"] == "Churn" else "result-retain"
    title = "CHURN RISK" if row["Prediction"] == "Churn" else "LIKELY TO STAY"

    st.markdown("---")
    st.markdown(
        f"""
        <div class="{card_class}">
          <div class="result-label" style="color:{color};">{title}</div>
          <div class="result-prob" style="color:{color};">{row['ChurnProbability']:.1f}%</div>
          <div class="result-desc">{level} risk - {description}. Powered by Logistic Regression.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-card"><div class="section-title">Risk Gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(probability), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card"><div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(make_donut(probability), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-card"><div class="section-title">Risk Factor Scores</div>', unsafe_allow_html=True)
        st.plotly_chart(make_risk_bar(row), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card"><div class="section-title">Retention Recommendations</div>', unsafe_allow_html=True)
        for title, body in get_recommendations(row, probability):
            st.markdown(f'<div class="rec-card"><div class="rec-title">{title}</div><div class="rec-body">{body}</div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_batch_results(result_df):
    total = len(result_df)
    churn_count = int((result_df["Prediction"] == "Churn").sum())
    high_count = int((result_df["RiskLevel"] == "High").sum())
    avg_prob = float(result_df["ChurnProbability"].mean()) if total else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers scored", total)
    m2.metric("Predicted churn", churn_count)
    m3.metric("High risk", high_count)
    m4.metric("Average churn risk", f"{avg_prob:.1f}%")

    chart_df = result_df.copy()
    fig = px.histogram(chart_df, x="ChurnProbability", color="RiskLevel", nbins=20, color_discrete_map={"Low": "#34d399", "Medium": "#fbbf24", "High": "#f87171"})
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8d8f0"), xaxis_title="Churn probability (%)", yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    display_cols = ["CustomerID", "Prediction", "RiskLevel", "ChurnProbability"] + FEATURE_COLUMNS
    st.dataframe(result_df[display_cols].sort_values("ChurnProbability", ascending=False), use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions CSV",
        data=result_df[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


model = load_model()
metrics = load_metrics()

with st.sidebar:
    st.markdown("""<div class=""brand-lockup""><div class=""brand-name"">ChurnGuard</div><div class=""brand-tag"">Telecom retention intelligence</div></div>""", unsafe_allow_html=True)
    nav = st.radio("Navigation", ["Predict Churn", "Feedback Dashboard", "About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Production Model**")
    if model is None:
        st.error("Model missing")
    else:
        st.success("Logistic Regression loaded")
    st.caption("Run `python train_model.py` after changing training data.")

st.markdown(
    """
<div class="hero-banner">
  <p class="hero-kicker">Telecom Customer Churn System</p>
  <p class="hero-title">Predict churn before customers leave.</p>
  <p class="hero-sub">Score one subscriber manually or upload a full CSV batch. ChurnGuard uses the trained Logistic Regression model to surface risk, confidence, and retention actions in one dashboard.</p>
  <div class="hero-pills">
    <span class="hero-pill">Single customer scoring</span>
    <span class="hero-pill">Bulk CSV prediction</span>
    <span class="hero-pill">Retention recommendations</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if nav == "Predict Churn":
    if model is None:
        st.error("No Logistic Regression model found. Run `python train_model.py` in the project folder first.")
        st.stop()

    mode = st.radio("Prediction mode", ["Manual single customer", "Batch CSV upload"], horizontal=True)

    if mode == "Manual single customer":
        st.markdown('<div class="section-card"><div class="section-title">Customer Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", CATEGORICAL_VALUES["Gender"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", CATEGORICAL_VALUES["Partner"])
            dependents = st.selectbox("Dependents", CATEGORICAL_VALUES["Dependents"])
        with c2:
            tenure = st.slider("Tenure (months)", 0, 100, 12)
            phone_service = st.selectbox("Phone Service", CATEGORICAL_VALUES["PhoneService"])
            internet_service = st.selectbox("Internet Service", CATEGORICAL_VALUES["InternetService"])
        with c3:
            contract = st.selectbox("Contract Type", CATEGORICAL_VALUES["Contract"])
            payment_method = st.selectbox("Payment Method", CATEGORICAL_VALUES["PaymentMethod"])
            monthly_charges = st.number_input("Monthly Charges", 0.0, 500.0, 50.0, step=0.5)
            total_charges = st.number_input("Total Charges", 0.0, 50000.0, float(monthly_charges * tenure), step=1.0)
        st.markdown("</div>", unsafe_allow_html=True)

        input_df = pd.DataFrame([
            {
                "CustomerID": "MANUAL-1",
                "Gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure": tenure,
                "PhoneService": phone_service,
                "InternetService": internet_service,
                "Contract": contract,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ])

        _, button_col, __ = st.columns([1, 2, 1])
        with button_col:
            predict_btn = st.button("Predict single customer", use_container_width=True)
        if predict_btn:
            render_single_result(predict_dataframe(model, input_df))

    else:
        st.markdown('<div class="section-card"><div class="section-title">Upload Customer CSV</div>', unsafe_allow_html=True)
        st.write("Upload a CSV with these columns:")
        st.code(", ".join(["CustomerID (optional)"] + FEATURE_COLUMNS), language="text")
        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                result_df = predict_dataframe(model, uploaded_df)
                st.success(f"Scored {len(result_df)} customers successfully.")
                render_batch_results(result_df)
            except Exception as exc:
                st.error(f"Could not score this file: {exc}")

elif nav == "Feedback Dashboard":
    st.markdown('<div class="section-card"><div class="section-title">Share Feedback</div>', unsafe_allow_html=True)
    with st.form("feedback_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            fb_name = st.text_input("Your Name")
            fb_email = st.text_input("Email Address")
            fb_type = st.selectbox("Feedback Type", ["Feature Request", "Bug Report", "General Feedback", "Question", "Appreciation"])
        with c2:
            fb_rating = st.select_slider("Overall Rating", options=[1, 2, 3, 4, 5], value=5)
            fb_message = st.text_area("Your Message", height=120)
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            if fb_name.strip() and fb_message.strip():
                save_feedback({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "name": fb_name, "email": fb_email, "type": fb_type, "rating": fb_rating, "message": fb_message})
                st.success(f"Thank you, {fb_name}. Your feedback has been recorded.")
            else:
                st.warning("Please enter at least your name and message.")
    st.markdown("</div>", unsafe_allow_html=True)

    feedback_data = load_feedback()
    if feedback_data:
        fdf = pd.DataFrame(feedback_data)
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Rating", f"{fdf['rating'].mean():.1f} / 5")
        c2.metric("Total Responses", len(fdf))
        c3.metric("Top Feedback", fdf["type"].mode()[0])
        st.dataframe(fdf.tail(10).sort_index(ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No feedback submitted yet.")

else:
    st.markdown('<div class="section-card"><div class="section-title">About ChurnGuard</div>', unsafe_allow_html=True)
    st.write("This version uses only the Logistic Regression model for production predictions. The training script still checks Random Forest, Decision Tree, and Logistic Regression accuracy, then saves Logistic Regression as the deployed model.")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    c3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    c4.metric("F1 Score", f"{metrics['f1'] * 100:.2f}%")
    c5.metric("ROC-AUC", f"{metrics['roc_auc'] * 100:.2f}%")
    if os.path.exists(COMPARISON_FILE):
        st.write("Latest algorithm accuracy check:")
        comparison = pd.read_csv(COMPARISON_FILE)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.markdown(
        """
Risk thresholds:
<span class="risk-low">Low</span> below 35%,
<span class="risk-medium">Medium</span> from 35% to 64.99%,
<span class="risk-high">High</span> at 65% and above.

Run commands:
`pip install -r requirements.txt`
`python train_model.py`
`streamlit run app.py`
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)



