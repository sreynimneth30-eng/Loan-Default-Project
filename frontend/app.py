from __future__ import annotations

import io
import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "final_rf_pipeline_with_metadata.joblib"
METRICS_PATH = PROJECT_ROOT / "metrics.json"
DATA_PATH = PROJECT_ROOT / "data" / "Loan_default.csv"

st.set_page_config(
    page_title="LoanSense",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_pipeline():
    pipeline, metadata = joblib.load(MODEL_PATH)
    return pipeline, metadata

@st.cache_resource
def build_explainer(_pipeline):
    classifier = _pipeline.named_steps["classifier"]
    return shap.TreeExplainer(classifier)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None

def class_one_shap(shap_output) -> np.ndarray:
    if isinstance(shap_output, list):
        return np.asarray(shap_output[1])
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr

def score_locally(pipeline, explainer, payload: dict, threshold: float):
    df = pd.DataFrame([payload])
    probability = float(pipeline.predict_proba(df)[0, 1])
    prediction = int(probability >= threshold)

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    transformed = preprocessor.transform(df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    shap_row = class_one_shap(explainer.shap_values(transformed))[0]
    values_row = np.asarray(transformed)[0]

    ranked = sorted(
        zip(feature_names, values_row, shap_row),
        key=lambda item: abs(item[2]),
        reverse=True,
    )[:10]

    return {
        "prediction": prediction,
        "probability_of_default": probability,
        "threshold_used": threshold,
        "risk_label": "High Risk of Default" if prediction == 1 else "Low Risk of Default",
        "top_feature_contributions": [
            {"feature": n, "value": float(v), "shap_value": float(s)} for n, v, s in ranked
        ],
    }

def render_shap_chart(contributions: list[dict]) -> None:
    contributions = list(reversed(contributions))
    features = [c["feature"] for c in contributions]
    values = [c["shap_value"] for c in contributions]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in values]

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(features))))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on default probability")
    ax.set_title("Feature contributions for this applicant")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", transparent=False)
    plt.close(fig)
    
    st.image(buffer.getvalue(), width="stretch")

# ==========================================
# 🎨 CUSTOM SIDEBAR & UI STYLING
# ==========================================
def inject_style() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background-color: #F4F5F9; }
        
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-radius: 20px;
            margin: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            height: calc(100vh - 30px);
        }
        
        [data-testid="collapsedControl"] { display: none; }


        [data-testid="stSidebar"] .stRadio > div { gap: 10px; margin-top: 20px; }
        
        [data-testid="stSidebar"] .stRadio label {
            display: flex;
            align-items: center;
            padding: 12px 20px;
            border-radius: 10px;
            cursor: pointer;
            color: #7A869A;
            font-weight: 600;
            transition: all 0.2s ease;
            border: none;
        }

        [data-testid="stSidebar"] .stRadio label > div:first-child { display: none; }

        [data-testid="stSidebar"] .stRadio label:hover { background-color: #F4F5F9; color: #2D3436; }

        [data-testid="stSidebar"] .stRadio label[data-checked="true"] {
            background-color: #6C5CE7;
            color: white;
            box-shadow: 0 4px 15px rgba(108, 92, 231, 0.4);
        }
        
        .profile-info h4 { margin: 0; color: #2D3436; font-size: 16px; }
        .profile-info p { margin: 0; color: #A0AAB5; font-size: 12px; }

        [data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E9F2;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.03);
            text-align: center;
        }
        [data-testid="stMetricValue"] {font-size: 2.2rem; color: #6C5CE7; font-weight: bold;}
        [data-testid="stMetricLabel"] {justify-content: center; font-size: 1rem;}
        
        [data-testid="stForm"] {
            background-color: #FFFFFF;
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #E5E9F2;
            box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        }

        .hero-section {
            background: linear-gradient(135deg, #6C5CE7, #a29bfe);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 20px rgba(108, 92, 231, 0.2);
        }
        .hero-section h1 { color: white; margin-bottom: 10px; font-size: 2.5rem;}
        .hero-section p { font-size: 1.1rem; opacity: 0.9; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# PAGE 1: HOME
# ==========================================
def render_home_page():
    st.markdown(
        """
        <div class="hero-section">
            <h1>Welcome to LoanSense </h1>
            <p>The Intelligent Risk Assessment System for modern financial institutions.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    df = load_data()
    metrics = load_metrics()
    
    if df is not None:
        st.subheader("📊 Portfolio Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Applications", f"{len(df):,}")
        col2.metric("Historical Default Rate", f"{df['Default'].mean() * 100:.1f}%")
        col3.metric("Average Credit Score", f"{df['CreditScore'].mean():.0f}")
        col4.metric("Avg Loan Amount", f"${df['LoanAmount'].mean():,.0f}")
        
    st.write("")
    
    if metrics is not None:
        st.subheader("⚙️ System Reliability Metrics")
        st.caption("These metrics prove the underlying predictive analytics are highly accurate.")
        optimal = metrics.get("optimal_threshold_metrics", {})
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC Score", f"{metrics.get('roc_auc', 0):.3f}", help="Model's ability to distinguish classes. 1.0 is perfect.")
        m2.metric("F1-Score", f"{optimal.get('f1', 0):.3f}", help="Balance between Precision and Recall.")
        m3.metric("Precision", f"{optimal.get('precision', 0):.3f}", help="When predicting default, how often is it correct?")
        m4.metric("Recall", f"{optimal.get('recall', 0):.3f}", help="Out of all actual defaulters, how many were caught?")

    st.divider()


    st.subheader("How to use the system")
    st.write("Navigate through the platform using the sidebar menu on the left:")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**📊 Dashboard**\n\nEnter a new applicant's profile to instantly calculate their probability of default and see exactly *why* they received that score.")
        st.info("**📁 Data Explorer**\n\nBrowse through the historical dataset. Use dynamic sliders to filter applicants by Age and Income.")
    with col_b:
        st.success("**🤖 Explainable AI**\n\nWe don't just give you a score. Our transparent system shows you which factors pushed the risk up or down.")
        st.success("**📥 Excel Export**\n\nEasily download individual risk reports to include in your official loan officer documentation.")

# ==========================================
# PAGE 2: PREDICTION DASHBOARD
# ==========================================
def render_predict_page(pipeline, explainer, metrics):
    st.subheader("Applicant Profile")
    st.write("Enter the applicant's details below to predict default risk.")
    
    threshold = metrics.get("optimal_threshold", 0.5) if metrics else 0.5
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=55000, step=1000)
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000, value=50000, step=1000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=60)
            
        with col2:
            num_credit_lines = st.number_input("Num Credit Lines", min_value=1, max_value=10, value=3)
            interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=25.0, value=8.0, step=0.1)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
            dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
            
        with col3:
            education = st.selectbox("Education", ["Bachelor's", "High School", "Master's", "PhD"])
            employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
            cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

        st.divider()
        submit_button = st.form_submit_button("Predict Default Risk", type="primary", use_container_width=True)

    if submit_button:
        payload = {
            "Age": age, "Income": income, "LoanAmount": loan_amount, "CreditScore": credit_score,
            "MonthsEmployed": months_employed, "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate, "LoanTerm": loan_term, "DTIRatio": dti_ratio,
            "Education": education, "EmploymentType": employment, "MaritalStatus": marital,
            "HasMortgage": mortgage, "HasDependents": dependents, "LoanPurpose": purpose, "HasCoSigner": cosigner
        }
        
        result = score_locally(pipeline, explainer, payload, threshold)
        st.session_state["prediction_result"] = result
        st.session_state["input_payload"] = payload


    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]
        payload = st.session_state["input_payload"]
        
        st.divider()
        st.subheader("Prediction Result")
        
        col_a, col_b = st.columns(2)
        col_a.metric("Probability of default", f"{result['probability_of_default']:.2%}")
        col_b.metric("Decision", "High risk" if result['prediction'] == 1 else "Low risk")
        
        if result["prediction"] == 1:
            st.error(result["risk_label"])
        else:
            st.success(result["risk_label"])
        st.progress(min(max(result['probability_of_default'], 0.0), 1.0))
        
        dl_df = pd.DataFrame([payload])
        dl_df["Predicted_Default"] = result["prediction"]
        dl_df["Default_Probability"] = result["probability_of_default"]
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            dl_df.to_excel(writer, index=False, sheet_name='Prediction')
        
        st.download_button(
            label="📥 Download Prediction Result (Excel)",
            data=excel_buffer.getvalue(),
            file_name="loan_prediction_result.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("### Why this prediction?")
        st.caption("Green bars push probability down (toward approval), red bars push it up (toward default).")
        render_shap_chart(result["top_feature_contributions"])

# ==========================================
# PAGE 3: DATA EXPLORER
# ==========================================
def render_data_explorer():
    df = load_data()
    if df is None:
        st.error("Dataset not found. Please ensure `data/Loan_default.csv` exists.")
        return

    st.subheader("Data Explorer")
    st.write("Use the filters on the left to explore the loan portfolio dataset.")

    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Filter by Age", min_age, max_age, (min_age, max_age))
    
    min_inc, max_inc = int(df["Income"].min()), int(df["Income"].max())
    inc_range = st.sidebar.slider("Filter by Income", min_inc, max_inc, (min_inc, max_inc), step=1000)

    filtered_df = df[
        (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) &
        (df["Income"] >= inc_range[0]) & (df["Income"] <= inc_range[1])
    ]
    
    filtered_df = filtered_df.sort_values(by=["Age", "Income"], ascending=[True, False])

    st.write(f"Showing **{len(filtered_df):,}** records matching your filters.")
    st.dataframe(filtered_df, use_container_width=True)

# ==========================================
# PAGE 4: ABOUT
# ==========================================
def render_about_page():
    st.subheader("About the Project")
    
    st.write("""
    This app predicts loan default risk using a Random Forest model trained on heavily imbalanced financial data (only ~11.6% default rate). 
    
    I built this end-to-end system to demonstrate how to handle imbalanced data safely, integrate **Explainable AI (SHAP)** so loan officers can understand *why* a decision was made, and deploy it as a clean, interactive web application.
    """)
    
    st.info("**Note:** This application was developed as part of the Data Science Web App midterm project.")

# ==========================================
# MAIN ROUTING
# ==========================================
def main() -> None:
    inject_style()
    
    pipeline, _ = load_pipeline()
    explainer = build_explainer(pipeline)
    metrics = load_metrics()


    # --- PROFILE HEADER ---
    with st.sidebar:
        col_img, col_text = st.columns([1, 2.5])
        with col_img:
            try:
                st.image("logo.png", width=50)
            except:
                st.write("")
        with col_text:
            st.markdown(
                """
                <div class="profile-info">
                    <h4>Loan Analyst</h4>
                    <p>Risk Management</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    # --- NAVIGATION ---
    page = st.sidebar.radio(
        "Menu", 
        ["🏠 Home", "📊 Dashboard", "📁 Data Explorer", "ℹ️ About"],
        label_visibility="collapsed"
    )

    # --- ROUTING ---
    if "Home" in page:
        render_home_page()
    elif "Dashboard" in page:
        render_predict_page(pipeline, explainer, metrics)
    elif "Data Explorer" in page:
        render_data_explorer()
    elif "About" in page:
        render_about_page()

if __name__ == "__main__":
    main()
