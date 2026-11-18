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
ARTIFACTS_DIR = PROJECT_ROOT / "evaluation_artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "Loan_default.csv"

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
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

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(features))))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on default probability)")
    ax.set_title("Feature contributions for this applicant")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    st.image(buffer.getvalue(), use_column_width=True)

def inject_style() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1000px;}
          [data-testid="stMetricValue"] {font-size: 1.9rem;}
          [data-testid="stMetricLabel"] {opacity: 0.75;}
          h1 {margin-bottom: 0.3rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# PAGE 1: PREDICTION
# ==========================================
def render_predict_page(pipeline, explainer, metrics):
    st.subheader("Applicant Profile")
    st.write("Enter the applicant's details below to predict default risk.")
    
    with st.form("prediction_form"):
        # Use columns to keep the form looking neat and centered
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
        
        # Decision Threshold
        tuned_threshold = metrics.get("optimal_threshold", 0.5) if metrics else 0.5
        threshold = st.slider(
            "Decision threshold", min_value=0.05, max_value=0.95, value=float(tuned_threshold), step=0.01,
            help="Applicants with probability at or above this value are flagged as high risk."
        )
        
        submit_button = st.form_submit_button("Predict Default Risk", type="primary", use_container_width=True)

    if submit_button:
        # Gather inputs
        payload = {
            "Age": age, "Income": income, "LoanAmount": loan_amount, "CreditScore": credit_score,
            "MonthsEmployed": months_employed, "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate, "LoanTerm": loan_term, "DTIRatio": dti_ratio,
            "Education": education, "EmploymentType": employment, "MaritalStatus": marital,
            "HasMortgage": mortgage, "HasDependents": dependents, "LoanPurpose": purpose, "HasCoSigner": cosigner
        }
        
        # Predict
        result = score_locally(pipeline, explainer, payload, threshold)
        st.session_state["prediction_result"] = result
        st.session_state["input_payload"] = payload

    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]
        payload = st.session_state["input_payload"]
        
        st.divider()
        st.subheader("Prediction Result")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Probability of default", f"{result['probability_of_default']:.2%}")
        col_b.metric("Threshold applied", f"{result['threshold_used']:.2f}")
        col_c.metric("Decision", "High risk" if result['prediction'] == 1 else "Low risk")
        
        if result["prediction"] == 1:
            st.error(result["risk_label"])
        else:
            st.success(result["risk_label"])
        st.progress(min(max(result['probability_of_default'], 0.0), 1.0))
        
        # Download Button
        dl_df = pd.DataFrame([payload])
        dl_df["Predicted_Default"] = result["prediction"]
        dl_df["Default_Probability"] = result["probability_of_default"]
        csv = dl_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Prediction Result (CSV)",
            data=csv,
            file_name="loan_prediction_result.csv",
            mime="text/csv",
        )

        st.markdown("### Why this prediction?")
        st.caption("Green bars push probability down (toward approval), red bars push it up (toward default).")
        render_shap_chart(result["top_feature_contributions"])


# ==========================================
# PAGE 2: DATA EXPLORER
# ==========================================
def render_data_explorer():
    df = load_data()
    if df is None:
        st.error("Dataset not found. Please ensure `data/Loan_default.csv` exists.")
        return

    st.subheader("Data Explorer")
    st.write("Use the filters on the left to explore the loan portfolio dataset.")

    # Sidebar filters
    st.sidebar.markdown("### Dynamic Filters")
    
    # Age Filter
    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider("Filter by Age", min_age, max_age, (min_age, max_age))
    
    # Income Filter
    min_inc, max_inc = int(df["Income"].min()), int(df["Income"].max())
    inc_range = st.sidebar.slider("Filter by Income", min_inc, max_inc, (min_inc, max_inc), step=1000)

    # Apply filters
    filtered_df = df[
        (df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) &
        (df["Income"] >= inc_range[0]) & (df["Income"] <= inc_range[1])
    ]

    st.write(f"Showing **{len(filtered_df):,}** records matching your filters.")
    st.dataframe(filtered_df, use_container_width=True)


# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================
def render_performance_page(metrics):
    if metrics is None:
        st.warning("metrics.json not found. Run `python ml/evaluate.py` to generate it.")
        return

    st.subheader("Model performance on held-out test set")
    optimal = metrics["optimal_threshold_metrics"]
    default = metrics["default_threshold_metrics"]

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        col2.metric("F1 (tuned)", f"{optimal['f1']:.3f}")
        col3.metric("Precision (tuned)", f"{optimal['precision']:.3f}")
        col4.metric("Recall (tuned)", f"{optimal['recall']:.3f}")
        st.caption(
            f"Optimal threshold = {metrics['optimal_threshold']:.3f}. "
            f"At default 0.5 threshold: F1 = {default['f1']:.3f}."
        )

    st.markdown("### Diagnostic charts")
    col_a, col_b = st.columns(2)
    
    # Check if images exist and show them
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    roc_path = ARTIFACTS_DIR / "roc_curve.png"
    pr_path = ARTIFACTS_DIR / "pr_curve.png"
    fi_path = ARTIFACTS_DIR / "feature_importance.png"
    
    if cm_path.exists(): col_a.image(cm_path.read_bytes(), caption="Confusion Matrix", use_column_width=True)
    if roc_path.exists(): col_b.image(roc_path.read_bytes(), caption="ROC Curve", use_column_width=True)
    if pr_path.exists(): col_a.image(pr_path.read_bytes(), caption="Precision-Recall Curve", use_column_width=True)
    if fi_path.exists(): col_b.image(fi_path.read_bytes(), caption="Feature Importance", use_column_width=True)


# ==========================================
# MAIN ROUTING
# ==========================================
def main() -> None:
    inject_style()
    
    pipeline, _ = load_pipeline()
    explainer = build_explainer(pipeline)
    metrics = load_metrics()

    # Sidebar Navigation
    st.sidebar.title("🏦 Navigation")
    page = st.sidebar.radio("Go to", ["Predict Risk", "Data Explorer", "Model Performance", "About"])

    st.title("Loan Default Prediction")
    st.caption("Random Forest classifier with threshold tuning and SHAP explanations.")

    if page == "Predict Risk":
        render_predict_page(pipeline, explainer, metrics)
    elif page == "Data Explorer":
        render_data_explorer()
    elif page == "Model Performance":
        render_performance_page(metrics)
    elif page == "About":
        st.subheader("About this project")
        st.write("This application predicts the probability that a loan applicant will default. It uses a Random Forest classifier and provides SHAP explanations to show exactly which factors influenced the decision.")
        st.write("Built for the Data Science Web App midterm project.")

if __name__ == "__main__":
    main()