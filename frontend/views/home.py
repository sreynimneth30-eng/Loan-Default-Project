import streamlit as st

st.title("🏦 Intelligent Loan Default Prediction")
st.markdown("""
Welcome to the **Loan Risk Assessment System**. 
This application uses advanced Machine Learning to predict the likelihood of a borrower defaulting on a loan.
""")

st.divider()

# Create metric columns
col1, col2, col3 = st.columns(3)
col1.metric("Model Type", "LightGBM", "Gradient Boosting")
col2.metric("Dataset Size", "255,347", "Kaggle Loan Dataset")
col3.metric("Explainability", "SHAP", "Feature Importance")

st.divider()

st.subheader("How to use this app:")
st.markdown("""
1. **Data Explorer:** Understand the dataset we used to train the model.
2. **Predict Risk:** Enter a borrower's details to get an instant default probability and an explanation of *why*.
3. **Model Performance:** Review the ROC curves and accuracy metrics of our ML models.
""")