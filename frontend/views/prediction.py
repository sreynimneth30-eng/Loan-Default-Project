import streamlit as st
import requests

def show():
    st.title("🔮 Predict Loan Default Risk")
    st.write("Enter the borrower's details below.")
    
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 18, 100, 35)
            income = st.number_input("Annual Income ($)", 0, 1000000, 55000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
        with col2:
            loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 50000)
            interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 8.0)
            months_employed = st.number_input("Months Employed", 0, 600, 60)
        with col3:
            dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
            num_credit_lines = st.slider("Number of Credit Lines", 1, 10, 3)

        col4, col5, col6 = st.columns(3)
        with col4:
            education = st.selectbox("Education", ["Bachelor's", "High School", "Master's", "PhD"])
            employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        with col5:
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
        with col6:
            mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
            cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

        submit = st.form_submit_button("Predict Risk", type="primary")

    if submit:
        # Prepare payload for FastAPI
        payload = {
            "Age": age, "Income": income, "LoanAmount": loan_amount, "CreditScore": credit_score,
            "MonthsEmployed": months_employed, "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate, "LoanTerm": loan_term, "DTIRatio": dti_ratio,
            "Education": education, "EmploymentType": employment, "MaritalStatus": marital,
            "LoanPurpose": purpose, "HasMortgage": mortgage, "HasDependents": dependents, "HasCoSigner": cosigner
        }
        
        # Send request to FastAPI
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            response.raise_for_status()
            prob = response.json()["default_probability"]
            
            st.divider()
            if prob >= 0.5:
                st.error(f"### ⚠️ HIGH RISK\nDefault Probability: **{prob:.1%}**")
            else:
                st.success(f"### ✅ LOW RISK\nDefault Probability: **{prob:.1%}**")
            st.progress(prob)
            
        except requests.exceptions.ConnectionError:
            st.error("🚨 Failed to connect to the Backend API. Make sure the FastAPI server is running!")