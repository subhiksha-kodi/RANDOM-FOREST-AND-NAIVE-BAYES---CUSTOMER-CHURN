import streamlit as st
import pandas as pd
import pickle

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # Make sure you saved this during training

st.title("Customer Churn Prediction")
st.subheader("Enter customer details")

monthly_charges = st.number_input("Monthly Charges ($):", min_value=0.0, step=1.0)
tenure = st.number_input("Tenure (months):", min_value=0, step=1)
support_calls = st.number_input("Number of Support Calls:", min_value=0, step=1)
contract = st.selectbox("Contract Type:", ["Month-to-month", "One year", "Two year"])

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract_encoded = contract_map[contract]

if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "MonthlyCharges": monthly_charges,
        "tenure": tenure,
        "SupportCalls": support_calls,
        "Contract": contract_encoded
    }])

    # Scale the input
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to Churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is likely to Stay (Probability: {prob:.2f})")
