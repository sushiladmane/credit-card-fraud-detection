import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(page_title="Credit Card Fraud Detection",
                   page_icon="💳",
                   layout="wide")

# Load Model
model = joblib.load("fraud_model.pkl")

# Title
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Real-time Fraud Risk Prediction Dashboard")

# Layout
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("📝 Transaction Details")

    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
    transaction_hour = st.slider("Transaction Hour", 0, 23, 12)

    merchant_category = st.selectbox(
        "Merchant Category",
        ["grocery", "electronics", "travel", "entertainment", "others"]
    )

    foreign_transaction = st.selectbox("Foreign Transaction?", [0,1])
    location_mismatch = st.selectbox("Location Mismatch?", [0,1])
    device_trust_score = st.slider("Device Trust Score", 0, 100, 50)
    velocity_last_24h = st.slider("Transactions in Last 24h", 0, 50, 5)
    cardholder_age = st.slider("Cardholder Age", 18, 80, 30)

# Simple encoding for merchant category
merchant_mapping = {
    "grocery":0,
    "electronics":1,
    "travel":2,
    "entertainment":3,
    "others":4
}

merchant_encoded = merchant_mapping[merchant_category]

# Feature array
features = np.array([[amount,
                      transaction_hour,
                      merchant_encoded,
                      foreign_transaction,
                      location_mismatch,
                      device_trust_score,
                      velocity_last_24h,
                      cardholder_age]])

# Prediction
if st.button("🔍 Predict Fraud Risk"):

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    with col2:
        st.subheader("📊 Prediction Result")

        st.write("Fraud Probability:", round(probability*100, 2), "%")

        # Risk Level Logic
        if probability < 0.30:
            st.success("Low Risk - Transaction Safe ✅")
        elif probability < 0.70:
            st.warning("Medium Risk - Manual Review Suggested ⚠️")
        else:
            st.error("High Risk - Fraud Alert 🚨")

        # Progress Bar
        st.progress(float(probability))

        # Feature Importance (only for tree models)
        if hasattr(model, "feature_importances_"):

            st.subheader("📈 Feature Importance")

            importance = model.feature_importances_

            feature_names = [
                "Amount",
                "Hour",
                "Merchant",
                "Foreign",
                "Location",
                "Device Score",
                "Velocity",
                "Age"
            ]

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))