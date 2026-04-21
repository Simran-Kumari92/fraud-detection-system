
import streamlit as st
import joblib
import pandas as pd
import os

# -------------------------------
# 📂 Load model + preprocessor
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "fraud_detection_pipeline.pkl")

data = joblib.load(model_path)

model = data["model"]
preprocessor = data["preprocessor"]

# -------------------------------
# 🎨 UI CONFIG
# -------------------------------
st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 Fraud Detection System")
st.markdown("### Enter transaction details to check fraud risk")

st.divider()

# -------------------------------
# 🧾 Input Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    )
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)

with col2:
    newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=900.0)
    oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

st.divider()

# -------------------------------
# 🔮 Prediction
# -------------------------------
if st.button("🔍 Predict Fraud", use_container_width=True):

    # -------------------------------
    # 🔴 Rule-based validation
    # -------------------------------
    if oldbalanceOrg == 0 and amount > 0:
        st.error("❌ Invalid transaction — cannot withdraw from zero balance")
        st.stop()

    if newbalanceOrig < 0:
        st.error("❌ Invalid: Negative balance not allowed")
        st.stop()

    if amount > oldbalanceOrg and oldbalanceOrg > 0:
        st.warning("⚠️ Amount exceeds available balance")

    # -------------------------------
    # 📥 Input Data
    # -------------------------------
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # -------------------------------
    # 🔥 Feature Engineering  #-------------------------------
    input_data["balance_error"] = (
        input_data["oldbalanceOrg"]
        - input_data["amount"]
        - input_data["newbalanceOrig"]
    )

    input_data["balance_ratio"] = (
        input_data["amount"] / (input_data["oldbalanceOrg"] + 1)
    )

    input_data["is_full_withdrawal"] = (
        (input_data["newbalanceOrig"] == 0).astype(int)
    )


    # -------------------------------
    # 🔮 Prediction
    # -------------------------------
    input_transformed = preprocessor.transform(input_data)

    prediction = model.predict(input_transformed)[0]
    probability = model.predict_proba(input_transformed)[0][1]

    st.divider()

    # -------------------------------
    # 🔍 Hybrid Logic
    # -------------------------------
    balance_error = abs(oldbalanceOrg - amount - newbalanceOrig)
    balance_ratio = amount / (oldbalanceOrg + 1)
    full_withdrawal = (oldbalanceOrg > 0 and newbalanceOrig == 0)
    high_amount = amount > 50000

    suspicious_rule = (
        balance_error > 1 or
        balance_ratio > 0.9 or
        full_withdrawal or
        high_amount
    )

    # -------------------------------
    # 🎯 Final Decision
    # -------------------------------
    if prediction == 1:
        st.error("🚨 Fraud Detected — Immediate attention required")

    elif suspicious_rule:
        st.warning("⚠️ Suspicious Transaction — Needs Manual Review")

    else:
        st.success("✅ Safe Transaction — No fraud risk detected")

    # -------------------------------
    # 📊 Risk Score
    # -------------------------------
    st.metric("Fraud Risk Score", f"{probability*100:.2f}%")
    st.progress(float(probability))
    st.caption("Confidence of fraud prediction")

    # -------------------------------
    # 🔥 Risk Level
    # -------------------------------
    if probability > 0.7:
        st.error("🔴 High Risk Transaction")
    elif probability > 0.4:
        st.warning("🟠 Medium Risk Transaction")
    else:
        st.success("🟢 Low Risk Transaction")

    # -------------------------------
    # 🔍 Explain Prediction
    # -------------------------------
    st.subheader("🔍 Why this prediction?")

    colA, colB = st.columns(2)
    with colA:
        st.write(f"Type: {transaction_type}")
        st.write(f"Amount: ₹{amount:,.0f}")
    with colB:
        st.write(f"Sender Balance Change: ₹{(oldbalanceOrg - newbalanceOrig):,.0f}")

    # -------------------------------
    # ⚠️ Additional warnings
    # -------------------------------
    if amount > 50000:
        st.warning("⚠️ High transaction amount detected")

    if oldbalanceOrg > 0 and newbalanceOrig == 0:
        st.warning("⚠️ Entire balance withdrawn — suspicious pattern")

    # -------------------------------
    # ℹ️ About Model
    # -------------------------------
    with st.expander("ℹ️ About the Model"):
        st.write("""
        - Model: Random Forest
        - Technique: SMOTE for class imbalance
        - Features: Transaction type, amount, balances + engineered features
        - Hybrid system: ML + rule-based validation
        """)

    # -------------------------------
    # 📌 Footer
    # -------------------------------
    st.markdown("---")
    st.caption("Built using Machine Learning (Random Forest + SMOTE) | By Simran Kumari")