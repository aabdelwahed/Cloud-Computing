import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and feature columns
model = pickle.load(open('churn_model.pkl', 'rb'))
columns = pickle.load(open('churn_columns.pkl', 'rb'))

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("🔮 Telco Customer Churn Predictor")

# Collect user inputs
tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
monthly = st.slider("Monthly Charges", min_value=0.0, max_value=120.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, value=840.0)

contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
payment = st.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])
internet = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

# Initialize empty sample row
sample = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Fill numerical values
sample['tenure'] = tenure
sample['MonthlyCharges'] = monthly
sample['TotalCharges'] = total

# Fill dummy variables
if contract == 'Month-to-month':
    sample['Contract_Month-to-month'] = 1
elif contract == 'One year':
    sample['Contract_One year'] = 1
elif contract == 'Two year':
    sample['Contract_Two year'] = 1

if payment == 'Electronic check':
    sample['PaymentMethod_Electronic check'] = 1
elif payment == 'Mailed check':
    sample['PaymentMethod_Mailed check'] = 1
elif payment == 'Bank transfer (automatic)':
    sample['PaymentMethod_Bank transfer (automatic)'] = 1
elif payment == 'Credit card (automatic)':
    sample['PaymentMethod_Credit card (automatic)'] = 1

if internet == 'DSL':
    sample['InternetService_DSL'] = 1
elif internet == 'Fiber optic':
    sample['InternetService_Fiber optic'] = 1
elif internet == 'No':
    sample['InternetService_No'] = 1

# Ensure all columns are present and ordered
missing = set(columns) - set(sample.columns)
for col in missing:
    sample[col] = 0
sample = sample[columns]

# Predict
if st.button("Predict"):
    result = model.predict(sample)[0]
    st.subheader("📢 Prediction Result:")
    if result == 0:
        st.success("✅ This customer is likely to stay (Active).")
    else:
        st.error("⚠️ This customer is likely to churn.")
