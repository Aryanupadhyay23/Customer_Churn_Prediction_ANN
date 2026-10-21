import os
import sys
import streamlit as st

# Ensure project root is in sys.path so 'src' can be imported cleanly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.pipeline.predict_pipeline import PredictPipeline, CustomData


# Cache pipeline initialization to avoid reloading model on every interaction
@st.cache_resource
def get_prediction_pipeline():
    return PredictPipeline()


pipeline = get_prediction_pipeline()

# App Header
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🏦", layout="centered")
st.title("🏦 Customer Churn Prediction (ANN)")
st.markdown("Enter customer financial and demographic details to assess churn risk.")

st.divider()

# Input layout using Streamlit columns for a clean UI
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=18, max_value=100, value=40)
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)

with col2:
    balance = st.number_input("Account Balance", min_value=0.0, max_value=250000.0, value=50000.0, step=1000.0)
    num_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
    has_cr_card = st.selectbox("Has Credit Card", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active = st.selectbox("Is Active Member", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)

st.divider()

# Prediction Trigger
if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing customer data through ANN pipeline..."):
        # Package UI input into CustomData
        customer_data = CustomData(
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            num_products=num_products,
            has_cr_card=has_cr_card,
            is_active=is_active,
            salary=salary
        )

        # Get DataFrame from user input
        features_df = customer_data.get_data_as_data_frame()

        # Delegate prediction to dedicated prediction pipeline
        prob, is_churn = pipeline.predict(features_df)

    # Display Results
    st.subheader("Prediction Result")
    progress_val = min(max(float(prob), 0.0), 1.0)
    st.progress(progress_val)
    st.write(f"**Churn Probability:** `{prob * 100:.2f}%`")

    if is_churn:
        st.error(f"⚠️ **High Churn Risk!** Customer is likely to leave the bank (Probability: {prob:.2f}).")
    else:
        st.success(f"✅ **Low Churn Risk.** Customer is likely to remain with the bank (Probability: {prob:.2f}).")