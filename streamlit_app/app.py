import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# Load Model & Preprocessing

model = load_model("model\model.h5")

with open("artifacts\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("artifacts\label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("artifacts\onehot_encoder_geo.pkl", "rb") as f:
    encoder = pickle.load(f)


# UI Title

st.title("Customer Churn Prediction (ANN)")
st.write("Enter customer details to predict churn")


# User Inputs

credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 40)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)


# Prediction Button

if st.button("Predict"):

    # Create DataFrame
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary
    }])


    # Preprocessing


    # Encode Gender
    input_data["Gender"] = label_encoder_gender.transform(input_data["Gender"])

    # One-Hot Encode Geography
    geo_encoded = encoder.transform(input_data[["Geography"]])
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=encoder.get_feature_names_out(["Geography"])
    )

    input_data = input_data.drop("Geography", axis=1)
    input_data = pd.concat([input_data, geo_df], axis=1)

    # Column Order 
    columns_order = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]

    input_data = input_data.reindex(columns=columns_order, fill_value=0)

    # Scale Numerical Features
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'EstimatedSalary']

    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Prediction

    prob = model.predict(input_data)[0][0]

    st.subheader(f"Churn Probability: {prob:.2f}")

    if prob > 0.5:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")