import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model

# get current file directory and project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# load trained ANN model
model = load_model(os.path.join(ROOT_DIR, "model", "model.h5"))

# load preprocessing objects
with open(os.path.join(ROOT_DIR, "artifacts", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(ROOT_DIR, "artifacts", "label_encoder_gender.pkl"), "rb") as f:
    label_encoder_gender = pickle.load(f)

with open(os.path.join(ROOT_DIR, "artifacts", "onehot_encoder_geo.pkl"), "rb") as f:
    encoder = pickle.load(f)

# app title and description
st.title("Customer Churn Prediction (ANN)")
st.write("Enter customer details to predict churn")

# user input fields
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

# prediction trigger
if st.button("Predict"):

    # create dataframe from user input
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

    # apply label encoding to gender
    input_data["Gender"] = label_encoder_gender.transform(input_data["Gender"])

    # apply one-hot encoding to geography
    geo_encoded = encoder.transform(input_data[["Geography"]])
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=encoder.get_feature_names_out(["Geography"])
    )

    # replace geography column with encoded columns
    input_data = input_data.drop("Geography", axis=1)
    input_data = pd.concat([input_data, geo_df], axis=1)

    # ensure same column order as training
    columns_order = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]

    input_data = input_data.reindex(columns=columns_order, fill_value=0)

    # scale numerical features
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance',
                'NumOfProducts', 'EstimatedSalary']

    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # model prediction
    prob = model.predict(input_data)[0][0]

    # display probability
    st.subheader(f"Churn Probability: {prob:.2f}")

    # final decision
    if prob > 0.5:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")