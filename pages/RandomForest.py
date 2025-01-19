import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
# Example model (replace with your model)
import joblib

model = joblib.load('random_forest_model (1).pkl')
print("Model loaded successfully!")


# Streamlit app interface
st.title("Cardiovascular Risk Prediction")

# Encoding user inputs
#gender = 1 if gender == "Male" else 0
#alco = 1 if alco == "Yes" else 0
#smoke = 1 if smoke == "Yes" else 0
#cholesterol = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]

age = st.slider('Age', min_value=20, max_value=80, value=50)
gender = st.selectbox("Gender", ["Female", "Male"])
ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
alco = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])

# Create raw input DataFrame
input_data = pd.DataFrame({
    'item_code': [item_code],
    'premise_type': [premise_type],
    'district': [district],
    'month': [month]
})


# Predict the cardiovascular risk
if st.button("Submit"):
    prediction = model.predict(input_data)[0]
    risk = "Risk of Cardiovascular Disease" if prediction[0] == 1 else "No Risk"
    st.write(f"Prediction: {risk}")

