import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
import joblib

# Load the model
model = joblib.load('random_forest_model (1).pkl')
print("Model loaded successfully!")

# Streamlit app interface
st.title("Cardiovascular Risk Prediction")

# User input
age = st.slider('Age', min_value=20, max_value=80, value=50)
gender = st.selectbox("Gender", ["Female", "Male"])
ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
alco = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])

# Create raw input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'ap_lo': [ap_lo],
    'ap_hi': [ap_hi],
    'cholesterol': [cholesterol],
    'smoke': [smoke],
    'alco': [alco],
})

# Predict the cardiovascular risk
if st.button("Submit"):
    prediction = model.predict(input_data)[0]
    risk = "Risk of Cardiovascular Disease" if prediction == 1 else "No Risk"
    st.write(f"Prediction: {risk}")

