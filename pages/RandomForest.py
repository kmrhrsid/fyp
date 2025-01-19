import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
# Example model (replace with your model)


# Load your trained model (Replace with actual model loading step)
# model = joblib.load('random_forest_model(1).pkl')  # Example, use joblib or pickle to load your trained model


# Fit a random forest classifier (replace with your actual trained model)
model = RandomForestClassifier()
model.fit(data, target)

# Streamlit app interface
st.title("Cardiovascular Risk Prediction")

# Encoding user inputs
gender = 1 if gender == "Male" else 0
alco = 1 if alco == "Yes" else 0
smoke = 1 if smoke == "Yes" else 0
cholesterol = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]

# Prepare the input data for prediction
input_data = np.array([[gender, age, ap_lo, ap_hi, alco, smoke, cholesterol]])
# User inputs for the selected columns
gender = st.selectbox("Gender", ["Female", "Male"])
ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
alco = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])

# Predict the cardiovascular risk
if st.button("Submit"):
    prediction = model.predict(input_data)
    risk = "Risk of Cardiovascular Disease" if prediction[0] == 1 else "No Risk"
    st.write(f"Prediction: {risk}")

