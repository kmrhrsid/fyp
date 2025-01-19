import streamlit as st
import pandas as pd
import joblib

# Load the trained model
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

# Preprocessing
gender_encoded = 1 if gender == "Male" else 0
alco_encoded = 1 if alco == "Yes" else 0
smoke_encoded = 1 if smoke == "Yes" else 0
cholesterol_encoded = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]

# Create raw input DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender_encoded],
    'ap_lo': [ap_lo],
    'ap_hi': [ap_hi],
    'cholesterol': [cholesterol_encoded],
    'smoke': [smoke_encoded],
    'alco': [alco_encoded],
})

# Match feature names
expected_features = model.feature_names_in_  # Get feature names from the model
input_data = input_data[expected_features]  # Ensure column alignment

# Predict the cardiovascular risk
if st.button("Submit"):
    prediction = model.predict(input_data)[0]
    risk = "Risk of Cardiovascular Disease" if prediction == 1 else "No Risk"
    st.write(f"Prediction: {risk}")
