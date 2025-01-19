import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model (1).pkl')
print("Model loaded successfully!")

# Streamlit app interface
st.title("Cardiovascular Risk Prediction")

# User input
age = st.slider('Age (years)', min_value=20, max_value=80, value=50)
gender = st.selectbox("Gender", ["Female", "Male"])
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)

# Preprocessing
gender_encoded = 1 if gender == "Male" else 0

# Create raw input DataFrame
input_data = pd.DataFrame({
    'age_years': [age],   # Renamed to match the model
    'gender': [gender_encoded],
    'height': [height],   # New input field
    'weight': [weight],   # New input field
    'ap_lo': [ap_lo],
    'ap_hi': [ap_hi],
})

# Debugging: Print feature names
expected_features = model.feature_names_in_  # Get feature names from the model
print("Expected features:", expected_features)
print("Input features:", input_data.columns)

# Align features
missing_features = set(expected_features) - set(input_data.columns)
extra_features = set(input_data.columns) - set(expected_features)

if missing_features or extra_features:
    st.error(f"Feature mismatch! Missing: {missing_features}, Extra: {extra_features}")
else:
    input_data = input_data[expected_features]  # Ensure column alignment

    # Predict the cardiovascular risk
    if st.button("Submit"):
        prediction = model.predict(input_data)[0]
        risk = "Risk of Cardiovascular Disease" if prediction == 1 else "No Risk"
        st.write(f"Prediction: {risk}")
