import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
# Example model (replace with your model)
from PIL import Image

# Streamlit app interface
st.title("Cardiovascular Risk Prediction")

# Display image buttons for gender selection
col1, col2 = st.columns(2)

with col1:
    male_img = Image.open("male_image.png")  # Replace with actual path to male image
    if st.button("Male", key="male"):
        gender = "Male"
        
with col2:
    female_img = Image.open("female_image.png")  # Replace with actual path to female image
    if st.button("Female", key="female"):
        gender = "Female"

# Proceed with other inputs
age = st.number_input("Age (years)", min_value=0, max_value=120)
ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
alco = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])

# Process gender input
if gender == "Male":
    gender_value = 1
elif gender == "Female":
    gender_value = 0

# Encoding other inputs (for simplicity, using placeholder values)
alco_value = 1 if alco == "Yes" else 0
smoke_value = 1 if smoke == "Yes" else 0
cholesterol_value = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]

# Prepare input data for prediction (example model)
input_data = [gender_value, age, ap_lo, ap_hi, alco_value, smoke_value, cholesterol_value]

if st.button("Submit"):
    # Here you would use your actual model to predict risk, for now a dummy result
    prediction = 1  # Dummy risk prediction
    risk = "Risk of Cardiovascular Disease" if prediction == 1 else "No Risk"
    st.write(f"Prediction: {risk}")

# Load your trained model (Replace with actual model loading step)
# model = joblib.load('your_model.pkl')  # Example, use joblib or pickle to load your trained model

# Sample data for the model (replace with actual feature engineering)
data = pd.DataFrame({
    'gender': [0, 1],  # 0: Female, 1: Male (example encoding)
    'age_years': [45, 60],
    'ap_lo': [80, 90],  # Low blood pressure (example)
    'ap_hi': [120, 140],  # High blood pressure (example)
    'alco': [0, 1],  # 0: No, 1: Yes (alcohol consumption)
    'smoke': [0, 1],  # 0: No, 1: Yes (smoking status)
    'cholesterol': [1, 2],  # 1: Normal, 2: Above Normal (cholesterol)
})

# Dummy target for the model (replace with actual target variable)
target = [0, 1]  # 0: No risk, 1: Risk (example)

# Fit a random forest classifier (replace with your actual trained model)
model = RandomForestClassifier()
model.fit(data, target)

# Streamlit app interface
st.title("Cardiovascular Risk Prediction")

# User inputs for the selected columns
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age (years)", min_value=0, max_value=120)
ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
alco = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
smoke = st.selectbox("Do you smoke?", ["No", "Yes"])
cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])

# Encoding user inputs
gender = 1 if gender == "Male" else 0
alco = 1 if alco == "Yes" else 0
smoke = 1 if smoke == "Yes" else 0
cholesterol = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]

# Prepare the input data for prediction
input_data = np.array([[gender, age, ap_lo, ap_hi, alco, smoke, cholesterol]])

# Predict the cardiovascular risk
if st.button("Submit"):
    prediction = model.predict(input_data)
    risk = "Risk of Cardiovascular Disease" if prediction[0] == 1 else "No Risk"
    st.write(f"Prediction: {risk}")
