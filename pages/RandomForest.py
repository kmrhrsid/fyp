import streamlit as st

st.title("Simple Streamlit App")
st.write("Enter your details below:")

name = st.text_input("What's your name?")
age = st.number_input("What's your age?", min_value=0)

# Blood pressure input
systolic = st.number_input("Systolic Pressure (mmHg)", min_value=0)
diastolic = st.number_input("Diastolic Pressure (mmHg)", min_value=0)

if st.button("Submit"):
    st.success(f"Hello, {name}! You are {age} years old.")

import streamlit as st

st.title("Cardiovascular Risk Prediction")

# User inputs for the selected columns
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age (years)", min_value=0, max_value=120)
ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
alco = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
smoke = st.selectbox("Do you smoke?", ["Yes", "No"])
cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])

# Show entered details on submit
if st.button("Submit"):
    st.write(f"Gender: {gender}")
    st.write(f"Age: {age} years")
    st.write(f"Blood Pressure: {ap_hi}/{ap_lo} mmHg")
    st.write(f"Alcohol: {alco}")
    st.write(f"Smoking: {smoke}")
    st.write(f"Cholesterol: {cholesterol}")

