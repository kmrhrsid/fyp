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
