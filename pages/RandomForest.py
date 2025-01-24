import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Cardiovascular Risk Prediction", page_icon="🫀")

# Function to encode the image in Base64
def add_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    background_style = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

# Add your background image here
add_background_image("images.jpg")


# Load the trained model
model = joblib.load('random_forest_model (1).pkl')

# Define feature importance values (based on your data)
feature_importances = {
    'ap_hi': 0.236094,
    'weight': 0.210486,
    'height': 0.185892,
    'age_years': 0.179032,
    'ap_lo': 0.115439,
    'cholesterol': 0.055300,
    'gender': 0.017758
}

# Convert to a pandas Series for easier manipulation
feature_importance_series = pd.Series(feature_importances).sort_values(ascending=False)

# App title and description
st.set_page_config(page_title="Cardiovascular Risk Prediction", page_icon="🫀")

# Sidebar Navigation
page = st.sidebar.radio("Select a Page", ["Home", "Predict", "Key Insights"])

if page == "Home":
    # Home Page Content
    st.markdown(
        """
        <h1 style="font-family: 'Arial', cursive; color: Black; font-size: 65px; text-align: center;">
        Cardiovascular Risk Prediction🫀
        </h1>
        <p style="font-family: 'CabinSketch Bold', cursive; color: Green ; font-size: 20px; text-align: center;">
        <i>"The greatest wealth is health"</i>
        </p>
        <p style="font-family: 'Arial', cursive; color: Black ; font-size: 20px; text-align: center;">
        This app helps you predict the likelihood of cardiovascular disease based on health metrics such as age, weight, blood pressure, cholesterol levels, and more. Enter your details to check your cardiovascular risk and understand the key factors influencing your health.
        </p>
        """,
        unsafe_allow_html=True
    )
    st.image("https://i.pinimg.com/originals/27/a1/2d/27a12d7efb1a509b27731c5b9e6a39a1.jpg", caption="Heart Health")

elif page == "Predict":
    # Prediction Page Content
    st.markdown(
        """
        <h1 style="font-family: 'Arial', cursive; color: Black; font-size: 65px; text-align: center;">
        Cardiovascular Risk Prediction🫀
        </h1>
        <p style="font-family: 'CabinSketch Bold', cursive; color: Green ; font-size: 20px; text-align: center;">
        <i>"The greatest wealth is health"</i>
        </p>
        """,
        unsafe_allow_html=True
    )

    # User input form
    with st.form("user_input_form"):
        st.subheader("Hi Dear, Enter Your Details")
        age = st.slider('Age (years)', min_value=20, max_value=80, value=50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
        ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
        ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
        cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])

        # Submit button
        submitted = st.form_submit_button("Submit")
