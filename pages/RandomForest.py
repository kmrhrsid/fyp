import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Function to encode and apply background images dynamically
def set_background(image_path):
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

# Load the trained model
model = joblib.load('random_forest_model (1).pkl')

# Define feature importance values
feature_importances = {
    'ap_hi': 0.236094,
    'weight': 0.210486,
    'height': 0.185892,
    'age_years': 0.179032,
    'ap_lo': 0.115439,
    'cholesterol': 0.055300,
    'gender': 0.017758
}
feature_importance_series = pd.Series(feature_importances).sort_values(ascending=False)

# Home Page
def homepage():
    set_background("hp3 (1).jpg")
    st.markdown(
        """
        <h1 style="font-family: 'Arial', cursive; color:#eca714 ; font-size: 70px; text-align: center;">
        Cardiovascular Risk Prediction
        </h1>
        <p style="font-family: 'CabinSketch Bold', cursive; color:white ; font-size: 40px; text-align: center;">
        <i>"The greatest wealth is health"</i>
        </p>
        <p style="font-family: 'Arial', cursive; color:white; font-size: 15px; text-align: center;">
        Welcome to the Cardiovascular Risk Prediction system. Use this app to predict your cardiovascular disease risk and gain insights into your health.
        </p>
        """,
        unsafe_allow_html=True
    )

# Prediction Page
def prediction_page():
    set_background("hp44.jpg")
    st.markdown(
        """
        <h2 style="color: white;">Enter Your Details for Prediction</h2>
        """, unsafe_allow_html=True
    )
    with st.form("user_input_form"):
        age = st.slider('Age (years)', min_value=20, max_value=80, value=50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
        ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
        ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
        cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])
        submitted = st.form_submit_button("Submit")

    if submitted:
        gender_encoded = 1 if gender == "Male" else 0
        cholesterol_encoded = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]
        bmi = round(weight / ((height / 100) ** 2), 1)

        input_data = pd.DataFrame({
            'age_years': [age],
            'gender': [gender_encoded],
            'height': [height],
            'weight': [weight],
            'ap_lo': [ap_lo],
            'ap_hi': [ap_hi],
            'cholesterol': [cholesterol_encoded],
        })

        expected_features = model.feature_names_in_
        input_data = input_data[expected_features]

        prediction = model.predict_proba(input_data)[0][1]
        risk_percentage = round(prediction * 100, 1)

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
            },
            title={'text': "Risk Percentage"}
        ))

        st.subheader("Prediction Results")
        st.plotly_chart(gauge_fig)

# Insights Page
def insights_page():
    set_background("hp44.jpg")
    st.markdown(
        """
        <h2 style="color: white;">Risk Factor Insights</h2>
        <p style="color: white;">Below is a feature importance chart to help you understand the most influential factors in predicting cardiovascular risk:</p>
        """,
        unsafe_allow_html=True
    )

    colors = ['#1f77b4', '#6baed6', '#9ecae1', '#d62728', '#ff9896', '#e377c2', '#ff7f0e']
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=feature_importance_series.values,
        y=feature_importance_series.index,
        palette=colors,
        ax=ax
    )
    ax.set_title("Feature Importance", fontsize=16, weight='bold')
    ax.set_xlabel("Importance Score", fontsize=12, weight='bold')
    ax.set_ylabel("Features", fontsize=12, weight='bold')
    st.pyplot(fig)

    st.markdown(
        """
        <h3 style="color: white;">Tips for Reducing Cardiovascular Risk:</h3>
        <ul style="color: white;">
            <li>Maintain healthy blood pressure levels with regular exercise and a low-sodium diet.</li>
            <li>Manage your weight through balanced nutrition and physical activity.</li>
            <li>Monitor your cholesterol and consult a doctor if levels are high.</li>
        </ul>
        """, unsafe_allow_html=True
    )

# Main Navigation
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home", "Prediction", "Insights"])
    
    if page == "Home":
        homepage()
    elif page == "Prediction":
        prediction_page()
    elif page == "Insights":
        insights_page()

if __name__ == "__main__":
    main()
