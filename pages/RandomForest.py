import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# Load the trained model
model = joblib.load('random_forest_model (1).pkl')

# File to save user health data
DATA_FILE = "user_health_data.csv"

# Initialize health data file if it doesn't exist
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=['date', 'age_years', 'gender', 'height', 'weight', 'ap_lo', 'ap_hi', 'cholesterol', 'risk_percentage', 'bmi']).to_csv(DATA_FILE, index=False)

# App title and description
st.markdown(
    """
    <h1 style="font-family: 'Arial', cursive; color: Black; font-size: 65px; text-align: center;">
    Cardiovascular Risk Prediction & Health Tracking
    </h1>
    <p style="font-family: 'CabinSketch Bold', cursive; color: Green ; font-size: 20px; text-align: center;">
    <i>"Your health journey starts here."</i>
    </p>
    """,
    unsafe_allow_html=True
)

# User input form
with st.form("user_input_form"):
    st.subheader("Enter Your Health Metrics")
    age = st.slider('Age (years)', min_value=20, max_value=80, value=50)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (cm)", min_value=100, max_value=250)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
    ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
    ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
    cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])
    
    # Submit button
    submitted = st.form_submit_button("Submit")

if submitted:
    # Preprocessing
    gender_encoded = 1 if gender == "Male" else 0
    cholesterol_encoded = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]
    bmi = round(weight / ((height / 100) ** 2), 1)

    # Create input DataFrame
    input_data = pd.DataFrame({
        'age_years': [age],
        'gender': [gender_encoded],
        'height': [height],
        'weight': [weight],
        'ap_lo': [ap_lo],
        'ap_hi': [ap_hi],
        'cholesterol': [cholesterol_encoded],
    })

    # Align with model features
    expected_features = model.feature_names_in_
    input_data = input_data[expected_features]

    # Prediction
    prediction = model.predict_proba(input_data)[0][1]  # Probability of cardiovascular risk
    risk_percentage = round(prediction * 100, 1)

    # Save health metrics to CSV
    user_data = pd.DataFrame({
        'date': [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
        'age_years': [age],
        'gender': [gender],
        'height': [height],
        'weight': [weight],
        'ap_lo': [ap_lo],
        'ap_hi': [ap_hi],
        'cholesterol': [cholesterol],
        'risk_percentage': [risk_percentage],
        'bmi': [bmi],
    })
    user_data.to_csv(DATA_FILE, mode='a', header=False, index=False)

    # Results Section
    st.subheader("Prediction Results")
    st.write(f"Your cardiovascular risk is estimated to be **{risk_percentage}%**.")
    st.write(f"Your BMI is **{bmi}**.")

    # Gauge chart visualization
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
    st.plotly_chart(gauge_fig)

# Health Tracking Dashboard
st.subheader("Your Health Metrics Over Time")

if os.path.exists(DATA_FILE):
    health_data = pd.read_csv(DATA_FILE)

    if not health_data.empty:
        # Line charts for tracking
        metric = st.selectbox("Select a metric to track:", ['risk_percentage', 'bmi', 'ap_lo', 'ap_hi', 'weight'])

        fig = px.line(
            health_data,
            x='date',
            y=metric,
            title=f'Trend of {metric.replace("_", " ").title()} Over Time',
            labels={'date': 'Date', metric: metric.replace("_", " ").title()},
            markers=True
        )
        st.plotly_chart(fig)

        # Display raw data
        with st.expander("View Raw Data"):
            st.dataframe(health_data)
    else:
        st.write("No health data found. Submit your health metrics to start tracking!")
else:
    st.write("No health data file found. Submit your health metrics to start tracking!")
