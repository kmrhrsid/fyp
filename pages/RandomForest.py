import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib

# Load the trained model
model = joblib.load('random_forest_model (1).pkl')

# App title and description
st.markdown(
    """
    <h1 style="font-family: 'Arial', cursive; color: Black; font-size: 65px; text-align: center;">
    Cardiovascular Risk Prediction
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
    
    # Color Picker
    color = st.color_picker("Pick a color for your result", '#FF5733')  # Default color is orange
    
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

    # Results Section
    st.subheader("Prediction Results")

    # Gauge chart visualization
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percentage,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},  # Use selected color
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "red"}
            ],
        },
        title={'text': "Risk Percentage"}
    ))
    st.plotly_chart(gauge_fig)

    # Display metrics in boxes with relevant size
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Cardiovascular Risk with thumbs up or down
        thumbs_icon_risk = "❤️" if risk_percentage <= 50 else "👎"
        st.markdown(
            """
            <div style="width: 250px; height: 250px; border: 2px solid #ccc; padding: 10px; border-radius: 10px; text-align: center; font-family: 'CabinSketch', cursive;">
            <h3 style="font-size: 18px;">Cardiovascular Risk (%)</h3>
            <p style="font-size: 24px; color: DarkSlateGray;">{}</p>
            <p style="font-size: 20px; color: {};">{}</p>
            <p style="font-size: 40px;">{}</p>
            </div>
            """.format(risk_percentage, "red" if risk_percentage > 50 else "green", "High" if risk_percentage > 50 else "Low", thumbs_icon_risk),
            unsafe_allow_html=True
        )
    
    with col2:
        # BMI with thumbs up for healthy
        thumbs_icon_bmi = "❤️" if 18.5 <= bmi <= 24.9 else "👎"
        st.markdown(
            """
            <div style="width: 250px; height: 250px; border: 2px solid #ccc; padding: 10px; border-radius: 10px; text-align: center; font-family: 'CabinSketch', cursive;">
            <h3 style="font-size: 18px;">BMI (Body Mass Index)</h3>
            <p style="font-size: 24px; color: DarkSlateGray;">{}</p>
            <p style="font-size: 20px; color: {};">{}</p>
            <p style="font-size: 40px;">{}</p>
            </div>
            """.format(bmi, "green" if 18.5 <= bmi <= 24.9 else "red", "Healthy" if 18.5 <= bmi <= 24.9 else "Unhealthy", thumbs_icon_bmi),
            unsafe_allow_html=True
        )

    # Motivational Quotes
    st.markdown(
        """
        <p style="font-family: 'CabinSketch', cursive; color: Green ; font-size: 60px; text-align: center;">
        <i>من جدّ وجد</i>
        </p>
        <p style="font-family: 'Arial', cursive; color: Black ; font-size: 40px; text-align: center;">
        <i>"Whoever works really hard, will succeed"</i>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Feedback Section
    st.subheader("Your Feedback")
    feedback = st.text_area("Please leave your feedback on the application:", height=150)
    if st.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter some feedback.")
