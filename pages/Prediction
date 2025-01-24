import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load the trained model
model = joblib.load('random_forest_model (1).pkl')

# Define prediction logic here
def predict_risk():
    st.subheader("Enter Your Details")
    age = st.slider('Age (years)', 20, 80, 50)
    gender = st.selectbox("Gender", ["Female", "Male"])
    height = st.number_input("Height (cm)", 100, 250)
    weight = st.number_input("Weight (kg)", 30, 200)
    ap_lo = st.number_input("Low Blood Pressure (mmHg)", 0)
    ap_hi = st.number_input("High Blood Pressure (mmHg)", 0)
    cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])
    
    if st.button("Submit"):
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
        input_data = input_data[model.feature_names_in_]
        prediction = model.predict_proba(input_data)[0][1]
        risk_percentage = round(prediction * 100, 1)
        
        # Display results
        st.success(f"Cardiovascular Risk: {risk_percentage}%")
        st.plotly_chart(go.Figure(go.Indicator(
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
        )))

predict_risk()
