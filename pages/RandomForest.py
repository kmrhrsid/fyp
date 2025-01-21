# Load necessary libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt

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

    # Display feature importance
    st.subheader("Risk Factor Insights")
    feature_importances = pd.Series(model.feature_importances_, index=expected_features).sort_values(ascending=False)
    top_features = feature_importances.head(5)

    # Bar chart of feature importances
    fig, ax = plt.subplots()
    top_features.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Top Contributing Factors")
    ax.set_ylabel("Feature Importance")
    st.pyplot(fig)

    # Provide tips for reducing risk
    st.markdown("### Tips for Reducing Cardiovascular Risk:")
    if 'age_years' in top_features.index:
        st.write("- **Age**: Maintain regular health checkups and a heart-healthy lifestyle as you age.")
    if 'ap_hi' in top_features.index or 'ap_lo' in top_features.index:
        st.write("- **Blood Pressure**: Manage blood pressure through a low-sodium diet, exercise, and medication if needed.")
    if 'cholesterol' in top_features.index:
        st.write("- **Cholesterol**: Adopt a diet low in saturated fats and cholesterol; include more fiber.")
    if 'weight' in top_features.index or 'height' in top_features.index:
        st.write("- **Weight/BMI**: Achieve a healthy BMI through balanced nutrition and physical activity.")

    # Motivational quotes
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
