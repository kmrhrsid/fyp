import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Go to", ["Home", "Prediction", "Insights"])

# Page 1: Home
if pages == "Home":
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

# Page 2: Prediction
elif pages == "Prediction":
    st.subheader("Hi Dear, Enter Your Details")
    with st.form("user_input_form"):
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

        # Results Section
        st.subheader("Prediction Results")
        col1, col2 = st.columns([1, 1])
        with col1:
            thumbs_icon_risk = "❤️" if risk_percentage <= 50 else "👎"
            st.markdown(
                f"""
                <div style="text-align: center;">
                <h3>Cardiovascular Risk (%)</h3>
                <p style="font-size: 24px;">{risk_percentage}%</p>
                <p>{thumbs_icon_risk}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            thumbs_icon_bmi = "❤️" if 18.5 <= bmi <= 24.9 else "👎"
            st.markdown(
                f"""
                <div style="text-align: center;">
                <h3>BMI (Body Mass Index)</h3>
                <p style="font-size: 24px;">{bmi}</p>
                <p>{thumbs_icon_bmi}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.plotly_chart(gauge_fig)

# Page 3: Insights
elif pages == "Insights":
    st.subheader("Risk Factor Insights")
    colors = ['#1f77b4', '#6baed6', '#9ecae1', '#d62728', '#ff9896', '#e377c2', '#ff7f0e']

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=feature_importance_series.values,
        y=feature_importance_series.index,
        palette=colors,
        ax=ax,
        linewidth=2,
        edgecolor="black"
    )
    ax.set_title("Feature Importance", fontsize=16, weight='bold')
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    st.pyplot(fig)
    
    st.markdown("### Tips for Reducing Cardiovascular Risk:")
    st.write("- **Systolic Blood Pressure (ap_hi)**: Regular exercise, a low-sodium diet, and stress management can help.")
    st.write("- **Weight**: Maintain a healthy weight through a balanced diet and regular physical activity.")
    st.write("- **Cholesterol**: Eat more fiber, reduce saturated fats, and consult a doctor if levels are high.")
