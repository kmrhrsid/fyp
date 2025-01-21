import streamlit as st
import pandas as pd
import joblib
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
import plotly.graph_objects as go

# Load the trained model
model = joblib.load('random_forest_model (1).pkl')

# App title and description
st.sidebar.title("Cardiovascular Risk Prediction")
page = st.sidebar.selectbox("Navigate", ["Home", "Predict Risk", "Key Insights"])

# Home Page
if page == "Home":
    st.markdown(
        """
        <h1 style="font-family: 'Arial', cursive; color: Black; font-size: 65px; text-align: center;">
        Cardiovascular Risk Prediction 🫀
        </h1>
        <p style="font-family: 'CabinSketch Bold', cursive; color: Green ; font-size: 20px; text-align: center;">
        <i>"The greatest wealth is health"</i>
        </p>
        """,
        unsafe_allow_html=True
    )
    st.write(
        "This app predicts cardiovascular risk based on your health metrics. "
        "Navigate to **Predict Risk** to estimate your risk, or explore **Key Insights** for health metric analysis."
    )

# Prediction Page
elif page == "Predict Risk":
    st.title("Cardiovascular Risk Prediction")
    with st.form("user_input_form"):
        st.subheader("Enter Your Details")
        age = st.slider('Age (years)', min_value=20, max_value=80, value=50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0, value=80)
        ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0, value=120)
        cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "High"])
        
        # Real-time BMI Calculation and Interpretation
        bmi = round(weight / ((height / 100) ** 2), 1)
        st.write(f"Your BMI: **{bmi}**")
        if bmi < 18.5:
            st.warning("You are underweight.")
        elif 18.5 <= bmi <= 24.9:
            st.success("You are in the healthy weight range.")
        elif 25 <= bmi <= 29.9:
            st.warning("You are overweight.")
        else:
            st.error("You are obese.")
        
        # Submit Button
        submitted = st.form_submit_button("Submit")

    if submitted:
        # Preprocessing
        gender_encoded = 1 if gender == "Male" else 0
        cholesterol_encoded = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]
        
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

        # Gauge Chart for Risk Percentage
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

# Key Insights Page
elif page == "Key Insights":
    st.title("Key Insights")
    st.markdown(
        "Explore the relationship between Body Mass Index (BMI) and cardiovascular risk."
    )

    # Sample Data for Visualization
    data = {
        'BMI': [18, 22, 27, 30, 35],
        'Risk': [10, 20, 50, 70, 90]
    }
    source = ColumnDataSource(data)

    # Create Bokeh Plot
    bokeh_plot = figure(
        title="BMI vs Cardiovascular Risk",
        x_axis_label="BMI",
        y_axis_label="Risk Percentage",
        plot_height=400,
        plot_width=600,
        tools="pan,box_zoom,reset"
    )
    bokeh_plot.line('BMI', 'Risk', source=source, line_width=2, color="blue", legend_label="Risk")
    bokeh_plot.circle('BMI', 'Risk', size=8, source=source, color="red", legend_label="Data Points")
    bokeh_plot.legend.location = "top_left"

    # Render Bokeh Plot in Streamlit
    st.bokeh_chart(bokeh_plot)

    st.markdown(
        """
        **Insights:**
        - A healthy BMI (18.5–24.9) is associated with lower cardiovascular risk.
        - Higher BMI correlates with increased risk percentages.
        """
    )
