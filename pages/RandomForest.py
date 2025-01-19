import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import plotly.express as px

# Load the trained model
model = joblib.load('random_forest_model (1).pkl')
print("Model loaded successfully!")

# Main title with customized font size, style, and centered
st.markdown(
    """
    <h1 style="font-family: HelloFirstieBig; color: DarkSlateGray; font-size: 40px; text-align: center;">
    Cardiovascular Risk Prediction
    </h1>
    """,
    unsafe_allow_html=True
)

# Quote with a different font, style, and centered
st.markdown(
    """
    <p style="font-family: Georgia; color: Green ; font-size: 20px; text-align: center;">
    <i>The greatest wealth is health</i>
    </p>
    """,
    unsafe_allow_html=True
)

# User input
age = st.slider('Age (years)', min_value=20, max_value=80, value=50)
gender = st.selectbox("Gender", ["Female", "Male"])
height = st.number_input("Height (cm)", min_value=100, max_value=250)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
ap_lo = st.number_input("Low Blood Pressure (mmHg)", min_value=0)
ap_hi = st.number_input("High Blood Pressure (mmHg)", min_value=0)
cholesterol = st.selectbox("Cholesterol level", ["Normal", "Above Normal", "High"])

# Preprocessing
gender_encoded = 1 if gender == "Male" else 0
cholesterol_encoded = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]

# Create raw input DataFrame
input_data = pd.DataFrame({
    'age_years': [age],
    'gender': [gender_encoded],
    'height': [height],
    'weight': [weight],
    'ap_lo': [ap_lo],
    'ap_hi': [ap_hi],
    'cholesterol': [cholesterol_encoded],
})

# Align features
expected_features = model.feature_names_in_
input_data = input_data[expected_features]

# Predict the cardiovascular risk
if st.button("Submit"):
    prediction = model.predict_proba(input_data)[0][1]  # Probability of cardiovascular risk
    risk_percentage = round(prediction * 100, 1)

    # Gauge chart visualization
    fig = go.Figure(go.Indicator(
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

    st.plotly_chart(fig)

    # Display additional metrics
    st.subheader("Prediction Summary")
    st.metric("Risk of Cardiovascular Disease", f"{risk_percentage}%", delta="High" if risk_percentage > 50 else "Low")
    st.metric("BMI (Body Mass Index)", f"{round(weight / (height / 100) ** 2, 1)} kg/m²")

    # Add a line chart for risk
    risk_data_line = pd.DataFrame({
        "Risk Type": ["Cardiovascular Risk", "No Risk"],
        "Percentage": [risk_percentage, 100 - risk_percentage]
    })

    # Plotting line chart
    line_fig = px.line(
        risk_data_line,
        x="Risk Type",
        y="Percentage",
        markers=True,
        title="Risk of Cardiovascular Disease",
        labels={"Percentage": "Percentage (%)"},
        line_shape="spline"  # Smooth the line for better appearance
    )

    # Customize the line chart's appearance
    line_fig.update_traces(line=dict(color="red", width=3), marker=dict(size=10))

    st.plotly_chart(line_fig)

    # Motivational Quote
    st.markdown(
        """
        <p style="font-family: Georgia; color: Green ; font-size: 60px; text-align: center;">
        <i>من جدّ وجد</i>
        </p>
        """,
        unsafe_allow_html=True
    )

    # Quote with a meaning 
    st.markdown(
        """
        <p style="font-family: Georgia; color: Green ; font-size: 40px; text-align: center;">
        <i>"Whoever works really hard, will succeed"</i>
        </p>
        """,
        unsafe_allow_html=True
    )
