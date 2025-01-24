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

# App title and description
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

# Streamlit Sidebar for navigation between pages
page = st.sidebar.radio("Select a Page", ("Home", "Prediction", "Risk Insights"))

# --- Page 1: Home ---
if page == "Home":
    st.markdown("### Welcome to the Cardiovascular Risk Prediction app!")
    st.write("""
        This app helps to predict the cardiovascular risk based on various health parameters such as age, 
        blood pressure, cholesterol level, and more. You can enter your health information and receive 
        an estimate of your risk.
    """)
    st.markdown("### How it Works:")
    st.write("""
        1. **Input**: Enter details like age, height, weight, etc.
        2. **Prediction**: The model predicts your risk of cardiovascular disease based on your inputs.
        3. **Risk Insights**: Learn about the importance of different factors and tips to reduce your risk.
    """)

# --- Page 2: Prediction ---
if page == "Prediction":
    with st.form("user_input_form"):
        st.subheader("Enter Your Health Information")
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
        st.session_state.risk_percentage = risk_percentage
        st.session_state.bmi = bmi
        st.session_state.age = age

        # Displaying the results
        st.success("Prediction successful!")
        st.markdown(f"### Your Cardiovascular Risk: {risk_percentage}%")
        st.markdown(f"### Your BMI: {bmi}")

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

# --- Page 3: Risk Insights ---
if page == "Risk Insights":
    st.subheader("Risk Factor Insights")
    st.write("The following chart shows the relative importance of each feature in predicting cardiovascular risk:")

    # Gradient colors for bars
    colors = ['#1f77b4', '#6baed6', '#9ecae1', '#d62728', '#ff9896', '#e377c2', '#ff7f0e']

    # Create a horizontal bar chart with bold edges
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=feature_importance_series.values,
        y=feature_importance_series.index,
        palette=colors,
        ax=ax,
        linewidth=3,  # Make edges bold
        edgecolor="black"
    )

    # Set titles and labels with bold styling
    ax.set_title("Feature Importance", fontsize=16, weight='bold')
    ax.set_xlabel("Importance Score", fontsize=12, weight='bold')
    ax.set_ylabel("Features", fontsize=12, weight='bold')

    # Format the bars with rounded edges and bold borders
    for bar in ax.patches:
        bar.set_linewidth(3)
        bar.set_edgecolor("black")
        bar.set_capstyle('round')

    st.pyplot(fig)

    # Provide tips based on feature importance
    st.markdown("### Tips for Reducing Cardiovascular Risk:")
    if 'ap_hi' in feature_importance_series.index:
        st.write("- *Systolic Blood Pressure (ap_hi)*: Regular exercise, a low-sodium diet, and stress management can help.")
    if 'weight' in feature_importance_series.index:
        st.write("- *Weight*: Maintain a healthy weight through a balanced diet and regular physical activity.")
    if 'height' in feature_importance_series.index:
        st.write("- *Height (BMI)*: Focus on achieving a healthy BMI through diet and exercise.")
    if 'age_years' in feature_importance_series.index:
        st.write("- *Age*: Regular health checkups and a heart-healthy lifestyle become more crucial as you age.")
    if 'ap_lo' in feature_importance_series.index:
        st.write("- *Diastolic Blood Pressure (ap_lo)*: Monitor and manage through diet, exercise, and medication if needed.")
    if 'cholesterol' in feature_importance_series.index:
        st.write("- *Cholesterol*: Eat more fiber, reduce saturated fats, and consult a doctor if levels are high.")
    if 'gender' in feature_importance_series.index:
        st.write("- *Gender*: Risk differences may exist, but focus on modifiable factors for prevention.")


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
