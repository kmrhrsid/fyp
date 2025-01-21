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

    if submitted:
        # Preprocessing
        gender_encoded = 1 if gender == "Male" else 0
        cholesterol_encoded = {"Normal": 1, "Above Normal": 2, "High": 3}[cholesterol]
        bmi = round(weight / ((height / 100) ** 2), 1)

        # Categorize BMI
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "red"
        elif 18.5 <= bmi <= 24.9:
            bmi_category = "Normal"
            bmi_color = "green"
        elif 25 <= bmi <= 29.9:
            bmi_category = "Overweight"
            bmi_color = "yellow"
        else:
            bmi_category = "Obese"
            bmi_color = "red"

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

        # Display the gauge chart above the results
        st.plotly_chart(gauge_fig)

        # Results Section
        st.subheader("Prediction Results")
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
            # BMI with thumbs up or down
            thumbs_icon_bmi = "❤️" if bmi_category == "Normal" else "👎"
            st.markdown(
                """
                <div style="width: 250px; height: 250px; border: 2px solid #ccc; padding: 10px; border-radius: 10px; text-align: center; font-family: 'CabinSketch', cursive;">
                <h3 style="font-size: 18px;">BMI Category</h3>
                <p style="font-size: 24px; color: DarkSlateGray;">{}</p>
                <p style="font-size: 20px; color: {};">{}</p>
                <p style="font-size: 40px;">{}</p>
                </div>
                """.format(bmi, bmi_color, bmi_category, thumbs_icon_bmi),
                unsafe_allow_html=True
            )

elif page == "Key Insights":
    # Key Insights Page Content
    st.subheader("Key Insights into Cardiovascular Risk Factors")
    st.write("Understanding which factors play a significant role in predicting cardiovascular risk is essential for prevention. Below is a chart showing the importance of different features in predicting your cardiovascular health.")

    # Gradient colors for bars
    colors = ['#1f77b4', '#6baed6', '#9ecae1', '#d62728', '#ff9896', '#e377c2', '#ff7f0e'][:len(feature_importance_series)]

    # Create a curved barplot with Seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=feature_importance_series.values, 
        y=feature_importance_series.index, 
        palette=colors, 
        ax=ax, 
        edgecolor="black"
    )

    # Make bars slightly rounded
    for patch in ax.patches:
        patch.set_linewidth(1.5)
        patch.set_edgecolor("black")
        patch.set_capstyle("round")

    # Chart aesthetics
    ax.set_title("Feature Importance", fontsize=16, weight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    sns.despine(left=True, bottom=True)

    st.pyplot(fig)

    # Provide tips based on feature importance
    st.markdown("### Tips for Reducing Cardiovascular Risk:")
    if 'ap_hi' in feature_importance_series.index:
        st.write("- **Systolic Blood Pressure (ap_hi)**: Regular exercise, a low-sodium diet, and stress management can help.")
    if 'weight' in feature_importance_series.index:
        st.write("- **Weight**: Maintain a healthy weight through a balanced diet and regular physical activity.")
    if 'height' in feature_importance_series.index:
        st.write("- **Height (BMI)**: Focus on achieving a healthy BMI through diet and exercise.")
    if 'age_years' in feature_importance_series.index:
        st.write("- **Age**: Regular health checkups and a heart-healthy lifestyle become more crucial as you age.")
    if 'ap_lo' in feature_importance_series.index:
        st.write("- **Diastolic Blood Pressure (ap_lo)**: Monitor and manage through diet, exercise, and medication if needed.")
    if 'cholesterol' in feature_importance_series.index:
        st.write("- **Cholesterol**: Eat more fiber, reduce saturated fats, and consult a doctor if levels are high.")
    if 'gender' in feature_importance_series.index:
        st.write("- **Gender**: Risk differences may exist, but focus on modifiable factors for prevention.")


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
