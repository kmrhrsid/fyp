import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def show_insights():
    st.subheader("Risk Factor Insights")
    st.write("The following chart shows the relative importance of each feature:")
    
    feature_importance_series = pd.Series(feature_importances).sort_values(ascending=False)
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
    st.write("- **Blood Pressure**: Exercise, diet, and medication if necessary.")
    st.write("- **Weight**: Maintain a healthy weight with balanced diet and activity.")
    st.write("- **Cholesterol**: Eat more fiber, avoid saturated fats.")

show_insights()
