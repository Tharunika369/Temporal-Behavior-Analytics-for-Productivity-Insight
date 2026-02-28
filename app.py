import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Behaviour Analytics Dashboard")

st.title("Behaviour Analytics Risk Dashboard")

# Load dataset
df = pd.read_csv("final_behavior_dataset.csv")

# Load model
rf_model = joblib.load("rf_model.pkl")

# Risk distribution
st.subheader("Risk Distribution")
risk_counts = df["high_risk_flag"].value_counts()
st.bar_chart(risk_counts)

# Feature importance (global)
st.subheader("Top Behaviour Drivers (Global)")

explainer = shap.TreeExplainer(rf_model)

X = df.drop(columns=["id", "high_risk_flag"])
shap_values = explainer(X.sample(200))

fig, ax = plt.subplots()
shap.summary_plot(shap_values.values[:, :, 2], X.sample(200), show=False)
st.pyplot(fig)

st.subheader("Dataset Preview")
st.dataframe(df.head())
