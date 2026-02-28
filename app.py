import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Behaviour Analytics Dashboard")
st.title("Behaviour Analytics Risk Dashboard")

# Load dataset
df = pd.read_csv("final_behavior_500.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Risk distribution
st.subheader("Risk Distribution")
st.bar_chart(df["high_risk_flag"].value_counts())

# Feature importance (SHAP summary)
st.subheader("Top Behaviour Drivers (Global)")
shap_img = Image.open("shap_summary.png")
st.image(shap_img, caption="SHAP Summary Plot", use_column_width=True)

# Interactive filtering
st.subheader("Filter Data")
risk_min, risk_max = st.slider("Filter by Risk Score", 
                               float(df["risk_score"].min()), 
                               float(df["risk_score"].max()), 
                               (float(df["risk_score"].min()), float(df["risk_score"].max())))
filtered_df = df[(df["risk_score"] >= risk_min) & (df["risk_score"] <= risk_max)]
st.dataframe(filtered_df)
