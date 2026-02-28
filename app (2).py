
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(page_title="Behaviour Analytics Dashboard")
st.title("Behaviour Analytics Risk Dashboard")

# ----------------------------
# Load dataset and model safely
# ----------------------------
@st.cache_data
def load_data(sample_size=1000):
    df = pd.read_csv("final_behavior_dataset.csv")
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load("rf_model.pkl")
    except:
        st.warning("Model file not found. Skipping model loading.")
        return None

df = load_data()
rf_model = load_model()

# ----------------------------
# Risk distribution chart
# ----------------------------
st.subheader("Risk Distribution")
if "high_risk_flag" in df.columns:
    risk_counts = df["high_risk_flag"].value_counts()
    st.bar_chart(risk_counts)
else:
    st.warning("'high_risk_flag' column not found in dataset.")

# ----------------------------
# Dataset preview
# ----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())
