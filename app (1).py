import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(page_title="Behaviour Analytics Dashboard", layout="wide")
st.title("Behaviour Analytics Risk Dashboard")

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_behavior_500.csv")

df = load_data()

# ----------------------------
# Dataset preview
# ----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# ----------------------------
# Risk distribution
# ----------------------------
st.subheader("Risk Distribution")
risk_counts = df["high_risk_flag"].value_counts()
st.bar_chart(risk_counts)

# ----------------------------
# Feature importance (SHAP summary)
# ----------------------------
st.subheader("Top Behaviour Drivers (Global)")
shap_img = Image.open("shap_summary.png")
st.image(shap_img, caption="SHAP Summary Plot", use_column_width=True)

# ----------------------------
# Interactive filtering
# ----------------------------
st.subheader("Filter Data")

# Filter by risk_score
risk_min, risk_max = st.slider(
    "Filter by Risk Score", 
    float(df["risk_score"].min()), 
    float(df["risk_score"].max()), 
    (float(df["risk_score"].min()), float(df["risk_score"].max()))
)

filtered_df = df[(df["risk_score"] >= risk_min) & (df["risk_score"] <= risk_max)]

st.write(f"Showing {len(filtered_df)} rows")
st.dataframe(filtered_df)

# ----------------------------
# Optional: scatter plot between features
# ----------------------------
st.subheader("Feature Relationships")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if len(numeric_cols) >= 2:
    x_feature = st.selectbox("X-axis feature", numeric_cols, index=0)
    y_feature = st.selectbox("Y-axis feature", numeric_cols, index=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(filtered_df[x_feature], filtered_df[y_feature], c=filtered_df["risk_score"], cmap="coolwarm")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"{y_feature} vs {x_feature} (colored by risk score)")
    st.pyplot(fig)
else:
    st.info("Not enough numeric features for scatter plot.")
