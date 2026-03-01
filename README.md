**Behaviour Analytics : Temporal behaviour analytics for productivity insight**

An Explainable behavioural risk prediction system that detects early signs of burnout, disengagement, and digital overexposure using machine learning and interpretable analytics.


**Project Overview**

This project analyzes behavioural, lifestyle, and digital activity patterns to predict whether an individual falls under:

- 🟢 Low Risk  
- 🟡 Medium Risk  
- 🔴 High Risk  

The system uses machine learning (Random Forest) combined with SHAP explainability to provide both predictions and reasoning behind those predictions.

The goal is to move from reactive intervention to proactive behavioural intelligence.


Problem Statement

Organizations and institutions often detect burnout or disengagement only after productivity drops.

This project aims to:

- Identify behavioural risk early  
- Detect digital overexposure patterns    
- Support data-driven HR or academic decisions  



**Dataset**

**Source:** HuggingFace  
**Dataset Name:** tarekmasryo/digital-lifestyle-benchmark-dataset  

The dataset includes:

### Demographic Features
- age  
- gender  
- region  
- income_level  
- education_level  
- daily_role  

### Digital Behaviour Features
- device_hours_per_day  
- phone_unlocks  
- notifications_per_day  
- social_media_mins  

### Lifestyle Features
- sleep_hours  
- physical_activity_days  

### Well-being Indicators
- stress_level  
- anxiety_score  
- depression_score  
- focus_score  
- happiness_score  

### Target Variable
- high_risk_flag  

---

##  Feature Engineering

Additional behavioural indicators were created:

- 7-day rolling averages  
- Screen time volatility  
- Normalized behavioural metrics  
- Composite behavioural risk score  

These engineered features help capture behavioural trends rather than just static values.



##  Machine Learning Model

**Algorithm Used:** Random Forest Classifier  

### Why Random Forest?
- Handles mixed data types  
- Captures non-linear relationships  
- Provides strong baseline performance  
- Supports feature importance analysis  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

---

##  Explainability with SHAP

SHAP (SHapley Additive Explanations) was used to:

- Identify top risk-driving features  
- Explain individual predictions  
- Provide model transparency  

This ensures the system is interpretable and suitable for real-world decision-making.

---

##  Streamlit Dashboard

An interactive dashboard was built using Streamlit to:

- Visualize risk distribution    
- Show behavioural trend insights  
- Provide real-time model predictions  

---

##  Use Cases

- HR analytics for burnout detection  
- Student behavioural risk monitoring  
- Workforce productivity analysis  
- Digital well-being assessment  

---

##  Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- SHAP  
- Matplotlib  
- Seaborn  
- Streamlit  

---

##  Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── data/
│   └── dataset.csv
├── app.py
├── model_training.ipynb
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Add real-time behavioural monitoring  
- Integrate API-based predictions  
- Enhance model performance using XGBoost or LightGBM  
- Add user authentication layer  




## 📜 License

This project is for academic and research purposes.
