import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(
    page_title="CardioVision",
    page_icon="🫀",
    layout="wide"
)


model = joblib.load("cardiovision_model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("heart.csv")


st.title("🫀 CardioVision")
st.markdown("#### AI-Powered Heart Disease Risk Prediction")
st.markdown("---")


st.sidebar.header("📋 Enter Patient Details")

age = st.sidebar.slider("Age", 20, 80, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", [
    "0 - Typical Angina",
    "1 - Atypical Angina",
    "2 - Non-anginal Pain",
    "3 - Asymptomatic"
])
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.sidebar.selectbox("Resting ECG Results", [
    "0 - Normal",
    "1 - ST-T Wave Abnormality",
    "2 - Left Ventricular Hypertrophy"
])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST", [
    "0 - Upsloping",
    "1 - Flat",
    "2 - Downsloping"
])
ca = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
thal = st.sidebar.selectbox("Thalassemia", [
    "1 - Normal",
    "2 - Fixed Defect",
    "3 - Reversible Defect"
])

sex_val = 1 if sex == "Male" else 0
cp_val = int(cp[0])
fbs_val = 1 if fbs == "Yes" else 0
restecg_val = int(restecg[0])
exang_val = 1 if exang == "Yes" else 0
slope_val = int(slope[0])
thal_val = int(thal[0])

input_data = np.array([[age, sex_val, cp_val, trestbps, chol,
                        fbs_val, restecg_val, thalach, exang_val,
                        oldpeak, slope_val, ca, thal_val]])

input_scaled = scaler.transform(input_data)


prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0]


col1, col2, col3 = st.columns(3)

with col1:
    risk = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOW RISK"
    st.metric("Prediction", risk)

with col2:
    st.metric("Heart Disease Probability", f"{probability[1]*100:.1f}%")

with col3:
    st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")

st.markdown("---")


fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=probability[1] * 100,
    title={"text": "Risk Score (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "darkred"},
        "steps": [
            {"range": [0, 40], "color": "lightgreen"},
            {"range": [40, 70], "color": "yellow"},
            {"range": [70, 100], "color": "salmon"},
        ],
    }
))

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.subheader("🎯 Risk Gauge")
    st.plotly_chart(fig_gauge, use_container_width=True)


with col_g2:
    st.subheader("📊 Top Factors Affecting Prediction")
    feature_names = ["age", "sex", "cp", "trestbps", "chol",
                     "fbs", "restecg", "thalach", "exang",
                     "oldpeak", "slope", "ca", "thal"]
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True).tail(8)

    fig_feat = px.bar(feat_df, x="Importance", y="Feature",
                      orientation="h", color="Importance",
                      color_continuous_scale="Reds")
    st.plotly_chart(fig_feat, use_container_width=True)

st.markdown("---")


st.subheader("📈 Dataset Insights")
col_d1, col_d2 = st.columns(2)

with col_d1:
    fig_age = px.histogram(df, x="age", color="target",
                           title="Age Distribution by Heart Disease",
                           labels={"target": "Has Disease"},
                           color_discrete_map={0: "green", 1: "red"})
    st.plotly_chart(fig_age, use_container_width=True)

with col_d2:
    fig_chol = px.scatter(df, x="chol", y="thalach",
                          color="target",
                          title="Cholesterol vs Max Heart Rate",
                          labels={"target": "Has Disease"},
                          color_discrete_map={0: "green", 1: "red"})
    st.plotly_chart(fig_chol, use_container_width=True)


st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only. Always consult a qualified doctor for medical advice.")
