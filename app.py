import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Custom CSS (Dark Production UI)
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:bold;
}
.result-box {
    padding:20px;
    border-radius:10px;
    font-size:20px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<div class='big-title'>🩺 Diabetes Risk Predictor</div>", unsafe_allow_html=True)
st.write("Enter patient health details to predict diabetes probability using ML.")

st.divider()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Patient Details")

gender = st.sidebar.radio("Gender", ["Male", "Female"])
pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
age = st.sidebar.slider("Age", 10, 90, 30)
glucose = st.sidebar.slider("Glucose Level", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 150, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

predict_btn = st.sidebar.button("🔍 Predict Diabetes")

# Convert gender
gender = 1 if gender == "Male" else 0

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("diabetes_model.pkl")

# -----------------------------
# MAIN CONTENT
# -----------------------------
if predict_btn:

    input_data = np.array([[pregnancies, glucose, bp, skin,
                            insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Glucose", glucose)

    with col2:
        st.metric("BMI", bmi)

    with col3:
        st.metric("Age", age)

    st.divider()

    # Result Box
    if prediction == 1:
        st.markdown(
            "<div class='result-box' style='background-color:#ff4b4b;color:white;'>⚠️ DIABETES DETECTED</div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='result-box' style='background-color:#00cc66;color:white;'>✅ NO DIABETES</div>",
            unsafe_allow_html=True)

    st.divider()

    # Probability Section
    st.subheader("Prediction Probability")

    col4, col5 = st.columns(2)

    with col4:
        st.metric("Diabetes Probability", f"{probability[1]*100:.2f}%")

    with col5:
        st.metric("No Diabetes Probability", f"{probability[0]*100:.2f}%")

    # Probability Chart
    fig, ax = plt.subplots()
    ax.bar(["No Diabetes", "Diabetes"], probability)
    st.pyplot(fig)
