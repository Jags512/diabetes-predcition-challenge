import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes ML Predictor", layout="wide")

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("diabetes_model.pkl")

# Load train data to get column structure
train = pd.read_csv("train.csv")

target_col = train.columns[-1]
feature_cols = train.drop(columns=[target_col]).columns

st.title("🔬 Advanced ML Prediction App")
st.write("Built using LightGBM | Kaggle Playground S5E12")

st.divider()

# ---------------------------
# SIDEBAR INPUT FORM
# ---------------------------
st.sidebar.header("Enter Feature Values")

input_dict = {}

for col in feature_cols:
    if train[col].dtype == "object":
        options = train[col].unique()
        input_dict[col] = st.sidebar.selectbox(col, options)
    else:
        min_val = float(train[col].min())
        max_val = float(train[col].max())
        mean_val = float(train[col].mean())
        input_dict[col] = st.sidebar.slider(
            col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

predict_btn = st.sidebar.button("🚀 Predict")

# ---------------------------
# ENCODING (SAME AS TRAINING)
# ---------------------------
if predict_btn:

    input_df = pd.DataFrame([input_dict])

    # Combine with train for consistent encoding
    full = pd.concat([train.drop(columns=[target_col]), input_df], axis=0)

    for col in full.columns:
        if full[col].dtype == "object":
            full[col] = full[col].astype("category").cat.codes

    input_encoded = full.iloc[-1:].values

    prediction = model.predict(input_encoded)[0]

    # ---------------------------
    # DISPLAY RESULT
    # ---------------------------
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Value", f"{prediction:.4f}")

    with col2:
        st.metric("Model Type", "LightGBM Regressor")

    st.success("Prediction Generated Successfully!")

    st.divider()

    # ---------------------------
    # FEATURE IMPORTANCE
    # ---------------------------
    st.subheader("📈 Feature Importance")

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(importance_df["Feature"][:10],
            importance_df["Importance"][:10])
    ax.invert_yaxis()
    st.pyplot(fig)
