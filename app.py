import streamlit as st
import joblib
import numpy as np

# Saved model load karein
model = joblib.load('salary_model.pkl')

st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("💼 Employee Salary Predictor")
st.write("Machine Learning model to predict salary based on experience and test score.")

# User inputs
exp = st.number_input("Years of Experience", min_value=0.0, max_value=40.0, value=2.0, step=0.5)
score = st.number_input("Test Score (0 - 100)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

if st.button("Predict Salary"):
    input_data = np.array([[exp, score]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Salary: ₹ {prediction[0]:,.2f}")