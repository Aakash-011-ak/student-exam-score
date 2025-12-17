import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Student Exam Score Prediction", page_icon="📘")

# Load model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Safety check
if isinstance(model, pd.DataFrame):
    st.error("❌ saldf.pkl contains a DataFrame, not a trained ML model.")
    st.stop()

st.title("📘 Student Exam Score Prediction")

study_hours = st.number_input("Study Hours per Day", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous_score = st.number_input("Previous Exam Score", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0)

if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, previous_score, sleep_hours]])
    prediction = model.predict(input_data)
    st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")


