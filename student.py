import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Student Exam Score Prediction", page_icon="📘")

# ================= LOAD MODEL =================
with open("student_model.pkl", "rb") as f:
    model = pickle.load(f)

# Try loading scaler (if used in training)
scaler = None
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    pass

st.title("📘 Student Exam Score Prediction App")
st.write("Enter student details to predict exam score")

# ================= USER INPUTS =================
study_hours = st.number_input("Study Hours per Day", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous_score = st.number_input("Previous Exam Score", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0)

# ================= PREDICTION =================
if st.button("Predict Exam Score"):

    # Full feature list (safe maximum)
    full_input = np.array([[study_hours,
                             attendance,
                             previous_score,
                             sleep_hours]])

    # Match model feature count automatically
    expected_features = model.n_features_in_
    input_data = full_input[:, :expected_features]

    # Apply scaler if exists
    if scaler is not None:
        input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")

# ================= DEBUG (OPTIONAL) =================
with st.expander("Debug Info"):
    st.write("Model expects features:", model.n_features_in_)
