import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

st.set_page_config(page_title="Student Exam Score Prediction", page_icon="📘")

# ================= LOAD MODEL =================
with open("saldf.pkl", "rb") as f:
    model = pickle.load(f)

# SAFETY CHECK (very important)
if isinstance(model, pd.DataFrame):
    st.error("❌ saldf.pkl contains a DataFrame, not a trained ML model.")
    st.info("Please re-save the trained model (LinearRegression) in your notebook.")
    st.stop()

# ================= UI =================
st.title("📘 Student Exam Score Prediction")
st.write("Enter student details to predict exam score")

study_hours = st.number_input("Study Hours per Day", min_value=0.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
previous_score = st.number_input("Previous Exam Score", min_value=0.0)
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0.0)

# ================= PREDICTION =================
if st.button("Predict Exam Score"):

    input_data = np.array([[study_hours,
                             attendance,
                             previous_score,
                             sleep_hours]])

    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")

