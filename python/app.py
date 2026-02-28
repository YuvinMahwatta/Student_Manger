import streamlit as st
import pickle
import numpy as np
import os

# Get absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build correct path to model
model_path = os.path.join(BASE_DIR, "..", "ml_model", "Linear_Regression.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# App Title
st.title("🎓 Student CGPA Prediction")
st.write("Enter student lifestyle details to estimate current semester CGPA.")

st.markdown("---")

# User Inputs

st.subheader("📊 Enter Student Details")

attendance = st.slider("Attendance Percentage", 0, 100, 75)

offline_study = st.number_input(
    "Offline Study Hours (per day)",
    min_value=0.0,
    max_value=12.0,
    value=3.0,
    step=0.1
)

online_study = st.number_input(
    "Online Study Hours (per day)",
    min_value=0.0,
    max_value=12.0,
    value=2.0,
    step=0.1
)

sleep = st.number_input(
    "Sleep Hours (per day)",
    min_value=0.0,
    max_value=12.0,
    value=7.0,
    step=0.1
)

screen_time = st.number_input(
    "Daily Screen Time Hours",
    min_value=0.0,
    max_value=14.0,
    value=4.0,
    step=0.1
)

st.markdown("---")

input_data = np.array([[screen_time, online_study, sleep, attendance, offline_study]])

# Prediction

if st.button("Predict CGPA"):

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted CGPA: {prediction:.2f}")
 
    # Reliability Warning System
   
    if attendance < 50:
        st.warning("⚠️ Very low attendance — prediction may be less reliable.")

    if screen_time > 9:
        st.warning("⚠️ Extremely high screen time compared to training data.")

    if sleep < 5:
        st.warning("⚠️ Low sleep hours may reduce prediction reliability.")

    if offline_study + online_study > 10:
        st.info("ℹ️ Very high study hours — ensure this reflects realistic behavior.")

    # General explanation
    st.caption(
        "Note: The model predicts based on patterns learned from historical data. "
        "Predictions for unusual behavior are extrapolations and may be less accurate."
    )