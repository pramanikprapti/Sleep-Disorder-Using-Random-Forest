import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("random_forest_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Load model feature columns
with open("model_columns.pkl", "rb") as columns_file:
    model_columns = pickle.load(columns_file)

# Streamlit app UI
st.title(" Sleep Disorder Prediction")

st.markdown("Enter your details to predict whether you have a sleep disorder.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, step=1)
occupation = st.selectbox("Occupation", ["Student", "Office Worker", "Self-Employed", "Other"])
sleep_duration = st.slider("Sleep Duration (hours)", min_value=1.0, max_value=12.0, step=0.5)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", min_value=1, max_value=10, step=1)
physical_activity = st.slider("Physical Activity Level (1-10)", min_value=1, max_value=10, step=1)
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, step=1)
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, step=1)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, step=1)
daily_steps = st.number_input("Daily Steps", min_value=100, max_value=30000, step=100)

# Convert input into DataFrame
input_data = pd.DataFrame([[gender, age, occupation, sleep_duration, quality_of_sleep,
                            physical_activity, stress_level, bmi_category, blood_pressure,
                            heart_rate, daily_steps]],
                          columns=["Gender", "Age", "Occupation", "Sleep Duration",
                                   "Quality of Sleep", "Physical Activity Level", "Stress Level",
                                   "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps"])

# One-hot encoding to match training data
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict"):
    prediction = rf_model.predict(input_data)[0]
    st.success(f" Predicted Sleep Disorder: {prediction}")
