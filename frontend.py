import streamlit as st
import requests

st.title("Student Performance Predictor")
hours = st.number_input("Hours Studied")
prev_score = st.number_input("Previous Score")
sleep_hours = st.number_input("Sleep Hours")
paper_practiced = st.number_input("Sample Question Papers Practiced")
eca = st.selectbox("Extracurricular Activity", ("Yes", "No"))

# ... more inputs
if st.button("Predict"):
    response = requests.post("http://localhost:8000/predict", json={
        "hours_studied": hours,
        "previous_score": prev_score,
        "sleep_hours": sleep_hours,
        "paper_practiced": paper_practiced,
        "extracurricular_activity": eca
        })
    st.write(response.json())