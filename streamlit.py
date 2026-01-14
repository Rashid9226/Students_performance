import streamlit as st
import pandas as pd
import joblib

model=joblib.load("model.pkl")
scaler=joblib.load("scaler.pkl")
encoder=joblib.load("encoder.pkl")


hour_studied= st.number_input("Hours studied")
prev_score=st.number_input("Previous Score")
sleep_hours= st.number_input("Sleep Hours")
paper=st.number_input("Sample Question paper practiced")
eca=st.selectbox("Extracurricular Activity",("Yes","No"))
eca=encoder.transform([eca])

data=pd.DataFrame({"Hours Studied":hour_studied,"Previous Scores":prev_score,"Extracurricular Activities":eca,"Sleep Hours":sleep_hours,"Sample Question Papers Practiced":paper})
# st.write(data)

scaled_data=scaler.transform(data)
# st.write(scaled_data)

if st.button("predict"):
    prediction=model.predict(scaled_data)[0]
    st.write(prediction)