from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Literal
import numpy as np

app = FastAPI()

# Enable CORS so the frontend (served from a different origin/file) can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

class StudentPerformanceInput(BaseModel):
    hours_studied: float
    previous_score: float
    sleep_hours: float
    paper_practiced: float
    extracurricular_activity: Literal["Yes", "No"]
    

@app.post("/predict")
def predict_performance(input_data: StudentPerformanceInput):
    eca_encoded = encoder.transform([input_data.extracurricular_activity])
    
    data = pd.DataFrame({
        "Hours Studied": [input_data.hours_studied],
        "Previous Scores": [input_data.previous_score],
        "Extracurricular Activities": eca_encoded,
        "Sleep Hours": [input_data.sleep_hours],
        "Sample Question Papers Practiced": [input_data.paper_practiced]
    })
    
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    
    return {"predicted_score": np.round(prediction, 2)}