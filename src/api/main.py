from fastapi import FastAPI
from src.api.pydantic_models import PredictionRequest, PredictionResponse
import joblib
import pandas as pd

app = FastAPI()

# Load your best model
model = joblib.load("src/api/best_model.pkl")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    # Convert validated pydantic data to DataFrame for the model
    input_df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "prediction": int(prediction),
        "risk_probability": float(probability)
    }