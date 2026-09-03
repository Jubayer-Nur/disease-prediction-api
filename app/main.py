from fastapi import FastAPI, HTTPException
from app.schemas import HeartFeatures, PredictionResponse, HealthResponse, InfoResponse
from app.model_loader import model, FEATURES
import numpy as np
import uvicorn

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart disease presence using a Random Forest model.",
    version="1.0.0"
)

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(status="healthy", service="heart-disease-prediction")

@app.get("/info", response_model=InfoResponse, tags=["Info"])
async def get_info():
    return InfoResponse(
        model_type="Random Forest Classifier",
        features=FEATURES,
        description="This model predicts the presence of heart disease based on 13 clinical features."
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: HeartFeatures):
    try:
        input_data = np.array([[
            features.age,
            features.sex,
            features.cp,
            features.trestbps,
            features.chol,
            features.fbs,
            features.restecg,
            features.thalach,
            features.exang,
            features.oldpeak,
            features.slope,
            features.ca,
            features.thal
        ]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return PredictionResponse(
            heart_disease=bool(prediction),
            probability=round(float(probability), 4)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)