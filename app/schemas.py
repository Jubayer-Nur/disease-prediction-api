from pydantic import BaseModel, Field

class HeartFeatures(BaseModel):
    age: float = Field(..., description="Age in years")
    sex: float = Field(..., description="Gender (1 = Male, 0 = Female)")
    cp: float = Field(..., description="Chest pain type (0-3)")
    trestbps: float = Field(..., description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., description="Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)")
    restecg: float = Field(..., description="Resting ECG results (0-2)")
    thalach: float = Field(..., description="Maximum heart rate achieved")
    exang: float = Field(..., description="Exercise induced angina (1 = Yes, 0 = No)")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: float = Field(..., description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: float = Field(..., description="Thalassemia (0 = Normal, 1 = Fixed defect, 2 = Reversible defect)")

class PredictionResponse(BaseModel):
    heart_disease: bool
    probability: float

class HealthResponse(BaseModel):
    status: str
    service: str

class InfoResponse(BaseModel):
    model_type: str
    features: list[str]
    description: str