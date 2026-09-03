import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'heart_model.joblib')

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train_model.py first.")
    return joblib.load(MODEL_PATH)

model = load_model()
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']