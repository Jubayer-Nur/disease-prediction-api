import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

df = pd.read_csv('heart.csv')

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("Columns:", df.columns.tolist())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/heart_model.joblib')
print("Model saved to model/heart_model.joblib")