# backend/src/predict.py
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics

MODEL_PATH = "backend/models/lstm_dual_output.h5"
SCALER_PATH = "backend/src/utils_scaler.pkl"

# Risk level function
def get_risk_level(prob):
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"

# Custom objects for Keras
custom_objects = {"mse": metrics.MeanSquaredError}

# -----------------------------
# Load Model & Scaler
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")

model = load_model(MODEL_PATH, custom_objects=custom_objects)
print("✅ Model loaded successfully.")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run preprocess.py first.")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print("✅ Scaler loaded successfully.")

# -----------------------------
# Prediction Function
# -----------------------------
def predict_spoilage(sequence: np.ndarray) -> dict:
    """
    sequence: np.ndarray of shape (1, timesteps, features)
              Raw sensor values [temperature, humidity, vibration]
    returns: dict with probability, risk_level, remaining_hours, confidence
    """
    # 1. Scale the input sequence using the saved MinMaxScaler
    timesteps, features = sequence.shape[1], sequence.shape[2]
    
    # Reshape to 2D for scaling, then back to 3D for LSTM
    sequence_2d = sequence.reshape(-1, features)
    scaled_sequence_2d = scaler.transform(sequence_2d)
    scaled_sequence = scaled_sequence_2d.reshape(1, timesteps, features)
    
    # 2. Make Prediction
    spoilage_prob, remaining_hours = model.predict(scaled_sequence, verbose=0)
    
    # Extract scalar values
    spoilage_prob = float(spoilage_prob[0][0])
    remaining_hours = float(remaining_hours[0][0])
    risk_level = get_risk_level(spoilage_prob)
    
    return {
        "probability": round(spoilage_prob, 3),
        "risk_level": risk_level,
        "remaining_hours": round(remaining_hours, 2),
        "confidence": round(spoilage_prob, 3)
    }

# -----------------------------
# Demo run
# -----------------------------
if __name__ == "__main__":
    # Simulating 1 sample of 10 timesteps with realistic raw values 
    # (Temp: ~4C, Humidity: ~75%, Vibration: ~1)
    np.random.seed(42)
    sample_input = np.array([
        [[4.5, 75.0, 1.2] for _ in range(10)]
    ], dtype=np.float32)
    
    print("\n--- Testing Prediction ---")
    result = predict_spoilage(sample_input)
    print(result)