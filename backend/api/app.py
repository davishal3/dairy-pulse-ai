from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import json
import os

# Import Tumhare Modules (Ensure paths are correct)
from backend.src.predict import predict_spoilage
from backend.src.route_optimizer import suggest_route
from backend.src.mandi_analysis import suggest_best_location, estimate_profit_saving

app = FastAPI(title="Dairy-Pulse AI API")

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Buffer file for Streamlit dashboard to read real-time IoT data
BUFFER_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../data/live_buffer.json'))
MANDI_CSV = "data/raw/mandi_data.csv"

# Load mandi data safely
try:
    df_mandi = pd.read_csv(MANDI_CSV)
except FileNotFoundError:
    print("⚠️ Warning: Mandi CSV not found. Ensure simulation script ran.")
    df_mandi = pd.DataFrame()

# Initialize buffer file with safe defaults if it doesn't exist
os.makedirs(os.path.dirname(BUFFER_FILE), exist_ok=True)
if not os.path.exists(BUFFER_FILE):
    with open(BUFFER_FILE, "w") as f:
        json.dump([[4.0, 75.0, 1.0] for _ in range(10)], f)

# ==========================================
# 2. PYDANTIC MODELS (Data Validation)
# ==========================================


class MilkSample(BaseModel):
    sequence: list  # 2D list: timesteps x features
    quantity_liters: float  # For profit estimation


class TelemetryData(BaseModel):
    temp: float = 4.5
    humid: float = 75.0
    vib: float = 0.0

# ==========================================
# 3. ENDPOINTS
# ==========================================


@app.get("/")
def read_root():
    return {"status": "Dairy-Pulse AI API running (FastAPI)"}

# 🔴 NEW ENDPOINT: For M5Stack (IoT Device)


@app.post("/api/telemetry")
def receive_telemetry(data: TelemetryData):
    """
    Receives live data from M5StickC Plus2 and updates the sliding window buffer.
    """
    try:
        new_reading = [data.temp, data.humid, data.vib]
        print(f"📡 Live M5 Data: {new_reading}")

        # Read existing buffer
        with open(BUFFER_FILE, "r") as f:
            buffer = json.load(f)

        # Append new reading and keep only the latest 10
        buffer.append(new_reading)
        if len(buffer) > 10:
            buffer.pop(0)

        # Save back to file (Streamlit will read this)
        with open(BUFFER_FILE, "w") as f:
            json.dump(buffer, f)

        return {"status": "success", "message": "IoT data buffered"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 🟢 EXISTING ENDPOINT: For Full Prediction logic


@app.post("/predict")
def full_prediction(sample: MilkSample):
    """
    Returns:
    - Spoilage probability & risk
    - Remaining shelf-life
    - Suggested route
    - Best mandi location & estimated profit
    """
    # Convert input sequence to numpy array
    seq_array = np.array(sample.sequence)
    if len(seq_array.shape) == 2:
        seq_array = seq_array.reshape(
            1, seq_array.shape[0], seq_array.shape[1])

    # 1️⃣ ML Prediction
    spoilage_res = predict_spoilage(seq_array)

    # 2️⃣ Route suggestion
    route_res = suggest_route(spoilage_res["risk_level"])

    # 3️⃣ Mandi analytics
    profit = 0
    best_location = {}
    if not df_mandi.empty:
        best_location = suggest_best_location(df_mandi)
        profit = estimate_profit_saving(
            quantity_liters=sample.quantity_liters,
            predicted_price=best_location["predicted_price_next_hour"],
            current_price=50  # Example current price (can be dynamic)
        )

    # Combine response
    response = {
        "spoilage": spoilage_res,
        "route_suggestion": route_res,
        "mandi_suggestion": {
            "best_location": best_location,
            "estimated_profit": profit
        }
    }

    return response
