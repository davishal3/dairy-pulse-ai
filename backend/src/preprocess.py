import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

# Ensure processed folder exists
os.makedirs("data/processed", exist_ok=True)

# Load raw data
df = pd.read_csv("data/raw/sensor_data.csv")

# Features: temperature, humidity, vibration
features = ["temperature", "humidity", "vibration"]
target = "spoilage"           # 0/1 label
remaining_hours_col = "remaining_hours"

# Scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features])

# Save scaler for later inference
with open("backend/src/utils_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Feature scaler saved at backend/src/utils_scaler.pkl")

# LSTM sequence creation
sequence_length = 10  # timesteps per sample
X = []
y_spoilage = []
y_hours = []

for i in range(len(scaled_features) - sequence_length):
    X.append(scaled_features[i:i+sequence_length])
    y_spoilage.append(df[target].iloc[i+sequence_length])
    y_hours.append(df[remaining_hours_col].iloc[i+sequence_length])

X = np.array(X)
y_spoilage = np.array(y_spoilage)
y_hours = np.array(y_hours)

# Save processed data
np.save("data/processed/X.npy", X)
np.save("data/processed/y_spoilage.npy", y_spoilage)
np.save("data/processed/y_hours.npy", y_hours)

print(f"✅ Preprocessing done. X shape: {X.shape}, y_spoilage shape: {y_spoilage.shape}, y_hours shape: {y_hours.shape}")

# Optional: check class balance
unique, counts = np.unique(y_spoilage, return_counts=True)
print("✅ Spoilage label distribution:", dict(zip(unique, counts)))