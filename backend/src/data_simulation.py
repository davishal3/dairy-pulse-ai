import pandas as pd
import numpy as np
import os

# Ensure directories exist
os.makedirs("data/raw", exist_ok=True)


def generate_data(n_rows=10000, sequence_hours=12):
    np.random.seed(42)
    data = []

    # We generate data in chunks of 50 steps to maintain time-series continuity for LSTM
    for chunk in range(n_rows // 50):
        # Randomly decide the "Theme" of this chunk
        chunk_type = np.random.choice(["SAFE", "MEDIUM", "DANGER"])

        # Set base conditions based on the theme
        if chunk_type == "SAFE":
            temp = np.random.uniform(2.0, 5.0)
            vibration = np.random.uniform(0.5, 1.2)
        elif chunk_type == "MEDIUM":
            temp = np.random.uniform(5.5, 8.0)
            vibration = np.random.uniform(1.0, 2.5)
        else:
            temp = np.random.uniform(8.5, 14.0)
            vibration = np.random.uniform(2.0, 4.5)

        humidity = np.random.uniform(65, 80)

        for step in range(50):
            # Add small random noise for realism
            temp += np.random.normal(0, 0.3)
            humidity += np.random.normal(0, 0.5)
            vibration += np.random.normal(0, 0.1)

            # Keep values within physical bounds
            temp = max(2.0, min(temp, 15.0))
            humidity = max(50.0, min(humidity, 95.0))
            vibration = max(0.0, min(vibration, 5.0))

            # Strictly assign Spoilage Logic based on current state
            if temp <= 5.5:
                spoilage = 0   # LOW risk
                remaining_hours = sequence_hours
            elif temp <= 8.5:
                # Medium risk boundary
                spoilage = np.random.choice([0, 1], p=[0.7, 0.3])
                remaining_hours = sequence_hours * 0.6
            else:
                spoilage = 1   # HIGH risk
                remaining_hours = sequence_hours * 0.2

            data.append([
                len(data),
                round(temp, 2),
                round(humidity, 2),
                round(vibration, 2),
                spoilage,
                round(remaining_hours, 2)
            ])

    df = pd.DataFrame(data, columns=[
        "time", "temperature", "humidity", "vibration", "spoilage", "remaining_hours"
    ])

    df.to_csv("data/raw/sensor_data.csv", index=False)
    print(f"✅ Generated {len(data)} rows successfully!")
    print("✅ BALANCED Spoilage distribution (Should be close to 50/50):")
    print(df['spoilage'].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    generate_data()
