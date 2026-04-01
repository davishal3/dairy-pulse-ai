# backend/src/train_lstm.py
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
MODEL_DIR = "backend/models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_dual_output.h5")

# -----------------------------
# Hyperparameters
# -----------------------------
SEQ_LEN = 10
FEATURES = 3
BATCH_SIZE = 32
EPOCHS = 30

# -----------------------------
# Load Processed Actual Data
# -----------------------------
print("Loading processed data from data/processed/...")
try:
    X = np.load("data/processed/X.npy")
    y_spoilage = np.load("data/processed/y_spoilage.npy")
    y_remaining = np.load("data/processed/y_hours.npy")
    print(f"✅ Data loaded successfully. X shape: {X.shape}")
except FileNotFoundError:
    raise FileNotFoundError("Processed data not found! Please run 'python backend/src/preprocess.py' first.")

# Train/test split
X_train, X_test, y_train_spoilage, y_test_spoilage, y_train_hours, y_test_hours = train_test_split(
    X, y_spoilage, y_remaining, test_size=0.2, random_state=42
)

# -----------------------------
# Model Architecture
# -----------------------------
inputs = Input(shape=(SEQ_LEN, FEATURES), name="input_layer")
x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)

spoilage_output = Dense(1, activation="sigmoid", name="spoilage")(x)
remaining_hours_output = Dense(1, activation="linear", name="remaining_hours")(x)

model = Model(inputs=inputs, outputs=[spoilage_output, remaining_hours_output])
model.compile(
    optimizer="adam",
    loss={"spoilage": "binary_crossentropy", "remaining_hours": "mse"},
    metrics={"spoilage": "accuracy", "remaining_hours": "mae"}
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_spoilage_loss",
    patience=5,
    restore_best_weights=True,
    mode="min"
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_spoilage_loss",
    save_best_only=True,
    mode="min"
)

# -----------------------------
# Training
# -----------------------------
print("Starting model training...")
history = model.fit(
    X_train,
    {"spoilage": y_train_spoilage, "remaining_hours": y_train_hours},
    validation_data=(X_test, {"spoilage": y_test_spoilage, "remaining_hours": y_test_hours}),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print(f"✅ Training completed. Model saved at: {MODEL_PATH}")