# backend/src/mandi_analysis.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# -----------------------------
# Ensure data folder exists
# -----------------------------
os.makedirs("data/raw", exist_ok=True)

# -----------------------------
# Generate Sample Mandi Data
# -----------------------------
def generate_sample_mandi_data():
    """
    Generates a sample mandi dataset for demo / hackathon.
    Columns: hour, location, price, demand
    """
    data = []
    # Updated locations to make it more relevant for the regional demo
    locations = ["Delhi", "Agra", "Mathura", "Jaipur"]
    
    for hour in range(24):
        for loc in locations:
            price = np.random.uniform(45, 55) + hour * 0.1  # slight upward trend
            demand = np.random.randint(50, 200)
            data.append([hour, loc, round(price, 2), demand])
    
    df = pd.DataFrame(data, columns=["hour", "location", "price", "demand"])
    df.to_csv("data/raw/mandi_data.csv", index=False)
    print("✅ Sample mandi data generated")


# -----------------------------
# Predict Prices for Next Hours
# -----------------------------
def predict_prices(df: pd.DataFrame, next_hours: list = [24,25,26]) -> dict:
    """
    Predicts price for the next hours using simple Linear Regression
    Returns a dictionary: {location: [pred_hour1, pred_hour2, ...]}
    """
    predictions = {}
    for loc in df["location"].unique():
        df_loc = df[df["location"] == loc]
        X = df_loc[["hour"]].values
        y = df_loc["price"].values
        model = LinearRegression()
        model.fit(X, y)
        future_hours = np.array(next_hours).reshape(-1, 1)
        pred_prices = model.predict(future_hours)
        predictions[loc] = [round(p, 2) for p in pred_prices]
    return predictions


# -----------------------------
# Suggest Best Location
# -----------------------------
def suggest_best_location(df: pd.DataFrame) -> dict:
    """
    Returns the location with max predicted price in next hour
    """
    pred_prices = predict_prices(df)
    best_loc = max(pred_prices.items(), key=lambda x: x[1][0])
    return {
        "location": best_loc[0],
        "predicted_price_next_hour": best_loc[1][0],
        "predicted_prices_next_3_hours": best_loc[1]
    }


# -----------------------------
# Estimate Profit Saving
# -----------------------------
def estimate_profit_saving(quantity_liters: float, predicted_price: float, current_price: float) -> float:
    """
    Returns estimated profit if sold at predicted price
    """
    return round((predicted_price - current_price) * quantity_liters, 2)


# -----------------------------
# Dynamic Pricing Engine (AI Quality-Based)
# -----------------------------
def apply_dynamic_pricing(base_price: float, remaining_hours: float) -> dict:
    """
    Adjusts price based on remaining shelf life.
    - If < 5 hours: 20% Discount (Clearance Sale to avoid waste)
    - If > 15 hours: 10% Premium (High Quality Freshness)
    - Else: Base Price
    """
    if remaining_hours < 5.0:
        discount = 0.20
        final_price = base_price * (1 - discount)
        strategy = "Clearance Discount (Urgent Sale)"
    elif remaining_hours > 15.0:
        premium = 0.10
        final_price = base_price * (1 + premium)
        strategy = "Premium Pricing (High Freshness)"
    else:
        final_price = base_price
        strategy = "Standard Pricing"
        
    return {
        "final_price_per_L": round(final_price, 2),
        "strategy": strategy,
        "price_diff_pct": round(((final_price - base_price) / base_price) * 100, 1)
    }


# -----------------------------
# Demo Run
# -----------------------------
if __name__ == "__main__":
    generate_sample_mandi_data()
    df = pd.read_csv("data/raw/mandi_data.csv")
    
    suggestion = suggest_best_location(df)
    print("\n✅ Best selling suggestion:", suggestion)
    
    profit = estimate_profit_saving(
        quantity_liters=100,
        predicted_price=suggestion["predicted_price_next_hour"],
        current_price=45  # Assuming base cost is 45
    )
    print(f"💰 Estimated profit if followed: ₹{profit}")
    
    # Test Dynamic Pricing
    print("\n--- Testing Dynamic Pricing ---")
    for hours in [3.0, 10.0, 20.0]:
        price_info = apply_dynamic_pricing(suggestion["predicted_price_next_hour"], remaining_hours=hours)
        print(f"Shelf Life: {hours}hrs -> Strategy: {price_info['strategy']} -> Final Price: ₹{price_info['final_price_per_L']}/L")