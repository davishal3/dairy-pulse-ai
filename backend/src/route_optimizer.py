# backend/src/route_optimizer.py
import math
import numpy as np

# -----------------------------
# Sample Cold Storage / Processing Unit Coordinates
# -----------------------------
COLD_STORAGES = [
    {"name": "Cold Storage A", "lat": 28.6139, "lon": 77.2090},  # Delhi
    {"name": "Cold Storage B", "lat": 27.1767, "lon": 78.0081},  # Agra
    {"name": "Processing Unit C", "lat": 26.9124, "lon": 75.7873},  # Jaipur
]

# -----------------------------
# Haversine Function
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points in km
    """
    R = 6371  # Earth radius
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# -----------------------------
# Suggest Nearest Storage
# -----------------------------
def suggest_nearest_storage(current_lat: float, current_lon: float) -> dict:
    """
    Given current location, suggest nearest storage
    """
    distances = [
        haversine(current_lat, current_lon, storage["lat"], storage["lon"])
        for storage in COLD_STORAGES
    ]
    
    min_index = np.argmin(distances)
    nearest = COLD_STORAGES[min_index]
    return {
        "name": nearest["name"],
        "distance_km": round(distances[min_index], 2)
    }

# -----------------------------
# Eco-Routing / Carbon Footprint Logic
# -----------------------------
def calculate_eco_impact(distance_km: float) -> dict:
    """
    Standard logistics truck emission is approx 0.8 kg CO2 per km.
    Calculates carbon footprint for the journey and savings if optimized.
    """
    co2_emitted = round(distance_km * 0.8, 2)
    
    # Assuming dynamic routing saves around 15% distance compared to standard static routes
    co2_saved = round(co2_emitted * 0.15, 2) 
    
    return {
        "co2_emitted_kg": co2_emitted,
        "co2_saved_kg": co2_saved,
        "trees_equivalent": round(co2_saved / 21, 2) # 1 mature tree absorbs ~21kg CO2/year
    }

# -----------------------------
# Predictive Route Based on Spoilage Risk
# -----------------------------
def suggest_route(risk_level: str) -> dict:
    """
    Suggest routing action based on spoilage risk:
    - HIGH/MEDIUM: reroute to nearest storage to save the batch
    - LOW: proceed on standard planned route
    """
    # Example current location (This can be hooked to a live GPS API later)
    current_lat, current_lon = 28.70, 77.10

    if risk_level.upper() in ["HIGH", "MEDIUM"]:
        route = suggest_nearest_storage(current_lat, current_lon)
        route["message"] = "Risk Detected: Reroute recommended to nearest safe facility."
        route["eco_metrics"] = calculate_eco_impact(route["distance_km"])
        return route
    else:
        # Default standard distance (e.g., approx 150km for a safe, non-diverted route)
        return {
            "message": "Spoilage risk low: Proceeding on planned standard route.",
            "eco_metrics": calculate_eco_impact(150.0)
        }

# -----------------------------
# Demo Run
# -----------------------------
if __name__ == "__main__":
    # Example risk levels to test the logic
    for risk in ["LOW", "MEDIUM", "HIGH"]:
        route = suggest_route(risk)
        print(f"✅ Risk Level: {risk}")
        print(f"   Route Info: {route['message']}")
        if 'name' in route:
            print(f"   Destination: {route['name']} ({route['distance_km']} km)")
        print(f"   Eco Impact: Saved {route['eco_metrics']['co2_saved_kg']} kg CO2 ({route['eco_metrics']['trees_equivalent']} trees)\n")   