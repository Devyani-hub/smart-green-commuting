import joblib
import pandas as pd
import numpy as np

# Load trained model and preprocessing objects
model = joblib.load("green_commute_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")


FEATURE_ORDER = [
    "distance_km",
    "travel_time_min",
    "transport_mode",
    "cost_rs",
    "traffic_level",
    "weather_condition",
    "temperature_c",
]


    # Central Pune
location_distance_map = {
    # West Pune
    ("Kothrud", "Karve Nagar"): 3,
    ("Kothrud", "Shivajinagar"): 8,
    ("Kothrud", "Deccan"): 6,
    ("Kothrud", "Hinjewadi"): 15,

    # Central Pune
    ("Shivajinagar", "Deccan"): 2,
    ("Shivajinagar", "Pune Station"): 3,
    ("Shivajinagar", "Swargate"): 4,

    # IT Hubs
    ("Baner", "Hinjewadi"): 7,
    ("Aundh", "Hinjewadi"): 9,
    ("Wakad", "Hinjewadi"): 5,

    # East Pune
    ("Hadapsar", "Magarpatta"): 4,
    ("Hadapsar", "Kharadi"): 6,
    ("Kharadi", "Viman Nagar"): 5,

    # South Pune
    ("Katraj", "Swargate"): 6,
    ("Katraj", "Bibwewadi"): 4,
    ("Bibwewadi", "Swargate"): 3,

    # Railway & Bus Areas
    ("Pune Station", "Swargate"): 4,
    ("Pune Station", "Camp"): 2,

    # Default short commutes
    ("Aundh", "Baner"): 3,
    ("Viman Nagar", "Yerwada"): 4
}


def get_distance(source, destination):
    return location_distance_map.get((source, destination), 10)

def infer_traffic(time_of_day):
    if time_of_day in ["Morning", "Evening"]:
        return "High"
    if time_of_day == "Afternoon":
        return "Medium"
    return "Low"

def get_weather(user_weather=None):
    if user_weather is not None and user_weather != "":
        return user_weather
    return "Sunny"   # default / simulated weather

transport_modes = [
    "Car",
    "Two-Wheeler",
    "Bus",
    "Metro",
    "Bicycle",
    "Walk"
]

MODE_CONFIG = {
    "Car": {"time_factor": 4.0, "cost_per_km": 12},
    "Two-Wheeler": {"time_factor": 3.5, "cost_per_km": 6},
    "Bus": {"time_factor": 5.0, "cost_per_km": 2},
    "Metro": {"time_factor": 3.0, "cost_per_km": 3},
    "Bicycle": {"time_factor": 6.0, "cost_per_km": 0},
    "Walk": {"time_factor": 8.0, "cost_per_km": 0}
}

def build_commute_options(distance, traffic, weather):
    temperature = 28
    rows = []

    for mode in transport_modes:
        config = MODE_CONFIG[mode]

        travel_time = distance * config["time_factor"]
        cost = distance * config["cost_per_km"]

        rows.append({
            "distance_km": distance,
            "travel_time_min": travel_time,
            "transport_mode": mode,
            "cost_rs": cost,
            "traffic_level": traffic,
            "weather_condition": weather,
            "temperature_c": temperature
        })

    return pd.DataFrame(rows)

def preprocess_input(df):
    # Encode categorical columns
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    # Force correct column order
    df = df[FEATURE_ORDER]

    # Scale using trained scaler
    return scaler.transform(df)

def predict_emissions(commute_df):
    X_scaled = preprocess_input(commute_df.copy())
    commute_df["predicted_co2"] = model.predict(X_scaled)
    return commute_df

def apply_feasibility_rules(options, weather, distance):
    feasible_options = []

    for _, row in options.iterrows():
        mode = row["transport_mode"]
        travel_time = row["travel_time_min"]

        # 🚶 Walk rules
        if mode == "Walk":
            if distance > 3 or travel_time > 40:
                continue
            if weather == "Rainy":
                continue

        # 🚴 Bicycle rules
        if mode == "Bicycle":
            if distance > 8 or travel_time > 60:
                continue
            if weather == "Rainy":
                continue

        # 🛵 Two-Wheeler rules
        if mode == "Two-Wheeler":
            if weather == "Rainy":
                continue

        # ✅ Bus, Metro, Car always allowed
        feasible_options.append(row)

    return pd.DataFrame(feasible_options)


def recommend_green_commute(distance, traffic, weather):
    options = build_commute_options(distance, traffic, weather)
    options = apply_feasibility_rules(options, weather, distance)
    options = predict_emissions(options)

    best_option = options.sort_values("predicted_co2").iloc[0]
    return best_option, options

def run_recommendation(source, destination, time_of_day, user_weather=None):
    distance = get_distance(source, destination)
    traffic = infer_traffic(time_of_day)
    weather = get_weather(user_weather)

    best_option, all_options = recommend_green_commute(
        distance, traffic, weather
    )

    return best_option, all_options

