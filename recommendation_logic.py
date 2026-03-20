import joblib
import pandas as pd
import numpy as np
import requests

# =========================
# LOAD MODEL
# =========================
model = joblib.load(r"green_commute_model.pkl")
scaler = joblib.load(r"scaler.pkl")
encoders = joblib.load(r"encoders.pkl")

# =========================
# API KEYS
# =========================
import streamlit as st

OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
ORS_API_KEY = st.secrets["ORS_API_KEY"]

FEATURE_ORDER = [
    "distance_km",
    "travel_time_min",
    "transport_mode",
    "cost_rs",
    "traffic_level",
    "weather_condition",
    "temperature_c",
]

# =========================
# GET COORDINATES
# =========================
def get_coordinates(city):
    try:
        url = f"https://api.openrouteservice.org/geocode/search?api_key={ORS_API_KEY}&text={city}"
        res = requests.get(url).json()
        return res['features'][0]['geometry']['coordinates']
    except:
        return [73.8567, 18.5204]  # Pune fallback

# =========================
# WEATHER (FIXED)
# =========================
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        data = requests.get(url).json()

        weather_raw = data['weather'][0]['main']
        temp = data['main']['temp']

        # mapping
        if weather_raw == "Clear":
            weather = "Sunny"
        elif weather_raw in ["Clouds", "Mist", "Haze"]:
            weather = "Foggy"
        elif weather_raw in ["Rain", "Drizzle", "Thunderstorm"]:
            weather = "Rainy"
        else:
            weather = "Sunny"

        return weather, temp

    except:
        return "Sunny", 28

# =========================
# DISTANCE + TIME
# =========================
def get_distance_api(src, dest):
    try:
        url = "https://api.openrouteservice.org/v2/directions/driving-car"

        headers = {
            "Authorization": ORS_API_KEY,
            "Content-Type": "application/json"
        }

        body = {"coordinates": [src, dest]}

        res = requests.post(url, json=body, headers=headers).json()

        distance = res['routes'][0]['summary']['distance'] / 1000
        duration = res['routes'][0]['summary']['duration'] / 60

        return distance, duration

    except:
        return 10, 20

# =========================
# TRAFFIC
# =========================
def infer_traffic(time):
    if time in ["Morning", "Evening"]:
        return "High"
    elif time == "Afternoon":
        return "Medium"
    return "Low"

# =========================
# MODES
# =========================
transport_modes = ["Car", "Two-Wheeler", "Bus", "Metro", "Bicycle", "Walk"]

MODE_COST = {
    "Car": 12,
    "Two-Wheeler": 6,
    "Bus": 2,
    "Metro": 3,
    "Bicycle": 0,
    "Walk": 0
}

MODE_SPEED = {
    "Car": 1.0,
    "Two-Wheeler": 1.2,
    "Bus": 0.8,
    "Metro": 1.3,
    "Bicycle": 0.4,
    "Walk": 0.2
}

# =========================
# BUILD OPTIONS
# =========================
def build_options(distance, duration, traffic, weather, temp):
    rows = []

    for mode in transport_modes:
        time = duration / MODE_SPEED[mode]
        cost = distance * MODE_COST[mode]

        rows.append({
            "distance_km": distance,
            "travel_time_min": time,
            "transport_mode": mode,
            "cost_rs": cost,
            "traffic_level": traffic,
            "weather_condition": weather,
            "temperature_c": temp
        })

    return pd.DataFrame(rows)

#===============================
#Feasibility rule
#==============================
def apply_feasibility_rules(df, distance):

    filtered = []

    for _, row in df.iterrows():
        mode = row["transport_mode"]
        time = row["travel_time_min"]

        # 🚶 Walk
        if mode == "Walk":
            if distance > 3 or time > 40:
                continue

        # 🚴 Bicycle
        if mode == "Bicycle":
            if distance > 8 or time > 60:
                continue

        # 🛵 Two-Wheeler
        if mode == "Two-Wheeler":
            if distance > 100:
                continue

        filtered.append(row)

    return pd.DataFrame(filtered)

# =========================
# PREPROCESS (FIXED)
# =========================
def preprocess(df):
    for col, enc in encoders.items():
        df[col] = df[col].apply(
            lambda x: x if x in enc.classes_ else enc.classes_[0]
        )
        df[col] = enc.transform(df[col])

    df = df[FEATURE_ORDER]
    return scaler.transform(df)

# =========================
# PREDICT (FIXED)
# =========================
def predict(df):
    X = preprocess(df.copy())
    df["predicted_co2"] = model.predict(X)

    # add realistic variation
    factor = {
        "Car": 1.0,
        "Two-Wheeler": 0.7,
        "Bus": 0.5,
        "Metro": 0.3,
        "Bicycle": 0.1,
        "Walk": 0.05
    }

    df["predicted_co2"] = df.apply(
        lambda r: r["predicted_co2"] * factor[r["transport_mode"]],
        axis=1
    )

    return df

# =========================
# MAIN
# =========================
def run_recommendation(src, dest, time):

    src_c = get_coordinates(src)
    dest_c = get_coordinates(dest)

    dist, dur = get_distance_api(src_c, dest_c)

    traffic = infer_traffic(time)
    weather, temp = get_weather(src)

    df = build_options(dist, dur, traffic, weather, temp)
    df = apply_feasibility_rules(df, dist)
    df = predict(df)

    best = df.sort_values("predicted_co2").iloc[0]

    return best, df
