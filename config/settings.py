"""
config/settings.py
------------------
Central configuration: loads API keys from .env, defines constants,
model paths, transport parameters, and CO₂ emission factors.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH    = BASE_DIR / "models" / "green_commute_model.pkl"
SCALER_PATH   = BASE_DIR / "models" / "scaler.pkl"
ENCODERS_PATH = BASE_DIR / "models" / "encoders.pkl"
DATA_PATH     = BASE_DIR / "data"   / "green_commuting_dataset.csv"

# ── API Keys (never hard-code) ────────────────────────────────────────────────
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
ORS_API_KEY:         str = os.getenv("ORS_API_KEY", "")
GEMINI_API_KEY:      str = os.getenv("GEMINI_API_KEY", "")

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Feature order must match scaler / model training ─────────────────────────
FEATURE_ORDER = [
    "distance_km",
    "travel_time_min",
    "transport_mode",
    "cost_rs",
    "traffic_level",
    "weather_condition",
    "temperature_c",
]

# ── Transport mode definitions ────────────────────────────────────────────────
# cost_per_km in ₹, speed_factor relative to car driving time
TRANSPORT_MODES = {
    "Car": {
        "cost_per_km": 12,
        "speed_factor": 1.0,
        "co2_factor":   1.0,
        "emoji":        "🚗",
        "max_km":       None,
    },
    "Two-Wheeler": {
        "cost_per_km": 6,
        "speed_factor": 1.15,
        "co2_factor":   0.65,
        "emoji":        "🛵",
        "max_km":       120,
    },
    "Bus": {
        "cost_per_km": 2,
        "speed_factor": 0.75,
        "co2_factor":   0.40,
        "emoji":        "🚌",
        "max_km":       None,
    },
    "Metro": {
        "cost_per_km": 3,
        "speed_factor": 1.25,
        "co2_factor":   0.20,
        "emoji":        "🚇",
        "max_km":       None,
    },
    "Bicycle": {
        "cost_per_km": 0,
        "speed_factor": 0.35,
        "co2_factor":   0.00,
        "emoji":        "🚴",
        "max_km":       10,
    },
    "Walk": {
        "cost_per_km": 0,
        "speed_factor": 0.18,
        "co2_factor":   0.00,
        "emoji":        "🚶",
        "max_km":       3,
    },
}

# ── Weather → model label mapping ─────────────────────────────────────────────
WEATHER_MAP: dict[str, str] = {
    "Clear":        "Sunny",
    "Clouds":       "Foggy",
    "Mist":         "Foggy",
    "Haze":         "Foggy",
    "Fog":          "Foggy",
    "Rain":         "Rainy",
    "Drizzle":      "Rainy",
    "Thunderstorm": "Rainy",
    "Snow":         "Cold",
    "Sleet":        "Cold",
}

VALID_WEATHER_LABELS = ["Sunny", "Foggy", "Rainy", "Cold"]

# ── Traffic inference from time-of-day ───────────────────────────────────────
TRAFFIC_MAP: dict[str, str] = {
    "Early Morning (5–8 AM)":  "Low",
    "Morning Peak (8–11 AM)":  "High",
    "Afternoon (11 AM–4 PM)":  "Medium",
    "Evening Peak (4–8 PM)":   "High",
    "Night (8 PM–12 AM)":      "Low",
    "Late Night (12–5 AM)":    "Low",
}
