"""
services/weather_service.py
----------------------------
Fetches live weather data from OpenWeatherMap.
Maps raw API condition strings to the four labels used by the ML model:
Sunny | Foggy | Rainy | Cold
"""

import logging
from typing import Optional

import requests

from config.settings import OPENWEATHER_API_KEY, WEATHER_MAP, VALID_WEATHER_LABELS

logger = logging.getLogger(__name__)

OWM_URL = "https://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT = 8


def get_weather(city: str) -> dict:
    """
    Return a dict with:
        weather_label  – one of Sunny / Foggy / Rainy / Cold
        temperature_c  – float
        humidity       – int %
        wind_speed     – float m/s
        description    – human-readable string
        source         – "live" | "fallback"
    """
    if OPENWEATHER_API_KEY and not OPENWEATHER_API_KEY.startswith("your_"):
        try:
            params = {
                "q":     city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
            }
            resp = requests.get(OWM_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            raw_condition = data["weather"][0]["main"]
            description   = data["weather"][0]["description"].capitalize()
            temp          = round(data["main"]["temp"], 1)
            humidity      = data["main"]["humidity"]
            wind_speed    = data["wind"]["speed"]

            label = WEATHER_MAP.get(raw_condition, "Sunny")
            logger.info(
                "Weather for '%s': %s (%.1f °C, hum=%d%%, wind=%.1f m/s)",
                city, label, temp, humidity, wind_speed,
            )
            return {
                "weather_label": label,
                "temperature_c": temp,
                "humidity":      humidity,
                "wind_speed":    wind_speed,
                "description":   description,
                "source":        "live",
            }

        except requests.RequestException as exc:
            logger.warning("Weather API failed for '%s': %s – using fallback.", city, exc)
        except (KeyError, IndexError) as exc:
            logger.warning("Unexpected weather API response: %s – using fallback.", exc)

    # ── Fallback ──────────────────────────────────────────────────────────────
    logger.warning("Returning fallback weather for '%s'.", city)
    return {
        "weather_label": "Sunny",
        "temperature_c": 28.0,
        "humidity":      60,
        "wind_speed":    3.0,
        "description":   "Data unavailable – using default",
        "source":        "fallback",
    }
