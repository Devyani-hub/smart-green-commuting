"""
pipeline.py
-----------
Orchestrates the full recommendation pipeline:
  1. Geocoding (ORS)
  2. Route distance + duration (ORS / haversine fallback)
  3. Weather (OpenWeatherMap)
  4. Traffic inference from time-of-day
  5. ML CO₂ prediction per transport mode
  6. GenAI reasoning & recommendation (Gemini)

Returns a single dict that the Streamlit app renders.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config.settings import TRAFFIC_MAP
from services.geo_service     import get_coordinates, get_route
from services.weather_service import get_weather
from ml.predictor             import build_transport_options, apply_feasibility_rules, predict_co2
from genai.advisor            import get_ai_recommendation

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for all data produced by the pipeline."""
    source:           str
    destination:      str
    time_of_day:      str
    distance_km:      float
    base_duration_min: float
    traffic_level:    str
    weather_info:     dict
    src_coords:       Optional[tuple]
    dest_coords:      Optional[tuple]
    options_df:       pd.DataFrame
    recommendation:   dict
    errors:           list[str] = field(default_factory=list)
    warnings:         list[str] = field(default_factory=list)


def run_pipeline(source: str, destination: str, time_of_day: str) -> PipelineResult:
    """
    Execute the full recommendation pipeline and return a PipelineResult.
    Never raises – errors are collected in result.errors.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # ── 1. Geocoding ──────────────────────────────────────────────────────────
    src_coords  = get_coordinates(source)
    dest_coords = get_coordinates(destination)

    if src_coords is None:
        warnings.append(f"Could not geocode '{source}'; using haversine fallback.")
    if dest_coords is None:
        warnings.append(f"Could not geocode '{destination}'; using haversine fallback.")

    # Provide dummy coords for haversine only if both are missing
    _src  = src_coords  or (72.8777, 19.0760)   # Mumbai
    _dest = dest_coords or (73.8567, 18.5204)   # Pune

    # ── 2. Route ──────────────────────────────────────────────────────────────
    try:
        distance_km, base_duration_min = get_route(_src, _dest)
    except Exception as exc:
        logger.error("Route computation failed: %s", exc)
        errors.append(f"Route computation error: {exc}")
        distance_km, base_duration_min = 15.0, 30.0   # safe defaults

    # ── 3. Weather ────────────────────────────────────────────────────────────
    try:
        weather_info = get_weather(source)
        if weather_info["source"] == "fallback":
            warnings.append("Live weather unavailable – using default weather values.")
    except Exception as exc:
        logger.error("Weather fetch failed: %s", exc)
        weather_info = {
            "weather_label": "Sunny", "temperature_c": 28.0,
            "humidity": 60, "wind_speed": 3.0,
            "description": "Unknown", "source": "fallback",
        }
        warnings.append("Weather service error – using defaults.")

    # ── 4. Traffic ────────────────────────────────────────────────────────────
    traffic_level = TRAFFIC_MAP.get(time_of_day, "Medium")
    logger.info("Traffic level for '%s': %s", time_of_day, traffic_level)

    # ── 5. ML predictions ─────────────────────────────────────────────────────
    try:
        options_df = build_transport_options(
            distance_km       = distance_km,
            base_duration_min = base_duration_min,
            traffic_level     = traffic_level,
            weather_label     = weather_info["weather_label"],
            temperature_c     = weather_info["temperature_c"],
        )
        options_df = apply_feasibility_rules(options_df, distance_km)

        if options_df.empty:
            errors.append("No feasible transport modes found for this journey.")
            # Return Car as last resort
            from config.settings import TRANSPORT_MODES
            from ml.predictor import build_transport_options as bto
            options_df = bto(distance_km, base_duration_min, traffic_level,
                             weather_info["weather_label"], weather_info["temperature_c"])
            options_df = options_df[options_df["transport_mode"] == "Car"].reset_index(drop=True)

        options_df = predict_co2(options_df)

    except Exception as exc:
        logger.error("ML prediction failed: %s", exc)
        errors.append(f"ML prediction error: {exc}")
        # Return an empty frame so downstream code doesn't crash
        options_df = pd.DataFrame()

    # ── 6. GenAI recommendation ───────────────────────────────────────────────
    try:
        if not options_df.empty:
            recommendation = get_ai_recommendation(
                source       = source,
                destination  = destination,
                distance_km  = distance_km,
                time_of_day  = time_of_day,
                weather_info = weather_info,
                traffic      = traffic_level,
                options_df   = options_df,
            )
        else:
            recommendation = {
                "recommended_mode": "N/A",
                "summary": "Unable to generate recommendation due to pipeline errors.",
                "_source": "error",
            }
    except Exception as exc:
        logger.error("GenAI advisor failed: %s", exc)
        errors.append(f"AI advisor error: {exc}")
        recommendation = {
            "recommended_mode": "N/A",
            "summary": str(exc),
            "_source": "error",
        }

    return PipelineResult(
        source            = source,
        destination       = destination,
        time_of_day       = time_of_day,
        distance_km       = distance_km,
        base_duration_min = base_duration_min,
        traffic_level     = traffic_level,
        weather_info      = weather_info,
        src_coords        = src_coords,
        dest_coords       = dest_coords,
        options_df        = options_df,
        recommendation    = recommendation,
        errors            = errors,
        warnings          = warnings,
    )
