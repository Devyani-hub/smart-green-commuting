"""
ml/predictor.py
---------------
Loads the trained Random Forest model, scaler, and label encoders.
Builds per-mode option rows, applies feasibility rules, and predicts
CO₂ emissions for each feasible transport mode.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODERS_PATH,
    FEATURE_ORDER,
    TRANSPORT_MODES,
)

logger = logging.getLogger(__name__)


# ── Model loading (cached – loaded once per process) ─────────────────────────

@lru_cache(maxsize=1)
def _load_artifacts():
    """Load and cache model, scaler, encoders from disk."""
    try:
        model    = joblib.load(MODEL_PATH)
        scaler   = joblib.load(SCALER_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        logger.info("ML artifacts loaded from %s", MODEL_PATH.parent)
        return model, scaler, encoders
    except FileNotFoundError as exc:
        logger.critical("Model file not found: %s", exc)
        raise


# ── Option builder ────────────────────────────────────────────────────────────

def build_transport_options(
    distance_km:       float,
    base_duration_min: float,
    traffic_level:     str,
    weather_label:     str,
    temperature_c:     float,
) -> pd.DataFrame:
    """
    Create one row per transport mode with raw (un-encoded) feature values.
    """
    rows = []
    for mode, params in TRANSPORT_MODES.items():
        travel_time = round(base_duration_min / params["speed_factor"], 1)
        cost_rs     = round(distance_km * params["cost_per_km"], 1)
        rows.append({
            "transport_mode":   mode,
            "distance_km":      distance_km,
            "travel_time_min":  travel_time,
            "cost_rs":          cost_rs,
            "traffic_level":    traffic_level,
            "weather_condition": weather_label,
            "temperature_c":    temperature_c,
            "emoji":            params["emoji"],
        })
    return pd.DataFrame(rows)


# ── Feasibility filter ────────────────────────────────────────────────────────

def apply_feasibility_rules(df: pd.DataFrame, distance_km: float) -> pd.DataFrame:
    """
    Remove modes that are physically unrealistic for the given distance.
    """
    def _keep(row) -> bool:
        mode = row["transport_mode"]
        max_km = TRANSPORT_MODES[mode]["max_km"]
        if max_km is not None and distance_km > max_km:
            logger.debug("Filtered out %s (distance %.1f > %.1f km)", mode, distance_km, max_km)
            return False
        return True

    mask = df.apply(_keep, axis=1)
    result = df[mask].reset_index(drop=True)
    logger.info("Feasibility filter: %d → %d modes", len(df), len(result))
    return result


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(df: pd.DataFrame) -> np.ndarray:
    """Encode categoricals and scale features; returns numpy array."""
    _, scaler, encoders = _load_artifacts()

    df_enc = df[FEATURE_ORDER].copy()

    for col, enc in encoders.items():
        # Safely handle unseen labels – map to first known class
        df_enc[col] = df_enc[col].apply(
            lambda x: x if x in enc.classes_ else enc.classes_[0]
        )
        df_enc[col] = enc.transform(df_enc[col])

    return scaler.transform(df_enc)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict_co2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `predicted_co2_g` column (grams) to the options DataFrame.
    The model predicts a base CO₂ value; we then apply mode-specific
    emission factors to ensure physical plausibility across modes.
    """
    model, _, _ = _load_artifacts()

    X = _preprocess(df.copy())
    raw_predictions = model.predict(X)

    result = df.copy()
    result["predicted_co2_g"] = [
        max(0.0, round(pred * TRANSPORT_MODES[mode]["co2_factor"], 1))
        for pred, mode in zip(raw_predictions, result["transport_mode"])
    ]

    # Convert grams to kg for display convenience
    result["predicted_co2_kg"] = (result["predicted_co2_g"] / 1000).round(3)

    logger.info("CO₂ predictions: %s", dict(zip(result["transport_mode"], result["predicted_co2_g"])))
    return result
