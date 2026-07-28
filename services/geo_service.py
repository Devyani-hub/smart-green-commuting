"""
services/geo_service.py
-----------------------
Handles geocoding (city → coordinates) and route distance/duration
via OpenRouteService.  Falls back to a haversine estimate if the API
is unavailable or the key is missing.
"""

import math
import logging
from typing import Optional

import requests

from config.settings import ORS_API_KEY

logger = logging.getLogger(__name__)

ORS_GEOCODE_URL  = "https://api.openrouteservice.org/geocode/search"
ORS_ROUTING_URL  = "https://api.openrouteservice.org/v2/directions/driving-car"
REQUEST_TIMEOUT  = 10  # seconds


# ── Geocoding ─────────────────────────────────────────────────────────────────

def get_coordinates(city: str) -> Optional[tuple[float, float]]:
    """
    Return (longitude, latitude) for *city*.
    Returns None if geocoding fails so the caller can handle it gracefully.
    """
    if not ORS_API_KEY or ORS_API_KEY.startswith("your_"):
        logger.warning("ORS_API_KEY not configured – geocoding unavailable.")
        return None

    try:
        params = {
            "api_key": ORS_API_KEY,
            "text":    city,
            "size":    1,
        }
        resp = requests.get(ORS_GEOCODE_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            logger.warning("No geocoding results for '%s'.", city)
            return None

        lon, lat = features[0]["geometry"]["coordinates"]
        logger.info("Geocoded '%s' → (%.4f, %.4f)", city, lon, lat)
        return (lon, lat)

    except requests.RequestException as exc:
        logger.error("Geocoding request failed for '%s': %s", city, exc)
        return None


# ── Routing ───────────────────────────────────────────────────────────────────

def get_route(
    src_coords: tuple[float, float],
    dest_coords: tuple[float, float],
) -> tuple[float, float]:
    """
    Return (distance_km, duration_min) for a driving route.
    Falls back to haversine straight-line estimate (× 1.3 detour factor)
    if the API call fails.
    """
    if ORS_API_KEY and not ORS_API_KEY.startswith("your_"):
        try:
            headers = {
                "Authorization": ORS_API_KEY,
                "Content-Type":  "application/json",
            }
            body = {"coordinates": [list(src_coords), list(dest_coords)]}
            resp = requests.post(
                ORS_ROUTING_URL, json=body, headers=headers, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()

            summary  = data["routes"][0]["summary"]
            distance = round(summary["distance"] / 1000, 2)   # metres → km
            duration = round(summary["duration"] / 60,  1)    # seconds → min
            logger.info("ORS route: %.2f km, %.1f min", distance, duration)
            return distance, duration

        except (requests.RequestException, KeyError, IndexError) as exc:
            logger.warning("ORS routing failed (%s); falling back to haversine.", exc)

    # ── Haversine fallback ────────────────────────────────────────────────────
    distance = _haversine(src_coords, dest_coords)
    duration = round((distance / 40) * 60, 1)   # assume 40 km/h avg
    logger.info("Haversine fallback: %.2f km, %.1f min", distance, duration)
    return distance, duration


def _haversine(coord1: tuple[float, float], coord2: tuple[float, float]) -> float:
    """Straight-line distance in km between two (lon, lat) points × 1.3."""
    R = 6371.0
    lon1, lat1 = map(math.radians, coord1)
    lon2, lat2 = map(math.radians, coord2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    straight = 2 * R * math.asin(math.sqrt(a))
    return round(straight * 1.3, 2)   # detour factor
