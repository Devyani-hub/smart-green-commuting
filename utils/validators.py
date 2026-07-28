"""
utils/validators.py
-------------------
Input validation helpers for the Streamlit UI.
"""

import re
from typing import Optional


MIN_CITY_LEN = 2
MAX_CITY_LEN = 80

# Simple pattern: letters (incl. unicode), spaces, hyphens, commas, dots
_CITY_PATTERN = re.compile(r"^[\w\s\-,\.]{2,80}$", re.UNICODE)


def validate_city(name: str) -> Optional[str]:
    """
    Return None if *name* is valid, or an error message string if invalid.
    """
    name = name.strip()
    if not name:
        return "City name cannot be empty."
    if len(name) < MIN_CITY_LEN:
        return f"City name must be at least {MIN_CITY_LEN} characters."
    if len(name) > MAX_CITY_LEN:
        return f"City name must be at most {MAX_CITY_LEN} characters."
    if not _CITY_PATTERN.match(name):
        return "City name contains invalid characters."
    return None


def validate_inputs(source: str, destination: str) -> list[str]:
    """
    Validate both source and destination.
    Returns a list of error messages (empty list = all good).
    """
    errors = []
    src_err = validate_city(source)
    dst_err = validate_city(destination)
    if src_err:
        errors.append(f"Source: {src_err}")
    if dst_err:
        errors.append(f"Destination: {dst_err}")
    if not errors and source.strip().lower() == destination.strip().lower():
        errors.append("Source and destination cannot be the same city.")
    return errors
