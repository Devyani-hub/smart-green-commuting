"""
utils/formatters.py
-------------------
Small formatting helpers used by the Streamlit dashboard.
"""


def fmt_co2(grams: float) -> str:
    if grams < 1:
        return "0 g CO₂"
    if grams >= 1000:
        return f"{grams / 1000:.2f} kg CO₂"
    return f"{grams:.0f} g CO₂"


def fmt_time(minutes: float) -> str:
    if minutes < 60:
        return f"{minutes:.0f} min"
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m" if m else f"{h}h"


def fmt_cost(rupees: float) -> str:
    if rupees == 0:
        return "Free"
    return f"₹{rupees:.0f}"


def sustainability_badge(score: int) -> str:
    """Return an emoji badge based on sustainability score 1–10."""
    if score >= 8:
        return "🌱 Excellent"
    elif score >= 6:
        return "✅ Good"
    elif score >= 4:
        return "⚠️ Moderate"
    else:
        return "🔴 Poor"


def traffic_color(level: str) -> str:
    return {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}.get(level, "#95a5a6")


def weather_emoji(label: str) -> str:
    return {"Sunny": "☀️", "Foggy": "🌫️", "Rainy": "🌧️", "Cold": "❄️"}.get(label, "🌤️")
