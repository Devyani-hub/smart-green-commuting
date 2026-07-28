"""
genai/advisor.py
----------------
Uses Google Gemini (gemini-1.5-flash) to reason over all transport
options and produce a structured JSON recommendation with a natural-
language explanation.  Falls back to a deterministic lowest-CO₂ rule
if the API is unavailable.
"""

import json
import logging
import re
from typing import Optional

import pandas as pd

from config.settings import GEMINI_API_KEY

logger = logging.getLogger(__name__)


# ── Sustainability context injected into every prompt (RAG-lite) ──────────────
SUSTAINABILITY_CONTEXT = """
SUSTAINABILITY REFERENCE DATA:
- Walking and Cycling produce zero direct CO₂ emissions.
- Metro/rail: ~40–60 g CO₂/passenger-km (electric traction).
- Bus (full load): ~80–100 g CO₂/passenger-km.
- Two-Wheeler (petrol): ~60–80 g CO₂/km.
- Car (petrol, single occupant): ~120–180 g CO₂/km.
- India government target: 45% reduction in CO₂ intensity by 2030 (NDC).
- WHO recommends at least 150 min of moderate physical activity/week —
  cycling and walking help achieve this.
- Urban heat-island effect is worsened by traffic; public transit helps.
- EV modes (e-bike, e-auto) can reduce emissions by up to 70% vs petrol.
"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    source:       str,
    destination:  str,
    distance_km:  float,
    time_of_day:  str,
    weather_info: dict,
    traffic:      str,
    options_df:   pd.DataFrame,
) -> str:
    options_summary = []
    for _, row in options_df.iterrows():
        options_summary.append(
            f"  - {row['emoji']} {row['transport_mode']}: "
            f"{row['travel_time_min']:.0f} min, "
            f"₹{row['cost_rs']:.0f}, "
            f"{row['predicted_co2_g']:.0f} g CO₂"
        )

    options_text = "\n".join(options_summary)

    prompt = f"""
You are an intelligent, data-driven sustainable transportation advisor for Indian cities.

JOURNEY DETAILS:
  Source:       {source}
  Destination:  {destination}
  Distance:     {distance_km:.1f} km
  Time of day:  {time_of_day}
  Weather:      {weather_info['description']} ({weather_info['weather_label']}, {weather_info['temperature_c']}°C, humidity {weather_info['humidity']}%)
  Wind speed:   {weather_info['wind_speed']} m/s
  Traffic:      {traffic}

AVAILABLE TRANSPORT OPTIONS (ML-predicted values):
{options_text}

{SUSTAINABILITY_CONTEXT}

INSTRUCTIONS:
Analyse all transport options carefully considering:
1. Sustainability (CO₂ emissions, environmental impact)
2. Practicality (travel time, distance, weather suitability)
3. Cost (economic burden on commuter)
4. Comfort and safety (weather, traffic, time of day)
5. Health benefits (active travel)

Rules:
- If weather is Rainy or temperature < 15°C, deprioritise Cycling and Walking.
- If distance > 10 km, do NOT recommend Walk or Bicycle.
- If traffic is High, prefer Metro/Bus over Car.
- Always explain trade-offs.
- Be specific to Indian urban context.

Respond ONLY with valid JSON — no markdown, no explanation outside the JSON.
Schema:
{{
  "recommended_mode": "<mode name>",
  "confidence": "<High|Medium|Low>",
  "sustainability_score": <1-10>,
  "summary": "<2–3 sentence summary for the user>",
  "reasoning": {{
    "sustainability": "<why this is/isn't the greenest>",
    "time": "<time trade-off analysis>",
    "cost": "<cost analysis>",
    "weather_impact": "<how weather affects this choice>",
    "traffic_impact": "<how traffic affects this choice>",
    "health": "<health/activity benefit>"
  }},
  "alternatives": [
    {{
      "mode": "<mode>",
      "trade_off": "<brief trade-off vs recommended>"
    }}
  ],
  "eco_tip": "<one actionable sustainability tip>"
}}
"""
    return prompt.strip()


# ── Gemini call ───────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini API and return raw text response."""
    try:
        import google.generativeai as genai  # optional dependency
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        logger.error("google-generativeai not installed. Run: pip install google-generativeai")
        return None
    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        return None


def _parse_json(raw: str) -> Optional[dict]:
    """Extract and parse JSON from the model's text response."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed (%s); raw: %s", exc, raw[:200])
        return None


# ── Fallback rule-based recommendation ───────────────────────────────────────

def _fallback_recommendation(options_df: pd.DataFrame) -> dict:
    """Deterministic fallback: pick lowest CO₂ mode."""
    best = options_df.sort_values("predicted_co2_g").iloc[0]
    return {
        "recommended_mode":    best["transport_mode"],
        "confidence":          "Medium",
        "sustainability_score": 7,
        "summary": (
            f"{best['emoji']} {best['transport_mode']} is recommended as the lowest CO₂ "
            f"option ({best['predicted_co2_g']:.0f} g) for this {best['distance_km']:.1f} km journey. "
            "AI advisor unavailable – configure GEMINI_API_KEY for intelligent reasoning."
        ),
        "reasoning": {
            "sustainability": "Lowest predicted CO₂ emissions among feasible modes.",
            "time": f"Estimated travel time: {best['travel_time_min']:.0f} min.",
            "cost": f"Estimated cost: ₹{best['cost_rs']:.0f}.",
            "weather_impact": "Weather data considered in feasibility filtering.",
            "traffic_impact": "Traffic inferred from time of day.",
            "health": "Active transport modes offer additional health benefits.",
        },
        "alternatives": [
            {"mode": row["transport_mode"], "trade_off": f"CO₂: {row['predicted_co2_g']:.0f} g"}
            for _, row in options_df.sort_values("predicted_co2_g").iloc[1:3].iterrows()
        ],
        "eco_tip": "Consider combining public transit with cycling for the last mile.",
        "_source": "fallback",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_ai_recommendation(
    source:       str,
    destination:  str,
    distance_km:  float,
    time_of_day:  str,
    weather_info: dict,
    traffic:      str,
    options_df:   pd.DataFrame,
) -> dict:
    """
    Build a prompt, call Gemini, parse JSON.
    Returns the recommendation dict (AI-generated or fallback).
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("your_"):
        logger.warning("GEMINI_API_KEY not configured – using fallback recommendation.")
        result = _fallback_recommendation(options_df)
        result["_source"] = "fallback_no_key"
        return result

    prompt = _build_prompt(
        source, destination, distance_km, time_of_day,
        weather_info, traffic, options_df,
    )

    logger.info("Sending prompt to Gemini (%d chars)…", len(prompt))
    raw_text = _call_gemini(prompt)

    if raw_text is None:
        result = _fallback_recommendation(options_df)
        result["_source"] = "fallback_api_error"
        return result

    parsed = _parse_json(raw_text)
    if parsed is None:
        result = _fallback_recommendation(options_df)
        result["_source"] = "fallback_parse_error"
        return result

    parsed["_source"] = "gemini"
    logger.info("Gemini recommended: %s (confidence: %s)", parsed.get("recommended_mode"), parsed.get("confidence"))
    return parsed
