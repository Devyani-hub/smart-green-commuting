# 🌱 Smart AI Green Commuting Recommendation System

An industry-grade AI capstone project demonstrating the integration of
**Machine Learning**, **Real-Time APIs**, and **Generative AI** to recommend
the most sustainable transport mode for any journey.

---

## 🏗️ Architecture

```
User Input
    ↓
Input Validation
    ↓
OpenRouteService  ──── Geocoding + Route (distance, duration)
    ↓
OpenWeatherMap    ──── Live weather (label + temperature + humidity)
    ↓
Traffic Inference ──── Time-of-day → High / Medium / Low
    ↓
ML Pipeline       ──── Random Forest → CO₂ prediction per mode (R² = 0.999)
    ↓
Google Gemini     ──── Structured JSON reasoning with sustainability context
    ↓
Streamlit Dashboard ── Interactive charts, radar, metrics, explanation
```

---

## 📁 Project Structure

```
green_commute/
├── app.py                  # Streamlit dashboard (entry point)
├── pipeline.py             # Orchestrates the full pipeline
├── config/
│   └── settings.py         # All config, constants, and env vars
├── services/
│   ├── geo_service.py      # Geocoding + routing (ORS / haversine)
│   └── weather_service.py  # Weather (OpenWeatherMap)
├── ml/
│   └── predictor.py        # Model loading, feature prep, CO₂ prediction
├── genai/
│   └── advisor.py          # Gemini prompt + RAG context + JSON parsing
├── utils/
│   ├── validators.py       # Input validation
│   └── formatters.py       # Display formatters
├── models/
│   ├── green_commute_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
├── data/
│   └── green_commuting_dataset.csv
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env with your keys:
```

| Key | Where to get |
|-----|-------------|
| `OPENWEATHER_API_KEY` | https://openweathermap.org/api (free tier) |
| `ORS_API_KEY` | https://openrouteservice.org/ (free tier) |
| `GEMINI_API_KEY` | https://aistudio.google.com/ (free) |

### 3. Run
```bash
streamlit run app.py
```

> **Graceful fallback**: The system works even without API keys using
> haversine distance estimation, default weather, and a rule-based
> lowest-CO₂ recommendation.

---

## 🧠 ML Model

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Target | CO₂ emissions (grams) per trip |
| Features | distance, travel time, mode, cost, traffic, weather, temperature |
| R² Score | **0.999** |
| CV R² (5-fold) | 0.997 ± 0.001 |
| Training samples | 500 |

---

## 🤖 AI Recommendation

Google Gemini 1.5 Flash receives:
- All ML-predicted transport options (CO₂, time, cost)
- Live weather and traffic conditions
- Injected sustainability reference data (RAG-lite)

It returns structured JSON with:
- Recommended mode + confidence
- Sustainability score (1–10)
- Natural-language summary
- Per-dimension reasoning (sustainability, time, cost, weather, traffic, health)
- Alternative modes with trade-offs
- Eco tip

---

## 📡 APIs Used

| API | Purpose | Fallback |
|-----|---------|---------|
| OpenRouteService | Geocoding + driving route | Haversine ×1.3 |
| OpenWeatherMap | Live weather | Sunny, 28°C |
| Google Gemini | AI reasoning | Lowest-CO₂ rule |

---

## 📊 Dashboard Features

- Journey summary strip (distance, time, traffic, weather)
- AI Recommendation card with full reasoning
- Transport comparison table
- CO₂ bar chart (Plotly)
- Time vs Cost scatter plot
- Multi-dimension radar chart
- Environmental impact (CO₂ saved vs car, tree equivalent)

---

## ✅ Issues Fixed from Original

| # | Original Issue | Fix Applied |
|---|---------------|------------|
| 1 | Hardcoded Windows absolute paths | `pathlib.Path` relative to `BASE_DIR` |
| 2 | API keys hardcoded in source | `python-dotenv` + `.env` file |
| 3 | Silent bare `except:` clauses | Typed `except` with logging |
| 4 | Rule-based if-else recommendation | Google Gemini reasoning pipeline |
| 5 | Fake hardcoded fallback distances | Haversine with detour factor |
| 6 | No input validation | `validators.py` with clear error messages |
| 7 | sklearn version mismatch warning | Model retrained with current sklearn |
| 8 | Random CO₂ multiplier post-prediction | Clean co2_factor in config |
| 9 | No loading indicators | `st.spinner` throughout |
| 10 | Minimal UI (4 lines of output) | Full professional dashboard |
| 11 | No logging | `logging` module throughout |
| 12 | Duplicate code across functions | Service/util modules |
| 13 | No requirements pinning | `requirements.txt` with versions |
| 14 | No traffic API | Time-of-day inference documented |
| 15 | Model cached every call | `@lru_cache` – loaded once |
