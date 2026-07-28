"""
app.py
------
Smart AI Green Commuting Recommendation System
Professional Streamlit dashboard — entry point.

Run:  streamlit run app.py
"""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import TRAFFIC_MAP, TRANSPORT_MODES
from pipeline import run_pipeline
from utils.formatters import (
    fmt_co2, fmt_cost, fmt_time,
    sustainability_badge, traffic_color, weather_emoji,
)
from utils.validators import validate_inputs

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Green Commute",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ── */
[data-testid="stAppViewContainer"] { background: #f0f4f0; }
[data-testid="stAppViewContainer"] * { color: #1b5e20 !important; }
[data-testid="stSidebar"]          { background: #1a2f1a; }
[data-testid="stSidebar"] *        { color: #e8f5e9 !important; }

/* ── Metric cards ── */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
    border-left: 5px solid #2e7d32;
    margin-bottom: 12px;
}
.metric-title  { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.metric-value  { font-size: 28px; font-weight: 800; color: #1b5e20; margin: 4px 0; }
.metric-sub    { font-size: 12px; color: #888; }

/* ── Recommendation card ── */
.rec-card {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: white;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 4px 20px rgba(46,125,50,.4);
    margin-bottom: 16px;
}
.rec-title  { font-size: 14px; opacity: .8; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.rec-mode   { font-size: 42px; font-weight: 900; margin: 8px 0; }
.rec-score  { font-size: 14px; opacity: .9; }

/* ── Warning / info banners ── */
.warn-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 13px;
    color: #5d4037;
}
.info-box {
    background: #e8f5e9;
    border-left: 4px solid #43a047;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 13px;
    color: #1b5e20;
}

/* ── Table ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Headings ── */
h1, h2, h3 { color: #1b5e20; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/seedling.png", width=64)
    st.title("🌱 Green Commute")
    st.caption("AI-Powered Sustainable Transport Advisor")

    st.divider()
    st.subheader("🗺️ Journey Details")

    source = st.text_input(
        "📍 Source City",
        value="Pune",
        placeholder="e.g. Pune, Mumbai, Delhi",
        help="Enter the city or area you are starting from.",
    )

    destination = st.text_input(
        "🏁 Destination City",
        value="Mumbai",
        placeholder="e.g. Mumbai, Bangalore, Chennai",
        help="Enter your destination city or area.",
    )

    time_of_day = st.selectbox(
        "⏰ Time of Travel",
        options=list(TRAFFIC_MAP.keys()),
        index=1,
        help="Select the approximate time of your journey (affects traffic modelling).",
    )

    st.divider()
    recommend_btn = st.button(
        "🚀 Get Recommendation",
        use_container_width=True,
        type="primary",
    )

    st.divider()
    st.caption("ℹ️ Powered by: OpenRouteService · OpenWeatherMap · Google Gemini · RandomForest ML")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌱 Smart AI Green Commuting System")
st.markdown(
    "Enter your journey details in the sidebar and click **Get Recommendation** "
    "for an AI-powered, real-time, sustainability-focused transport suggestion."
)

if not recommend_btn:
    # Landing illustration
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">🤖 AI-Powered</div>
        <div class="metric-value" style="font-size:18px">Google Gemini</div>
        <div class="metric-sub">Natural language reasoning over real-time data</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">🧠 Machine Learning</div>
        <div class="metric-value" style="font-size:18px">Random Forest</div>
        <div class="metric-sub">R² = 0.999 · CO₂ prediction per mode</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-title">📡 Real-Time APIs</div>
        <div class="metric-value" style="font-size:18px">Live Data</div>
        <div class="metric-sub">ORS routing · OpenWeatherMap · Traffic</div>
        </div>""", unsafe_allow_html=True)
    st.stop()


# ── Validation ────────────────────────────────────────────────────────────────
errors = validate_inputs(source, destination)
if errors:
    for err in errors:
        st.error(f"❌ {err}")
    st.stop()


# ── Run pipeline ──────────────────────────────────────────────────────────────
with st.spinner("🔄 Fetching live data and running AI analysis…"):
    result = run_pipeline(source.strip(), destination.strip(), time_of_day)

# Surface pipeline errors
if result.errors:
    for err in result.errors:
        st.error(f"❌ Pipeline error: {err}")

# Surface warnings
for warn in result.warnings:
    st.markdown(f'<div class="warn-box">⚠️ {warn}</div>', unsafe_allow_html=True)

# If completely empty options bail out
if result.options_df.empty:
    st.error("No transport options could be computed. Please check your inputs and API keys.")
    st.stop()


# ── Journey summary strip ─────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b, col_c, col_d = st.columns(4)

w = result.weather_info
with col_a:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">📏 Distance</div>
    <div class="metric-value">{result.distance_km:.1f} km</div>
    <div class="metric-sub">{source} → {destination}</div>
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">🚗 Base Drive Time</div>
    <div class="metric-value">{fmt_time(result.base_duration_min)}</div>
    <div class="metric-sub">By car in current traffic</div>
    </div>""", unsafe_allow_html=True)

with col_c:
    tc = traffic_color(result.traffic_level)
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{tc}">
    <div class="metric-title">🚦 Traffic</div>
    <div class="metric-value" style="color:{tc}">{result.traffic_level}</div>
    <div class="metric-sub">{time_of_day}</div>
    </div>""", unsafe_allow_html=True)

with col_d:
    we = weather_emoji(w["weather_label"])
    st.markdown(f"""
    <div class="metric-card">
    <div class="metric-title">🌤️ Weather</div>
    <div class="metric-value" style="font-size:20px">{we} {w['weather_label']}</div>
    <div class="metric-sub">{w['temperature_c']}°C · {w['description']} {'(live)' if w['source']=='live' else '(default)'}</div>
    </div>""", unsafe_allow_html=True)


# ── AI Recommendation card ────────────────────────────────────────────────────
rec = result.recommendation
rec_mode = rec.get("recommended_mode", "N/A")
rec_score = rec.get("sustainability_score", 0)
rec_conf  = rec.get("confidence", "—")

mode_emoji = TRANSPORT_MODES.get(rec_mode, {}).get("emoji", "🚀")

st.markdown("---")
st.subheader("🤖 AI Recommendation")

rc1, rc2 = st.columns([1, 2])
with rc1:
    src_label = "🌐 Gemini AI" if rec.get("_source") == "gemini" else "🔧 Rule-based"
    st.markdown(f"""
    <div class="rec-card">
    <div class="rec-title">✅ Recommended Mode · {src_label}</div>
    <div class="rec-mode">{mode_emoji} {rec_mode}</div>
    <div class="rec-score">
        Sustainability {rec_score}/10 &nbsp;·&nbsp; Confidence: {rec_conf}<br>
        {sustainability_badge(rec_score)}
    </div>
    </div>""", unsafe_allow_html=True)

with rc2:
    st.markdown(f'<div class="info-box">💬 <strong>AI Summary</strong><br>{rec.get("summary","")}</div>',
                unsafe_allow_html=True)

    if "reasoning" in rec:
        with st.expander("🔍 Detailed AI Reasoning", expanded=False):
            r = rec["reasoning"]
            cols = st.columns(2)
            fields = [
                ("🌍 Sustainability", r.get("sustainability", "")),
                ("⏱️ Time",           r.get("time", "")),
                ("💰 Cost",           r.get("cost", "")),
                ("🌧️ Weather",        r.get("weather_impact", "")),
                ("🚦 Traffic",        r.get("traffic_impact", "")),
                ("🏃 Health",         r.get("health", "")),
            ]
            for i, (label, text) in enumerate(fields):
                with cols[i % 2]:
                    st.markdown(f"**{label}**")
                    st.write(text)

    if "eco_tip" in rec:
        st.markdown(f'<div class="info-box">🌿 <strong>Eco Tip:</strong> {rec["eco_tip"]}</div>',
                    unsafe_allow_html=True)

    if rec.get("alternatives"):
        with st.expander("🔄 Alternative Modes"):
            for alt in rec["alternatives"]:
                st.markdown(f"- **{alt['mode']}**: {alt['trade_off']}")


# ── Transport options table ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 All Transport Options Comparison")

df = result.options_df.copy()
df["CO₂"]       = df["predicted_co2_g"].apply(fmt_co2)
df["Time"]       = df["travel_time_min"].apply(fmt_time)
df["Cost"]       = df["cost_rs"].apply(fmt_cost)
df["Mode"]       = df["emoji"] + " " + df["transport_mode"]
df["Recommended"] = df["transport_mode"].apply(lambda m: "✅" if m == rec_mode else "")

display_df = df[["Mode", "Time", "Cost", "CO₂", "Recommended"]].rename(
    columns={"Mode": "Transport Mode"}
)
st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🌿 CO₂ Emissions by Mode")
    fig_co2 = px.bar(
        df,
        x="transport_mode",
        y="predicted_co2_g",
        color="predicted_co2_g",
        color_continuous_scale=["#1b5e20", "#66bb6a", "#ffeb3b", "#e53935"],
        labels={"transport_mode": "Mode", "predicted_co2_g": "CO₂ (g)"},
        text=df["predicted_co2_g"].apply(lambda v: f"{v:.0f}g"),
    )
    fig_co2.update_traces(textposition="outside")
    fig_co2.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20),
        xaxis_title="",
        yaxis_title="CO₂ (grams)",
    )
    st.plotly_chart(fig_co2, use_container_width=True)

with chart_col2:
    st.subheader("⏱️ Travel Time vs Cost")
    fig_scatter = px.scatter(
        df,
        x="travel_time_min",
        y="cost_rs",
        size="predicted_co2_g",
        color="transport_mode",
        text="transport_mode",
        labels={
            "travel_time_min": "Travel Time (min)",
            "cost_rs": "Cost (₹)",
            "transport_mode": "Mode",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Radar chart
st.subheader("🕸️ Multi-Dimension Mode Comparison")

categories = ["Low CO₂", "Low Cost", "Low Time", "Weather OK", "Traffic OK"]

# Normalise each dimension 0–1 (1 = best)
max_co2  = df["predicted_co2_g"].max() or 1
max_cost = df["cost_rs"].max() or 1
max_time = df["travel_time_min"].max() or 1

weather_score = {"Sunny": 1.0, "Foggy": 0.7, "Rainy": 0.4, "Cold": 0.5}
traffic_score = {"Low": 1.0, "Medium": 0.6, "High": 0.3}

fig_radar = go.Figure()
for _, row in df.iterrows():
    scores = [
        1 - row["predicted_co2_g"] / max_co2,
        1 - row["cost_rs"] / max_cost if max_cost else 1,
        1 - row["travel_time_min"] / max_time,
        weather_score.get(row["weather_condition"], 0.7),
        traffic_score.get(row["traffic_level"], 0.6),
    ]
    fig_radar.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        name=f"{row['emoji']} {row['transport_mode']}",
        fill="toself",
        opacity=0.6,
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=20, b=20),
    height=420,
)
st.plotly_chart(fig_radar, use_container_width=True)


# ── Environmental impact summary ──────────────────────────────────────────────
st.markdown("---")
st.subheader("🌍 Environmental Impact")

if rec_mode in df["transport_mode"].values:
    car_co2 = df.loc[df["transport_mode"] == "Car", "predicted_co2_g"]
    rec_co2 = df.loc[df["transport_mode"] == rec_mode, "predicted_co2_g"]

    if not car_co2.empty and not rec_co2.empty:
        saved = float(car_co2.iloc[0]) - float(rec_co2.iloc[0])
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-title">🚗 Car CO₂</div>
            <div class="metric-value">{fmt_co2(float(car_co2.iloc[0]))}</div>
            <div class="metric-sub">Baseline reference</div>
            </div>""", unsafe_allow_html=True)
        with e2:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-title">{mode_emoji} {rec_mode} CO₂</div>
            <div class="metric-value">{fmt_co2(float(rec_co2.iloc[0]))}</div>
            <div class="metric-sub">Recommended mode</div>
            </div>""", unsafe_allow_html=True)
        with e3:
            pct = (saved / float(car_co2.iloc[0]) * 100) if float(car_co2.iloc[0]) else 0
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#43a047">
            <div class="metric-title">🌱 CO₂ Saved vs Car</div>
            <div class="metric-value" style="color:#2e7d32">{fmt_co2(max(0, saved))}</div>
            <div class="metric-sub">{pct:.0f}% reduction</div>
            </div>""", unsafe_allow_html=True)

        if saved > 0:
            trees = saved / 21000  # avg tree absorbs ~21 kg CO₂/year
            st.info(
                f"🌳 Choosing **{rec_mode}** instead of a car for this trip saves "
                f"**{fmt_co2(saved)}** of CO₂ — equivalent to planting "
                f"**{trees:.4f} trees** per trip."
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Smart AI Green Commuting System · "
    "ML: Random Forest (R²=0.999) · "
    "AI: Google Gemini 1.5 Flash · "
    "Data: OpenRouteService, OpenWeatherMap · "
    "Built for AI/ML Capstone"
)
