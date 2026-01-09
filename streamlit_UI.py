import streamlit as st
from recommendation_logic import run_recommendation

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Smart Green Commuting",
    page_icon="🌱",
    layout="centered"
)

# ---------------- Title ----------------
st.title("🌱 Smart AI Green Commuting System")
st.write(
    "This application recommends the most **eco‑friendly commuting option** "
    "based on source, destination, time of travel, and weather."
)

st.divider()

# ---------------- User Inputs ----------------
st.subheader("🚏 Enter Travel Details")

source = st.selectbox(
    "Source Location",
    [
        "Kothrud", "Baner", "Hadapsar", "Shivajinagar", "Aundh",
        "Wakad", "Kharadi", "Bibwewadi", "Katraj",
        "Pune Station", "Viman Nagar"
    ]
)

destination = st.selectbox(
    "Destination Location",
    [
        "Hinjewadi", "Shivajinagar", "Magarpatta", "Karve Nagar",
        "Deccan", "Pune Station", "Swargate", "Kharadi",
        "Viman Nagar", "Bibwewadi", "Camp", "Yerwada", "Baner"
    ]
)

time_of_day = st.selectbox(
    "Time of Travel",
    ["Morning", "Afternoon", "Evening"]
)

weather = st.selectbox(
    "Weather Condition",
    ["Sunny", "Rainy", "Cold", "Foggy"]
)

st.divider()

# ---------------- Recommendation Button ----------------
if st.button("🌍 Recommend Green Commute"):
    with st.spinner("Analyzing green commuting options..."):
        best_option, all_options = run_recommendation(
            source=source,
            destination=destination,
            time_of_day=time_of_day,
            user_weather=weather
        )

    # ---------------- Best Recommendation ----------------
    st.success("✅ Recommended Green Commute")
    st.markdown(f"""
    **🚲 Mode:** {best_option['transport_mode']}  
    **🌿 Predicted CO₂ Emission:** {best_option['predicted_co2']:.2f}  
    **⏱ Travel Time (min):** {best_option['travel_time_min']}  
    **💰 Cost (₹):** {best_option['cost_rs']}
    """)

    # ---------------- Comparison Table ----------------
    st.subheader("📊 Comparison of All Transport Options")

    display_columns = [
        "transport_mode",
        "travel_time_min",
        "cost_rs",
        "traffic_level",
        "weather_condition",
        "predicted_co2"
    ]

    all_options_sorted = all_options.sort_values(
        by="predicted_co2",
        ascending=True
    )

    st.dataframe(all_options_sorted[display_columns])

    # ---------------- CO₂ Chart ----------------
    st.subheader("📉 CO₂ Emission Comparison")

    chart_data = all_options_sorted.set_index("transport_mode")["predicted_co2"]
    st.bar_chart(chart_data)

# ---------------- Footer ----------------
st.divider()
st.caption("Capstone Project | Smart AI Green Commuting System 🌿")
