import streamlit as st
from recommendation_logic import run_recommendation

st.set_page_config(page_title="Green Commute", page_icon="🌱")

st.title("🌱 Smart AI Green Commuting")

source = st.text_input("Source City", "Pune")
dest = st.text_input("Destination City", "Mumbai")

time = st.selectbox("Time", ["Morning", "Afternoon", "Evening"])

if st.button("Recommend"):

    best, df = run_recommendation(source, dest, time)

    st.success("✅ Best Option")

    st.write(f"Mode: {best['transport_mode']}")
    st.write(f"CO₂: {best['predicted_co2']:.2f}")
    st.write(f"Time: {best['travel_time_min']:.1f} min")
    st.write(f"Cost: ₹{best['cost_rs']:.1f}")

    st.subheader("All Options")
    st.dataframe(df)

    st.subheader("CO₂ Comparison")
    st.bar_chart(df.set_index("transport_mode")["predicted_co2"])
