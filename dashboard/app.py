import streamlit as st
import pandas as pd
import os
import time

st.set_page_config(
    page_title="AI Traffic Monitoring",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI-Based Traffic Monitoring System")
st.subheader("Real-Time Vehicle Analytics Dashboard")

CSV_FILE = "results/traffic_data.csv"

# -----------------------------
# Check CSV
# -----------------------------

if not os.path.exists(CSV_FILE):

    st.error(
        "Traffic data not found. "
        "Run src/data_logger.py first."
    )

    st.stop()

# -----------------------------
# Load data
# -----------------------------

data = pd.read_csv(CSV_FILE)

if data.empty:

    st.warning("No traffic data available.")

    st.stop()

# -----------------------------
# Latest data
# -----------------------------

latest = data.iloc[-1]

cars = int(latest["Cars"])
motorcycles = int(latest["Motorcycles"])
buses = int(latest["Buses"])
trucks = int(latest["Trucks"])

total = int(latest["Total"])
up = int(latest["UP"])
down = int(latest["DOWN"])

current = int(latest["Current Vehicles"])

density = latest["Density"]

# -----------------------------
# Metrics
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🚗 Cars",
        cars
    )

with col2:
    st.metric(
        "🏍️ Motorcycles",
        motorcycles
    )

with col3:
    st.metric(
        "🚌 Buses",
        buses
    )

with col4:
    st.metric(
        "🚚 Trucks",
        trucks
    )

# -----------------------------
# Total / Density
# -----------------------------

st.divider()

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🚘 Total Vehicles",
        total
    )

with col2:

    st.metric(
        "🚦 Current Vehicles",
        current
    )

with col3:

    st.metric(
        "Traffic Density",
        density
    )

# -----------------------------
# Direction
# -----------------------------

st.divider()

st.subheader("Vehicle Direction")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "⬆️ UP",
        up
    )

with col2:

    st.metric(
        "⬇️ DOWN",
        down
    )

# -----------------------------
# Vehicle Count Chart
# -----------------------------

st.divider()

st.subheader("📊 Vehicle Count")

chart_data = data[
    [
        "Cars",
        "Motorcycles",
        "Buses",
        "Trucks"
    ]
]

st.line_chart(chart_data)

# -----------------------------
# Total Traffic Chart
# -----------------------------

st.subheader("📈 Total Traffic")

total_chart = data[
    ["Total"]
]

st.line_chart(total_chart)

# -----------------------------
# Traffic Density Distribution
# -----------------------------

st.subheader("🚦 Traffic Density")

density_counts = (
    data["Density"]
    .value_counts()
)

st.bar_chart(density_counts)

# -----------------------------
# Recent Records
# -----------------------------

st.subheader("📋 Recent Traffic Records")

st.dataframe(
    data.tail(20),
    use_container_width=True
)

# -----------------------------
# Download CSV
# -----------------------------

st.subheader("📥 Download Data")

with open(CSV_FILE, "rb") as file:

    st.download_button(
        label="Download Traffic CSV",
        data=file,
        file_name="traffic_data.csv",
        mime="text/csv"
    )

# -----------------------------
# Refresh
# -----------------------------

st.divider()

if st.button("🔄 Refresh Dashboard"):

    st.rerun()