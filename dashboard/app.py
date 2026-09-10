import streamlit as st
import cv2
import av
import threading
import csv
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VehixAI | Intelligent Traffic Control",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# High-Contrast Professional Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Main Background & Streamlit Container Overrides */
    .stApp {
        background-color: #0A0E17 !important;
        color: #F8FAFC !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: rgba(10, 14, 23, 0.8) !important;
        backdrop-filter: blur(8px);
    }
    
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }
    
    /* Sidebar Overrides */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #E2E8F0 !important;
    }
    
    /* Header Card Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .hero-title {
        font-size: 24px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .hero-sub {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #1E293B !important;
        padding: 6px !important;
        border-radius: 10px !important;
        border: 1px solid #334155 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        background-color: transparent !important;
        border-radius: 6px !important;
        color: #CBD5E1 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 0 20px !important;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4) !important;
    }
    
    /* Metric Card Styling */
    .stat-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stat-title {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: 800;
        color: #FFFFFF;
        margin-top: 4px;
    }
    
    .stat-value-blue { color: #38BDF8; }
    .stat-value-green { color: #4ADE80; }
    .stat-value-amber { color: #FBBF24; }
    .stat-value-purple { color: #A78BFA; }
    .stat-value-rose { color: #FB7185; }

    /* Density Badges */
    .badge-status {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        border: 1px solid #22C55E;
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid #F59E0B;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Logging Logic
# ---------------------------------------------------------
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO IDs for Car, Motorcycle, Bus, Truck
CSV_FILE = "results/live_traffic_data.csv"
HISTORICAL_CSV = "results/traffic_data.csv"

os.makedirs("results", exist_ok=True)

def save_data(cars, motorcycles, buses, trucks, up, down, current, density):
    file_exists = os.path.exists(CSV_FILE)
    total = cars + motorcycles + buses + trucks
    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Cars", "Motorcycles", "Buses", "Trucks",
                "Total", "UP", "DOWN", "Current Vehicles", "Density"
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cars, motorcycles, buses, trucks, total, up, down, current, density
        ])

# ---------------------------------------------------------
# OpenCV Video Processor Thread
# ---------------------------------------------------------
class TrafficProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = YOLO("yolo26n.pt")
        self.previous_positions = {}
        self.counted_ids = set()

        self.car_count = 0
        self.motorcycle_count = 0
        self.bus_count = 0
        self.truck_count = 0

        self.up_count = 0
        self.down_count = 0

        self.current_vehicles = 0
        self.last_saved_second = -1
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        line_y = int(height * 0.60)

        results = self.model.track(
            img,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=0.40,
            verbose=False
        )

        result = results[0]
        current_count = 0

        # Counting Line
        cv2.line(img, (0, line_y), (width, line_y), (0, 215, 255), 2)
        cv2.putText(img, "TRAFFIC COUNTING LINE", (15, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            current_count = len(track_ids)

            for box, cls, track_id in zip(boxes, classes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                class_id = int(cls)

                names = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
                name = names.get(class_id, "Vehicle")

                # Sleek Neon Green bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (50, 255, 126), 2)
                cv2.circle(img, (center_x, center_y), 4, (0, 165, 255), -1)

                # Label tag
                label_text = f"{name} #{track_id}"
                cv2.rectangle(img, (x1, max(y1 - 24, 0)), (x1 + len(label_text)*10 + 20, max(y1, 24)), (50, 255, 126), -1)
                cv2.putText(img, label_text, (x1 + 6, max(y1 - 7, 17)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 20, 30), 1, cv2.LINE_AA)

                if track_id in self.previous_positions:
                    previous_y = self.previous_positions[track_id]

                    # DOWN crossing
                    if previous_y < line_y and center_y >= line_y and track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        with self.lock:
                            self.down_count += 1
                            if class_id == 2: self.car_count += 1
                            elif class_id == 3: self.motorcycle_count += 1
                            elif class_id == 5: self.bus_count += 1
                            elif class_id == 7: self.truck_count += 1

                    # UP crossing
                    elif previous_y > line_y and center_y <= line_y and track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        with self.lock:
                            self.up_count += 1
                            if class_id == 2: self.car_count += 1
                            elif class_id == 3: self.motorcycle_count += 1
                            elif class_id == 5: self.bus_count += 1
                            elif class_id == 7: self.truck_count += 1

                self.previous_positions[track_id] = center_y

        with self.lock:
            self.current_vehicles = current_count
            cars = self.car_count
            motorcycles = self.motorcycle_count
            buses = self.bus_count
            trucks = self.truck_count
            up = self.up_count
            down = self.down_count

        if current_count <= 5:
            density = "LOW"
        elif current_count <= 10:
            density = "MEDIUM"
        else:
            density = "HIGH"

        # Save data periodically
        current_second = datetime.now().second
        if current_second % 5 == 0 and current_second != self.last_saved_second:
            self.last_saved_second = current_second
            save_data(cars, motorcycles, buses, trucks, up, down, current_count, density)

        # Video HUD Box
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (250, 155), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (10, 10), (250, 155), (51, 65, 85), 1)

        cv2.putText(img, "VEHIXAI LIVE TELEMETRY", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(img, f"Cars: {cars} | Bikes: {motorcycles}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Buses: {buses} | Trucks: {trucks}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"UP: {up} | DOWN: {down}", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        density_color = (74, 222, 128) if density == "LOW" else ((251, 191, 36) if density == "MEDIUM" else (248, 113, 113))
        cv2.putText(img, f"Density: {density}", (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.5, density_color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------------------------------------
# Sidebar Navigation & System Information
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="color: #FFFFFF !important; margin: 0; font-weight: 800; font-size: 22px;">🚦 VehixAI</h2>
        <p style="color: #94A3B8 !important; font-size: 12px; margin-top: 2px;">Smart Traffic Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ System Status")
    st.markdown("🟢 **Detector**: YOLO PyTorch Engine")
    st.markdown("📹 **Stream Processor**: OpenCV / WebRTC")
    st.markdown(f"⏱️ **Active Time**: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    st.markdown("### 📊 Preset Density Thresholds")
    st.markdown("- **LOW**: 0 - 5 Vehicles")
    st.markdown("- **MEDIUM**: 6 - 10 Vehicles")
    st.markdown("- **HIGH**: 11+ Vehicles")

    st.divider()
    st.markdown("<p style='font-size: 11px; color: #64748B !important; text-align: center;'>VehixAI Control System © 2026</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Top Hero Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div>
        <div class="hero-title">🚦 VehixAI Traffic Control Center</div>
        <div class="hero-sub">Autonomous AI Multi-Class Vehicle Detection, Direction Tracking & Density Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------
tab_feed, tab_analytics, tab_logs = st.tabs([
    "📹 Live Camera Feed & Metrics", 
    "📊 Analytics & Distribution", 
    "📄 Log Records & Export"
])

# Default values
cars, motorcycles, buses, trucks, up, down, current, total = 0, 0, 0, 0, 0, 0, 0, 0
density = "LOW"

with tab_feed:
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown('<div class="section-title">📹 Camera Stream</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="traffic-monitor",
            video_processor_factory=TrafficProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

    # Fetch live counters from processor thread
    if ctx.video_processor:
        processor = ctx.video_processor
        with processor.lock:
            cars = processor.car_count
            motorcycles = processor.motorcycle_count
            buses = processor.bus_count
            trucks = processor.truck_count
            up = processor.up_count
            down = processor.down_count
            current = processor.current_vehicles

        total = cars + motorcycles + buses + trucks
        density = "LOW" if current <= 5 else ("MEDIUM" if current <= 10 else "HIGH")

    with col_right:
        st.markdown('<div class="section-title">📊 Live Dashboard Metrics</div>', unsafe_allow_html=True)

        # Overview Stats (Total & Density)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">Total Processed</div>
                <div class="stat-value">{total}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            badge_cls = "badge-low" if density == "LOW" else ("badge-med" if density == "MEDIUM" else "badge-high")
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">Traffic Density</div>
                <div style="margin-top: 8px;">
                    <span class="badge-status {badge_cls}">{density} ({current} Active)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Vehicle Breakdown 2x2 grid
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">🚗 Cars</div>
                <div class="stat-value stat-value-blue">{cars}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">🚌 Buses</div>
                <div class="stat-value stat-value-purple">{buses}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">🏍️ Motorcycles</div>
                <div class="stat-value stat-value-amber">{motorcycles}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">🚚 Trucks</div>
                <div class="stat-value stat-value-green">{trucks}</div>
            </div>
            """, unsafe_allow_html=True)

        # Direction Flow Cards
        f1, f2 = st.columns(2)
        with f1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">⬆️ UP Direction</div>
                <div class="stat-value stat-value-blue">{up}</div>
            </div>
            """, unsafe_allow_html=True)

        with f2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-title">⬇️ DOWN Direction</div>
                <div class="stat-value stat-value-rose">{down}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Analytics & Graphical Visualizations
# ---------------------------------------------------------
with tab_analytics:
    st.markdown('<div class="section-title">📈 Real-Time Traffic Analytics</div>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        labels = ['Cars', 'Motorcycles', 'Buses', 'Trucks']
        values = [cars, motorcycles, buses, trucks]
        if sum(values) == 0:
            values = [1, 1, 1, 1]

        fig_donut = px.pie(
            names=labels,
            values=values,
            hole=0.6,
            title="Vehicle Type Breakdown",
            color_discrete_sequence=['#38BDF8', '#FBBF24', '#A78BFA', '#4ADE80']
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC', family="Plus Jakarta Sans"),
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15)
        )
        st.plotly_chart(fig_donut, width='stretch')

    with chart_col2:
        fig_flow = go.Figure(data=[
            go.Bar(name='UP (North)', x=['Flow Volume'], y=[up], marker_color='#38BDF8'),
            go.Bar(name='DOWN (South)', x=['Flow Volume'], y=[down], marker_color='#FB7185')
        ])
        fig_flow.update_layout(
            title="Directional Traffic Volume",
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC', family="Plus Jakarta Sans"),
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15)
        )
        st.plotly_chart(fig_flow, width='stretch')

    st.divider()
    st.markdown('<div class="section-title">⏱️ Historical Traffic Trend</div>', unsafe_allow_html=True)

    target_csv = CSV_FILE if os.path.exists(CSV_FILE) else (HISTORICAL_CSV if os.path.exists(HISTORICAL_CSV) else None)

    if target_csv and os.path.getsize(target_csv) > 50:
        try:
            df = pd.read_csv(target_csv)
            if not df.empty and "Timestamp" in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                fig_trend = px.line(
                    df,
                    x='Timestamp',
                    y=['Cars', 'Motorcycles', 'Buses', 'Trucks'],
                    title="Vehicle Flow History",
                    color_discrete_map={'Cars': '#38BDF8', 'Motorcycles': '#FBBF24', 'Buses': '#A78BFA', 'Trucks': '#4ADE80'}
                )
                fig_trend.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.6)',
                    font=dict(color='#F8FAFC', family="Plus Jakarta Sans"),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#334155'),
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_trend, width='stretch')
        except Exception:
            st.info("Loading time series dataset...")
    else:
        st.info("Live data logging active. Time-series chart will populate as data points record.")

# ---------------------------------------------------------
# Tab 3: Data Log Export
# ---------------------------------------------------------
with tab_logs:
    st.markdown('<div class="section-title">📋 Live Telemetry Log Table</div>', unsafe_allow_html=True)

    target_csv = CSV_FILE if os.path.exists(CSV_FILE) else (HISTORICAL_CSV if os.path.exists(HISTORICAL_CSV) else None)

    if target_csv and os.path.exists(target_csv):
        df_log = pd.read_csv(target_csv)
        st.dataframe(df_log.tail(100), width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        with open(target_csv, "rb") as file:
            st.download_button(
                label="📥 Download Telemetry Log (CSV)",
                data=file,
                file_name=f"vehixai_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
    else:
        st.warning("No log data recorded yet. Turn on the video stream to begin logging.")