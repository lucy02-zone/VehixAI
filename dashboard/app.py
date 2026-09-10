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
    page_title="VehixAI - Intelligent Traffic Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Enterprise Dark Theme Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Main App Background */
    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: #0F172A !important;
        border-bottom: 1px solid #1E293B;
    }
    
    .main .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }
    
    /* Sidebar Overrides */
    section[data-testid="stSidebar"] {
        background-color: #090D16 !important;
        border-right: 1px solid #1E293B !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #CBD5E1 !important;
    }

    .sidebar-brand {
        font-size: 20px;
        font-weight: 700;
        color: #FFFFFF !important;
        letter-spacing: -0.3px;
        margin-bottom: 2px;
    }
    
    .sidebar-sub {
        font-size: 12px;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Top Control Banner */
    .control-header {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .control-title {
        font-size: 20px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    
    .control-sub {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Status Indicator Dot */
    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background-color: #1E293B !important;
        padding: 4px !important;
        border-radius: 6px !important;
        border: 1px solid #334155 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 38px !important;
        background-color: transparent !important;
        border-radius: 4px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 0 16px !important;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    
    .metric-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
    }
    
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
    }

    .val-cyan { color: #38BDF8; }
    .val-amber { color: #F59E0B; }
    .val-purple { color: #A78BFA; }
    .val-emerald { color: #10B981; }
    .val-rose { color: #F43F5E; }

    /* Density Badges */
    .density-tag {
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 700;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .density-low {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .density-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .density-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .section-label {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #E2E8F0;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Logging Logic
# ---------------------------------------------------------
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO IDs: Car, Motorcycle, Bus, Truck
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

        # Counting Reference Line
        cv2.line(img, (0, line_y), (width, line_y), (0, 191, 255), 2)
        cv2.putText(img, "TRAFFIC COUNTING LINE", (15, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 191, 255), 1, cv2.LINE_AA)

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

                # Clean Bounding Box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 118), 2)
                cv2.circle(img, (center_x, center_y), 4, (0, 191, 255), -1)

                # Label tag
                label_text = f"{name} ID:{track_id}"
                cv2.rectangle(img, (x1, max(y1 - 22, 0)), (x1 + len(label_text)*9 + 15, max(y1, 22)), (0, 230, 118), -1)
                cv2.putText(img, label_text, (x1 + 4, max(y1 - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 20, 30), 1, cv2.LINE_AA)

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
        cv2.rectangle(overlay, (10, 10), (250, 150), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (10, 10), (250, 150), (51, 65, 85), 1)

        cv2.putText(img, "TELEMETRY OVERLAY", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(img, f"Cars: {cars} | Bikes: {motorcycles}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(img, f"Buses: {buses} | Trucks: {trucks}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(img, f"North: {up} | South: {down}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (241, 245, 249), 1, cv2.LINE_AA)

        density_color = (74, 222, 128) if density == "LOW" else ((251, 191, 36) if density == "MEDIUM" else (248, 113, 113))
        cv2.putText(img, f"Density: {density}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, density_color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------------------------------------
# Sidebar System Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 15px 0; border-bottom: 1px solid #1E293B;">
        <div class="sidebar-brand">VehixAI Systems</div>
        <div class="sidebar-sub">Traffic Control Center</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="metric-label">System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 13px; color: #94A3B8; margin-top: 8px; line-height: 1.6;">
        <div><span class="status-dot"></span><b>Inference:</b> YOLO PyTorch</div>
        <div><span class="status-dot"></span><b>Tracker:</b> ByteTrack</div>
        <div><span class="status-dot"></span><b>Ingestion:</b> OpenCV / WebRTC</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Density Threshold Config</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 12px; color: #64748B; margin-top: 8px; line-height: 1.6;">
        • <b>Low:</b> 0 – 5 vehicles<br>
        • <b>Medium:</b> 6 – 10 vehicles<br>
        • <b>High:</b> 11+ vehicles
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 11px; color: #475569;'>System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="control-header">
    <div>
        <div class="control-title">VehixAI Traffic Monitoring Center</div>
        <div class="control-sub">Real-Time Vehicle Analytics, Directional Tracking, and Density Profiling</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------
tab_feed, tab_analytics, tab_logs = st.tabs([
    "Live Video Feed", 
    "Analytics & Distribution", 
    "Telemetry Logs"
])

cars, motorcycles, buses, trucks, up, down, current, total = 0, 0, 0, 0, 0, 0, 0, 0
density = "LOW"

with tab_feed:
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown('<div class="section-label">Camera Stream Ingestion</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="traffic-monitor",
            video_processor_factory=TrafficProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

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
        st.markdown('<div class="section-label">Real-Time Telemetry</div>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Volume</div>
                <div class="metric-val">{total}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            d_class = "density-low" if density == "LOW" else ("density-medium" if density == "MEDIUM" else "density-high")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Traffic Density</div>
                <div style="margin-top: 6px;">
                    <span class="density-tag {d_class}">{density} ({current} Active)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Cars</div>
                <div class="metric-val val-cyan">{cars}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Buses</div>
                <div class="metric-val val-purple">{buses}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Motorcycles</div>
                <div class="metric-val val-amber">{motorcycles}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Trucks</div>
                <div class="metric-val val-emerald">{trucks}</div>
            </div>
            """, unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        with f1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Northbound (UP)</div>
                <div class="metric-val val-cyan">{up}</div>
            </div>
            """, unsafe_allow_html=True)

        with f2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Southbound (DOWN)</div>
                <div class="metric-val val-rose">{down}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Analytics & Visualizations
# ---------------------------------------------------------
with tab_analytics:
    st.markdown('<div class="section-label">Traffic Distribution Analytics</div>', unsafe_allow_html=True)
    
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
            title="Vehicle Category Breakdown",
            color_discrete_sequence=['#38BDF8', '#F59E0B', '#A78BFA', '#10B981']
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC', family="Inter"),
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15)
        )
        st.plotly_chart(fig_donut, width='stretch')

    with chart_col2:
        fig_flow = go.Figure(data=[
            go.Bar(name='Northbound (UP)', x=['Directional Traffic'], y=[up], marker_color='#38BDF8'),
            go.Bar(name='Southbound (DOWN)', x=['Directional Traffic'], y=[down], marker_color='#F43F5E')
        ])
        fig_flow.update_layout(
            title="Directional Traffic Volume",
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC', family="Inter"),
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15)
        )
        st.plotly_chart(fig_flow, width='stretch')

    st.divider()
    st.markdown('<div class="section-label">Time Series Traffic Trend</div>', unsafe_allow_html=True)

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
                    title="Historical Vehicle Counts",
                    color_discrete_map={'Cars': '#38BDF8', 'Motorcycles': '#F59E0B', 'Buses': '#A78BFA', 'Trucks': '#10B981'}
                )
                fig_trend.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.6)',
                    font=dict(color='#F8FAFC', family="Inter"),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#334155'),
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_trend, width='stretch')
        except Exception:
            st.info("Loading time series dataset...")
    else:
        st.info("Telemetry data logging active. Time-series chart populates automatically.")

# ---------------------------------------------------------
# Tab 3: Data Export
# ---------------------------------------------------------
with tab_logs:
    st.markdown('<div class="section-label">Live Telemetry Logs</div>', unsafe_allow_html=True)

    target_csv = CSV_FILE if os.path.exists(CSV_FILE) else (HISTORICAL_CSV if os.path.exists(HISTORICAL_CSV) else None)

    if target_csv and os.path.exists(target_csv):
        df_log = pd.read_csv(target_csv)
        st.dataframe(df_log.tail(100), width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        with open(target_csv, "rb") as file:
            st.download_button(
                label="Download Log Dataset (CSV)",
                data=file,
                file_name=f"vehixai_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
    else:
        st.warning("No telemetry log records available.")