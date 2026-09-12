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
# Enterprise Dark Theme & Glassmorphism Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Main App Background with ambient dark radial glow */
    .stApp {
        background-color: #0B0F17 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.08) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(14, 165, 233, 0.06) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(99, 102, 241, 0.05) 0px, transparent 50%) !important;
        color: #F8FAFC !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: rgba(11, 15, 23, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }
    
    /* Sidebar Overrides */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #CBD5E1 !important;
    }

    .sidebar-brand-container {
        padding: 14px 16px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 18px;
    }
    
    .sidebar-brand {
        font-size: 21px;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .sidebar-sub {
        font-size: 11px;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }

    .sidebar-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    /* Executive Top Banner */
    .control-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .control-title {
        font-size: 22px;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.4px;
    }
    
    .control-sub {
        font-size: 13.5px;
        color: #94A3B8;
        margin-top: 4px;
        font-weight: 400;
    }

    .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(37, 99, 235, 0.3);
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        color: #60A5FA;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }

    /* Glowing Live Status Indicator */
    .live-pulse {
        height: 7px;
        width: 7px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 10px #22C55E;
        animation: pulse_glow 2s infinite;
    }

    @keyframes pulse_glow {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    .status-dot {
        height: 8px;
        width: 8px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 6px rgba(34, 197, 94, 0.6);
    }

    /* Custom Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        padding: 5px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 18px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 13.5px !important;
        padding: 0 22px !important;
        border: none !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }
    
    /* High-Tech Telemetry Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.4), 0 0 20px rgba(56, 189, 248, 0.08);
    }

    .metric-card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .metric-label {
        font-size: 11.5px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94A3B8;
    }

    .metric-icon-badge {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .icon-indigo { background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.25); color: #818CF8; }
    .icon-cyan { background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.25); color: #38BDF8; }
    .icon-amber { background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.25); color: #F59E0B; }
    .icon-purple { background: rgba(167, 139, 250, 0.12); border: 1px solid rgba(167, 139, 250, 0.25); color: #A78BFA; }
    .icon-emerald { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.25); color: #10B981; }
    .icon-rose { background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.25); color: #F43F5E; }
    
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 6px;
        letter-spacing: -0.5px;
    }

    .val-indigo { color: #818CF8; }
    .val-cyan { color: #38BDF8; }
    .val-amber { color: #F59E0B; }
    .val-purple { color: #C084FC; }
    .val-emerald { color: #34D399; }
    .val-rose { color: #FB7185; }

    /* Density Tags */
    .density-tag {
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .density-low {
        background-color: rgba(34, 197, 94, 0.12);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
    .density-medium {
        background-color: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.25);
    }
    .density-high {
        background-color: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }

    .section-label {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #CBD5E1;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::before {
        content: "";
        display: inline-block;
        width: 3px;
        height: 14px;
        background: #3B82F6;
        border-radius: 2px;
    }

    /* Container Glass Panel */
    .glass-panel {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Download Button override */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.45) !important;
    }

    /* Dataframe Table styling */
    [data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
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
        model_path = "yolo26n.pt" if os.path.exists("yolo26n.pt") else "yolov8n.pt"
        self.model = YOLO(model_path)
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
# Sidebar System Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-container">
        <div class="sidebar-brand">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="2" width="20" height="8" rx="2"></rect>
                <rect x="2" y="14" width="20" height="8" rx="2"></rect>
                <line x1="6" y1="6" x2="6.01" y2="6"></line>
                <line x1="6" y1="18" x2="6.01" y2="18"></line>
            </svg>
            VehixAI Core
        </div>
        <div class="sidebar-sub">Traffic Operations Center</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="metric-label" style="margin-bottom: 8px;">Architecture Specifications</div>
        <div style="font-size: 12.5px; color: #CBD5E1; line-height: 1.8;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span class="status-dot"></span><b>Inference</b></span>
                <span style="color: #60A5FA; font-weight: 600;">YOLOv8 PyTorch</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span class="status-dot"></span><b>Tracking Engine</b></span>
                <span style="color: #34D399; font-weight: 600;">ByteTrack</span>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span><span class="status-dot"></span><b>Stream Protocol</b></span>
                <span style="color: #FBBF24; font-weight: 600;">OpenCV / WebRTC</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="metric-label" style="margin-bottom: 8px;">Density Threshold Profiling</div>
        <div style="font-size: 12px; color: #94A3B8; line-height: 1.7;">
            <div style="display: flex; justify-content: space-between; padding: 2px 0;">
                <span style="color: #4ADE80; font-weight: 600;">● Low Flow</span>
                <span>0 – 5 vehicles</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 2px 0;">
                <span style="color: #FBBF24; font-weight: 600;">● Medium Flow</span>
                <span>6 – 10 vehicles</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 2px 0;">
                <span style="color: #F87171; font-weight: 600;">● High Density</span>
                <span>11+ vehicles</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="padding: 12px 16px; background: rgba(15, 23, 42, 0.4); border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); margin-top: 20px;">
        <div style="font-size: 10.5px; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">System Runtime Clock</div>
        <div style="font-size: 12.5px; color: #94A3B8; font-weight: 600; margin-top: 2px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Executive Top Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="control-header">
    <div>
        <div class="brand-badge">
            <span class="live-pulse"></span> SYSTEM ONLINE
        </div>
        <div class="control-title">VehixAI Intelligent Traffic Analytics Platform</div>
        <div class="control-sub">Real-Time Autonomous Multi-Vehicle Telemetry, Flow Profiling, and Directional Tracking</div>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
        <div style="padding: 8px 16px; background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; text-align: right;">
            <div style="font-size: 10px; color: #64748B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px;">Inference Latency</div>
            <div style="font-size: 13px; color: #34D399; font-weight: 700;">Real-Time <span style="font-size: 10px; color: #94A3B8;">(GPU/CPU)</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------
tab_feed, tab_analytics, tab_logs = st.tabs([
    "📹 Live Camera Feed & Telemetry", 
    "📊 Flow Analytics & Trends", 
    "📁 Telemetry Data Logs"
])

cars, motorcycles, buses, trucks, up, down, current, total = 0, 0, 0, 0, 0, 0, 0, 0
density = "LOW"

with tab_feed:
    col_left, col_right = st.columns([1.55, 1])

    with col_left:
        st.markdown('<div class="section-label">Live Camera Stream & Object Detection</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-label">Real-Time Telemetry Breakdown</div>', unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-top">
                    <div class="metric-label">Total Volume</div>
                    <div class="metric-icon-badge icon-indigo">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                    </div>
                </div>
                <div class="metric-val val-indigo">{total}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            d_class = "density-low" if density == "LOW" else ("density-medium" if density == "MEDIUM" else "density-high")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-top">
                    <div class="metric-label">Traffic Density</div>
                    <div class="metric-icon-badge icon-emerald">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                    </div>
                </div>
                <div style="margin-top: 10px;">
                    <span class="density-tag {d_class}">
                        <span class="live-pulse" style="width:5px; height:5px;"></span> {density} ({current} Active)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-top">
                    <div class="metric-label">Cars</div>
                    <div class="metric-icon-badge icon-cyan">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H7c-.7 0-1.3.3-1.8.7C4.3 8.6 3 10 3 10s-2.7.6-4.5 1.1C.7 11.3 0 12.1 0 13v3c0 .6.4 1 1 1h2"></path><circle cx="7" cy="17" r="2"></circle><circle cx="17" cy="17" r="2"></circle></svg>
                    </div>
                </div>
                <div class="metric-val val-cyan">{cars}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-top">
                    <div class="metric-label">Buses</div>
                    <div class="metric-icon-badge icon-purple">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="15" rx="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><circle cx="7" cy="15" r="1"></circle><circle cx="17" cy="15" r="1"></circle></svg>
                    </div>
                </div>
                <div class="metric-val val-purple">{buses}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-top">
                    <div class="metric-label">Motorcycles</div>
                    <div class="metric-icon-badge icon-amber">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="5.5" cy="17.5" r="3.5"></circle><circle cx="18.5" cy="17.5" r="3.5"></circle><path d="M15 6h2.5l2.5 4.5M9 17.5L12 8h4.5l2 4"></path></svg>
                    </div>
                </div>
                <div class="metric-val val-amber">{motorcycles}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-top">
                    <div class="metric-label">Trucks</div>
                    <div class="metric-icon-badge icon-emerald">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="3" width="15" height="13" rx="2"></rect><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"></polygon><circle cx="5.5" cy="18.5" r="2.5"></circle><circle cx="18.5" cy="18.5" r="2.5"></circle></svg>
                    </div>
                </div>
                <div class="metric-val val-emerald">{trucks}</div>
            </div>
            """, unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        with f1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-top">
                    <div class="metric-label">Northbound (UP)</div>
                    <div class="metric-icon-badge icon-cyan">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>
                    </div>
                </div>
                <div class="metric-val val-cyan">{up}</div>
            </div>
            """, unsafe_allow_html=True)

        with f2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-top">
                    <div class="metric-label">Southbound (DOWN)</div>
                    <div class="metric-icon-badge icon-rose">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><polyline points="19 12 12 19 5 12"></polyline></svg>
                    </div>
                </div>
                <div class="metric-val val-rose">{down}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Analytics & Visualizations
# ---------------------------------------------------------
with tab_analytics:
    st.markdown('<div class="section-label">Traffic Distribution & Class Distribution</div>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        labels = ['Cars', 'Motorcycles', 'Buses', 'Trucks']
        values = [cars, motorcycles, buses, trucks]
        if sum(values) == 0:
            values = [1, 1, 1, 1]

        fig_donut = px.pie(
            names=labels,
            values=values,
            hole=0.62,
            title="Vehicle Class Share",
            color_discrete_sequence=['#38BDF8', '#F59E0B', '#C084FC', '#34D399']
        )
        fig_donut.update_traces(
            hoverinfo='label+percent+value',
            textinfo='percent',
            marker=dict(line=dict(color='#0B0F17', width=2))
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(15, 23, 42, 0.4)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC', family="Plus Jakarta Sans"),
            margin=dict(t=50, b=30, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(color="#CBD5E1")),
            title_font=dict(size=16, color="#F8FAFC")
        )
        st.plotly_chart(fig_donut, width='stretch')

    with chart_col2:
        fig_flow = go.Figure(data=[
            go.Bar(
                name='Northbound (UP)',
                x=['Directional Volume'],
                y=[up],
                marker_color='#38BDF8',
                marker_line=dict(color='rgba(56, 189, 248, 0.5)', width=1.5)
            ),
            go.Bar(
                name='Southbound (DOWN)',
                x=['Directional Volume'],
                y=[down],
                marker_color='#FB7185',
                marker_line=dict(color='rgba(251, 113, 133, 0.5)', width=1.5)
            )
        ])
        fig_flow.update_layout(
            title="Directional Traffic Breakdown",
            barmode='group',
            paper_bgcolor='rgba(15, 23, 42, 0.4)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC', family="Plus Jakarta Sans"),
            margin=dict(t=50, b=30, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(color="#CBD5E1")),
            yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.06)', zeroline=False),
            title_font=dict(size=16, color="#F8FAFC")
        )
        st.plotly_chart(fig_flow, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Historical Time Series Traffic Trend</div>', unsafe_allow_html=True)

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
                    title="Time-Series Vehicle Counts",
                    color_discrete_map={'Cars': '#38BDF8', 'Motorcycles': '#F59E0B', 'Buses': '#C084FC', 'Trucks': '#34D399'}
                )
                fig_trend.update_layout(
                    paper_bgcolor='rgba(15, 23, 42, 0.4)',
                    plot_bgcolor='rgba(11, 15, 23, 0.6)',
                    font=dict(color='#F8FAFC', family="Plus Jakarta Sans"),
                    xaxis=dict(showgrid=False, gridcolor='rgba(255, 255, 255, 0.06)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.06)'),
                    margin=dict(t=50, b=30, l=20, r=20),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(color="#CBD5E1")),
                    title_font=dict(size=16, color="#F8FAFC")
                )
                st.plotly_chart(fig_trend, width='stretch')
        except Exception:
            st.info("Loading time series dataset...")
    else:
        st.info("Telemetry logging active. Time-series chart populates automatically as data streams in.")

# ---------------------------------------------------------
# Tab 3: Data Export
# ---------------------------------------------------------
with tab_logs:
    st.markdown('<div class="section-label">Real-Time Telemetry Log Records</div>', unsafe_allow_html=True)

    target_csv = CSV_FILE if os.path.exists(CSV_FILE) else (HISTORICAL_CSV if os.path.exists(HISTORICAL_CSV) else None)

    if target_csv and os.path.exists(target_csv):
        df_log = pd.read_csv(target_csv)
        st.dataframe(df_log.tail(100), width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        with open(target_csv, "rb") as file:
            st.download_button(
                label="📥 Export Full Telemetry Log Dataset (CSV)",
                data=file,
                file_name=f"vehixai_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch'
            )
    else:
        st.warning("No telemetry log records available yet.")