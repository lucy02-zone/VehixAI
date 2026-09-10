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
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="VehixAI | Smart Traffic Management & Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Executive Professional Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Padding */
    .stApp {
        background-color: #0B0F17;
        color: #F1F5F9;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header-subtitle {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8;
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 6px;
    }

    /* Traffic Badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
    }
    
    .badge-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .badge-high {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Video Box Wrapper */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #334155;
        background-color: #020617;
    }
    
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Tabs Customization */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Configuration
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
# Video Processor Class
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

        # Draw Monitoring Counting Line (Professional Cyan/Blue overlay)
        cv2.line(img, (0, line_y), (width, line_y), (255, 191, 0), 2)
        cv2.putText(img, "TRAFFIC COUNTING LINE", (15, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 191, 0), 1)

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

                # Bounding boxes in sleek green/cyan
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 118), 2)
                cv2.circle(img, (center_x, center_y), 4, (0, 150, 255), -1)

                # Label badge
                cv2.rectangle(img, (x1, max(y1 - 22, 0)), (x1 + len(name)*10 + 65, max(y1, 22)), (0, 230, 118), -1)
                cv2.putText(img, f"{name} #{track_id}", (x1 + 4, max(y1 - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 25, 47), 1, cv2.LINE_AA)

                if track_id in self.previous_positions:
                    previous_y = self.previous_positions[track_id]

                    # DOWN Crossing
                    if previous_y < line_y and center_y >= line_y and track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        with self.lock:
                            self.down_count += 1
                            if class_id == 2: self.car_count += 1
                            elif class_id == 3: self.motorcycle_count += 1
                            elif class_id == 5: self.bus_count += 1
                            elif class_id == 7: self.truck_count += 1

                    # UP Crossing
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

        # Save data periodically (every 5 seconds)
        current_second = datetime.now().second
        if current_second % 5 == 0 and current_second != self.last_saved_second:
            self.last_saved_second = current_second
            save_data(cars, motorcycles, buses, trucks, up, down, current_count, density)

        # Overlay dashboard info box (semi-transparent slate box)
        overlay = img.copy()
        cv2.rectangle(overlay, (12, 12), (240, 160), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (12, 12), (240, 160), (51, 65, 85), 1)

        cv2.putText(img, "LIVE METRICS", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(img, f"Cars: {cars} | Bikes: {motorcycles}", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(img, f"Buses: {buses} | Trucks: {trucks}", (24, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 245, 249), 1, cv2.LINE_AA)
        cv2.putText(img, f"Flow: UP {up} | DOWN {down}", (24, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (241, 245, 249), 1, cv2.LINE_AA)
        
        density_color = (16, 185, 129) if density == "LOW" else ((245, 158, 11) if density == "MEDIUM" else (239, 68, 68))
        cv2.putText(img, f"Density: {density}", (24, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, density_color, 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/traffic-jam.png", width=70)
    st.title("VehixAI Control")
    st.markdown("<span style='color: #94A3B8; font-size: 13px;'>AI-Powered Traffic Intelligence</span>", unsafe_allow_html=True)
    st.divider()

    st.subheader("⚙️ System Status")
    st.markdown("🟢 **Model**: YOLO (Real-time Tracker)")
    st.markdown("📹 **WebRTC Stream**: Active")
    st.markdown(f"🕒 **System Time**: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    st.subheader("🎯 Auto-Refresh Rate")
    refresh_rate = st.slider("Update Interval (seconds)", min_value=1, max_value=10, value=2)
    st_autorefresh(interval=refresh_rate * 1000, key="traffic_refresh")

    st.divider()
    st.caption("VehixAI Traffic Analytics © 2026")

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">
        <span>🚦 VehixAI Traffic Intelligence Center</span>
    </div>
    <div class="header-subtitle">
        Real-time multi-class vehicle detection, object tracking, flow direction analysis & automated traffic density profiling.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📹 Live Feed & Metrics", "📊 Analytics & Insights", "📄 Data Logs & Export"])

# Data extraction helpers
cars, motorcycles, buses, trucks, up, down, current, total = 0, 0, 0, 0, 0, 0, 0, 0
density = "LOW"

with tab1:
    col_video, col_stats = st.columns([1.6, 1])

    with col_video:
        st.subheader("📹 Real-Time Camera Stream")
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

    with col_stats:
        st.subheader("📊 Executive Summary")

        # Top summary metrics
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Cumulative</div>
                <div class="metric-value">{total}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            badge_class = "badge-low" if density == "LOW" else ("badge-medium" if density == "MEDIUM" else "badge-high")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Density</div>
                <div class="metric-value" style="font-size: 20px; margin-top: 10px;">
                    <span class="badge {badge_class}">{density} ({current} Active)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Vehicle Breakdown Cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚗 Cars</div>
                <div class="metric-value">{cars}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚌 Buses</div>
                <div class="metric-value">{buses}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🏍️ Motorcycles</div>
                <div class="metric-value">{motorcycles}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🚚 Trucks</div>
                <div class="metric-value">{trucks}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Flow directional summary
        f1, f2 = st.columns(2)
        with f1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">⬆️ Northbound (UP)</div>
                <div class="metric-value" style="color: #38BDF8;">{up}</div>
            </div>
            """, unsafe_allow_html=True)
        with f2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">⬇️ Southbound (DOWN)</div>
                <div class="metric-value" style="color: #F43F5E;">{down}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 2: Analytics & Graphs
# ---------------------------------------------------------
with tab2:
    st.subheader("📈 Traffic Distribution & Flow Insights")
    
    col_chart1, col_chart2 = st.columns(2)
    
    # Donut Chart for Vehicle Categories
    with col_chart1:
        labels = ['Cars', 'Motorcycles', 'Buses', 'Trucks']
        values = [cars, motorcycles, buses, trucks]
        if sum(values) == 0:
            values = [1, 1, 1, 1]  # Placeholder visual if initial counts are zero
            
        fig_donut = px.pie(
            names=labels,
            values=values,
            hole=0.55,
            title="Vehicle Type Distribution",
            color_discrete_sequence=['#38BDF8', '#F59E0B', '#8B5CF6', '#10B981']
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC'),
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_donut, width='stretch')

    # Direction Flow Bar Chart
    with col_chart2:
        fig_flow = go.Figure(data=[
            go.Bar(name='Northbound (UP)', x=['Traffic Direction'], y=[up], marker_color='#38BDF8'),
            go.Bar(name='Southbound (DOWN)', x=['Traffic Direction'], y=[down], marker_color='#F43F5E')
        ])
        fig_flow.update_layout(
            title="Directional Traffic Volume",
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#F8FAFC'),
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_flow, width='stretch')

    # Time series graph from logged data if available
    st.divider()
    st.subheader("⏱️ Historical Traffic Trend (Time Series)")

    target_csv = CSV_FILE if os.path.exists(CSV_FILE) else (HISTORICAL_CSV if os.path.exists(HISTORICAL_CSV) else None)

    if target_csv and os.path.getsize(target_csv) > 50:
        try:
            df = pd.read_csv(target_csv)
            if not df.empty and "Timestamp" in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                
                fig_time = px.line(
                    df,
                    x='Timestamp',
                    y=['Cars', 'Motorcycles', 'Buses', 'Trucks'],
                    title="Vehicle Counts Over Time",
                    labels={'value': 'Vehicle Count', 'variable': 'Vehicle Category'},
                    color_discrete_map={'Cars': '#38BDF8', 'Motorcycles': '#F59E0B', 'Buses': '#8B5CF6', 'Trucks': '#10B981'}
                )
                fig_time.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15, 23, 42, 0.5)',
                    font=dict(color='#F8FAFC'),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='#334155'),
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_time, width='stretch')
        except Exception as e:
            st.info("Collecting live trend data... graph will populate shortly.")
    else:
        st.info("Accumulating live telemetry data for time-series analytics...")

# ---------------------------------------------------------
# Tab 3: Data Table & CSV Export
# ---------------------------------------------------------
with tab3:
    st.subheader("📋 Traffic Log Records")

    target_csv = CSV_FILE if os.path.exists(CSV_FILE) else (HISTORICAL_CSV if os.path.exists(HISTORICAL_CSV) else None)

    if target_csv and os.path.exists(target_csv):
        df_logs = pd.read_csv(target_csv)
        st.dataframe(df_logs.tail(50), width='stretch')

        col_dl, col_blank = st.columns([1, 3])
        with col_dl:
            with open(target_csv, "rb") as file:
                st.download_button(
                    label="📥 Download Full CSV Report",
                    data=file,
                    file_name=f"vehixai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.warning("No log data file generated yet. Start the camera stream to log data.")