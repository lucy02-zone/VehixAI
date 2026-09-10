import streamlit as st
import cv2
import av
import threading
import csv
import os
from datetime import datetime

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from streamlit_autorefresh import st_autorefresh


st.set_page_config(
    page_title="AI Traffic Monitoring",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI-Based Real-Time Traffic Monitoring")


VEHICLE_CLASSES = [2, 3, 5, 7]
CSV_FILE = "results/live_traffic_data.csv"


os.makedirs("results", exist_ok=True)


def save_data(
    cars,
    motorcycles,
    buses,
    trucks,
    up,
    down,
    current,
    density
):

    file_exists = os.path.exists(CSV_FILE)

    total = cars + motorcycles + buses + trucks

    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "Timestamp",
                "Cars",
                "Motorcycles",
                "Buses",
                "Trucks",
                "Total",
                "UP",
                "DOWN",
                "Current Vehicles",
                "Density"
            ])

        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            cars,
            motorcycles,
            buses,
            trucks,
            total,
            up,
            down,
            current,
            density
        ])


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

        img = frame.to_ndarray(
            format="bgr24"
        )

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

        cv2.line(
            img,
            (0, line_y),
            (width, line_y),
            (0, 0, 255),
            3
        )

        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            current_count = len(track_ids)

            for box, cls, track_id in zip(
                boxes,
                classes,
                track_ids
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                class_id = int(cls)

                names = {
                    2: "Car",
                    3: "Motorcycle",
                    5: "Bus",
                    7: "Truck"
                }

                name = names.get(
                    class_id,
                    "Vehicle"
                )

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    img,
                    (center_x, center_y),
                    5,
                    (255, 0, 0),
                    -1
                )

                cv2.putText(
                    img,
                    f"{name} ID:{track_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                if track_id in self.previous_positions:

                    previous_y = self.previous_positions[
                        track_id
                    ]

                    # DOWN
                    if (
                        previous_y < line_y
                        and center_y >= line_y
                        and track_id not in self.counted_ids
                    ):

                        self.counted_ids.add(
                            track_id
                        )

                        with self.lock:

                            self.down_count += 1

                            if class_id == 2:
                                self.car_count += 1

                            elif class_id == 3:
                                self.motorcycle_count += 1

                            elif class_id == 5:
                                self.bus_count += 1

                            elif class_id == 7:
                                self.truck_count += 1

                    # UP
                    elif (
                        previous_y > line_y
                        and center_y <= line_y
                        and track_id not in self.counted_ids
                    ):

                        self.counted_ids.add(
                            track_id
                        )

                        with self.lock:

                            self.up_count += 1

                            if class_id == 2:
                                self.car_count += 1

                            elif class_id == 3:
                                self.motorcycle_count += 1

                            elif class_id == 5:
                                self.bus_count += 1

                            elif class_id == 7:
                                self.truck_count += 1

                self.previous_positions[
                    track_id
                ] = center_y

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

        # Save once every 5 seconds
        current_second = datetime.now().second

        if (
            current_second % 5 == 0
            and current_second != self.last_saved_second
        ):

            self.last_saved_second = current_second

            save_data(
                cars,
                motorcycles,
                buses,
                trucks,
                up,
                down,
                current_count,
                density
            )

        # Dashboard overlay

        cv2.rectangle(
            img,
            (10, 10),
            (330, 220),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            img,
            f"Cars: {cars}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"Motorcycles: {motorcycles}",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"Buses: {buses}",
            (25, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"Trucks: {trucks}",
            (25, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"UP: {up}  DOWN: {down}",
            (25, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"Density: {density}",
            (25, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


st_autorefresh(
    interval=1000,
    key="traffic_refresh"
)


ctx = webrtc_streamer(
    key="traffic-monitor",
    video_processor_factory=TrafficProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
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

    total = (
        cars
        + motorcycles
        + buses
        + trucks
    )

    if current <= 5:

        density = "LOW"

    elif current <= 10:

        density = "MEDIUM"

    else:

        density = "HIGH"


    st.divider()

    st.subheader("📊 Real-Time Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🚗 Cars", cars)

    with col2:
        st.metric(
            "🏍️ Motorcycles",
            motorcycles
        )

    with col3:
        st.metric("🚌 Buses", buses)

    with col4:
        st.metric("🚚 Trucks", trucks)


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🚘 Total",
            total
        )

    with col2:
        st.metric(
            "🚦 Current",
            current
        )

    with col3:
        st.metric(
            "⬆️ UP",
            up
        )

    with col4:
        st.metric(
            "⬇️ DOWN",
            down
        )


    st.subheader("🚦 Traffic Density")

    if density == "LOW":
        st.success("LOW TRAFFIC")

    elif density == "MEDIUM":
        st.warning("MEDIUM TRAFFIC")

    else:
        st.error("HIGH TRAFFIC")


    st.divider()

    st.subheader("📥 Traffic Data")

    if os.path.exists(CSV_FILE):

        with open(
            CSV_FILE,
            "rb"
        ) as file:

            st.download_button(
                "Download Traffic CSV",
                file,
                file_name="live_traffic_data.csv",
                mime="text/csv"
            )

        st.success(
            f"Data saved to `{CSV_FILE}`"
        )