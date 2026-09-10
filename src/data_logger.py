from ultralytics import YOLO
import cv2
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------
# Load YOLO
# -----------------------------
model = YOLO("yolo26n.pt")

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

# -----------------------------
# Counting variables
# -----------------------------
line_y = 400

previous_positions = {}
counted_ids = set()

car_count = 0
motorcycle_count = 0
bus_count = 0
truck_count = 0

up_count = 0
down_count = 0

# -----------------------------
# CSV setup
# -----------------------------
os.makedirs("results", exist_ok=True)

csv_file = "results/traffic_data.csv"

with open(csv_file, "w", newline="") as file:

    writer = csv.writer(file)

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

    # -----------------------------
    # Main loop
    # -----------------------------
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=vehicle_classes,
            conf=0.40
        )

        result = results[0]

        current_vehicle_count = 0

        # Draw counting line
        cv2.line(
            frame,
            (0, line_y),
            (frame.shape[1], line_y),
            (0, 0, 255),
            3
        )

        if result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()

            current_vehicle_count = len(track_ids)

            for box, cls, track_id in zip(
                boxes,
                classes,
                track_ids
            ):

                x1, y1, x2, y2 = map(int, box)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                class_id = int(cls)

                # Draw box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                # Draw ID
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # Draw center
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (255, 0, 0),
                    -1
                )

                # Check crossing
                if track_id in previous_positions:

                    previous_y = previous_positions[track_id]

                    # DOWN
                    if (
                        previous_y < line_y
                        and center_y >= line_y
                        and track_id not in counted_ids
                    ):

                        counted_ids.add(track_id)

                        down_count += 1

                        if class_id == 2:
                            car_count += 1

                        elif class_id == 3:
                            motorcycle_count += 1

                        elif class_id == 5:
                            bus_count += 1

                        elif class_id == 7:
                            truck_count += 1

                    # UP
                    elif (
                        previous_y > line_y
                        and center_y <= line_y
                        and track_id not in counted_ids
                    ):

                        counted_ids.add(track_id)

                        up_count += 1

                        if class_id == 2:
                            car_count += 1

                        elif class_id == 3:
                            motorcycle_count += 1

                        elif class_id == 5:
                            bus_count += 1

                        elif class_id == 7:
                            truck_count += 1

                previous_positions[track_id] = center_y

        # -----------------------------
        # Calculate density
        # -----------------------------
        if current_vehicle_count <= 5:
            density = "LOW"

        elif current_vehicle_count <= 10:
            density = "MEDIUM"

        else:
            density = "HIGH"

        total_count = (
            car_count
            + motorcycle_count
            + bus_count
            + truck_count
        )

        # -----------------------------
        # Save data
        # -----------------------------
        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        writer.writerow([
            timestamp,
            car_count,
            motorcycle_count,
            bus_count,
            truck_count,
            total_count,
            up_count,
            down_count,
            current_vehicle_count,
            density
        ])

        file.flush()

        # -----------------------------
        # Display statistics
        # -----------------------------
        cv2.putText(
            frame,
            f"Cars: {car_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Motorcycles: {motorcycle_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Buses: {bus_count}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Trucks: {truck_count}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Total: {total_count}",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"UP: {up_count}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"DOWN: {down_count}",
            (20, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Current: {current_vehicle_count}",
            (20, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Density: {density}",
            (20, 325),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "AI Traffic Monitoring - Data Logger",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("\nTraffic monitoring stopped.")
print("Data saved to:", csv_file)

# -----------------------------
# Generate graph
# -----------------------------

try:

    import pandas as pd

    data = pd.read_csv(csv_file)

    # Remove duplicate timestamps if necessary
    data = data.drop_duplicates(
        subset=["Timestamp"]
    )

    # -----------------------------
    # Vehicle count graph
    # -----------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(
        data.index,
        data["Cars"],
        label="Cars"
    )

    plt.plot(
        data.index,
        data["Motorcycles"],
        label="Motorcycles"
    )

    plt.plot(
        data.index,
        data["Buses"],
        label="Buses"
    )

    plt.plot(
        data.index,
        data["Trucks"],
        label="Trucks"
    )

    plt.xlabel("Time")
    plt.ylabel("Vehicle Count")

    plt.title("Traffic Vehicle Count")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/vehicle_count_graph.png"
    )

    plt.show()

    # -----------------------------
    # Total traffic graph
    # -----------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(
        data.index,
        data["Total"]
    )

    plt.xlabel("Time")
    plt.ylabel("Total Vehicles")

    plt.title("Total Traffic Over Time")

    plt.tight_layout()

    plt.savefig(
        "results/total_traffic_graph.png"
    )

    plt.show()

except Exception as e:

    print("Graph generation error:", e)