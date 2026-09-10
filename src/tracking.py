from ultralytics import YOLO
import cv2

# --------------------------------
# 1. Load YOLO model
# --------------------------------
model = YOLO("yolo26n.pt")

# --------------------------------
# 2. Open webcam
# --------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully!")

# Vehicle classes from COCO
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
vehicle_classes = [2, 3, 5, 7]

# --------------------------------
# 3. Process frames
# --------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # --------------------------------
    # 4. YOLO + ByteTrack
    # --------------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        conf=0.40
    )

    # --------------------------------
    # 5. Draw tracking results
    # --------------------------------
    result = results[0]

if result.boxes.id is not None:

    track_ids = result.boxes.id.int().cpu().tolist()

    print("Current vehicle IDs:", track_ids)

annotated_frame = result.plot()

    # --------------------------------
    # 6. Display
    # --------------------------------
    cv2.imshow(
        "AI Traffic Monitoring - Tracking",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------
# 7. Release resources
# --------------------------------
cap.release()
cv2.destroyAllWindows()

print("Tracking stopped.")