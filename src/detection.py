from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo26n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam opened successfully!")

# Vehicle class IDs
# COCO dataset classes:
# car = 2
# motorcycle = 3
# bus = 5
# truck = 7

vehicle_classes = [2, 3, 5, 7]

while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO
    results = model(
        frame,
        classes=vehicle_classes,
        conf=0.40
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow(
        "AI Traffic Monitoring - Vehicles",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release
cap.release()
cv2.destroyAllWindows()

print("Program stopped.")