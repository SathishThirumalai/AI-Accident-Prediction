from ultralytics import YOLO
import cv2
import numpy as np
import datetime
import os

# -------------------------
# CONFIGURATION
# -------------------------
VIDEO_PATH = "sample_videos/near_miss_demo.mp4"
NEAR_MISS_THRESHOLD = 50   # pixel distance
VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]
VEHICLE_COLOR = (0, 255, 0)
NEAR_MISS_COLOR = (0, 0, 255)
ALERT_TEXT = "⚠️ Near Miss Detected!"

# -------------------------
# SETUP
# -------------------------
os.makedirs("results", exist_ok=True)
log_file = open("results/logs.txt", "w")

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Error: Video not found or cannot be opened.")
    exit()

frame_count = 0
print("✅ System Initialized. Press 'Q' to quit.")

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎞️ End of video.")
        break

    frame_count += 1
    results = model(frame, stream=True)

    vehicles = []

    # Detect vehicles
    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls
        for box, cls in zip(boxes, classes):
            label = model.names[int(cls)]
            if label in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                vehicles.append(((cx, cy), (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), VEHICLE_COLOR, 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, VEHICLE_COLOR, 1)

    # Check near misses
    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):
            (c1, box1), (c2, box2) = vehicles[i], vehicles[j]
            dist = np.linalg.norm(np.array(c1) - np.array(c2))

            if dist < NEAR_MISS_THRESHOLD:
                cv2.line(frame, c1, c2, NEAR_MISS_COLOR, 2)
                cv2.putText(frame, ALERT_TEXT, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, NEAR_MISS_COLOR, 3)

                # Log near miss event
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"[{timestamp}] Near miss between vehicles at {dist:.2f}px\n")

    cv2.imshow("AI Accident Prediction & Road Hazard Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
print("✅ Process Completed. Logs saved in /results/logs.txt")
