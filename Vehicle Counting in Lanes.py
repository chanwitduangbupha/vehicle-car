import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *
import os

# ======== ตั้งค่า path วิดีโอ ========
video_path = r"carsvid.mp4"

# ======== ตรวจสอบว่าไฟล์วิดีโอมีอยู่จริง ========
if not os.path.exists(video_path):
    print(f"❌ Error: Video file not found at {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Cannot open video file. Please check the file format or codec.")
    exit()

model = YOLO('yolov8n.pt')

classnames = []
with open(r"classes.txt", 'r') as f:
    classnames = f.read().splitlines()

road_zoneA = np.array([[308, 789], [711, 807], [694, 492], [415, 492], [309, 790]], np.int32)
road_zoneB = np.array([[727, 797], [1123, 812], [1001, 516], [741, 525], [730, 795]], np.int32)
road_zoneC = np.array([[1116, 701], [1533, 581], [1236, 367], [1009, 442], [1122, 698]], np.int32)

tracker = Sort()
zoneAcounter, zoneBcounter, zoneCcounter = [], [], []

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ End of video or cannot read frame.")
        break

    frame = cv2.resize(frame, (1920, 1080))
    results = model(frame)
    current_detections = np.empty((0, 5))

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100)
            class_detect = classnames[int(box.cls[0])]

            if (class_detect in ['car', 'truck', 'bus']) and conf > 60:
                detections = np.array([x1, y1, x2, y2, conf])
                current_detections = np.vstack([current_detections, detections])

    # วาดโซน
    cv2.polylines(frame, [road_zoneA], False, (0, 0, 255), 8)
    cv2.polylines(frame, [road_zoneB], False, (0, 255, 255), 8)
    cv2.polylines(frame, [road_zoneC], False, (255, 0, 0), 8)

    # ติดตามวัตถุ
    track_results = tracker.update(current_detections)
    for x1, y1, x2, y2, id in track_results:
        x1, y1, x2, y2, id = map(int, [x1, y1, x2, y2, id])
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2 - 40

        if cv2.pointPolygonTest(road_zoneA, (cx, cy), False) >= 0 and id not in zoneAcounter:
            zoneAcounter.append(id)
        if cv2.pointPolygonTest(road_zoneB, (cx, cy), False) >= 0 and id not in zoneBcounter:
            zoneBcounter.append(id)
        if cv2.pointPolygonTest(road_zoneC, (cx, cy), False) >= 0 and id not in zoneCcounter:
            zoneCcounter.append(id)

    # แสดงจำนวนรถ
    cvzone.putTextRect(frame, f'LANE A Vehicles = {len(zoneAcounter)}', [1000, 99], thickness=4, scale=2.3, border=1)
    cvzone.putTextRect(frame, f'LANE B Vehicles = {len(zoneBcounter)}', [1000, 140], thickness=4, scale=2.3, border=1)
    cvzone.putTextRect(frame, f'LANE C Vehicles = {len(zoneCcounter)}', [1000, 180], thickness=4, scale=2.3, border=1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

