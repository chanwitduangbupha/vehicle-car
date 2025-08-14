import cv2
import numpy as np

polygon_points = []

video_path = r"carsvid.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video file")
    exit()

def mouse_callback(event, x, y, flags, param):
    global polygon_points
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"Point Added: (X: {x}, Y: {y})")

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot read frame or video ended.")
        break

    frame = cv2.resize(frame, (1920, 1080))

    if len(polygon_points) > 1:
        cv2.polylines(frame, [np.array(polygon_points, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

print("\nPolygon Points:")
for point in polygon_points:
    print(f"X: {point[0]}, Y: {point[1]}")

