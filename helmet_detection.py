from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model (lightweight)
model = YOLO("yolov8n.pt")

# Load test image
img = cv2.imread("test.jpg")

# Run detection
results = model(img)

# Show results
results[0].show()