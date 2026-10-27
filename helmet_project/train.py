from ultralytics import YOLO

# load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# train the model on your dataset
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)