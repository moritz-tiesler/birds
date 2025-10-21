from ultralytics import solutions

inf = solutions.Inference(
    model="yolov8n.onnx",  # you can use any model that Ultralytics support, i.e. YOLO11, or custom trained model
    source="http://192.168.2.38:5000/video_feed"

)

inf.inference()