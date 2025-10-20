from ultralytics import YOLO
import cv2
import time

def detect_birds_yolov8(video_path, confidence_threshold=0.5):
    # Load a pre-trained YOLOv8 model
    # 'yolov8n.pt' is the nano model, good balance of speed/accuracy
    # For better accuracy, consider 'yolov8s.pt' or 'yolov8m.pt'
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    land_timestamps = []
    # To track multiple birds, you'd need an actual object tracker
    # For a simple "a bird appeared" timestamp, we just check presence
    prev_bird_present = False
    bird_present_since = None # Timestamp when bird first appeared

    # Define a class ID for 'bird'. You'll need to check the model's classes.
    # For COCO dataset, 'bird' is usually class 14.
    # Run `model.names` to see all classes.
    bird_class_id = 14 # This is often the bird class ID in COCO dataset

    print("Processing video with YOLOv8...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000.0

        # Perform inference
        # stream=True processes frames as a stream, slightly faster
        results = model(frame, stream=True, conf=confidence_threshold)

        current_bird_present = False
        for r in results:
            boxes = r.boxes # Bounding boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == bird_class_id: # Check if it's a bird
                    conf = box.conf[0]
                    # Optional: filter by size or location if needed
                    # x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # if (x2-x1)*(y2-y1) > min_bird_area:
                    current_bird_present = True
                    break # Found at least one bird, no need to check others

            if current_bird_present:
                break # Found a bird in this result set

        if current_bird_present and not prev_bird_present:
            # Bird just appeared
            land_timestamps.append(current_time_sec)
            print(f"Bird landed at {current_time_sec:.2f} seconds.")
            bird_present_since = current_time_sec # Mark the start time

        # If a bird was present and now isn't, or has been present for a while,
        # reset the state for next potential landing
        elif not current_bird_present and prev_bird_present:
            # Bird just left, reset "present" state
            bird_present_since = None
        elif current_bird_present and prev_bird_present and bird_present_since is not None:
            # If bird has been present for a long time (e.g., > 10 seconds),
            # you might want to consider it "gone" and re-land if it moves
            # or disappears briefly. This logic needs careful tuning.
            pass


        prev_bird_present = current_bird_present

        # Optional: display frame with detections
        # annotated_frame = r.plot()
        # cv2.imshow('YOLOv8 Detections', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection complete with YOLOv8.")
    return land_timestamps

# Example Usage:
video_file = '2025-10-20-165625.webm'
# IMPORTANT: You might need to change bird_class_id depending on your model.
# To find it:
model = YOLO('yolov8n.pt')
print(model.names)
bird_landings = detect_birds_yolov8(video_file, confidence_threshold=0.6)
print("\nFinal Bird Landing Timestamps (seconds):")
for ts in bird_landings:
    print(f"- {ts:.2f}")