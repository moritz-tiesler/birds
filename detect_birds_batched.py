from ultralytics import YOLO
import cv2
import time
import numpy as np

import sys

def detect_birds_yolov8_batched_with_skipping(
    video_path,
    confidence_threshold=0.5,
    batch_size=8, # Now explicitly passed to .predict()
    frame_skip_interval=5
):
    """
    Detects bird landing timestamps in a video using YOLOv8, with batch processing and frame skipping.
    Uses model.predict() with the 'batch' argument for explicit batch inference.

    Args:
        video_path (str): Path to the input video file.
        confidence_threshold (float): Minimum confidence to consider a detection valid.
        batch_size (int): Number of frames to process in a single batch (passed to .predict()).
        frame_skip_interval (int): Process every Nth frame (e.g., 1 for every frame, 5 for every 5th frame).

    Returns:
        list: A list of timestamps (in seconds) where birds are detected to have "landed".
    """
    # Load the ONNX model, explicitly defining the task
    # Ensure 'yolov8n_dynamic_half.onnx' was exported with dynamic=True and half=True
    model = YOLO('yolov8n.onnx', task='detect')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []

    land_timestamps = []
    prev_bird_present = False
    bird_class_id = 14 # Common COCO ID for 'bird'

    frame_buffer = []
    frame_info_buffer = [] # (current_time_sec, original_frame_idx)

    print(f"Processing video '{video_path}' with YOLOv8 (batch size: {batch_size}, skip: {frame_skip_interval})...")

    frame_count = 0
    # Capture FPS for a more accurate skip calculation for final batch
    # Not strictly needed if processing final batch as is, but good for reporting
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video or error, process any remaining frames in the buffer
            if frame_buffer:
                print(f"Processing final batch of {len(frame_buffer)} frames.")
                # Pass remaining frames to .predict()
                # Use stream=False for batching, but 'batch' arg is now primary
                batch_results = model.predict(
                    source=frame_buffer,
                    conf=confidence_threshold,
                    imgsz=640, # Explicitly set imgsz, especially if model was exported with this.
                    stream=False, # Important for processing the list as a single batch
                    verbose=True, # Set to True temporarily to see detailed speed stats for the batch
                    batch=batch_size # <--- The key addition!
                )

                for i, r in enumerate(batch_results):
                    current_time_sec, original_frame_idx = frame_info_buffer[i]
                    current_bird_present = False
                    boxes = r.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id == bird_class_id:
                            current_bird_present = True
                            break

                    if current_bird_present and not prev_bird_present:
                        land_timestamps.append(current_time_sec)
                        print(f"Bird landed at {current_time_sec:.2f} seconds (frame {original_frame_idx}).")
                    prev_bird_present = current_bird_present
            break # Exit the loop after processing remaining frames

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000.0

        if frame_count % frame_skip_interval == 0:
            frame_buffer.append(frame)
            frame_info_buffer.append((current_time_sec, frame_count))

            if len(frame_buffer) == batch_size:
                # Perform batch inference using .predict() with the 'batch' argument
                batch_results = model.predict(
                    source=frame_buffer,
                    conf=confidence_threshold,
                    imgsz=640, # Explicitly set imgsz for consistency and clarity
                    stream=False, # Keep False for explicit batching
                    verbose=True, # Keep True to see the batch shape in the speed stats
                    batch=batch_size # <--- The key addition!
                )

                for i, r in enumerate(batch_results):
                    current_time_sec, original_frame_idx = frame_info_buffer[i]
                    current_bird_present = False

                    boxes = r.boxes
                    for box in boxes:
                        class_id = int(box.cls[0])
                        if class_id == bird_class_id:
                            current_bird_present = True
                            break

                    if current_bird_present and not prev_bird_present:
                        land_timestamps.append(current_time_sec)
                        print(f"Bird landed at {current_time_sec:.2f} seconds (frame {original_frame_idx}).")

                    prev_bird_present = current_bird_present

                # Clear buffers for the next batch
                frame_buffer = []
                frame_info_buffer = []

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection complete with YOLOv8 (batched & skipped using .predict).")
    return land_timestamps

# Example Usage:
# video_file = '2025-10-21-074016.webm'
video_file = sys.argv[1]

start_time = time.time()
bird_landings = detect_birds_yolov8_batched_with_skipping(
    video_file,
    confidence_threshold=0.6,
    batch_size=32,           # Try a higher batch size with proper batching
    frame_skip_interval=10
)
end_time = time.time()

print("\nFinal Bird Landing Timestamps (seconds):")
for ts in bird_landings:
    print(f"- {ts:.2f}")

print(f"\nProcessing time: {end_time - start_time:.2f} seconds")