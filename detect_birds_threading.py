from ultralytics import YOLO
import cv2
import time
import numpy as np
import os
from threading import Thread, Lock
from queue import Queue

# Global list to store timestamps, and a lock to protect it
all_land_timestamps = []
timestamps_lock = Lock()

# --- Worker Function for a single video chunk using a thread ---
def process_video_chunk_yolov8_threaded(
    video_path, start_frame, end_frame,
    confidence_threshold, bird_class_id,
    chunk_id
):
    print(f"[{chunk_id}] Starting thread for chunk: frames {start_frame}-{end_frame}")

    # Each thread gets its own YOLO model instance
    # This is crucial for CPU parallelism with released GIL.
    try:
        local_model = YOLO('yolov8n.pt')
        # Ensure it's explicitly on CPU, though it will default if no GPU found.
        local_model.to('cpu')
    except Exception as e:
        print(f"[{chunk_id}] Error loading YOLO model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{chunk_id}] Error: Could not open video file.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    chunk_land_timestamps = []
    prev_bird_present = False

    current_frame_idx = start_frame
    while current_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_time_sec = current_time_ms / 1000.0

        # Perform inference
        # stream=False and verbose=False for efficiency and clean output
        results = local_model(frame, stream=False, conf=confidence_threshold, verbose=False)

        current_bird_present = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == bird_class_id:
                    current_bird_present = True
                    break
            if current_bird_present:
                break

        if current_bird_present and not prev_bird_present:
            chunk_land_timestamps.append(current_time_sec)

        prev_bird_present = current_bird_present
        current_frame_idx += 1

    cap.release()
    print(f"[{chunk_id}] Finished chunk: frames {start_frame}-{end_frame}. Found {len(chunk_land_timestamps)} landings.")

    # Safely add the chunk's timestamps to the global list
    with timestamps_lock:
        all_land_timestamps.extend(chunk_land_timestamps)

# --- Main parallel processing function using threading ---
def threaded_detect_birds_yolov8(video_path, confidence_threshold=0.5, bird_class_id=14, num_threads=None):
    global all_land_timestamps # Declare intent to modify the global variable

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps == 0:
        print("Error: Could not get FPS from video. Cannot calculate accurate timestamps.")
        return []

    print(f"Video: {video_path}, Total Frames: {total_frames}, FPS: {fps}")

    if num_threads is None:
        num_threads = os.cpu_count() # Use all available CPU cores
    print(f"Splitting into {num_threads} chunks for threaded processing...")

    # Determine chunk boundaries
    chunk_size = total_frames // num_threads
    thread_args = []
    for i in range(num_threads):
        start_frame = i * chunk_size
        end_frame = (i + 1) * chunk_size
        if i == num_threads - 1: # Ensure the last chunk covers any remainder
            end_frame = total_frames

        # (video_path, start_frame, end_frame, confidence_threshold, bird_class_id, chunk_id)
        thread_args.append((video_path, start_frame, end_frame, confidence_threshold, bird_class_id, i + 1))

    threads = []
    all_land_timestamps = [] # Reset global list for a new run
    start_time = time.time()

    for args in thread_args:
        thread = Thread(target=process_video_chunk_yolov8_threaded, args=args)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Aggregate and sort all timestamps
    # Using a set to remove duplicates (due to `extend` and potential overlaps)
    final_timestamps = sorted(list(set(all_land_timestamps)))

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds.")
    print("Detection complete with YOLOv8 (CPU Threading).")

    return final_timestamps

# Example Usage:
video_file = '2025-10-20-165625.webm'
# IMPORTANT: You might need to change bird_class_id depending on your model.
# For COCO dataset, 'bird' is usually class 14.
# To find it:
# model = YOLO('yolov8n.pt')
# print(model.names)
#
# You can specify `num_threads` or let it use `os.cpu_count()`
bird_landings = threaded_detect_birds_yolov8(video_file, confidence_threshold=0.6, bird_class_id=14, num_threads=8)
print("\nFinal Bird Landing Timestamps (seconds):")
for ts in bird_landings:
    print(f"- {ts:.2f}")