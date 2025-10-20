from ultralytics import YOLO
import cv2
import time
import os
from multiprocessing import Pool, cpu_count

# --- Worker Function for a single video chunk with YOLOv8 ---
def process_video_chunk_yolov8(args):
    video_path, start_frame, end_frame, confidence_threshold, bird_class_id, chunk_id = args
    print(f"[{chunk_id}] Processing chunk: frames {start_frame}-{end_frame}")

    # Load YOLOv8 model - this happens in each process
    # If using GPU, ensure CUDA_VISIBLE_DEVICES is set correctly if you want to assign specific GPUs
    # For CPU, this is fine.
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{chunk_id}] Error: Could not open video file.")
        return []

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

        results = model(frame, stream=False, conf=confidence_threshold, verbose=False) # verbose=False to suppress output per frame

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
    return chunk_land_timestamps

# --- Main parallel processing function for YOLOv8 ---
def parallel_detect_birds_yolov8(video_path, confidence_threshold=0.5, bird_class_id=14, num_chunks=None):
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

    if num_chunks is None:
        num_chunks = cpu_count()
    print(f"Splitting into {num_chunks} chunks for parallel processing...")

    chunk_size = total_frames // num_chunks
    chunk_args = []
    for i in range(num_chunks):
        start_frame = i * chunk_size
        end_frame = (i + 1) * chunk_size
        if i == num_chunks - 1:
            end_frame = total_frames
        chunk_args.append((video_path, start_frame, end_frame, confidence_threshold, bird_class_id, i + 1))

    all_land_timestamps = []
    start_time = time.time()

    with Pool(processes=num_chunks) as pool:
        results = pool.map(process_video_chunk_yolov8, chunk_args)

    for chunk_timestamps in results:
        all_land_timestamps.extend(chunk_timestamps)

    all_land_timestamps = sorted(list(set(all_land_timestamps)))

    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds.")

    return all_land_timestamps

# Example Usage:
video_file = '2025-10-20-165625.webm'
bird_landings = parallel_detect_birds_yolov8(video_file, confidence_threshold=0.6, bird_class_id=14, num_chunks=2)
print("\nFinal Bird Landing Timestamps (seconds):")
for ts in bird_landings:
    print(f"- {ts:.2f}")