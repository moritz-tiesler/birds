import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import sys
import time
import argparse
import os

parser = argparse.ArgumentParser()

# use app args like this:
#   streamlit run detect_birds_streamlit.py -- --stream
parser.add_argument('--stream', action='store_true', default=False,
                    help="Stream frames via streamlit app")

try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    os._exit(e.code)

frame_count = 0

def report(data: str) -> None:

    # Print the string followed by '\r' to return the cursor to the beginning
    sys.stdout.write(f"\r{data}")
    sys.stdout.flush()

# --- Configuration ---
FLASK_STREAM_URL = 'http://192.168.2.38:5000/video_feed'
MODEL_PATH = "yolov8n.onnx"
FAIL_CAP = 1

# --- Streamlit Setup ---
st.title("YOLOv8 Live Inference from Flask Stream")
st.caption(f"Source: {FLASK_STREAM_URL}")
stop_button = st.button("Stop Inference")

# Create a placeholder for the video feed
live_feed_placeholder = st.empty()

# --- Processing Logic ---
try:
    # 1. Initialize Camera (Force FFMPEG)
    cap = cv2.VideoCapture(FLASK_STREAM_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        st.error(f"Failed to open stream at {FLASK_STREAM_URL}. Check Flask server status.")
    else:
        # 2. Load Model
        # YOLOv8 models can be loaded once outside the loop
        model = YOLO(MODEL_PATH, task="detect")

        # 3. Main Loop
        read_fails = 0
        while cap.isOpened() and not stop_button:
            if read_fails >= FAIL_CAP:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(FLASK_STREAM_URL, cv2.CAP_FFMPEG)
            # TODO: logic for reopening video caputre after hitting fail cap?
            success, frame = cap.read()

            if success:
                read_fails = 0
                # Run detection
                results = model.track(frame, persist=False, conf=0.5, verbose=False)

                # Plot results onto the frame
                result = results[0]
                annotated_frame = result.plot()

                class_ids = result.boxes.cls.tolist()
                class_name_map = result.names

                detected_classes = []

                for class_id in class_ids:
                    # Convert the numerical ID to its string name using the map
                    class_name = class_name_map[class_id]
                    detected_classes.append(class_name)


                if 'bird' in detected_classes:
                # Create a filename based on the current timestamp
                    # TODO: use tracked id to distinguish detected birds?
                    #   result.boxes.id
                    #   detection_frame-id_{n}-{date}.jpg
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captures/detection_frame_{timestamp}.jpg"

                    # Save the annotated frame (NumPy array) using OpenCV
                    # annotated_frame is already a BGR NumPy array (standard OpenCV format)
                    cv2.imwrite(filename, frame)
                    print(f"\n{timestamp}: saved bird")
                    st.sidebar.success(f"{timestamp}: Saved frame to {filename}")

                    # Reset the button state to prevent continuous saving (optional, but cleaner)
                    # save_button = False

                # Convert BGR frame to RGB for Streamlit display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Update the placeholder with the new frame
                if args.stream:
                    live_feed_placeholder.image(
                        annotated_frame_rgb,
                        channels="RGB",
                        caption="Live YOLO Detection"
                    )

                frame_count += 1
                report(f"{frame_count} frames")

                del annotated_frame
                del annotated_frame_rgb
                del result
                del results

            else:
                read_fails += 1
                frame_count = 0
                st.warning("Stream ended or frame reading failed.")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"\n{timestamp}: failed to read frame, skipping. fails: {read_fails}/{FAIL_CAP}")
                # break

except Exception as e:
    st.error(f"An error occurred during streaming: {e}")
    print(f"An error occurred during streaming: {e}")

finally:
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    st.success("Inference stopped and resources released.")