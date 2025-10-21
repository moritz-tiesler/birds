import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# --- Configuration ---
FLASK_STREAM_URL = 'http://192.168.2.38:5000/video_feed'
MODEL_PATH = "yolov8n.pt"

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
        while cap.isOpened() and not stop_button:
            success, frame = cap.read()

            if success:
                # Run detection
                results = model.track(frame, persist=True, conf=0.7, verbose=False)

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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_frame_{timestamp}.jpg"

                    # Save the annotated frame (NumPy array) using OpenCV
                    # annotated_frame is already a BGR NumPy array (standard OpenCV format)
                    cv2.imwrite(filename, frame)
                    print(f"{timestamp}: saved bird")
                    # st.sidebar.success(f"Saved frame to {filename}")

                    # Reset the button state to prevent continuous saving (optional, but cleaner)
                    # save_button = False

                # Convert BGR frame to RGB for Streamlit display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Update the placeholder with the new frame
                live_feed_placeholder.image(
                    annotated_frame_rgb,
                    channels="RGB",
                    caption="Live YOLO Detection"
                )
            else:
                st.warning("Stream ended or frame reading failed.")
                break

except Exception as e:
    st.error(f"An error occurred during streaming: {e}")

finally:
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    st.success("Inference stopped and resources released.")