import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to detect waist
def detect_waist(image):
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Hip points
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Convert to pixel coordinates
        height, width, _ = image.shape
        left_hip_px = (int(left_hip.x * width), int(left_hip.y * height))
        right_hip_px = (int(right_hip.x * width), int(right_hip.y * height))

        # Draw landmarks
        cv2.circle(image, left_hip_px, 5, (0, 255, 0), -1)
        cv2.circle(image, right_hip_px, 5, (0, 255, 0), -1)
        cv2.line(image, left_hip_px, right_hip_px, (255, 0, 0), 2)

        # Waist measurement (in pixels)
        waist_px = int(np.linalg.norm(np.array(left_hip_px) - np.array(right_hip_px)))
        return image, waist_px
    return image, None

# Calibration: Conversion factor
PIXELS_PER_CM = 0.5  # Example value

# Streamlit App
st.title("Body Waist Measurement Using Computer Vision")
st.sidebar.header("Instructions")
st.sidebar.write(
    """
1. Enable your webcam using the camera input below.
2. Stand in front of the camera so your waist is visible.
3. The app will detect your waist and dynamically update the estimated size.
"""
)

# Camera input
image_file = st.camera_input("Capture your image")

if image_file:
    # Convert uploaded image to OpenCV format
    image = Image.open(image_file)
    image = np.array(image)

    # Detect waist and calculate measurement
    annotated_image, waist_px = detect_waist(image)

    if waist_px is not None:
        # Convert to cm and dynamically update the text
        waist_cm = waist_px * PIXELS_PER_CM
        st.markdown(f"### Estimated Waist Size: {waist_cm:.2f} cm")
    else:
        st.markdown("### Estimated Waist Size: Unable to detect. Please adjust your position.")

    # Display the annotated image
    st.image(annotated_image, channels="RGB")
else:
    st.write("Please capture an image to start waist measurement.")