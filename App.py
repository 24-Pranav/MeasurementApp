import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to detect waist
def detect_waist(image):
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        # Waist points (hip bones)
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
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

# Calibration: Conversion factor (adjustable based on calibration object)
PIXELS_PER_CM = 0.5  # Example value (tune this based on a real calibration object)

# Streamlit App
st.title("Body Waist Measurement Using Computer Vision")
st.write("This app measures your body waist using your webcam and computer vision!")

# Sidebar for instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Ensure your webcam is enabled.
2. Stand in front of the camera, ensuring your waist is visible.
3. The app will detect your hip points and measure your waist size.
""")

# Start Webcam
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

# Capture video from webcam
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to access the webcam!")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture a frame from the webcam!")
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect waist and calculate measurement
            annotated_frame, waist_px = detect_waist(frame_rgb)

            # Display the result
            if waist_px is not None:
                waist_cm = waist_px * PIXELS_PER_CM
                st.markdown(f"### Estimated Waist Size: {waist_cm:.2f} cm")
            
            FRAME_WINDOW.image(annotated_frame, channels="RGB")
            cv2.waitKey(1)  # Allow a brief pause for rendering

        cap.release()
else:
    st.write("Click 'Start Webcam' to begin!")