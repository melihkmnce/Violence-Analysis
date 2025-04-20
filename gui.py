import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile

# Streamlit page title and description
st.title("Violence Detection AI")
st.write("Upload a video and the system will guess whether you are fighting or not.")

# Load the trained Keras model
model = load_model("violence_detection_model.h5")

# File uploader for selecting a video
video_file = st.file_uploader("Select a video file (.mp4)", type=["mp4"])

if video_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Get frames per second (FPS) from video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_rate = 1  # Process one frame per second
    count = 0
    predictions = []

    # Prepare placeholders for dynamic updates
    stframe = st.empty()
    status_text = st.empty()
    status_text.write("Analyzing video...")

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze frame at specified rate (e.g., 1 per second)
        if count % int(fps // frame_rate) == 0:
            resized = cv2.resize(frame, (128, 128))  # Resize frame to match model input
            img_array = img_to_array(resized) / 255.0  # Normalize pixels
            img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for prediction

            prediction = model.predict(img_array)[0][0]
            predictions.append(prediction)

        count += 1

    cap.release()
    status_text.empty()  # Clear the status message after processing

    # Display the result
    if predictions:
        mean_prediction = np.mean(predictions)

        if mean_prediction > 0.7:
            st.error(f"The probability of a fight is HIGH: %{mean_prediction * 100:.2f} rate.")
        elif mean_prediction > 0.4:
            st.warning(f"SUSPICIOUS movements: %{mean_prediction * 100:.2f} rate.")
        else:
            st.success(f"The fight is NOT VISIBLE: %{(1 - mean_prediction) * 100:.2f} rate of there is no fight.")
    else:
        st.info("The video could not be read or analyzed.")
