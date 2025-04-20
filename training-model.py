import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping


# This function splits the given video file into frames
def extract_frames(video_path, output_folder, frame_rate=1):
    # Create folder to store extracted frames (no error if it already exists)
    os.makedirs(output_folder, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    # Get the FPS (frames per second) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # If FPS is 0, the video cannot be read → skip the process
    if fps == 0:
        print(f"Invalid video file: {video_path}")
        return

    saved = 0  # Number of frames saved

    # Read frames sequentially until the video ends
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:  # If frame cannot be read → end of video
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Current frame number

        # Take 1 frame per second → e.g., take every 30th frame in a 30 FPS video
        if current_frame % int(fps // frame_rate) == 0:
            frame_name = os.path.join(output_folder, f"frame_{saved}.jpg")  # File name to save
            cv2.imwrite(frame_name, frame)  # Save frame as .jpg
            saved += 1  # Counter for number of saved frames

    cap.release()  # Release memory, close video file
    print(f"Extracted {saved} frames from: {video_path}")


# Set folders containing violence and non-violence videos
base_paths = {
    "violence": "dataset/violence",
    "non_violence": "dataset/non_violence"
}


# Process each label (violence, non_violence) separately
for label, folder_path in base_paths.items():
    # Prepare the main folder to save frames
    output_base = os.path.join("frames", label)
    os.makedirs(output_base, exist_ok=True)

    # Process all video files from 1 to 1000
    for i in range(1, 1001):
        # Generate filename
        if label == "violence":
            filename = f"V_{i}.mp4"
        else:
            filename = f"NV_{i}.mp4"

        # Create full path to video file
        video_path = os.path.join(folder_path, filename)

        # Create folder to save frames from this video
        video_output_folder = os.path.join(output_base, f"{label}_{i}")

        # If video file does not exist → warn and continue
        if not os.path.exists(video_path):
            print(f"File not found, skipping: {video_path}")
            continue

        # Start frame extraction
        extract_frames(video_path, video_output_folder)


img_size = (128, 128)  # Resize images for easier processing
batch_size = 32

# Image data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Training data loader
train_generator = datagen.flow_from_directory(
    'frames/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Validation data loader
val_generator = datagen.flow_from_directory(
    'frames/',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# CNN model definition
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.6),
    Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stop]
)

# Save the model as .h5 file
model.save("violence_detection_model.h5")
print("Model saved successfully.")
