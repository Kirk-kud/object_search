import streamlit as st
import os
import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained Inception model
model = tf.keras.applications.InceptionV3(weights='imagenet')

# Function to process video and extract frames
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        cv2.imwrite(os.path.join(output_folder, f'frame{count}.jpg'), frame)
        count += 1
    cap.release()

# Function to predict frame
def predict_frame(frame_path):
    img = cv2.imread(frame_path)
    img = cv2.resize(img, (299, 299))  # InceptionV3 expects 299x299 input size
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    preds = model.predict(img)
    return tf.keras.applications.inception_v3.decode_predictions(preds, top=1)[0][0][1]

# Streamlit app
st.title('Video Frame Search')

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

search_element = st.text_input("Search Element")

if uploaded_file is not None and search_element:
    video_path = os.path.join("uploads", uploaded_file.name)
    with open(video_path, mode='wb') as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    frames_folder = 'frames'
    extract_frames(video_path, frames_folder)
    frames = os.listdir(frames_folder)
    results = []

    for frame in frames:
        frame_path = os.path.join(frames_folder, frame)
        prediction = predict_frame(frame_path)
        if search_element.lower() in prediction.lower():
            results.append(frame_path)

    if results:
        st.write(f"Found {len(results)} frames containing '{search_element}':")
        for result in results:
            st.image(result, caption=result, use_column_width=True)
    else:
        st.write(f"No frames containing '{search_element}' were found.")
