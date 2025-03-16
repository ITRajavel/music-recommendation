import cv2
import numpy as np
import os
import random
import pygame
import streamlit as st
import time
from keras.models import load_model

# Initialize Pygame mixer
pygame.mixer.init()

# Load trained emotion detection model
model = load_model('emotion_model.h5')

# Emotion categories
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Available languages
languages = ['English', 'Tamil', 'Hindi']


# Base directory for songs
BASE_DIR = r"D:\Project-in-Hour\music recommendation\dataset\songs"

# Function to get song path based on emotion and language
def get_song_path(emotion, language):
    song_folder = os.path.join(BASE_DIR, language.lower(), emotion)
    
    if os.path.exists(song_folder):
        songs = [f for f in os.listdir(song_folder) if f.lower().endswith(('.mp3', '.wav'))]
        if songs:
            return os.path.join(song_folder, random.choice(songs))
    
    return None

# Cooldown timer settings (5 minutes)
COOLDOWN_TIME = 4 * 60  
last_played_time = 0
current_song_path = None

# Function to play a song with cooldown
def play_song_with_cooldown(emotion, language):
    global last_played_time, current_song_path
    current_time = time.time()

    # Enforce 5-minute cooldown (300 seconds)
    if current_time - last_played_time < COOLDOWN_TIME and current_song_path:
        return

    # Get a song path based on emotion & language
    song_path = get_song_path(emotion, language)

    if song_path:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        st.success(f"🎵 Now Playing: {os.path.basename(song_path)} ({language})")
        
        # Save current song path and reset cooldown
        current_song_path = song_path
        last_played_time = current_time
    else:
        st.warning(f"⚠️ No song found for {emotion} in {language}. Check your folder structure.")

# Improved UI CSS styling
stylish_css = """
    <style>
    body {
        background: linear-gradient(to right, #141E30, #243B55);
        color: #e1e1e1;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background: linear-gradient(145deg, #1f2933, #3b4a5a);
        color: #e1e1e1;
    }
    h1 {
        text-shadow: 2px 2px 10px #00ffff;
    }
    .stButton>button {
        background: rgba(0, 0, 0, 0.5);
        border: 1px solid #00ffff;
        color: #00ffff;
        box-shadow: 0 0 15px #00ffff;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #00ffff;
        color: #141E30;
        box-shadow: 0 0 20px #00ffff;
        transform: scale(1.05);
    }
    .stSelectbox>div {
        background: rgba(0, 0, 0, 0.7);
        color: #e1e1e1;
        border-radius: 8px;
        border: 1px solid #00ffff;
    }
    .stImage>img {
        border-radius: 15px;
        box-shadow: 0 0 30px #00ffff;
    }
    .stWarning {
        color: #FF4500;
    }
    .stSuccess {
        color: #32CD32;
    }

    /* Floating Circular Play Button */
    .floating-play-btn {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 70px;
        height: 70px;
        background: #00ffff;
        border: none;
        border-radius: 50%;
        box-shadow: 0 0 20px #00ffff;
        cursor: pointer;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        color: #141E30;
        font-weight: bold;
    }

    .floating-play-btn:hover {
        transform: scale(1.2);
        box-shadow: 0 0 30px #00ffff;
    }

    .floating-play-btn:active {
        transform: scale(0.9);
        box-shadow: 0 0 15px #00ffff;
    }
    </style>
"""

# Apply the CSS
st.markdown(stylish_css, unsafe_allow_html=True)

# Streamlit UI setup
st.title("🎵 Music Recommendations-Based on Facial Recognitions")
st.write("Detects your facial emotion and plays a song based on it!")

# Language selection
selected_language = st.selectbox("🌍 Choose Language:", languages)

# Video capture setup
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
emotion_text = st.empty()

# Floating Play Button with JS Trigger
st.markdown(
    """
    <button class='floating-play-btn' onclick="document.getElementById('hidden-button').click()">▶️</button>
    """,
    unsafe_allow_html=True
)

# Hidden button (Streamlit triggers can't run from raw HTML directly)
hidden_play_button = st.button("🎶 Detect Emotion & Play Music", key="hidden-button")

# Track last detected emotion
last_detected_emotion = None

# Start emotion detection and song playing
if hidden_play_button:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Camera not working.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        detected_emotion = "Neutral"  

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0).reshape(1, 48, 48, 1) / 255.0

            prediction = model.predict(face)
            top_emotion_index = np.argmax(prediction)
            detected_emotion = emotion_classes[top_emotion_index]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Emotion: {detected_emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display camera feed and detected emotion
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        emotion_text.write(f"😃 Detected Emotion: **{detected_emotion}**")

        # Play song only after 5 minutes cooldown
        play_song_with_cooldown(detected_emotion, selected_language)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
