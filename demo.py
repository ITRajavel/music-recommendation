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
languages = ['English', 'Tamil']

# Base directory for songs
BASE_DIR = r"D:\Project-in-Hour\music recommendation\dataset\songs"

# Function to get a playlist of songs based on emotion and language
def get_playlist(emotion, language):
    song_folder = os.path.join(BASE_DIR, language.lower(), emotion)
    
    if os.path.exists(song_folder):
        songs = [f for f in os.listdir(song_folder) if f.lower().endswith(('.mp3', '.wav'))]
        if songs:
            random.shuffle(songs)
            return [os.path.join(song_folder, song) for song in songs]
    
    return []

# Function to play a playlist
current_song_index = 0
playlist = []


def play_next_song():
    global current_song_index, playlist
    if playlist and current_song_index < len(playlist):
        song_path = playlist[current_song_index]
        pygame.mixer.music.load(song_path)
        pygame.mixer.music.play()
        st.success(f"🎵 Now Playing: {os.path.basename(song_path)}")
        current_song_index += 1
    else:
        st.warning("⚠️ No more songs in the playlist.")


# Function to handle skip button
def skip_song():
    pygame.mixer.music.stop()
    play_next_song()


# Attractive UI CSS styling
stylish_css = """
    <style>
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

st.markdown(stylish_css, unsafe_allow_html=True)

# Streamlit UI setup
st.title("🎵 Emotion-Based Music Playlist Recommendation")
st.write("Detects your facial emotion and plays a full playlist based on it!")

# Language selection
selected_language = st.selectbox("🌍 Choose Language:", languages)

# Volume control
volume = st.slider("🔊 Volume", 0, 100, 50)
pygame.mixer.music.set_volume(volume / 100)

# Video capture setup
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()
emotion_text = st.empty()

# Hidden button for emotion detection
hidden_play_button = st.button("🎶 Detect Emotion & Start Playlist")

# Skip button
if st.button("⏭️ Skip Song"):
    skip_song()

# Start emotion detection and playlist setup
if hidden_play_button:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Camera not working.")
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        detected_emotion = "Neutral"
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0).reshape(1, 48, 48, 1) / 255.0

            prediction = model.predict(face)
            detected_emotion = emotion_classes[np.argmax(prediction)]

        # Display detected emotion
        emotion_text.write(f"😃 Detected Emotion: **{detected_emotion}**")

        # Get the playlist based on emotion and language
        playlist = get_playlist(detected_emotion, selected_language)

        if playlist:
            current_song_index = 0
            play_next_song()
        else:
            st.warning(f"⚠️ No songs found for {detected_emotion} in {selected_language}.")

# Check if current song has ended and play next
if not pygame.mixer.music.get_busy() and playlist:
    play_next_song()

cap.release()
cv2.destroyAllWindows()