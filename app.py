import streamlit as st
import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import os
from transformers import pipeline

# Load sentiment and emotion models
sentiment_model = pipeline("sentiment-analysis")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def extract_audio_features(file_path):
    y, sr = librosa.load(file_path)
    
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.mean(pitch)
    pitch_std = np.std(pitch)

    loudness = librosa.feature.rms(y=y)[0]
    loudness_mean = np.mean(loudness)
    loudness_std = np.std(loudness)

    freqs = np.abs(np.fft.rfft(y))
    freqs_mean = np.mean(freqs)
    freqs_std = np.std(freqs)

    return {
        "Pitch (mean)": round(pitch_mean, 2),
        "Pitch (std)": round(pitch_std, 2),
        "Loudness (mean)": round(loudness_mean, 2),
        "Loudness (std)": round(loudness_std, 2),
        "Frequency (mean)": round(freqs_mean, 2),
        "Frequency (std)": round(freqs_std, 2),
    }

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Could not transcribe"

def analyze_text(text):
    sentiment = sentiment_model(text)[0]
    emotion = emotion_model(text)[0]
    return sentiment, emotion

# Streamlit UI
st.title("🎧 Voice Emotion & Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload audio file (MP3/WAV)", type=["mp3", "wav"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    file_path = "temp.wav"
    if uploaded_file.name.endswith(".mp3"):
        sound = AudioSegment.from_mp3(uploaded_file)
        sound.export(file_path, format="wav")
    else:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    st.subheader("📈 Extracted Audio Features:")
    features = extract_audio_features(file_path)
    st.json(features)

    st.subheader("📝 Transcription:")
    text = transcribe_audio(file_path)
    st.write(text)

    if text and text != "Could not transcribe":
        sentiment, emotion = analyze_text(text)

        st.subheader("📊 Sentiment from Text:")
        st.write(sentiment)

        st.subheader("💖 Emotion from Text:")
        st.write(emotion)

    os.remove(file_path)

