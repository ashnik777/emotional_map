import os
import tempfile
import warnings
import argparse
from time import perf_counter
import base64

import matplotlib.pyplot as plt
import matplotlib as mpl
import sounddevice as sd
import soundfile as sf
import google.generativeai as genai

# Gemini API setup
GEMINI_API_KEY = "AIzaSyAhobKANOGwmHcV4kNGSAi0PbgUnuCGe0c"
genai.configure(api_key=GEMINI_API_KEY)

# Configurations
SAMPLERATE = 16000
CHANNELS = 1
EMOTIONS = ["Happy", "Sad", "Angry", "Neutral"]

# Style settings
mpl.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.facecolor": "#f0f4f8",
    "axes.edgecolor": "#555",
    "grid.color": "#ccc",
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans"
})

emotion_mapping = {"Angry": -1, "Sad": -0.5, "Neutral": 0, "Happy": 1}
emotion_colors = {"Angry": "#e74c3c", "Sad": "#3498db", "Neutral": "#95a5a6", "Happy": "#27ae60"}

# Convert audio file to base64
def audio_to_base64(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

# Gemini API request for emotion detection
def detect_emotion(audio_path: str) -> str:
    try:
        audio_b64 = audio_to_base64(audio_path)
        prompt = (
            "Analyze the speaker's emotion from vocal tone and intonation in this WAV audio."
            " Respond strictly with one word: Happy, Sad, Angry, or Neutral."
        )
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}}
        ]}])
        emotion = response.text.strip().capitalize()
        return emotion if emotion in EMOTIONS else "Neutral"
    except Exception as e:
        print(f"[Gemini API Error]: {e}")
        return "Neutral"

# Initialize figure and axis globally to update continuously
fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor('#f9f9f9')

# Enhanced plotting function
def plot_emotions(timestamps, emotions):
    ax.clear()

    ax.set_title('Real-time Emotion Recognition', fontsize=24, pad=25)
    ax.set_xlabel('Time (seconds)', fontsize=16)
    ax.set_ylabel('Emotion Intensity', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_yticks([-1, -0.5, 0, 1])
    ax.set_yticklabels(['Angry 😡', 'Sad 😢', 'Neutral 😐', 'Happy 😄'])
    ax.set_ylim(-1.2, 1.2)

    ax.plot(timestamps, [emotion_mapping[emo] for emo in emotions], linestyle='-', linewidth=2, alpha=0.7, color='#2c3e50')

    for t, emo in zip(timestamps, emotions):
        score = emotion_mapping[emo]
        ax.scatter(t, score, s=120, color=emotion_colors[emo], edgecolor='black', linewidth=1.5, zorder=3)
        ax.annotate(
            emo, xy=(t, score), xytext=(t, score + 0.15), fontsize=14,
            ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc=emotion_colors[emo], ec="#222", lw=0.5)
        )

    plt.tight_layout()
    plt.pause(0.1)

# Continuous emotion detection loop
def start_emotion_detection(duration: int):
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "temp_audio.wav")

    timestamps, detected_emotions = [], []
    start_time = perf_counter()

    print("\n🎙️ Speak now (Press Ctrl+C to exit).")
    try:
        while True:
            audio = sd.rec(int(duration * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS)
            sd.wait()
            sf.write(audio_path, audio, SAMPLERATE)

            emotion = detect_emotion(audio_path)
            elapsed = round(perf_counter() - start_time, 2)

            timestamps.append(elapsed)
            detected_emotions.append(emotion)

            print(f"⏰ {elapsed}s: Detected Emotion: {emotion}")
            plot_emotions(timestamps, detected_emotions)

    except KeyboardInterrupt:
        print("\n⏹️ Emotion detection terminated.")
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Real-time Emotion Detection via Gemini API")
    parser.add_argument("--duration", default=5, type=int, help="Recording chunk duration in seconds")
    args = parser.parse_args()

    start_emotion_detection(args.duration)