import os
import io
import re
import torch
import warnings
import argparse
import tempfile
from time import perf_counter
import torchaudio
from pydub import AudioSegment
import matplotlib.pyplot as plt
import speech_recognition as sr
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

# Suppress warnings
warnings.filterwarnings("ignore")

# Argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--stop_word", default="exit", help="Stop word to abort transcription", type=str)
parser.add_argument("--verbose", default=False, help="Print verbose output", type=bool)
parser.add_argument("--energy", default=500, help="Energy level for mic to detect", type=int)
parser.add_argument("--dynamic_energy", default=False, help="Enable dynamic energy", type=bool)
parser.add_argument("--pause", default=0.5, help="Minimum silence duration (sec) to end phrase", type=float)
parser.add_argument("--prediction_interval", default=3.0, help="Interval in seconds for emotion prediction", type=float)
parser.add_argument("--plot_interval", default=0.5, help="Interval in seconds for plotting updates", type=float)
args = parser.parse_args()

# Temp file setup
temp_dir = tempfile.mkdtemp()
save_path = os.path.join(temp_dir, "temp.wav")

# Load model and extractor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "r-f/wav2vec-english-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)

# Emotion simplification
def simplify_emotion(raw_emotion):
    if raw_emotion in ["happy", "surprise"]:
        return "happy"
    elif raw_emotion in ["angry", "disgust", "fear", "sad"]:
        return "angry"
    else:
        return "neutral"

# Mapping for plotting
simplified_emotion_mapping = {
    "angry": -1,
    "neutral": 0,
    "happy": 1
}

# Stop word checker
def check_stop_word(predicted_text: str) -> bool:
    pattern = re.compile('[\W_]+', re.UNICODE)
    return pattern.sub('', predicted_text).lower() == args.stop_word

# Emotion prediction
def predict_emotion(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1)
        raw_emotion = model.config.id2label[predicted_label.item()]
    return simplify_emotion(raw_emotion)

# Transcription and plotting
def transcribe():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = args.energy
    recognizer.pause_threshold = args.pause
    recognizer.dynamic_energy_threshold = args.dynamic_energy

    fig, ax = plt.subplots()

    timestamps = []
    emotion_scores = []
    current_emotion = "neutral"
    last_prediction_time = perf_counter()

    with sr.Microphone(sample_rate=16000) as source:
        print(f"Listening... (say '{args.stop_word}' to stop)")
        while True:
            # Record small chunk of audio
            audio = recognizer.record(source, duration=args.plot_interval)
            now = perf_counter()

            # Save chunk to file
            data = io.BytesIO(audio.get_wav_data())
            audio_clip = AudioSegment.from_file(data)
            audio_clip.export(save_path, format="wav")

            # Predict emotion every N seconds
            if now - last_prediction_time >= args.prediction_interval:
                current_emotion = predict_emotion(save_path)
                last_prediction_time = now
                print("Detected Emotion (simplified):", current_emotion)

                if check_stop_word(current_emotion):
                    break

            # Store timestamp and emotion score
            timestamps.append(now - timestamps[0] if timestamps else 0)
            emotion_scores.append(simplified_emotion_mapping[current_emotion])

            # Plotting
            ax.clear()
            ax.plot(timestamps, emotion_scores, color='blue', marker='o')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Emotion')
            ax.set_title(f'Real-time Emotion (updated every {args.plot_interval}s)')
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(['Angry', 'Neutral', 'Happy'])
            plt.pause(0.05)

    plt.close()

if __name__ == "__main__":
    transcribe()
