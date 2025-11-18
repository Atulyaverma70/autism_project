# predictor.py
import os
import numpy as np
import librosa
import joblib
import sys

n_mfcc = 20

def extract_mfcc_avg(audio_path, n_mfcc=n_mfcc):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1).reshape(1, -1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predictor.py <model.pkl> <audio_file>")
        sys.exit(1)
    model_path = sys.argv[1]
    audio_path = sys.argv[2]
    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        sys.exit(1)
    if not os.path.exists(audio_path):
        print("Audio not found:", audio_path)
        sys.exit(1)
    model = joblib.load(model_path)
    feat = extract_mfcc_avg(audio_path)
    pred = model.predict(feat)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(feat)[0]
    label_map = {1: "Autistic", 0: "Non-autistic"}
    print("Prediction:", label_map.get(pred, pred))
    if prob is not None:
        print("Probabilities:", prob)
