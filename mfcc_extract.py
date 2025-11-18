# mfcc_extract.py
import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration
recordings_folder = "recordings"   # put your .m4a/.wav files here
features_folder = "features"
n_mfcc = 20

os.makedirs(features_folder, exist_ok=True)
os.makedirs(recordings_folder, exist_ok=True)

def extract_mfcc_features(audio_path, n_mfcc=n_mfcc):
    y, sr = librosa.load(audio_path, sr=None)  # preserve native sr
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Row-wise average like your original pipeline
    mfcc_avg = np.mean(mfcc, axis=1)  # shape (n_mfcc,)
    return mfcc_avg

def main():
    files = [f for f in os.listdir(recordings_folder) if f.lower().endswith(('.wav', '.m4a', '.mp3', '.flac'))]
    if not files:
        print("No audio files found in", recordings_folder)
        return
    for filename in files:
        src = os.path.join(recordings_folder, filename)
        try:
            mfcc_avg = extract_mfcc_features(src)
            outname = os.path.splitext(filename)[0] + ".npy"
            outpath = os.path.join(features_folder, outname)
            np.save(outpath, mfcc_avg)
            print("Saved MFCC:", outpath)
        except Exception as e:
            print("Failed to process", filename, "->", e)

if __name__ == "__main__":
    main()
