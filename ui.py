# ui.py
import streamlit as st
import joblib
import numpy as np
import librosa
import tempfile
import os
import matplotlib.pyplot as plt
from matplotlib import mlab
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(page_title="Autism Detection", layout="centered")

st.title("🎧 Autism Detection from Voice — Demo")
st.write("Upload an audio file (m4a / wav / mp3). Models are trained on averaged MFCC features (n_mfcc=20).")

# available models (files expected in working dir)
available_models = {
    "Random Forest": "rf.pkl",
    "Neural Net (MLP)": "ann.pkl",
    "Support Vector Machine": "svm.pkl",
    "Naive Bayes": "nb.pkl"
}

# show what files are present
present_models = {name: fname for name, fname in available_models.items() if os.path.exists(fname)}
if not present_models:
    st.warning("No pretrained model files found (rf.pkl, ann.pkl, svm.pkl, nb.pkl). Run model.py to train and save them.")
else:
    st.info("Loaded models found: " + ", ".join(present_models.keys()))

model_name = st.selectbox("Choose model", list(available_models.keys()))
model_file = available_models[model_name]

uploaded_file = st.file_uploader("Upload audio file", type=["m4a", "wav", "mp3", "flac"])

n_mfcc = 20

def extract_mfcc_from_tempfile(temp_path, n_mfcc=n_mfcc):
    # try librosa directly
    try:
        y, sr = librosa.load(temp_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        st.warning(f"librosa couldn't read file directly ({e}). Trying fallback...")
        # fallback: try reading using soundfile/pydub etc. If user has ffmpeg, pydub may work.
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_path)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sr = audio.frame_rate
            # if stereo, take mean of channels
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            samples = samples / (2 ** 15 - 1)
            mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=n_mfcc)
            return np.mean(mfcc, axis=1)
        except Exception as e2:
            st.error("Failed to read audio. Ensure ffmpeg is installed for m4a support.")
            raise e2

if uploaded_file:
    # Save uploaded file to a temporary file (so librosa/pydub can read it)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    try:
        tfile.write(uploaded_file.getvalue())
        tfile.flush()
        tfile.close()

        # Display player
        st.audio(tfile.name)

        # Show waveform (simple)
        try:
            y, sr = librosa.load(tfile.name, sr=None)
            fig, ax = plt.subplots(figsize=(8, 2))
            times = np.arange(len(y)) / float(sr)
            ax.plot(times, y, linewidth=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Waveform")
            st.pyplot(fig)
        except Exception:
            st.info("Couldn't display waveform (format may require ffmpeg).")

        # Check model file exists
        if not os.path.exists(model_file):
            st.error(f"Model file {model_file} not found. Train models by running `python model.py`.")
        else:
            # load model
            try:
                pipeline = joblib.load(model_file)
            except Exception as e:
                st.error("Failed to load model: " + str(e))
                pipeline = None

            if pipeline is not None:
                try:
                    feats = extract_mfcc_from_tempfile(tfile.name)
                    if feats.shape[0] != n_mfcc:
                        st.error(f"Extracted MFCC length {feats.shape[0]} != expected {n_mfcc}. Check pipeline consistency.")
                    else:
                        feats = feats.reshape(1, -1)
                        pred = pipeline.predict(feats)[0]
                        probs = None
                        if hasattr(pipeline, "predict_proba"):
                            probs = pipeline.predict_proba(feats)[0]
                        label_map = {1: "Autistic", 0: "Non-autistic"}
                        st.markdown("### Prediction")
                        if pred == 1:
                            st.error(f"Prediction: {label_map.get(pred)}")
                        else:
                            st.success(f"Prediction: {label_map.get(pred)}")
                        if probs is not None:
                            st.write("Probabilities:")
                            st.write(f"Non-autistic: {probs[0]:.3f} — Autistic: {probs[1]:.3f}")
                            st.progress(int(probs[1] * 100))  # autistic prob bar
                except Exception as e:
                    st.error("Error during feature extraction or prediction: " + str(e))
    finally:
        try:
            os.unlink(tfile.name)
        except Exception:
            pass

st.write("---")
st.write("Notes:")
st.markdown(
    """
- Make sure you have `rf.pkl` etc. in the same folder (run `python model.py` after creating features).
- For `.m4a` support you may need `ffmpeg` installed on your system.
- This demo uses averaged MFCC (n_mfcc=20). Keep the same `n_mfcc` across all scripts.
"""
)
