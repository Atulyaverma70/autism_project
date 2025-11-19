import streamlit as st
import joblib
import numpy as np
import librosa
import warnings
from pydub import AudioSegment
import io

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Autism Detection",
    page_icon="🎧",
    layout="centered"
)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
    <h1 style='text-align:center;color:#4C7BF3;'>🎧 Autism Detection Using Audio</h1>
    <p style='text-align:center;font-size:18px;color:gray;'>
        Upload a child's speech audio file and choose a model to detect Autism Spectrum Disorder (ASD).
    </p>
""", unsafe_allow_html=True)

# -------------------------------
# MODEL DROPDOWN
# -------------------------------
models = {
    'rf.pkl': 'Random Forest (90% accuracy)',
    'ann.pkl': 'ANN (72% accuracy)',
    'svm.pkl': 'SVM (54% accuracy)',
    'nb.pkl': 'Naive Bayes (81% accuracy)',
}

model_label = st.selectbox("🔍 Choose a Model", list(models.values()))
chosen_model_file = [k for k, v in models.items() if v == model_label][0]

with st.spinner("Loading model..."):
    model = joblib.load(chosen_model_file)

# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader("🎙 Upload an audio file", type=["m4a", "wav", "mp3"])

# -------------------------------
# STYLES FOR RESULT
# -------------------------------
YES_HTML = """
<h1 style="color:red;text-align:center;font-size:48px;">⚠️ Prediction: Autistic</h1>
"""

NO_HTML = """
<h1 style="color:green;text-align:center;font-size:48px;">✔️ Prediction: Non Autistic</h1>
"""

# -------------------------------
# PROCESS AUDIO
# -------------------------------
def process_audio(uploaded_audio):
    try:
        audio = AudioSegment.from_file(io.BytesIO(uploaded_audio.read()))
        samples = audio.get_array_of_samples()
        y = np.array(samples).astype(np.float32) / (2**15 - 1)
        sr = audio.frame_rate

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        if np.isnan(mfcc).any():
            return None

        mfcc_avg = np.mean(mfcc, axis=1).reshape(1, 20)
        return mfcc_avg

    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if uploaded_file:
    st.audio(uploaded_file, format="audio/m4a")

    if st.button("🚀 Predict"):
        with st.spinner("Extracting features & predicting..."):

            features = process_audio(uploaded_file)

            if features is None:
                st.error("Could not extract valid MFCC features. Try another audio.")
            else:
                prediction = model.predict(features)[0]

                if prediction == 1:
                    st.markdown(YES_HTML, unsafe_allow_html=True)
                else:
                    st.markdown(NO_HTML, unsafe_allow_html=True)
