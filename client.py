import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import streamlit as st
from glob import glob
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics.pairwise import euclidean_distances
import shutil

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y = librosa.util.fix_length(y, size=5 * sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    pitches = pitch[mag > np.median(mag)]
    pitch_mean = np.mean(pitches) if len(pitches) > 0 else 0

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    energy = np.mean(y ** 2)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return {
        "mfcc": np.mean(mfcc, axis=1).tolist(),
        "pitch": float(pitch_mean),
        "energy": float(energy),
        "zcr": float(zcr),
        "centroid": float(centroid)
    }

def find_nearest_audio(input_audio, features_csv, scaler_path="scaler.pkl", top_k=3):
    features = extract_audio_features(input_audio)
    input_features = [features["pitch"], features["energy"], features["zcr"], features["centroid"]] + features["mfcc"]

    scaler = joblib.load(scaler_path)
    input_scaled = scaler.transform([input_features]) 

    df_features = pd.read_csv(features_csv)
    filenames = df_features["filename"].values
    feature_cols = [col for col in df_features.columns if col != "filename"]
    feature_vectors = df_features[feature_cols].values  

    distances = euclidean_distances(input_scaled, feature_vectors)[0]

    nearest_indices = np.argsort(distances)[:top_k]
    nearest_files = [(filenames[i], distances[i]) for i in nearest_indices]

    return nearest_files


st.title("Ứng dụng Tìm File Âm Thanh Tương Đồng")

st.sidebar.header("Tải lên file âm thanh hoặc ghi âm trực tiếp")

uploaded_file = st.sidebar.file_uploader("Chọn file âm thanh", type=["wav"])

record_button = st.sidebar.button("Ghi âm")

if uploaded_file is not None:
    with open("input_sample.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format="audio/wav")

    nearest = find_nearest_audio("input_sample.wav", "voice_features_normalized.csv")

    for i, (filename, dist) in enumerate(nearest):
        output_path = os.path.join("output", filename)
        input_path = os.path.join("data", filename)
        shutil.copy(input_path, output_path)

    st.subheader("3 File Âm Thanh Gần Nhất")
    for i, (filename, _) in enumerate(nearest):
        st.audio(os.path.join("output", filename), format="audio/wav")

    y_input, sr_input = librosa.load("input_sample.wav", sr=16000)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    librosa.display.waveshow(y_input, sr=sr_input, ax=axes[0])
    axes[0].set_title("Waveform của Input Audio")
    axes[0].set_xlabel("Thời gian (s)")
    axes[0].set_ylabel("Biên độ")

    output_files = [os.path.join("output", filename) for filename, _ in nearest]
    for i, output_file in enumerate(output_files):
        y_output, sr_output = librosa.load(output_file, sr=16000)
        librosa.display.waveshow(y_output, sr=sr_output, ax=axes[i+1])
        axes[i+1].set_title(f"Waveform của Output {i+1}")
        axes[i+1].set_xlabel("Thời gian (s)")
        axes[i+1].set_ylabel("Biên độ")

    plt.tight_layout()
    st.pyplot(fig)

    mfcc_input = librosa.feature.mfcc(y=y_input, sr=sr_input, n_mfcc=13)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    librosa.display.specshow(mfcc_input, x_axis='time', sr=sr_input, ax=axes[0])
    axes[0].set_title("MFCC của Input Audio")
    axes[0].set_xlabel("Thời gian (s)")
    axes[0].set_ylabel("MFCC Coefficients")

    for i, output_file in enumerate(output_files):
        y_output, sr_output = librosa.load(output_file, sr=16000)
        mfcc_output = librosa.feature.mfcc(y=y_output, sr=sr_output, n_mfcc=13)

        librosa.display.specshow(mfcc_output, x_axis='time', sr=sr_output, ax=axes[i+1])
        axes[i+1].set_title(f"MFCC của Output {i+1}")
        axes[i+1].set_xlabel("Thời gian (s)")
        axes[i+1].set_ylabel("MFCC Coefficients")

    plt.tight_layout()
    st.pyplot(fig)

    D_input = librosa.amplitude_to_db(librosa.stft(y_input), ref=np.max)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    librosa.display.specshow(D_input, x_axis='time', y_axis='log', sr=sr_input, ax=axes[0])
    axes[0].set_title("Spectrogram của Input Audio")
    axes[0].set_xlabel("Thời gian (s)")
    axes[0].set_ylabel("Tần số (Hz)")

    for i, output_file in enumerate(output_files):
        y_output, sr_output = librosa.load(output_file, sr=16000)
        D_output = librosa.amplitude_to_db(librosa.stft(y_output), ref=np.max)

        librosa.display.specshow(D_output, x_axis='time', y_axis='log', sr=sr_output, ax=axes[i+1])
        axes[i+1].set_title(f"Spectrogram của Output {i+1}")
        axes[i+1].set_xlabel("Thời gian (s)")
        axes[i+1].set_ylabel("Tần số (Hz)")

    plt.tight_layout()
    st.pyplot(fig)

