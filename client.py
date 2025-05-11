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
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, manhattan_distances
import shutil
import time
import plotly.express as px

st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .stSidebar {
        background-color: #f0f2f6;
    }
    .stSpinner {
        color: #4CAF50;
    }
    h1, h2, h3 {
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

def extract_audio_features(audio_path):
    try:
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
    except Exception as e:
        st.error(f"Lỗi khi xử lý file âm thanh: {e}")
        return None

def find_nearest_audio(input_audio, features_csv, scaler_path="scaler.pkl", top_k=4, distance_metric="Euclidean"):
    start_time = time.time()
    features = extract_audio_features(input_audio)
    if features is None:
        return [], []

    input_features = [features["pitch"], features["energy"], features["zcr"], features["centroid"]] + features["mfcc"]

    try:
        scaler = joblib.load(scaler_path)
        input_scaled = scaler.transform([input_features])

        df_features = pd.read_csv(features_csv)
        if df_features.empty:
            st.warning("Không tìm thấy dữ liệu trong file CSV!")
            return [], input_features

        filenames = df_features["filename"].values
        feature_cols = [col for col in df_features.columns if col != "filename"]
        feature_vectors = df_features[feature_cols].values

        if distance_metric == "Euclidean":
            distances = euclidean_distances(input_scaled, feature_vectors)[0]
        elif distance_metric == "Cosine":
            distances = 1 - cosine_similarity(input_scaled, feature_vectors)[0]
        elif distance_metric == "Manhattan":
            distances = manhattan_distances(input_scaled, feature_vectors)[0]
        else:
            raise ValueError("Unsupported distance metric")

        nearest_indices = np.argsort(distances)[:top_k]
        nearest_files = [(filenames[i], distances[i], feature_vectors[i]) for i in nearest_indices]

        end_time = time.time()
        time_taken = end_time - start_time
        st.success(f"✅ Tìm thấy {len(nearest_indices)-1} file tương giống. Thời gian xử lý: {time_taken:.2f} giây.")

        return nearest_files, input_features
    except Exception as e:
        st.error(f"Lỗi khi tìm kiếm file tương đồng: {e}")
        return [], input_features

st.title("Ứng dụng Tìm File Âm Thanh Tương Đồng")

st.sidebar.subheader("📐 Cài đặt")
distance_metric = st.sidebar.selectbox("Thuật toán đo khoảng cách", ["Euclidean", "Cosine", "Manhattan"])

with st.sidebar.expander("🧪 Công thức các thuật toán"):
    st.markdown("**Euclidean Distance:**  \n"
                "$$d(p, q) = \\sqrt{\\sum_i (p_i - q_i)^2}$$")
    st.markdown("**Cosine Similarity:**  \n"
                "$$\\cos(\\theta) = \\frac{p \\cdot q}{\\|p\\| \\cdot \\|q\\|}$$")
    st.markdown("**Manhattan Distance:**  \n"
                "$$d(p, q) = \\sum_i |p_i - q_i|$$")

st.sidebar.header("Tải lên file âm thanh")
uploaded_file = st.sidebar.file_uploader("Chọn file âm thanh", type=["wav"])
if uploaded_file is not None:
    if uploaded_file.size > 10 * 1024 * 1024: 
        st.error("File quá lớn! Vui lòng tải file dưới 10MB.")
    else:
        with open("input_sample.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(uploaded_file, format="audio/wav")

        with st.spinner("Đang xử lý file âm thanh..."):
            nearest, input_feats = find_nearest_audio("input_sample.wav", "voice_features_normalized.csv",
                                                     top_k=4, distance_metric=distance_metric)

        if nearest:
            os.makedirs("output", exist_ok=True)
            for filename, dist, _ in nearest:
                output_path = os.path.join("output", filename)
                input_path = os.path.join("data", filename)
                shutil.copy(input_path, output_path)

            st.subheader(f"Kết quả tìm kiếm ({distance_metric})")
            
            # max_distance = max([dist for _, dist, _ in nearest]) if nearest else 1
            for filename, dist, _ in nearest:
                if filename != uploaded_file.name:
                    # similarity = (1 - dist / max_distance) * 100
                    
                    similarity = (1 / (1 + dist)) * 100  # Đảm bảo giá trị nằm trong (0, 100]
  
                    st.write(f"{filename}: Độ tương đồng {similarity:.2f}%")
                    st.audio(os.path.join("output", filename), format="audio/wav")

            st.subheader("Waveform")
            st.write("Biểu đồ hiển thị biên độ âm thanh theo thời gian.")
            y_input, sr_input = librosa.load("input_sample.wav", sr=16000)
            fig, axes = plt.subplots(1,len(nearest), figsize=(20, 4))

            if len(nearest) == 0:
                axes = [axes]
            else:
                axes = axes.flatten()

            librosa.display.waveshow(y_input, sr=sr_input, ax=axes[0])
            axes[0].set_title("Waveform của Input Audio")
            axes[0].set_xlabel("Thời gian (s)")
            axes[0].set_ylabel("Biên độ")

            output_files = []
            for filename, _, _ in nearest:
                if filename != uploaded_file.name:
                    output_files.append(os.path.join("output", filename))      
                    
            # output_files = [os.path.join("output", filename) for filename, _, _ in nearest]
            # print(output_files)
            for i, output_file in enumerate(output_files):
               
                    y_output, sr_output = librosa.load(output_file, sr=16000)
                    librosa.display.waveshow(y_output, sr=sr_output, ax=axes[i + 1])
                    axes[i + 1].set_title(f"Waveform của Output {i + 1}")
                    axes[i + 1].set_xlabel("Thời gian (s)")
                    axes[i + 1].set_ylabel("Biên độ")

            plt.tight_layout()
            st.pyplot(fig)
            st.subheader("MFCC")
            st.write("Biểu đồ hiển thị các hệ số MFCC, biểu thị đặc trưng âm sắc.")
            mfcc_input = librosa.feature.mfcc(y=y_input, sr=sr_input, n_mfcc=13)
            fig, axes = plt.subplots(1, len(nearest), figsize=(20, 4))

            if len(nearest) == 0:
                axes = [axes]
            else:
                axes = axes.flatten()

            librosa.display.specshow(mfcc_input, x_axis='time', sr=sr_input, ax=axes[0])
            axes[0].set_title("MFCC của Input Audio")
            axes[0].set_xlabel("Thời gian (s)")
            axes[0].set_ylabel("MFCC Coefficients")

            for i, output_file in enumerate(output_files):
               
                    y_output, sr_output = librosa.load(output_file, sr=16000)
                    mfcc_output = librosa.feature.mfcc(y=y_output, sr=sr_output, n_mfcc=13)
                    librosa.display.specshow(mfcc_output, x_axis='time', sr=sr_output, ax=axes[i + 1])
                    axes[i + 1].set_title(f"MFCC của Output {i + 1}")
                    axes[i + 1].set_xlabel("Thời gian (s)")
                    axes[i + 1].set_ylabel("MFCC Coefficients")

            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Spectrogram")
            st.write("Biểu đồ hiển thị phổ tần số theo thời gian.")
            D_input = librosa.amplitude_to_db(librosa.stft(y_input), ref=np.max)
            fig, axes = plt.subplots(1, len(nearest), figsize=(20, 4))

            if len(nearest) == 0:
                axes = [axes]
            else:
                axes = axes.flatten()

            librosa.display.specshow(D_input, x_axis='time', y_axis='log', sr=sr_input, ax=axes[0])
            axes[0].set_title("Spectrogram của Input Audio")
            axes[0].set_xlabel("Thời gian (s)")
            axes[0].set_ylabel("Tần số (Hz)")

            for i, output_file in enumerate(output_files):
                    y_output, sr_output = librosa.load(output_file, sr=16000)
                    D_output = librosa.amplitude_to_db(librosa.stft(y_output), ref=np.max)
                    librosa.display.specshow(D_output, x_axis='time', y_axis='log', sr=sr_output, ax=axes[i + 1])
                    axes[i + 1].set_title(f"Spectrogram của Output {i + 1}")
                    axes[i + 1].set_xlabel("Thời gian (s)")
                    axes[i + 1].set_ylabel("Tần số (Hz)")

            plt.tight_layout()
            st.pyplot(fig)

            feature_names = ["pitch", "energy", "zcr", "centroid"] + [f"mfcc_{i}" for i in range(13)]
            df_compare = pd.DataFrame(columns=["file", "distance"] + feature_names)

            input_row = ["Input", 0.0] + input_feats
            df_compare.loc[0] = input_row

            for idx, (fname, dist, feats) in enumerate(nearest):
                row = [fname, round(dist, 4)] + list(feats)
                df_compare.loc[idx + 1] = row

            st.subheader(f"So sánh đặc trưng ({distance_metric})")
            st.dataframe(df_compare, use_container_width=True)

            st.subheader("Biểu đồ so sánh đặc trưng")
            selected_features = st.multiselect("Chọn đặc trưng để so sánh", feature_names, default=["pitch", "energy", "zcr", "centroid"])
            if selected_features:
                fig = px.bar(df_compare, x="file", y=selected_features, barmode="group", title="So sánh đặc trưng")
                st.plotly_chart(fig)

            csv = df_compare.to_csv(index=False)
            st.download_button("Tải bảng so sánh", csv, "comparison.csv", "text/csv")
