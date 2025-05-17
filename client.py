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
import module as mod
import io

# Thiết lập giao diện
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
    .main-menu {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Tạo menu chính
st.title("Ứng dụng Xử Lý Âm Thanh")

menu = ["Trích Xuất File Âm Thanh", "Tìm File Âm Thanh Tương Đồng", "Xóa File Âm Thanh", "Về Ứng Dụng"]
choice = st.selectbox("Chọn chức năng:", menu)

# CHỨC NĂNG 1: TÌM FILE ÂM THANH TƯƠNG ĐỒNG
if choice == "Tìm File Âm Thanh Tương Đồng":
    st.header("Tìm File Âm Thanh Tương Đồng")
    
    st.sidebar.subheader("📐 Cài đặt")
    distance_metric = st.sidebar.selectbox("Thuật toán đo khoảng cách", ["Cosine", "Euclidean", "Manhattan"])

    with st.sidebar.expander("🧪 Công thức các thuật toán"):
        st.markdown("**Euclidean Distance:**  \n"
                    "$$d(p, q) = \\sqrt{\\sum_i (p_i - q_i)^2}$$")
        st.markdown("**Cosine Similarity:**  \n"
                    "$$\\cos(\\theta) = \\frac{p \\cdot q}{\\|p\\| \\cdot \\|q\\|}$$")
        st.markdown("**Manhattan Distance:**  \n"
                    "$$d(p, q) = \\sum_i |p_i - q_i|$$")

    st.sidebar.header("Tải lên file âm thanh")
    audio_files = [os.path.basename(f) for f in glob(os.path.join("data", "*.wav"))]
    upload_option = st.sidebar.radio("Chọn nguồn file âm thanh", ("Chọn từ thư mục data"))
    selected_file = None
    uploaded_file = None

    if upload_option == "Chọn từ thư mục data":
        selected_file = st.sidebar.selectbox("Chọn file âm thanh từ thư mục data", audio_files)
    if selected_file:
        file_path = os.path.join("data", selected_file)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        st.audio(file_bytes, format="audio/wav")
        with st.spinner("Đang xử lý file âm thanh..."):
            nearest, input_feats = mod.find_similarity_file(file_path, top_k=3, distance_metric=distance_metric)
        if nearest:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", f"{os.path.splitext(selected_file)[0]}_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            for filename, dist, _ in nearest:
                output_path = os.path.join(output_dir, filename)
                input_path = os.path.join("data", filename)
                shutil.copy(input_path, output_path)

            st.subheader(f"Kết quả tìm kiếm ({distance_metric})")

            for filename, dist, _ in nearest:
                if filename != selected_file:
                    similarity = (1 / (1 + dist)) * 100
                    st.write(f"{filename}: Độ tương đồng {similarity:.2f}%")
                    st.audio(os.path.join(output_dir, filename), format="audio/wav")
            y_input, sr_input = librosa.load(file_path, sr=16000)
            output_files = [os.path.join(output_dir, filename) for filename, _, _ in nearest if filename != selected_file]
            y_outputs = [librosa.load(output_file, sr=16000)[0] for output_file in output_files]
            sr_outputs = [16000] * len(y_outputs)

            def plot_2x2_ms(fig_title, plot_func, *args):
                fig, axes = plt.subplots(2, 2, figsize=(20, 10))
                axes = axes.flatten()
                titles = ["Input Audio"] + [f"Output {i+1}" for i in range(len(y_outputs))]
                for i, ax in enumerate(axes):
                    if i == 0:
                        plot_func(ax, y_input, sr_input, *args)
                    else:
                        plot_func(ax, y_outputs[i-1], sr_outputs[i-1], *args)
                    ax.set_title(titles[i], fontsize=14)
                    ax.set_xlabel("Thời gian (ms)", fontsize=12)
                    ax.set_ylabel(
                        "Biên độ" if "wave" in fig_title.lower() else
                        "MFCC Coefficients" if "mfcc" in fig_title.lower() else
                        "Tần số (Hz)",
                        fontsize=12
                    )
                plt.tight_layout()
                st.pyplot(fig)

            st.subheader("Waveform (ms)")
            st.write("Biểu đồ hiển thị biên độ âm thanh theo thời gian (ms).")
            def plot_waveform_ms(ax, y, sr):
                times_ms = np.arange(len(y)) / sr * 1000  
                ax.plot(times_ms, y)
                ax.xaxis.set_major_locator(plt.MaxNLocator(6))
            plot_2x2_ms("Waveform", plot_waveform_ms)

            def convert_xticks_to_ms(ax):
                ticks_s = ax.get_xticks()
                ax.set_xticklabels([f"{t*1000:.0f}" for t in ticks_s])

            st.subheader("MFCC (ms)")
            st.write("Biểu đồ hiển thị hệ số MFCC theo thời gian (ms).")
            def plot_mfcc_ms(ax, y, sr):
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                img = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=ax)
                convert_xticks_to_ms(ax)
            plot_2x2_ms("MFCC", plot_mfcc_ms)

            st.subheader("Spectrogram (ms)")
            st.write("Biểu đồ hiển thị phổ tần số theo thời gian.")
            def plot_spectrogram_ms(ax, y, sr):
                D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
                librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
                convert_xticks_to_ms(ax)
            plot_2x2_ms("Spectrogram", plot_spectrogram_ms)
            
            feature_names = ["pitch", "energy", "zcr", "centroid", "bandwidth", "silence_ratio"] \
                + [f"mfcc_{i}" for i in range(13)]
            rows = []
            rows.append({
                "file": "Input",
                "distance": 0.0,
                **dict(zip(feature_names, input_feats))
            })
            for fname, dist, feats in nearest:
                if feats is not None:
                    rows.append({
                        "file": fname,
                        "distance": round(dist, 4),
                        **dict(zip(feature_names, feats))
                    })
            df_compare = pd.DataFrame(rows, columns=["file", "distance"] + feature_names)
            st.subheader(f"So sánh đặc trưng ({distance_metric})")
            st.dataframe(df_compare, use_container_width=True)

            st.subheader("Biểu đồ so sánh đặc trưng")
            available_features = [f for f in feature_names if f in df_compare.columns]
            default_features = ["pitch", "energy", "centroid", "bandwidth", "silence_ratio"]
            default_features = [f for f in default_features if f in available_features]
            selected_features = st.multiselect("Chọn đặc trưng để so sánh", 
                                              available_features, 
                                              default=default_features[:4])
            
            if selected_features and len(df_compare) > 0:
                try:
                    fig = px.bar(
                        df_compare,
                        x="file",
                        y=selected_features,
                        barmode="group",
                        title="So sánh đặc trưng giữa input và các file tương đồng",
                        height=600,
                        color_discrete_sequence=px.colors.qualitative.Set2 
                    )
                    fig.update_layout(
                        font=dict(size=14),
                        title_font_size=18,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(t=80, b=40, l=40, r=40)
                    )
                    fig.update_xaxes(
                        tickangle=-45,
                        tickfont=dict(size=12)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if len(selected_features) >= 3:
                        radar_data = df_compare[["file"] + selected_features].copy()
                        fig_radar = px.line_polar(radar_data, r=selected_features, theta="file", line_close=True)
                        st.plotly_chart(fig_radar, use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi khi tạo biểu đồ: {str(e)}")

            if len(df_compare) > 0:
                st.subheader("Heat Map")
                available_heatmap_features = [f for f in feature_names if f in df_compare.columns]
                features_for_heatmap = available_heatmap_features[:min(8, len(available_heatmap_features))]
                
                if features_for_heatmap:
                    try:
                        heatmap_data = df_compare.set_index("file")[features_for_heatmap]
                        fig = px.imshow(heatmap_data, 
                                      labels=dict(x="Đặc trưng", y="File", color="Giá trị"),
                                      title="Heat Map so sánh các đặc trưng",
                                      height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi khi tạo heat map: {str(e)}")

            if not df_compare.empty:
                csv = df_compare.to_csv(index=False)
                st.download_button("Tải bảng so sánh", csv, "comparison.csv", "text/csv")

# CHỨC NĂNG 2: TRÍCH XUẤT FILE ÂM THANH
elif choice == "Trích Xuất File Âm Thanh":
    st.header("Trích Xuất File Âm Thanh")
    
    extract_option = st.radio("Chọn phương thức trích xuất:", ("Trích xuất thư mục vào cơ sở dữ liệu", "Thêm file âm thanh mới vào cơ sở dữ liệu"))
    
    if extract_option == "Trích xuất thư mục vào cơ sở dữ liệu":
        folder_path_input = st.text_input("Nhập đường dẫn thư mục chứa file âm thanh :", "data")
        paths = [path.strip() for path in folder_path_input.split(',')]
        folder_path = paths[0]
        
        if st.button("Bắt đầu trích xuất và lưu vào CSDL"):
            with st.spinner("Đang trích xuất đặc trưng từ tất cả file âm thanh và lưu vào cơ sở dữ liệu..."):
                results = mod.extract_and_save_all_audio_files(folder_path)
                df_results = pd.DataFrame(results["processed_files"])
                if not df_results.empty:
                    st.write(f"Đã xử lý {results['success']}/{results['total']} file thành công:")
                    st.dataframe(df_results)
                    csv = df_results.to_csv(index=False)
                    st.download_button("Tải kết quả xử lý", csv, "processing_results.csv", "text/csv")
    else:
        uploaded_file = st.file_uploader("Tải lên file âm thanh mới (wav)", type=["wav"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("Thêm vào cơ sở dữ liệu"):
                with st.spinner("Đang xử lý file âm thanh và lưu vào cơ sở dữ liệu..."):
                    os.makedirs("data", exist_ok=True)
                    file_path = os.path.join("data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    success = mod.save_audio_features_to_db(file_path)
                    if success:
                        st.success(f"Đã thêm file {uploaded_file.name} vào cơ sở dữ liệu thành công!")
                    else:
                        st.error(f"Không thể thêm file {uploaded_file.name} vào cơ sở dữ liệu.")

# CHỨC NĂNG 3: XÓA FILE ÂM THANH
elif choice == "Xóa File Âm Thanh":
    st.header("Xóa File Âm Thanh")
    
    audio_files = [os.path.basename(f) for f in glob(os.path.join("data", "*.wav"))]
    
    if not audio_files:
        st.warning("Không có file âm thanh nào trong thư mục data.")
        st.stop()
    
    selected_file = st.selectbox("Chọn file âm thanh để xóa:", audio_files)
    
    if selected_file:
        file_path = os.path.join("data", selected_file)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        st.audio(file_bytes, format="audio/wav")
        
        st.warning("Hành động này sẽ xóa file khỏi thư mục data và dữ liệu liên quan trong cơ sở dữ liệu. Bạn có chắc chắn muốn tiếp tục?")
        confirm_delete = st.checkbox("Xác nhận xóa file")
        
        if confirm_delete and st.button("Xóa File"):
            with st.spinner("Đang xóa file và dữ liệu..."):
                success, message = mod.delete_audio_file(selected_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)

# CHỨC NĂNG 4: VỀ ỨNG DỤNG
else:
    st.header("Về Ứng Dụng")
    st.write("""
    ### Ứng dụng Xử Lý Âm Thanh
    
    Đây là ứng dụng cho phép người dùng:
    
    1. **Tìm File Âm Thanh Tương Đồng**: Tìm những file âm thanh tương tự với file đã chọn dựa trên các đặc trưng âm thanh.
    2. **Trích Xuất File Âm Thanh**: Trích xuất và lưu các file âm thanh từ thư mục data.
    3. **Xóa File Âm Thanh**: Xóa file âm thanh khỏi thư mục data và dữ liệu liên quan trong cơ sở dữ liệu.
    
    Ứng dụng sử dụng thư viện librosa để phân tích âm thanh và trích xuất các đặc trưng như:
    - MFCC (Mel Frequency Cepstral Coefficients)
    - Pitch (Cao độ)
    - Energy (Năng lượng)
    - ZCR (Zero Crossing Rate)
    - Centroid (Trọng tâm phổ)
    - Bandwidth (Băng thông)
    - Silence Ratio (Tỷ lệ im lặng)
    
    ### Công Nghệ Sử Dụng
    - Python
    - Streamlit
    - Librosa
    - Pandas & NumPy
    - MySQL
    
    ### Tác Giả
    - Dự án này là một phần của môn học Hệ cơ sở dữ liệu đa phương tiện.
    """)