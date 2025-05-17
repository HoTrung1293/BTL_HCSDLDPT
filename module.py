import librosa
import time
import streamlit as st
import joblib
import calculation_module as calc
import os
import pymysql
from dotenv import load_dotenv
from glob import glob
import numpy as np
from scipy.fftpack import fft, dct
import pandas as pd

def connect_to_db():
    """Kết nối đến cơ sở dữ liệu từ biến môi trường"""
    load_dotenv() 
    conn = pymysql.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME')
    )
    return conn, conn.cursor()


def extract_audio_features(audio_path):
    try:
        import scipy.io.wavfile as wav
        
        # Load audio file
        sr, y = wav.read(audio_path)
        
        # Convert to float and normalize
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        
        # Ensure consistent length (5 seconds)
        target_length = 5 * sr
        if len(y.shape) > 1:  # Convert stereo to mono by averaging channels
            y = np.mean(y, axis=1)
        
        # Fix length
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, max(0, target_length - len(y))), 'constant')
        
        # Calculate ZCR (Zero Crossing Rate)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(y)))) / len(y)
        
        # Calculate Energy
        energy = np.mean(y ** 2)
        
        # Calculate Pitch using autocorrelation
        frame_size = 2048
        hop_length = 512
        correlation = np.correlate(y[:frame_size], y[:frame_size], mode='full')
        correlation = correlation[len(correlation)//2:]
        
        # Find peaks in autocorrelation
        peak_indices = np.where((correlation[1:-1] > correlation[:-2]) & 
                              (correlation[1:-1] > correlation[2:]))[0] + 1
        
        if len(peak_indices) > 0:
            # Get the first peak after the initial drop
            peak_values = correlation[peak_indices]
            sorted_peaks = sorted(zip(peak_indices, peak_values), key=lambda x: -x[1])
            for idx, val in sorted_peaks:
                if idx > 20:  # Minimum frequency threshold
                    pitch_idx = idx
                    break
            else:
                pitch_idx = sorted_peaks[0][0]
                
            pitch = sr / pitch_idx if pitch_idx > 0 else 0
        else:
            pitch = 0
            
        # Calculate Spectral Centroid
        # Compute magnitude spectrum
        n_fft = 2048
        half_n_fft = n_fft // 2 + 1
        
        # Get FFT of frame
        magnitude_spectrum = np.abs(fft(y[:n_fft]))[:half_n_fft]
        
        # Calculate frequency bins
        freqs = np.linspace(0, sr/2, half_n_fft)
        
        # Compute centroid
        centroid = np.sum(magnitude_spectrum * freqs) / (np.sum(magnitude_spectrum) + 1e-10)
        
        # Manual MFCC calculation
        n_mfcc = 13
        n_mels = 40
        
        # Calculate mel filterbank
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz/700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel/2595) - 1)
        
        # Create mel filterbank
        min_mel = hz_to_mel(0)
        max_mel = hz_to_mel(sr / 2)
        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        
        fbank = np.zeros((n_mels, half_n_fft))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_indices[m - 1]
            f_m = bin_indices[m]
            f_m_plus = bin_indices[m + 1]
            
            for k in range(f_m_minus, f_m):
                fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        
        # Apply mel filterbank to power spectrum
        power_spectrum = magnitude_spectrum ** 2
        mel_spectrum = np.dot(fbank, power_spectrum)
        
        # Apply log
        mel_spectrum = np.where(mel_spectrum > 0, np.log(mel_spectrum), 0)
        
        # Apply DCT to get MFCC
        mfcc = dct(mel_spectrum, type=2, norm='ortho')[:n_mfcc]
        
        # Calculate Spectral Bandwidth
        # Bandwidth là độ lệch chuẩn của phổ tần số quanh centroid
        bandwidth = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * magnitude_spectrum) / (np.sum(magnitude_spectrum) + 1e-10)
        )

        # Calculate Silence Ratio
        # Định nghĩa ngưỡng im lặng là 2% năng lượng tối đa
        silence_threshold = 0.02 * np.max(np.abs(y))
        silence_ratio = np.sum(np.abs(y) < silence_threshold) / len(y)

        return {
            "mfcc": mfcc.tolist(),
            "pitch": float(pitch),
            "energy": float(energy),
            "zcr": float(zero_crossings),
            "centroid": float(centroid),
            "bandwidth": float(bandwidth),
            "silence_ratio": float(silence_ratio)
        }
    except Exception as e:
        st.error(f"Lỗi khi xử lý file âm thanh: {e}")
        return None

def save_features_to_mysql(folder_path):
    """Lưu đặc trưng của các file âm thanh vào cơ sở dữ liệu MySQL"""
    conn, cursor = connect_to_db()
    data = []
    
    # Xóa dữ liệu cũ trước khi thêm mới
    cursor.execute("DELETE FROM voice")
    cursor.execute("DELETE FROM voice_normalized")
    conn.commit()
    
    # Đọc và tích trắc đặc trưng của các file âm thanh
    for file in glob(os.path.join(folder_path, "*.wav")):
        features = extract_audio_features(file)
        row = {
            "filename": os.path.basename(file),
            "pitch": features["pitch"],
            "energy": features["energy"],
            "zcr": features["zcr"],
            "centroid": features["centroid"],
            "bandwidth": features["bandwidth"],
            "silence_ratio": features["silence_ratio"]
        }
        # Thêm các hệ số MFCC vào row
        for i, mfcc_val in enumerate(features["mfcc"]):
            row[f"mfcc_{i+1}"] = float(mfcc_val)
            
        data.append(row)
    
    # Lưu đặc trưng vào bảng voice
    for item in data:
        cursor.execute("""
        INSERT INTO voice (filename, pitch, energy, zcr, centroid, bandwidth, silence_ratio,
                        mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7,
                        mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            item["filename"], item["pitch"], item["energy"], item["zcr"], item["centroid"],
            item["bandwidth"], item["silence_ratio"],
            item["mfcc_1"], item["mfcc_2"], item["mfcc_3"], item["mfcc_4"], item["mfcc_5"],
            item["mfcc_6"], item["mfcc_7"], item["mfcc_8"], item["mfcc_9"], item["mfcc_10"],
            item["mfcc_11"], item["mfcc_12"], item["mfcc_13"]
        ))
        
    conn.commit()
    print(f" Đã lưu đặc trưng vào cơ sở dữ liệu")


def save_audio_features_to_db(audio_path, conn=None, cursor=None):
    """
    Trích xuất đặc trưng âm thanh và lưu vào cơ sở dữ liệu
    
    Parameters:
    audio_path (str): Đường dẫn đến file âm thanh
    conn: Kết nối database (tùy chọn)
    cursor: Con trỏ database (tùy chọn)
    
    Returns:
    bool: True nếu lưu thành công, False nếu thất bại
    """
    try:
        # Tạo kết nối nếu chưa có
        if conn is None or cursor is None:
            conn, cursor = connect_to_db()
            close_after = True
        else:
            close_after = False
        
        # Trích xuất đặc trưng
        features = extract_audio_features(audio_path)
        if features is None:
            return False
              # Lưu vào cơ sở dữ liệu bảng voice
        filename = os.path.basename(audio_path)
        pitch = features["pitch"]
        energy = features["energy"]
        zcr = features["zcr"]
        centroid = features["centroid"]
        bandwidth = features["bandwidth"]
        silence_ratio = features["silence_ratio"]
        mfccs = features["mfcc"]
        
        # Tạo câu lệnh SQL cho bảng voice
        columns = ["filename", "pitch", "energy", "zcr", "centroid", "bandwidth", "silence_ratio"] + [f"mfcc_{i+1}" for i in range(len(mfccs))]
        placeholders = ", ".join(["%s"] * len(columns))
        
        sql = f"INSERT INTO voice ({', '.join(columns)}) VALUES ({placeholders})"
        
        # Chuẩn bị giá trị để chèn
        values = [filename, pitch, energy, zcr, centroid, bandwidth, silence_ratio] + mfccs
        
        # Thực thi câu lệnh SQL
        cursor.execute(sql, values)
        
        # Commit thay đổi
        conn.commit()
        
        # Chuẩn hóa và lưu dữ liệu vào bảng normalized
        success = standardize_voice_features(conn, cursor)
        
        # Đóng kết nối nếu được tạo trong hàm
        if close_after:
            cursor.close()
            conn.close()
            
        return True
    except Exception as e:
        st.error(f"Lỗi khi lưu đặc trưng âm thanh vào cơ sở dữ liệu: {e}")
        
        # Đóng kết nối nếu được tạo trong hàm và gặp lỗi
        if 'close_after' in locals() and close_after and conn is not None:
            if cursor is not None:
                cursor.close()
            conn.close()
            
        return False


def standardize_voice_features(conn=None, cursor=None):
    """
    Chuẩn hóa tất cả đặc trưng từ bảng voice sang bảng voice_normalized
    sử dụng chuẩn hóa min-max trong khoảng [-1,1]
    
    Parameters:
    conn: Kết nối database (tùy chọn)
    cursor: Con trỏ database (tùy chọn)
    
    Returns:
    bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Tạo kết nối nếu chưa có
        if conn is None or cursor is None:
            conn, cursor = connect_to_db()
            close_after = True
        else:
            close_after = False
          # Xóa dữ liệu cũ trong bảng normalized nếu có
        cursor.execute("TRUNCATE TABLE voice_normalized")
        
        # Lấy tất cả các đặc trưng từ bảng gốc
        cursor.execute("SELECT id, filename, pitch, energy, zcr, centroid, bandwidth, silence_ratio, mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13 FROM voice")
        rows = cursor.fetchall()
        
        if not rows:
            st.warning("Không có dữ liệu để chuẩn hóa!")
            return False
        
        # Chuyển đổi thành dataframe để dễ xử lý
        feature_names = ["id", "filename", "pitch", "energy", "zcr", "centroid", "bandwidth", "silence_ratio",
                         "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13"]
        
        df = pd.DataFrame(rows, columns=feature_names)
        
        # Tách các cột cần chuẩn hóa (không chuẩn hóa id và filename)
        features_to_normalize = feature_names[2:]  # Bỏ qua id và filename
        
        # Thực hiện chuẩn hóa min-max từ [-1,1]
        for feature in features_to_normalize:
            feature_min = df[feature].min()
            feature_max = df[feature].max()
            
            # Tránh chia cho 0 nếu min=max
            if feature_max == feature_min:
                df[feature + '_norm'] = 0  # Nếu không có sự khác biệt, gán giá trị 0
            else:
                # Công thức chuẩn hóa min-max scale từ [-1,1]
                df[feature + '_norm'] = 2 * (df[feature] - feature_min) / (feature_max - feature_min) - 1
          # Lưu dữ liệu đã chuẩn hóa vào bảng normalized
        for _, row in df.iterrows():
            sql = ("INSERT INTO voice_normalized (filename, pitch, energy, zcr, centroid, bandwidth, silence_ratio, "
                   "mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13) "
                   "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
            
            values = [
                row['filename'],
                row['pitch_norm'],
                row['energy_norm'],
                row['zcr_norm'],
                row['centroid_norm'],
                row['bandwidth_norm'],
                row['silence_ratio_norm'],
                row['mfcc_1_norm'],
                row['mfcc_2_norm'],
                row['mfcc_3_norm'],
                row['mfcc_4_norm'],
                row['mfcc_5_norm'],
                row['mfcc_6_norm'],
                row['mfcc_7_norm'],
                row['mfcc_8_norm'],
                row['mfcc_9_norm'],
                row['mfcc_10_norm'],
                row['mfcc_11_norm'],
                row['mfcc_12_norm'],
                row['mfcc_13_norm']
            ]
            
            cursor.execute(sql, values)
        
        # Commit thay đổi
        conn.commit()
        
        # Đóng kết nối nếu được tạo trong hàm
        if close_after:
            cursor.close()
            conn.close()
        
        return True
    except Exception as e:
        st.error(f"Lỗi khi chuẩn hóa đặc trưng âm thanh: {e}")
        
        # Đóng kết nối nếu được tạo trong hàm và gặp lỗi
        if 'close_after' in locals() and close_after and conn is not None:
            if cursor is not None:
                cursor.close()
            conn.close()
        
        return False


def find_similarity_file(input_audio_path, distance_metric, top_k=3):
    """
    Tìm các file âm thanh tương đồng với file đầu vào
    
    Parameters:
    -----------
    input_audio_path : str
        Đường dẫn đến file âm thanh đầu vào
    distance_metric : str
        Phương pháp tính khoảng cách ("Cosine", "Euclidean", "Manhattan")
    top_k : int, default=3
        Số lượng file tương đồng cần trả về
        
    Returns:
    --------
    list, dict
        Danh sách các file tương đồng và đặc trưng của file đầu vào
    """    
    start_time = time.time()
    # Trích xuất đặc trưng từ file input
    input_filename = os.path.basename(input_audio_path)  # Lấy tên file đầu vào để loại bỏ nếu trùng
    input_features = extract_audio_features(input_audio_path)
    
    if input_features is None:
        return [], None
        
    try:
        conn, cursor = connect_to_db()
        
        # Lấy dữ liệu từ bảng chuẩn hóa
        cursor.execute("SELECT filename, pitch, energy, zcr, centroid, bandwidth, silence_ratio, "
                "mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13 "
                "FROM voice_normalized")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Tạo vector chứa tất cả dữ liệu
        filenames = []
        features = []

        for row in rows:
            filenames.append(row[0])  # Lưu tên file riêng
            features.append(list(row[1:]))  # Lưu các đặc trưng

        # Chuyển đổi thành mảng NumPy
        filenames = np.array(filenames)
        features = np.array(features)
        
        # Tìm file input trong danh sách nếu có
        input_index = -1
        for i, filename in enumerate(filenames):
            if input_filename == filename:
                input_index = i
                break
        
        # Nếu input_file đã tồn tại trong database
        input_index != -1
        input_vector = features[input_index]
        
        # Tính độ tương đồng/khoảng cách
        if distance_metric == "Cosine":
            similarities = calc.cosine_similarity_np(input_vector, features)

            # Tạo các cặp (index, similarity, filename, features) bỏ qua chính file đó nếu có
            similarity_pairs = []
            for i in range(len(similarities)):
                if input_index == -1 or i != input_index:
                    similarity_pairs.append((i, similarities[i], filenames[i], features[i]))
            
            # Sắp xếp theo độ tương đồng giảm dần
            sorted_pairs = sorted(similarity_pairs, key=lambda x: -x[1])
            
            # Lấy top k kết quả
            nearest_files = [(filename, 1 - similarity, feats) 
                            for _, similarity, filename, feats in sorted_pairs[:top_k]]
            
        elif distance_metric == "Euclidean":
            distances = calc.euclidean_distance_np(input_vector, features)
            
            distance_pairs = []
            for i in range(len(distances)):
                if input_index == -1 or i != input_index:
                    distance_pairs.append((i, distances[i], filenames[i], features[i]))
            
            sorted_pairs = sorted(distance_pairs, key=lambda x: x[1])
            nearest_files = [(filename, distance, feats) 
                            for _, distance, filename, feats in sorted_pairs[:top_k]]
            
        elif distance_metric == "Manhattan":
            distances = calc.manhattan_distance_np(input_vector, features)
            
            distance_pairs = []
            for i in range(len(distances)):
                if input_index == -1 or i != input_index:
                    distance_pairs.append((i, distances[i], filenames[i], features[i]))
            
            sorted_pairs = sorted(distance_pairs, key=lambda x: x[1])
            nearest_files = [(filename, distance, feats) 
                            for _, distance, filename, feats in sorted_pairs[:top_k]]
        else:
            raise ValueError("Unsupported distance metric")
        
        end_time = time.time()
        time_taken = end_time - start_time
        st.success(f"✅ Tìm thấy {len(nearest_files)} file tương đồng. Thời gian xử lý: {time_taken:.2f} giây.")
        
        return nearest_files, input_features
    
    except Exception as e:
        st.error(f"Lỗi khi tìm kiếm file tương đồng: {e}")
        return [], input_features

def extract_and_save_all_audio_files(audio_path):
    """
    Trích xuất và lưu đặc trưng của tất cả các file âm thanh trong thư mục data
    sử dụng hàm save_features_to_mysql
    
    Returns:
    dict: Kết quả xử lý với số lượng file đã xử lý thành công và thất bại
    """
    import os
    
    data_dir = audio_path
    
    # Hiển thị thanh tiến trình
    if 'extract_progress_bar' not in st.session_state:
        st.session_state.extract_progress_bar = st.progress(0)
        st.session_state.extract_status = st.empty()
    
    st.session_state.extract_progress_bar.progress(0)
    st.session_state.extract_status.text("Đang chuẩn bị xử lý tất cả file âm thanh...")
    
    results = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "processed_files": []
    }
    
    try:
        # Sử dụng hàm save_features_to_mysql để lưu tất cả đặc trưng vào cơ sở dữ liệu
        st.session_state.extract_status.text("Đang trích xuất và lưu đặc trưng...")
        st.session_state.extract_progress_bar.progress(30)
        
        save_features_to_mysql(data_dir)
        standardize_voice_features()
        st.session_state.extract_progress_bar.progress(80)
        st.session_state.extract_status.text("Đang hoàn tất...")
        
        # Cập nhật thông tin kết quả
        from glob import glob
        audio_files = glob(os.path.join(data_dir, "*.wav"))
        results["total"] = len(audio_files)
        results["success"] = len(audio_files)
        
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            results["processed_files"].append({"filename": filename, "status": "success"})
        
        st.session_state.extract_progress_bar.progress(100)
    except Exception as e:
        st.error(f"Lỗi khi xử lý tất cả file âm thanh: {e}")
        results["failed"] = results["total"]
        
        # Thêm thông tin lỗi vào kết quả
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            results["processed_files"].append({
                "filename": filename, 
                "status": "failed", 
                "error": str(e)
            })
    
    # Hiển thị thông báo kết quả
    if results["failed"] == 0:
        st.success(f"✅ Đã xử lý thành công tất cả {results['total']} file âm thanh.")
    else:
        st.warning(f"⚠️ Đã xử lý {results['success']}/{results['total']} file âm thanh. {results['failed']} file bị lỗi.")
    
    return results

def extract_audio_file(filename, output_dir='wav_extracted'):
    """
    Trích xuất một file âm thanh từ thư mục data vào thư mục output
    
    Parameters:
    filename (str): Tên file âm thanh cần trích xuất
    output_dir (str): Thư mục đích để lưu file trích xuất
    
    Returns:
    dict: Thông tin về kết quả trích xuất và đường dẫn đến file trích xuất
    """
    import os
    import shutil
    import numpy as np
    from scipy.fftpack import fft, dct
    
    data_dir = "data"
    source_path = os.path.join(data_dir, filename)
    
    # Kiểm tra file tồn tại
    if not os.path.exists(source_path):
        return {
            "success": False,
            "error": f"File {filename} không tồn tại trong thư mục data."
        }
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Đường dẫn đích
    dest_path = os.path.join(output_dir, filename)
    
    try:
        # Copy file
        shutil.copy2(source_path, dest_path)
        
        # Trích xuất đặc trưng
        features = extract_audio_features(source_path)
        
        return {
            "success": True,
            "path": dest_path,
            "features": features
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
def delete_audio_file(filename):
    try:
        file_path = os.path.join("data", filename)
        
        # 1. Xóa file trong thư mục
        file_deleted = False
        if os.path.exists(file_path):
            os.remove(file_path)
            file_deleted = True
        else:
            return False, f"File {filename} không tồn tại trong thư mục data."

        # 2. Xóa dữ liệu trong cơ sở dữ liệu
        conn, cursor = connect_to_db()
        if conn is None or cursor is None:
            return False, f"Lỗi: Không thể kết nối đến cơ sở dữ liệu."

        # Kiểm tra xem dữ liệu có tồn tại trước khi xóa
        cursor.execute("SELECT COUNT(*) FROM voice WHERE filename = %s", (filename,))
        voice_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM voice_normalized WHERE filename = %s", (filename,))
        normalized_count = cursor.fetchone()[0]

        # Xóa bản ghi trong bảng voice
        cursor.execute("DELETE FROM voice WHERE filename = %s", (filename,))
        voice_deleted = cursor.rowcount
        
        # Xóa bản ghi trong bảng voice_normalized
        cursor.execute("DELETE FROM voice_normalized WHERE filename = %s", (filename,))
        normalized_deleted = cursor.rowcount
        
        conn.commit()
        cursor.close()
        conn.close()

        # Kiểm tra kết quả
        if file_deleted and (voice_deleted > 0 or normalized_deleted > 0):
            return True, f"Đã xóa file {filename} và dữ liệu liên quan thành công. (voice: {voice_deleted}, voice_normalized: {normalized_deleted})"
        elif file_deleted and voice_count == 0 and normalized_count == 0:
            return True, f"Đã xóa file {filename} khỏi thư mục, nhưng không có dữ liệu liên quan trong cơ sở dữ liệu."
        else:
            return False, f"Đã xóa file {filename}, nhưng không thể xóa dữ liệu trong cơ sở dữ liệu. (voice: {voice_deleted}, voice_normalized: {normalized_deleted})"
    except Exception as e:
        return False, f"Lỗi khi xóa file/dữ liệu: {str(e)}"