import tkinter as tk
from tkinter import filedialog, messagebox
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import mysql.connector
import json
import os

# Trích xuất đặc trưng từ file .wav
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.fix_length(y, 5 * sr)

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

# So sánh và tìm top 3 file gần nhất
def find_top3_similar(features):
    input_vec = features['mfcc'] + [
        features['pitch'],
        features['energy'],
        features['zcr'],
        features['centroid']
    ]

    conn = mysql.connector.connect(
        host="localhost", user="root", password="123456", database="voice_db"
    )
    cursor = conn.cursor()

    cursor.execute("SELECT name, mfcc, pitch, energy, zcr, centroid FROM voices")
    results = []
    for name, mfcc_str, pitch, energy, zcr, centroid in cursor.fetchall():
        db_vec = json.loads(mfcc_str) + [pitch, energy, zcr, centroid]
        similarity = 1 - cosine(input_vec, db_vec)
        results.append((name, similarity))

    conn.close()
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]

# Xử lý khi người dùng chọn file
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("input files", "*.mp3")])
    if not file_path:
        return

    try:
        features = extract_features(file_path)
        top3 = find_top3_similar(features)

        result_text = "\n".join([
            f"{i+1}. {name} (Tương đồng: {similarity:.4f})"
            for i, (name, similarity) in enumerate(top3)
        ])
        result_label.config(text=result_text)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể xử lý file âm thanh.\n{str(e)}")

# Giao diện chính
root = tk.Tk()
root.title("Tìm giọng nói giống nhất")
root.geometry("450x300")

frame = tk.Frame(root)
frame.pack(pady=20)

choose_btn = tk.Button(frame, text="Chọn file âm thanh (.wav)", command=browse_file, font=("Arial", 12))
choose_btn.pack(pady=10)

result_label = tk.Label(root, text="Chọn một file để tìm 3 giọng nói giống nhất", font=("Arial", 11), wraplength=400, justify="left")
result_label.pack(pady=10)

root.mainloop()
