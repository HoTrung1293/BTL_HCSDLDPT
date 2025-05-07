import os
import random
import numpy as np
import soundfile as sf
from scipy.signal import resample

input_folder =  "D:\\Project\\Python\\Nhan_dien_giong_noi_csdldpt\\wav_data"
output_folder = "D:\\Project\\Python\\Nhan_dien_giong_noi_csdldpt\\data"
target_duration_ms = 5000  # 5 seconds in milliseconds

# Thông số chuẩn
target_duration = 5  # giây
target_sr = 16000  # tần số mẫu
target_length = target_duration * target_sr


all_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
selected_files = random.sample(all_files, 100)
for filename in selected_files:
    filepath = os.path.join(input_folder, filename)

    # Đọc file
    audio, sr = sf.read(filepath)

    # Nếu là stereo → chuyển mono
    if len(audio.shape) == 2:
        audio = np.mean(audio, axis=1)

    # Nếu sample rate khác 16kHz → resample
    if sr != target_sr:
        duration = len(audio) / sr
        new_length = int(duration * target_sr)
        audio = resample(audio, new_length)

    # Cắt hoặc đệm về đúng 5s
    if len(audio) > target_length:
        start = random.randint(0, len(audio) - target_length)
        audio = audio[start:start + target_length]
    elif len(audio) < target_length:
        padding = np.zeros(target_length - len(audio))
        audio = np.concatenate([audio, padding])

    # Lưu file mới
    output_path = os.path.join(output_folder, filename)
    sf.write(output_path, audio, target_sr)

print("Script đã chạy vào được")
