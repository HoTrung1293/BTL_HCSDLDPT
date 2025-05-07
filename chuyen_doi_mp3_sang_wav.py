import os
from pydub import AudioSegment

# Thư mục chứa file mp3
input_folder = "D:\\Project\\Python\\Nhan_dien_giong_noi_csdldpt\\clips"  # thay bằng đường dẫn thực tế
output_folder = "D:\\Project\\Python\\Nhan_dien_giong_noi_csdldpt\\wav_data"  # có thể giống hoặc khác input_folder

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Lặp qua từng file trong thư mục
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".mp3"):
        mp3_path = os.path.join(input_folder, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(output_folder, wav_filename)

        # Đọc và chuyển đổi
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")

        print(f"Đã chuyển: {filename} → {wav_filename}")
