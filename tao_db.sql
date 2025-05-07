create database voice_db;
Use voice_db;
CREATE TABLE voices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255),        -- Tên file âm thanh (vd: man001.wav)
    mfcc TEXT,                -- MFCC lưu dưới dạng chuỗi JSON
    pitch FLOAT,              -- Độ cao (tần số cơ bản)
    energy FLOAT,             -- Năng lượng trung bình
    zcr FLOAT,                -- Zero Crossing Rate
    centroid FLOAT            -- Trọng tâm phổ (Spectral Centroid)
);
