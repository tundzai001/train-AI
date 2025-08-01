# train_and_pack_robust_model_V5_FINAL.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
import os
import math

# --- TỐI ƯU HÓA GPU ---
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"-> Đã kích hoạt Mixed Precision Policy: {policy.name}. Tăng tốc huấn luyện trên GPU.")
except Exception as e:
    print(f"-> Không thể kích hoạt Mixed Precision. Lỗi: {e}. Tiếp tục với độ chính xác mặc định.")

print(f"Bắt đầu quy trình huấn luyện và đóng gói MÔ HÌNH VỮNG VÀNG (PHIÊN BẢN HOÀN CHỈNH)...")
start_time = time.time()

# --- CẤU HÌNH ---
SEQUENCE_LENGTH = 180
BATCH_SIZE = 512
N_FEATURES = 4
VALIDATION_SPLIT = 0.1
DATA_FILE = 'normal_data_robust.csv'

# --- Tên file output ---
SCALER_PATH = 'universal_scaler.gz'
AUTOENCODER_TFLITE_PATH = 'foundation_model_autoencoder.tflite'
FORECASTER_TFLITE_PATH = 'foundation_model_forecaster.tflite'
AUTOENCODER_H5_PATH = 'temp_autoencoder.h5'
FORECASTER_H5_PATH = 'temp_forecaster.h5'

# 1. Tải và chuẩn bị dữ liệu
if not os.path.exists(DATA_FILE):
    print(f"LỖI: Không tìm thấy file '{DATA_FILE}'. Vui lòng chạy script tạo dữ liệu trước.")
    exit()
    
print(f"Đang tải dữ liệu từ '{DATA_FILE}'...")
df = pd.read_csv(DATA_FILE)
df.rename(columns={'height': 'h'}, inplace=True)
features_df = df[['lat', 'lon', 'h', 'water_level']]

# 2. Scale dữ liệu
print("Đang scale dữ liệu...")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(features_df)
joblib.dump(scaler, SCALER_PATH)
print(f"Đã lưu scaler vào '{SCALER_PATH}'")

# --- 3. Tạo chuỗi dữ liệu bằng tf.data hiệu quả cao ---
print("Đang tạo quy trình dữ liệu với tf.data (hiệu quả cao)...")
data_tensor = tf.convert_to_tensor(data_scaled, dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices(data_tensor)
dataset = dataset.window(size=SEQUENCE_LENGTH, shift=5, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(SEQUENCE_LENGTH))

n_sequences = math.ceil((len(data_scaled) - SEQUENCE_LENGTH) / 5) 
val_size = int(VALIDATION_SPLIT * n_sequences)
train_size = n_sequences - val_size
print(f"Tổng số chuỗi: {n_sequences}, Train: {train_size}, Validation: {val_size}")

# (SỬA LỖI #3) Tính toán số bước cho mỗi epoch để Keras biết khi nào dừng
steps_per_epoch = math.ceil(train_size / BATCH_SIZE)
validation_steps = math.ceil(val_size / BATCH_SIZE)
print(f"Số bước mỗi epoch: Train: {steps_per_epoch}, Validation: {validation_steps}")

dataset = dataset.shuffle(buffer_size=n_sequences)

# Tạo dataset cho Autoencoder
ae_dataset = dataset.map(lambda sequence: (sequence, sequence))

# Tạo dataset cho Forecaster
def create_forecaster_xy(chunk):
    return chunk[:-1], chunk[-1]
fc_dataset = dataset.map(create_forecaster_xy)

# (SỬA LỖI #2) Thêm .repeat() để dữ liệu có thể được tái sử dụng qua nhiều epoch
train_ae_dataset = ae_dataset.take(train_size).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_ae_dataset = ae_dataset.skip(train_size).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_fc_dataset = fc_dataset.take(train_size).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_fc_dataset = fc_dataset.skip(train_size).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 4. Huấn luyện Mô hình A: AUTOENCODER ---
print("\n--- Huấn luyện Autoencoder ---")
ae_inputs = Input(shape=(SEQUENCE_LENGTH, N_FEATURES))
# (SỬA LỖI #1) Đổi relu -> tanh để ổn định hơn
ae_encoded = LSTM(128, activation='tanh', return_sequences=True)(ae_inputs)
ae_encoded = LSTM(64, activation='tanh')(ae_encoded)
ae_decoded = RepeatVector(SEQUENCE_LENGTH)(ae_encoded)
ae_decoded = LSTM(64, activation='tanh', return_sequences=True)(ae_decoded)
ae_decoded = LSTM(N_FEATURES, activation='linear', return_sequences=True, dtype=tf.float32)(ae_decoded) 
autoencoder = Model(ae_inputs, ae_decoded)

# (SỬA LỖI #1) Thêm Gradient Clipping để chống bùng nổ gradient
optimizer_ae = tf.keras.optimizers.Adam(clipnorm=1.0)
autoencoder.compile(optimizer=optimizer_ae, loss=tf.keras.losses.Huber())

# (SỬA LỖI #3) Cung cấp số bước cho Keras
autoencoder.fit(
    train_ae_dataset, 
    epochs=25, 
    validation_data=validation_ae_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]
)
autoencoder.save(AUTOENCODER_H5_PATH)

# --- 5. Huấn luyện Mô hình B: FORECASTER ---
print("\n--- Huấn luyện Forecaster ---")
fc_inputs = Input(shape=(SEQUENCE_LENGTH - 1, N_FEATURES))
# (SỬA LỖI #1) Đổi relu -> tanh để ổn định hơn
fc_lstm_1 = LSTM(128, activation='tanh', return_sequences=True)(fc_inputs)
fc_dropout = Dropout(0.3)(fc_lstm_1)
fc_lstm_2 = LSTM(64, activation='tanh')(fc_dropout)
fc_outputs = Dense(N_FEATURES, activation='linear', dtype=tf.float32)(fc_lstm_2)
forecaster = Model(fc_inputs, fc_outputs)

# (SỬA LỖI #1) Thêm Gradient Clipping để chống bùng nổ gradient
optimizer_fc = tf.keras.optimizers.Adam(clipnorm=1.0)
forecaster.compile(optimizer=optimizer_fc, loss='mae')

# (SỬA LỖI #3) Cung cấp số bước cho Keras
forecaster.fit(
    train_fc_dataset, 
    epochs=25, 
    validation_data=validation_fc_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]
)
forecaster.save(FORECASTER_H5_PATH)

# --- 6. Chuyển đổi sang TFLite ---
print("\n--- Đang chuyển đổi mô hình sang định dạng TFLite tối ưu...")
converter_ae = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter_ae.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_autoencoder_model = converter_ae.convert()
with open(AUTOENCODER_TFLITE_PATH, 'wb') as f:
    f.write(tflite_autoencoder_model)
print(f"-> Đã lưu Autoencoder TFLite vào '{AUTOENCODER_TFLITE_PATH}'")

converter_fc = tf.lite.TFLiteConverter.from_keras_model(forecaster)
converter_fc.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_forecaster_model = converter_fc.convert()
with open(FORECASTER_TFLITE_PATH, 'wb') as f:
    f.write(tflite_forecaster_model)
print(f"-> Đã lưu Forecaster TFLite vào '{FORECASTER_TFLITE_PATH}'")

os.remove(AUTOENCODER_H5_PATH)
os.remove(FORECASTER_H5_PATH)

end_time = time.time()
print(f"\nQUY TRÌNH SẢN XUẤT MÔ HÌNH VỮNG VÀNG (PHIÊN BẢN HOÀN CHỈNH) HOÀN TẤT trong {end_time - start_time:.2f} giây.")
