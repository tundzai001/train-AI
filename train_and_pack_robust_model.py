import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
import os
import numpy as np

# Cấu hình tối ưu cho RTX 4060
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Cấu hình bộ nhớ: 7GB/8GB
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7168)]
        )
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Đã cấu hình GPU: NVIDIA GeForce RTX 4060 8GB")
    except RuntimeError as e:
        print(e)

# Kích hoạt mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f'Mixed precision: {policy.compute_dtype} tính toán, {policy.variable_dtype} biến')

print("Bắt đầu huấn luyện MÔ HÌNH VỮNG VÀNG trên RTX 4060...")
start_time = time.time()

# --- CẤU HÌNH TỐI ƯU ---
SEQUENCE_LENGTH = 180
N_FEATURES = 4
BATCH_SIZE = 4096  # Batch size lớn cho RTX 4060
DATA_FILE = 'normal_data_robust_gpu.csv'
EPOCHS = 50
print(f"GPU: Batch size={BATCH_SIZE} | Sequence={SEQUENCE_LENGTH} | Epochs={EPOCHS}")

# --- ĐẦU RA ---
SCALER_PATH = 'universal_scaler_gpu.gz'
AUTOENCODER_TFLITE = 'geotech_autoencoder_rtx4060.tflite'
FORECASTER_TFLITE = 'geotech_forecaster_rtx4060.tflite'
MODEL_SUMMARY = 'model_summary.txt'

# 1. Tải và chuẩn bị dữ liệu
print(f"GPU: Đang tải dữ liệu từ '{DATA_FILE}'...")
df = pd.read_csv(DATA_FILE)
features = df[['lat', 'lon', 'height', 'water_level']].values
total_samples = len(features)
print(f"GPU: Đã tải {total_samples:,} mẫu dữ liệu")

# 2. Chuẩn hóa dữ liệu
print("GPU: Đang chuẩn hóa dữ liệu...")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(features)
joblib.dump(scaler, SCALER_PATH)
print(f"GPU: Đã lưu scaler vào '{SCALER_PATH}'")

# 3. Tạo dataset tối ưu cho GPU
print("GPU: Tạo dataset pipeline cho GPU...")
def create_sequences(data, seq_length):
    sequences = []
    for i in range(0, len(data) - seq_length, 5):  # Bước 5 mẫu
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# Tạo dataset
dataset = create_sequences(data_scaled, SEQUENCE_LENGTH)
print(f"GPU: Đã tạo {len(dataset):,} chuỗi dữ liệu")

# Chia dataset
def create_tf_dataset(sequences):
    # Autoencoder: input = output
    ds_ae = tf.data.Dataset.from_tensor_slices((sequences, sequences))
    
    # Forecaster: input = [0:-1], output = [-1]
    X_fc = sequences[:, :-1, :]
    y_fc = sequences[:, -1, :]
    ds_fc = tf.data.Dataset.from_tensor_slices((X_fc, y_fc))
    
    return ds_ae, ds_fc

ds_ae, ds_fc = create_tf_dataset(dataset)

# Tối ưu hóa pipeline
def configure_ds(ds, batch_size):
    return ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ae = configure_ds(ds_ae, BATCH_SIZE)
train_fc = configure_ds(ds_fc, BATCH_SIZE)

# 4. Xây dựng mô hình Autoencoder
print("\nGPU: Xây dựng Autoencoder...")
inputs = Input(shape=(SEQUENCE_LENGTH, N_FEATURES))
x = LSTM(256, return_sequences=True)(inputs)
x = LSTM(128)(x)
x = RepeatVector(SEQUENCE_LENGTH)(x)
x = LSTM(128, return_sequences=True)(x)
outputs = LSTM(N_FEATURES, return_sequences=True, dtype='float32')(x)
autoencoder = Model(inputs, outputs)

autoencoder.compile(optimizer='adam', loss=tf.keras.losses.Huber())
print("GPU: Kiến trúc Autoencoder:")
autoencoder.summary()

# 5. Huấn luyện Autoencoder
print("\nGPU: Bắt đầu huấn luyện Autoencoder...")
ae_history = autoencoder.fit(
    train_ae,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]
)

# 6. Xây dựng mô hình Forecaster
print("\nGPU: Xây dựng Forecaster...")
fc_input = Input(shape=(SEQUENCE_LENGTH-1, N_FEATURES))
x = LSTM(256, return_sequences=True)(fc_input)
x = Dropout(0.3)(x)
x = LSTM(128)(x)
fc_output = Dense(N_FEATURES, dtype='float32')(x)
forecaster = Model(fc_input, fc_output)

forecaster.compile(optimizer='adam', loss='mse')
print("GPU: Kiến trúc Forecaster:")
forecaster.summary()

# 7. Huấn luyện Forecaster
print("\nGPU: Bắt đầu huấn luyện Forecaster...")
fc_history = forecaster.fit(
    train_fc,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]
)

# 8. Xuất mô hình dạng TFLite
print("\nGPU: Xuất mô hình sang TFLite...")
def export_tflite(model, filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Tối ưu cho GPU
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    tflite_model = converter.convert()
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"GPU: Đã xuất {filename} ({len(tflite_model)/1024:.1f} KB)")

export_tflite(autoencoder, AUTOENCODER_TFLITE)
export_tflite(forecaster, FORECASTER_TFLITE)

# 9. Lưu thông tin mô hình
with open(MODEL_SUMMARY, 'w') as f:
    f.write("=== Autoencoder Summary ===\n")
    autoencoder.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\n=== Forecaster Summary ===\n")
    forecaster.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write(f"\nTraining Time: {(time.time()-start_time)/60:.1f} minutes")
    f.write(f"\nGPU: {gpus[0].name if gpus else 'None'}")

# Kết thúc
total_time = (time.time() - start_time) / 60
print(f"\nGPU: HUẤN LUYỆN HOÀN TẤT TRONG {total_time:.1f} PHÚT")
print(f"GPU: Autoencoder: {AUTOENCODER_TFLITE}")
print(f"GPU: Forecaster: {FORECASTER_TFLITE}")
print(f"GPU: Model summary: {MODEL_SUMMARY}")

# Hiển thị thông tin GPU cuối cùng
if gpus:
    from tensorflow.python.client import device_lib
    print("\nThông tin GPU sử dụng:")
    print(device_lib.list_local_devices()[-1])