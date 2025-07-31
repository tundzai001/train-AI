import numpy as np
import pandas as pd
import time
import cupy as cp
import os

# Cấu hình GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Chỉ sử dụng RTX 4060

print("Bắt đầu tạo bộ dữ liệu huấn luyện THỬ THÁCH TỔNG HỢP bằng GPU...")

# --- CẤU HÌNH TỐI ƯU CHO RTX 4060 ---
N_SCENARIOS = 30
SAMPLES_PER_SCENARIO = 86400 * 2  # 2 ngày (172,800 mẫu/kịch bản)
BASE_LAT, BASE_LON, BASE_H = 21.0739, 105.7770, 25.0
BASE_WATER_LEVEL = -10.0
print(f"GPU: RTX 4060 8GB | Số kịch bản: {N_SCENARIOS} | Mẫu/kịch bản: {SAMPLES_PER_SCENARIO}")

def generate_scenario_gpu(i):
    start_time = time.time()
    print(f" -> GPU: Đang tạo Kịch bản #{i+1}/{N_SCENARIOS}...")
    
    # 1. Tạo vector thời gian trên GPU
    time_vector = cp.arange(SAMPLES_PER_SCENARIO)
    
    # 2. Mô phỏng Nước ngầm tối ưu hóa
    rise_duration = int(SAMPLES_PER_SCENARIO * cp.random.uniform(0.5, 0.8))
    stable_duration = SAMPLES_PER_SCENARIO - rise_duration
    
    # Tính toán vector hóa hoàn toàn
    t = cp.linspace(-6, 6, rise_duration)
    water_rise = (1 / (1 + cp.exp(-t))) * cp.random.uniform(7, 9)
    water_stable = cp.full(stable_duration, water_rise[-1]) + cp.random.normal(0, 0.05, stable_duration)
    water_levels = BASE_WATER_LEVEL + cp.concatenate([water_rise, water_stable])
    
    # 3. Tính toán biến dạng - Tận dụng tensor cores
    consolidation_strain = (water_levels - BASE_WATER_LEVEL) * cp.random.uniform(-0.005, -0.01)
    water_pressure_factor = cp.maximum(0, water_levels - (-3.0))
    creep_velocity = water_pressure_factor * cp.random.uniform(1e-9, 5e-9)
    creep_displacement = cp.cumsum(creep_velocity)
    
    # 4. Tính toán tọa độ - Tối ưu bộ nhớ
    lat_factor = cp.random.uniform(0.1, 0.3)
    lon_factor = cp.random.uniform(0.1, 0.3)
    lat_creep = cp.cumsum(creep_velocity * lat_factor) / 111111
    lon_creep = cp.cumsum(creep_velocity * lon_factor) / 111111
    
    hs = BASE_H + consolidation_strain + creep_displacement
    lats = BASE_LAT + lat_creep
    lons = BASE_LON + lon_creep
    
    # 5. THÊM CÁC THỬ THÁCH PHỨC TẠP
    # Thử thách A: Nhiễu địa chấn (tối ưu hóa)
    n_events = cp.random.randint(5, 15)
    event_starts = cp.random.randint(0, SAMPLES_PER_SCENARIO - 300, n_events)
    event_durations = cp.random.randint(60, 300, n_events)
    
    for start, duration in zip(event_starts.get(), event_durations.get()):
        t = cp.arange(duration)
        seismic_noise = (cp.sin(t * 0.5) + cp.sin(t * 1.5)) * cp.exp(-t/100) * 0.008
        hs[start:start+duration] += seismic_noise

    # Thử thách B: Hiệu ứng nhiệt độ (vector hóa)
    thermal_effect = -cp.cos(2 * cp.pi * time_vector / 86400) * 0.003
    hs += thermal_effect
    
    # Thử thách C: Lỗi cảm biến (tối ưu)
    n_glitches = cp.random.randint(1, 4)
    glitch_indices = cp.random.randint(0, SAMPLES_PER_SCENARIO, n_glitches)
    stuck_starts = cp.random.randint(0, SAMPLES_PER_SCENARIO - 600, n_glitches)
    stuck_durations = cp.random.randint(300, 600, n_glitches)
    
    for idx in glitch_indices.get():
        hs[idx] += cp.random.uniform(-0.1, 0.1).get()
    
    for start, duration in zip(stuck_starts.get(), stuck_durations.get()):
        water_levels[start:start+duration] = water_levels[start]

    # Nhiễu RTK chính xác cao
    lats += cp.random.normal(0, 0.00000002, SAMPLES_PER_SCENARIO)
    lons += cp.random.normal(0, 0.00000002, SAMPLES_PER_SCENARIO)
    hs += cp.random.normal(0, 0.005, SAMPLES_PER_SCENARIO)

    # Chuyển về CPU cho pandas
    return (
        cp.asnumpy(lats), 
        cp.asnumpy(lons), 
        cp.asnumpy(hs), 
        cp.asnumpy(water_levels),
        time.time() - start_time
    )

# --- SINH DỮ LIỆU BẰNG GPU ---
print(f"GPU: Bắt đầu sinh {N_SCENARIOS} kịch bản trên RTX 4060...")
start_total = time.time()
all_data = []

for i in range(N_SCENARIOS):
    lats, lons, hs, water_levels, duration = generate_scenario_gpu(i)
    df = pd.DataFrame({
        'lat': lats, 
        'lon': lons, 
        'height': hs, 
        'water_level': water_levels
    })
    all_data.append(df)
    print(f"  -> GPU: Hoàn thành kịch bản #{i+1} trong {duration:.2f} giây")

# Kết hợp dữ liệu
total_samples = sum(len(df) for df in all_data)
print("GPU: Đang kết hợp dữ liệu...")
final_df = pd.concat(all_data, ignore_index=True)
final_df['timestamp'] = np.arange(len(final_df)) + 1753872000
final_df = final_df[['timestamp', 'lat', 'lon', 'height', 'water_level']]

# Lưu file
output_file = 'normal_data_robust_gpu.csv'
final_df.to_csv(output_file, index=False)
total_time = time.time() - start_total
print(f"GPU: Hoàn tất! Tạo {total_samples:,} mẫu trong {total_time/60:.2f} phút")
print(f"GPU: Tốc độ: {total_samples/total_time:,.0f} mẫu/giây")
print(f"GPU: Dữ liệu đã lưu vào '{output_file}'")

# Kiểm tra bộ nhớ GPU
mem = cp.get_default_memory_pool().used_bytes() / (1024**3)
print(f"GPU: Bộ nhớ tối đa sử dụng: {mem:.2f} GB")