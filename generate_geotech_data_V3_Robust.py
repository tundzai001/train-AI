# generate_geotech_data_V4_Parallel.py
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count # (MỚI) Thêm thư viện multiprocessing

print("Bắt đầu tạo bộ dữ liệu huấn luyện THỬ THÁCH TỔNG HỢP (Song song)...")

# --- CẤU HÌNH ---
N_SCENARIOS = 30
SAMPLES_PER_SCENARIO = 86400 * 2
BASE_LAT, BASE_LON, BASE_H = 21.0739, 105.7770, 25.0
BASE_WATER_LEVEL = -10.0

# (MỚI) Tách logic tạo một kịch bản ra một hàm riêng
def generate_scenario(scenario_index):
    """Hàm này chứa logic để tạo MỘT kịch bản độc lập."""
    print(f" -> Bắt đầu tạo Kịch bản Thử thách #{scenario_index + 1}/{N_SCENARIOS}...")
    
    # Thiết lập seed ngẫu nhiên khác nhau cho mỗi tiến trình
    np.random.seed() 
    
    time_vector = np.arange(SAMPLES_PER_SCENARIO)
    
    # 1. Mô phỏng Nước ngầm và Dịch chuyển Vật lý
    rise_duration = int(SAMPLES_PER_SCENARIO * np.random.uniform(0.5, 0.8))
    stable_duration = SAMPLES_PER_SCENARIO - rise_duration
    water_rise = (1 / (1 + np.exp(- (np.linspace(-6, 6, rise_duration))))) * np.random.uniform(7, 9)
    water_stable = np.full(stable_duration, water_rise[-1]) + np.random.normal(0, 0.05, stable_duration)
    water_levels = BASE_WATER_LEVEL + np.concatenate([water_rise, water_stable])
    
    consolidation_strain = (water_levels - BASE_WATER_LEVEL) * np.random.uniform(-0.005, -0.01)
    water_pressure_factor = np.maximum(0, water_levels - (-3.0))
    creep_velocity = water_pressure_factor * np.random.uniform(1e-9, 5e-9)
    creep_displacement = np.cumsum(creep_velocity)
    
    hs = BASE_H + consolidation_strain + creep_displacement
    lat_creep = np.cumsum(creep_velocity * np.random.uniform(0.1, 0.3)) / 111111
    lon_creep = np.cumsum(creep_velocity * np.random.uniform(0.1, 0.3)) / 111111
    lats = BASE_LAT + lat_creep
    lons = BASE_LON + lon_creep
    
    # 2. THÊM CÁC THỬ THÁCH PHỨC TẠP
    for _ in range(np.random.randint(5, 15)):
        start = np.random.randint(0, SAMPLES_PER_SCENARIO - 300)
        duration = np.random.randint(60, 300)
        t = np.arange(duration)
        seismic_noise = (np.sin(t * 0.5) + np.sin(t * 1.5)) * np.exp(-t/100) * 0.008
        hs[start:start+duration] += seismic_noise

    thermal_effect = -np.cos(2 * np.pi * time_vector / 86400) * 0.003
    hs += thermal_effect
    
    for _ in range(np.random.randint(1, 4)):
        idx = np.random.randint(0, SAMPLES_PER_SCENARIO)
        hs[idx] += np.random.uniform(-0.1, 0.1)
        start_kẹt = np.random.randint(0, SAMPLES_PER_SCENARIO - 600)
        duration_kẹt = np.random.randint(300, 600)
        water_levels[start_kẹt:start_kẹt+duration_kẹt] = water_levels[start_kẹt]

    lats += np.random.normal(0, 0.00000002, SAMPLES_PER_SCENARIO)
    lons += np.random.normal(0, 0.00000002, SAMPLES_PER_SCENARIO)
    hs += np.random.normal(0, 0.005, SAMPLES_PER_SCENARIO)

    df = pd.DataFrame({'lat': lats, 'lon': lons, 'h': hs, 'water_level': water_levels})
    print(f" -> Hoàn thành Kịch bản #{scenario_index + 1}.")
    return df

# (MỚI) Bọc code chính trong 'if __name__ == "__main__":' để multiprocessing hoạt động an toàn
if __name__ == '__main__':
    start_time = time.time()

    num_processes = cpu_count()
    print(f"Sử dụng {num_processes} lõi CPU để tạo dữ liệu song song.")
    
    # (THAY ĐỔI) Sử dụng Pool để chạy hàm generate_scenario song song
    with Pool(processes=num_processes) as pool:
        all_dfs = pool.map(generate_scenario, range(N_SCENARIOS))

    print("Đang kết hợp các kịch bản và lưu file...")
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df['timestamp'] = np.arange(len(final_df)) + 1753872000
    final_df.rename(columns={'h': 'height'}, inplace=True)
    final_df = final_df[['timestamp', 'lat', 'lon', 'height', 'water_level']]

    final_df.to_csv('normal_data_robust.csv', index=False)
    
    end_time = time.time()
    total_samples = len(final_df)
    
    print(f"\nHoàn tất! Đã tạo 'normal_data_robust.csv' với {total_samples} dòng trong {end_time - start_time:.2f} giây.")
    print("File này chứa các kịch bản phức tạp để huấn luyện một AI vững vàng.")
