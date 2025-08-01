import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import deque

import joblib
import numpy as np
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite

# --- Vô hiệu hóa log không cần thiết của TensorFlow ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==============================================================================
# CẤU HÌNH LOGGING
# ==============================================================================
log_formatter = logging.Formatter('%(asctime)s - SHIFTING - %(levelname)s - %(message)s')
logger = logging.getLogger("ShiftingLogger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("shifting_activity.log", mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# ==============================================================================
# CÁC HẰNG SỐ VÀ CẤU HÌNH
# ==============================================================================
class Config:
    GNSS_BUFFER_SIZE = 300
    WATER_BUFFER_SIZE = 60
    SEQUENCE_LENGTH = 180
    THRESHOLD_BUFFER_SIZE = 10800
    THRESHOLD_PERCENTILE = 99.9
    MIN_SAMPLES_FOR_ADAPTATION = 600
    SCALER_PATH = 'universal_scaler.gz'
    AUTOENCODER_TFLITE_PATH = 'foundation_model_autoencoder.tflite'
    FORECASTER_TFLITE_PATH = 'foundation_model_forecaster.tflite'
    REPORTING_INTERVAL_SAFE = 60.0
    REPORTING_INTERVAL_MONITOR = 5.0
    REPORTING_INTERVAL_WARNING = 1.0

# ==============================================================================
# CÁC BIẾN TRẠNG THÁI TOÀN CỤC
# ==============================================================================
data_queue = queue.Queue()
gnss_buffer = deque(maxlen=Config.GNSS_BUFFER_SIZE)
water_buffer = deque(maxlen=Config.WATER_BUFFER_SIZE)
base_gnss_point = None
current_system_status = "INITIALIZING"
time_to_next_full_report = 0.0
VELOCITY_CLASSIFICATION_TABLE = [
    {"speed_class": 7, "name": "Extremely Rapid", "threshold_mmps": 5000.0},
    {"speed_class": 6, "name": "Very Rapid", "threshold_mmps": 50.0},
    {"speed_class": 5, "name": "Rapid", "threshold_mmps": 0.5},
    {"speed_class": 4, "name": "Moderate", "threshold_mmps": 0.016},
    {"speed_class": 3, "name": "Slow", "threshold_mmps": 0.0005},
    {"speed_class": 2, "name": "Very Slow", "threshold_mmps": 0.000005},
    {"speed_class": 1, "name": "Extremely Slow", "threshold_mmps": 0.0}
]

# ==============================================================================
# CÁC HÀM TIỆN ÍCH VÀ PARSER
# ==============================================================================
def convert_nmea_to_decimal(nmea_coord, direction):
    try:
        degrees = int(float(nmea_coord) / 100)
        minutes = float(nmea_coord) % 100
        decimal_degrees = degrees + (minutes / 60.0)
        if direction in ['S', 'W']: decimal_degrees = -decimal_degrees
        return decimal_degrees
    except (ValueError, TypeError): return None

def parse_gnss(payload_str):
    ACCEPTED_FIX_QUALITIES = ['4', '5']
    try:
        gngga_part = next((p for p in payload_str.strip().split('$') if p.startswith('GNGGA')), None)
        if not gngga_part: return None
        fields = gngga_part.split(',')
        if len(fields) < 15: return None
        if fields[6] in ACCEPTED_FIX_QUALITIES:
            lat = convert_nmea_to_decimal(fields[2], fields[3])
            lon = convert_nmea_to_decimal(fields[4], fields[5])
            h = float(fields[9]) - float(fields[11])
            if lat is not None and lon is not None:
                return (time.time(), lat, lon, h)
        else: return None
    except (ValueError, IndexError): return None

def parse_water(payload_dict):
    try:
        if "value" in payload_dict and "timestamp" in payload_dict:
            return (float(payload_dict["timestamp"]), float(payload_dict["value"]))
    except (ValueError, TypeError): return None

def haversine_3d(p1, p2):
    R = 6371000
    lat1, lon1, h1 = p1; lat2, lon2, h2 = p2
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2_rad - lat1_rad, lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    dist_2d = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return np.sqrt(dist_2d**2 + (h2-h1)**2)

def classify_velocity(velocity_mmps):
    global VELOCITY_CLASSIFICATION_TABLE
    for classification in VELOCITY_CLASSIFICATION_TABLE:
        if velocity_mmps >= classification["threshold_mmps"]:
            return classification
    return {"speed_class": 0, "name": "Undefined"}

# ==============================================================================
# LỚP AI PROFILER
# ==============================================================================
class AdaptiveAIProfiler:
    def __init__(self, scaler_path, autoencoder_path, forecaster_path):
        self.is_ready = False
        self.recon_error_buffer = deque(maxlen=Config.THRESHOLD_BUFFER_SIZE)
        self.pred_error_buffer = deque(maxlen=Config.THRESHOLD_BUFFER_SIZE)
        self.current_recon_threshold = 1.0
        self.current_pred_threshold = 1.0
        try:
            self.scaler = joblib.load(scaler_path)
            self.autoencoder = tflite.Interpreter(model_path=autoencoder_path); self.autoencoder.allocate_tensors()
            self.ae_input_details = self.autoencoder.get_input_details(); self.ae_output_details = self.autoencoder.get_output_details()
            self.forecaster = tflite.Interpreter(model_path=forecaster_path); self.forecaster.allocate_tensors()
            self.fc_input_details = self.forecaster.get_input_details(); self.fc_output_details = self.forecaster.get_output_details()
            self.is_ready = True
        except Exception as e:
            logger.critical(f"Lỗi tải mô hình nền tảng: {e}"); self.is_ready = False
            
    def update_thresholds(self):
        if len(self.recon_error_buffer) >= Config.MIN_SAMPLES_FOR_ADAPTATION:
            self.current_recon_threshold = np.percentile(self.recon_error_buffer, Config.THRESHOLD_PERCENTILE)
            self.current_pred_threshold = np.percentile(self.pred_error_buffer, Config.THRESHOLD_PERCENTILE)
            
    def predict(self, feature_matrix):
        if not self.is_ready or feature_matrix.shape[0] < Config.SEQUENCE_LENGTH: return {"reconstruction_error": 0.0, "prediction_error": 0.0, "is_anomaly": False}
        scaled_matrix = self.scaler.transform(feature_matrix)
        ae_input = np.expand_dims(scaled_matrix, axis=0).astype(np.float32)
        self.autoencoder.set_tensor(self.ae_input_details[0]['index'], ae_input); self.autoencoder.invoke()
        ae_output = self.autoencoder.get_tensor(self.ae_output_details[0]['index'])
        recon_error = float(np.mean(np.abs(ae_input - ae_output)))
        fc_input = np.expand_dims(scaled_matrix[:-1, :], axis=0).astype(np.float32)
        self.forecaster.set_tensor(self.fc_input_details[0]['index'], fc_input); self.forecaster.invoke()
        fc_output = self.forecaster.get_tensor(self.fc_output_details[0]['index'])
        pred_error = float(np.mean(np.abs(scaled_matrix[-1, :] - fc_output)))
        self.recon_error_buffer.append(recon_error); self.pred_error_buffer.append(pred_error); self.update_thresholds()
        is_anomaly = (recon_error > self.current_recon_threshold) or (pred_error > self.current_pred_threshold)
        return {"reconstruction_error": recon_error, "prediction_error": pred_error, "is_anomaly": is_anomaly}

# ==============================================================================
# CÁC TẦNG PHÂN TÍCH
# ==============================================================================
def interpolate_water_level(current_ts):
    if len(water_buffer) < 2: return 0.0, 0.0
    p1_ts, p1_val = water_buffer[-2]; p2_ts, p2_val = water_buffer[-1]
    if p2_ts <= p1_ts: return p2_val, 0.0
    interp_water = p1_val + (p2_val - p1_val) * ((current_ts - p1_ts) / (p2_ts - p1_ts))
    roc_m_per_s = (p2_val - p1_val) / (p2_ts - p1_ts)
    return interp_water, roc_m_per_s * 1000 * 60

def run_level_1_feature_extraction(current_ts):
    interp_water, water_roc = interpolate_water_level(current_ts)
    displacement, velocity = 0.0, 0.0
    if len(gnss_buffer) >= 2 and base_gnss_point is not None:
        current_point, prev_point = gnss_buffer[-1], gnss_buffer[-2]
        displacement = haversine_3d(current_point[1:], base_gnss_point[1:]) * 1000
        dt = current_point[0] - prev_point[0]
        velocity = (haversine_3d(current_point[1:], prev_point[1:]) / dt) * 1000 if dt > 0 else 0.0
    velocity_classification = classify_velocity(velocity)
    return {"interp_water_m": interp_water, "water_roc_mm_per_min": water_roc, "displacement_mm": displacement, "velocity_mmps": velocity, "velocity_class_info": velocity_classification}

def run_level_2_contextual_analysis(features, args):
    advisor_signals = {"water_level_is_warning": features["interp_water_m"] >= args.water_warn_threshold, "water_level_is_critical": features["interp_water_m"] >= args.water_crit_threshold}
    sentry_threat, sentry_reason = 0, "SYSTEM_STABLE"
    if advisor_signals["water_level_is_critical"]: sentry_threat, sentry_reason = 3, "WATER_LEVEL_CRITICAL"
    elif advisor_signals["water_level_is_warning"]: sentry_threat, sentry_reason = 2, "WATER_LEVEL_WARNING"
    elif features["water_roc_mm_per_min"] > 10.0: sentry_threat, sentry_reason = 1, "WATER_RISING_FAST"
    return {"advisor_signals": advisor_signals, "sentry_threat": sentry_threat, "sentry_reason": sentry_reason}

def run_level_3_ai_profiler(ai_profiler):
    if len(gnss_buffer) < Config.SEQUENCE_LENGTH or len(water_buffer) < 2: return {"is_anomaly": False}
    feature_matrix = []
    last_seq_gnss = list(gnss_buffer)[-Config.SEQUENCE_LENGTH:]
    for gnss_entry in last_seq_gnss:
        interp_water, _ = interpolate_water_level(gnss_entry[0])
        feature_matrix.append([gnss_entry[1], gnss_entry[2], gnss_entry[3], interp_water])
    return ai_profiler.predict(np.array(feature_matrix))

def run_level_4_final_assessment(features, context, ai_results, ai_profiler):
    threat_score = 0.0
    velocity_class_info = features.get("velocity_class_info", {}); speed_class = velocity_class_info.get("speed_class", 0)
    ai_is_anomaly = ai_results.get("is_anomaly", False)
    ai_confidence_multiplier = 1.0 + (speed_class / 7.0)**2
    if ai_is_anomaly:
        ai_raw_score = ai_results.get("reconstruction_error", 0) + ai_results.get("prediction_error", 0)
        threat_score += (80 + ai_raw_score * 300) * ai_confidence_multiplier
    threat_score += context["sentry_threat"] * 50
    status_code, velocity_name = "SAFE", velocity_class_info.get("name", "N/A")
    summary_text = f"Hệ thống ổn định. Phân loại vận tốc: {velocity_name}."
    if threat_score >= 150: status_code, summary_text = "CRITICAL", f"NGUY HIỂM KHẨN CẤP! Vận tốc ở mức '{velocity_name}' và/hoặc AI phát hiện bất thường nghiêm trọng."
    elif threat_score >= 80: status_code, summary_text = "WARNING", f"CẢNH BÁO! Vận tốc '{velocity_name}'. " + ("AI phát hiện sai lệch." if ai_is_anomaly else "")
    elif threat_score >= 25: status_code, summary_text = "MONITOR", f"THEO DÕI. Vận tốc '{velocity_name}'. Ghi nhận các thay đổi nhỏ."
    if len(ai_profiler.recon_error_buffer) < Config.MIN_SAMPLES_FOR_ADAPTATION:
        status_code, summary_text = "ADAPTING", f"AI đang thích ứng ({len(ai_profiler.recon_error_buffer)}/{Config.MIN_SAMPLES_FOR_ADAPTATION} mẫu). Giám sát bằng ngưỡng."
        if context["sentry_threat"] >= 2: status_code, summary_text = "WARNING", "CẢNH BÁO NGƯỠNG! Mực nước vượt ngưỡng trong giai đoạn AI đang thích ứng."
    return {"overall_threat_score": threat_score, "status_code": status_code, "summary_text": summary_text}

# ==============================================================================
# LUỒNG CHÍNH VÀ CÁC LUỒNG PHỤ
# ==============================================================================
def stdin_reader_thread():
    global VELOCITY_CLASSIFICATION_TABLE
    for line in sys.stdin:
        try:
            packet = json.loads(line)
            if packet.get("type") == "CONTROL_COMMAND" and packet.get("command") == "UPDATE_VELOCITY_CLASSIFICATION":
                new_table = packet.get("payload"); VELOCITY_CLASSIFICATION_TABLE = sorted(new_table, key=lambda x: x['threshold_mmps'], reverse=True)
                logger.info("Đã nhận và cập nhật động bảng phân loại vận tốc.")
        except: pass

def processing_loop(args, ai_profiler, client):
    global base_gnss_point, current_system_status, time_to_next_full_report
    status_translations = {"SAFE": "An toàn", "MONITOR": "Cần theo dõi", "WARNING": "Cảnh báo", "CRITICAL": "Nguy hiểm khẩn cấp", "ADAPTING": "Đang hiệu chỉnh"}
    while True:
        try:
            packet = data_queue.get(timeout=1.0)
            topic, payload = packet['topic'], packet['payload']
            if topic in args.gnss_topic:
                gnss_data = parse_gnss(payload)
                if gnss_data:
                    gnss_buffer.append(gnss_data)
                    if base_gnss_point is None: base_gnss_point = gnss_buffer[0]
            elif topic in args.water_topic:
                water_data = parse_water(json.loads(payload))
                if water_data: water_buffer.append(water_data)
            if not (len(gnss_buffer) >= Config.SEQUENCE_LENGTH and len(water_buffer) >= 2): continue
            
            current_ts = gnss_buffer[-1][0]
            features = run_level_1_feature_extraction(current_ts)
            context = run_level_2_contextual_analysis(features, args)
            ai_results = run_level_3_ai_profiler(ai_profiler)
            final_assessment = run_level_4_final_assessment(features, context, ai_results, ai_profiler)
            
            time_to_next_full_report -= 1
            new_status = final_assessment['status_code']
            send_full_report_now = (new_status != current_system_status) or (time_to_next_full_report <= 0)
            
            tts_message = None
            if new_status != current_system_status:
                logger.info(f"Hệ thống chuyển trạng thái: {current_system_status} -> {new_status}")
                if new_status in status_translations:
                    tts_message = f"Chú ý. Hệ thống chuyển sang trạng thái {status_translations[new_status]}."
                current_system_status = new_status
            
            full_report = {"timestamp": current_ts, "type": "ASTCS_ANALYSIS_REPORT", "primary_state": features, "contextual_analysis": context, "ai_analysis": ai_results, "final_assessment": final_assessment}
            if tts_message:
                full_report["tts_message"] = tts_message

            if send_full_report_now:
                print(json.dumps(full_report)); sys.stdout.flush()
                if args.publish_topic:
                    client.publish(args.publish_topic, json.dumps(full_report))
                if current_system_status in ["SAFE", "INITIALIZING"]: time_to_next_full_report = Config.REPORTING_INTERVAL_SAFE
                elif current_system_status in ["MONITOR", "ADAPTING"]: time_to_next_full_report = Config.REPORTING_INTERVAL_MONITOR
                else: time_to_next_full_report = Config.REPORTING_INTERVAL_WARNING
            else:
                heartbeat = {"type": "HEARTBEAT", "timestamp": current_ts, "status_code": new_status, "displacement_3d_mm": features["displacement_mm"], "water_level_m": features["interp_water_m"]}
                print(json.dumps(heartbeat)); sys.stdout.flush()
        except queue.Empty:
            time_to_next_full_report -= 1
            continue
        except Exception as e:
            logger.error(f"Lỗi không mong muốn: {e}\n{traceback.format_exc()}"); time.sleep(5)

# ==============================================================================
# KHỞI ĐỘNG
# ==============================================================================
def on_message_factory(args):
    def on_message(client, userdata, msg): data_queue.put({"topic": msg.topic, "payload": msg.payload.decode('utf-8')})
    return on_message
def on_connect_factory(args):
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            logger.info("MQTT Connected.")
            all_topics = (args.gnss_topic or []) + (args.water_topic or [])
            for topic in all_topics: client.subscribe(topic); logger.info(f"Subscribed to: {topic}")
        else: logger.error(f"MQTT Connection Failed with code {rc}")
    return on_connect
def main(args):
    logger.info("IAFS Shifting.py (Semantic, Interactive, Publishing) is starting...")
    ai_profiler = AdaptiveAIProfiler(scaler_path=Config.SCALER_PATH, autoencoder_path=Config.AUTOENCODER_TFLITE_PATH, forecaster_path=Config.FORECASTER_TFLITE_PATH)
    if not ai_profiler.is_ready: sys.exit(1)

    stdin_thread = threading.Thread(target=stdin_reader_thread, daemon=True); stdin_thread.start()
    logger.info("Luồng lắng nghe lệnh từ Bộ chỉ huy đã được kích hoạt.")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if args.username: client.username_pw_set(args.username, args.password)
    client.on_connect = on_connect_factory(args); client.on_message = on_message_factory(args)
    
    processing_thread = threading.Thread(target=processing_loop, args=(args, ai_profiler, client), daemon=True)
    processing_thread.start()
    try:
        client.connect(args.broker, args.port, 60); client.loop_forever()
    except Exception as e:
        logger.critical(f"MQTT Main Loop Error: {e}"); sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAFS - Shifting Intelligence Core")
    parser.add_argument('--broker', required=True); parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--username'); parser.add_argument('--password')
    parser.add_argument('--gnss-topic', action='append'); parser.add_argument('--water-topic', action='append')
    parser.add_argument('--publish-topic', help='Topic để publish kết quả phân tích.')
    parser.add_argument('--water-warn-threshold', type=float, required=True)
    parser.add_argument('--water-crit-threshold', type=float, required=True)
    parser.add_argument('--pid-file', required=True)
    args = parser.parse_args()
    pid_file_path = args.pid_file
    try:
        with open(pid_file_path, 'w') as f: f.write(str(os.getpid()))
        main(args)
    finally:
        if os.path.exists(pid_file_path): os.remove(pid_file_path)