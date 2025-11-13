import os
from datetime import datetime

# Dataset Configuration
DATASET_CONFIG = {
    'kaggle_dataset': 'madhavmalhotra/unb-cic-iot-dataset',
    'dataset_filename': 'NF-UQ-NIDS-v2.csv',
    'raw_data_path': 'data/raw/',
    'processed_data_path': 'data/processed/'
}

# Feature Selection - 20 temel özellik
SELECTED_FEATURES = [
    # TCP Bayrakları (6)
    'syn_count', 'ack_count', 'rst_count', 'fin_count', 'psh_count', 'urg_count',

    # Trafik Hızı (3)
    'Rate', 'Srate', 'Drate',

    # Zaman Metrikleri (3)
    'flow_duration', 'Duration', 'IAT',

    # Protokol Bilgisi (3)
    'TCP', 'UDP', 'ICMP',

    # İstatistikler (5)
    'Number', 'Tot_sum', 'AVG', 'Std', 'Weight'
]

# Target Column
TARGET_COLUMN = 'Attack'

# Model Configuration
MODEL_CONFIG = {
    'model_type': 'RandomForestClassifier',
    'n_estimators': 125,
    'max_depth': 18,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Data Splitting Configuration
DATA_SPLIT_CONFIG = {
    'test_size': 0.3,  # 30% test, 70% train
    'stratify': True,  # Stratified sampling
    'random_state': 42
}

# Output Configuration
OUTPUT_CONFIG = {
    'results_path': 'results/',
    'models_path': 'models/',
    'sample_size': 1000,  # İlk 1000 örnek için detaylı çıktı
    'save_misclassified': True,  # Yanlış tahminleri kaydet
    'save_feature_importance': True,
    'save_confusion_matrix': True
}

# File Paths Configuration
FILE_PATHS = {
    'raw_dataset': os.path.join(DATASET_CONFIG['raw_data_path'], DATASET_CONFIG['dataset_filename']),
    'processed_data': os.path.join(DATASET_CONFIG['processed_data_path'], 'processed_data.csv'),
    'model_file': os.path.join(OUTPUT_CONFIG['models_path'], 'multiclass_ddos_model.pkl'),
    'scaler_file': os.path.join(OUTPUT_CONFIG['models_path'], 'scaler.pkl'),
    'encoder_file': os.path.join(OUTPUT_CONFIG['models_path'], 'label_encoder.pkl'),
    'performance_summary': os.path.join(OUTPUT_CONFIG['results_path'], 'performance_summary.json'),
    'sample_results': os.path.join(OUTPUT_CONFIG['results_path'], 'sample_results.json'),
    'test_data': os.path.join(OUTPUT_CONFIG['results_path'], 'test_data.json'),
    'confusion_matrix': os.path.join(OUTPUT_CONFIG['results_path'], 'confusion_matrix.json')
}

# Runtime Information
RUNTIME_CONFIG = {
    'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'python_version': '3.8+',
    'created_by': 'DDoS Detection System'
}

# Class Labels (belirlenecek)
CLASS_LABELS = {
    0: 'Benign',
    1: 'DDoS-UDP_Flood',
    2: 'DDoS-SYN_Flood',
    3: 'DDoS-ICMP_Flood',
    4: 'DDoS-HTTP_Flood',
    5: 'DDoS-TCP_Flood',
    6: 'DDoS-Slowloris',
    7: 'DDoS-Other'
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'ddos_detection.log'
}