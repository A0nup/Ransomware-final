"""
Central Configuration Module
Hybrid Deep Learning Model for Behavioral Ransomware Detection
"""

from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
EDA_DIR = REPORTS_DIR / "eda"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
DATASET_PATH = DATA_DIR / "behavior_sequences.csv"
SAMPLE_BENIGN_PATH = DATA_DIR / "sample_benign.csv"
SAMPLE_RANSOMWARE_PATH = DATA_DIR / "sample_ransomware_like.csv"

MODEL_CHECKPOINT_PATH = MODELS_DIR / "ransomware_hybrid.pt"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.csv"
TRAINING_CURVE_PATH = REPORTS_DIR / "training_curve.png"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.png"
PR_CURVE_PATH = REPORTS_DIR / "precision_recall_curve.png"
THRESHOLD_ANALYSIS_PATH = REPORTS_DIR / "threshold_analysis.csv"
MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
ABLATION_RESULTS_PATH = REPORTS_DIR / "ablation_results.csv"
ABLATION_COMPARISON_PATH = REPORTS_DIR / "ablation_comparison.png"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PLOT_PATH = REPORTS_DIR / "feature_importance.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# Reproducibility
SEED = 42

# Dataset Specifications
N_SEQUENCES = 5000
BENIGN_RATIO = 0.50
RANSOMWARE_RATIO = 0.50
SEQ_LEN = 20
N_FEATURES = 20

# 20 Telemetry Features Required by Schema
FEATURES = [
    "file_write_rate",
    "file_rename_rate",
    "file_delete_rate",
    "entropy_delta",
    "extension_change_rate",
    "process_create_rate",
    "shadow_copy_cmd_count",
    "backup_delete_count",
    "cpu_percent",
    "memory_percent",
    "network_conn_rate",
    "file_open_rate",
    "directory_traversal_rate",
    "suspicious_api_rate",
    "unique_extensions",
    "encrypted_file_ratio",
    "failed_access_rate",
    "admin_action_rate",
    "bytes_written_mb",
    "new_unknown_process_rate",
]

METADATA_COLUMNS = ["sequence_id", "time_step", "label", "host_id"]

# Dataset Splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 6
GRADIENT_CLIP = 2.0

# Architecture Parameters
CNN_FILTERS = 64
CNN_KERNEL_SIZE = 3
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
DENSE_UNITS = 64
DROPOUT_RATE = 0.25

# Evaluation & Thresholds
DEFAULT_THRESHOLD = 0.50
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Risk Level Classification Rules
RISK_THRESHOLD_LOW = 0.30
RISK_THRESHOLD_HIGH = 0.70

# Hardware Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
