# Hybrid Deep Learning Model for Behavioral Ransomware Detection using AI

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, reproducible, defensive cybersecurity research prototype and final-year academic project for detecting ransomware-like temporal behavioral patterns from endpoint telemetry using a hybrid deep-learning architecture (**1D-CNN + BiLSTM + Attention Mechanism**).

---

## 1. Project Overview

Ransomware represents one of the most critical threats to modern computing infrastructure. Attackers continuously alter binary signatures, employ polymorphic packers, and leverage zero-day exploits to bypass traditional signature-based Endpoint Detection and Response (EDR) agents. 

This project demonstrates a proactive behavioral approach: instead of inspecting binary files, the system monitors **numerical endpoint telemetry** across fixed time-step sequences to detect early anomalous behaviors (such as sudden entropy spikes, shadow copy command execution, rapid directory traversal, and mass file renaming) before irreversible damage occurs.

```text
Endpoint Behavioral Telemetry (20 timesteps × 20 features)
                            │
                            ▼
               ┌─────────────────────────┐
               │ 1D CNN Feature Extractor│
               └────────────┬────────────┘
                            ▼
               ┌─────────────────────────┐
               │ Bidirectional LSTM      │
               └────────────┬────────────┘
                            ▼
               ┌─────────────────────────┐
               │ Additive Attention      │
               └────────────┬────────────┘
                            ▼
               ┌─────────────────────────┐
               │ Fully Connected Head    │
               └────────────┬────────────┘
                            ▼
                   BENIGN or RANSOMWARE
```

---

## 2. Problem Statement

Signature matching fails against previously unseen ransomware variants. Machine-learning models trained on static file features are easily bypassed by obfuscation and runtime packing. Ransomware, however, cannot hide its **behavioral footprint** during the encryption kill-chain:
- System enumeration & discovery
- Inhibiting system recovery (`vssadmin delete shadows`)
- Traversing file hierarchies
- High-entropy data rewriting
- Appending encrypted file extensions
- Dropping ransom notes

This project models behavioral endpoint telemetry as a multi-variate time-series classification problem.

---

## 3. Project Objectives

1. Generate a controlled synthetic behavioral dataset simulating benign activity and ransomware escalation.
2. Formulate telemetry as numerical time-window sequences.
3. Preprocess telemetry with strict anti-data-leakage protocols (sequence-level splitting, train-only scaling).
4. Train the proposed **1D-CNN + BiLSTM + Attention** model with early stopping.
5. Train baseline models (Logistic Regression, Random Forest, CNN-only, BiLSTM-only, CNN+BiLSTM).
6. Perform an architectural ablation study (Configurations A, B, C, D).
7. Conduct classification threshold sensitivity analysis ($0.10$ to $0.90$).
8. Calculate standard cybersecurity metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Specificity, FPR, FNR).
9. Extract interpretable attention weights and permutation feature importance.
10. Deliver an interactive multi-page Streamlit web dashboard.
11. Provide automated test suites (`pytest`).
12. Deliver academic report documentation for viva examination.

---

## 4. Proposed Hybrid Architecture

| Stage | Layer / Component | Hyperparameters & Configuration | Purpose |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **1D CNN** | 2 Conv1D layers (64 filters, kernel size 3, same padding), BatchNorm1d, ReLU, Dropout(0.20) | Extracts local spatial and short-term behavioral motifs |
| **Stage 2** | **BiLSTM** | 2 LSTM layers (hidden size 64, bidirectional=True, dropout=0.20) | Learns forward and backward temporal dependencies across 20 windows |
| **Stage 3** | **Attention** | Additive attention layer projecting 128-dim representations to alignment scores | Learns dynamic timestep weights and outputs a 128-dim context vector |
| **Stage 4** | **Classifier** | Linear(128 → 64), ReLU, Dropout(0.25), Linear(64 → 1) | Binary classification head producing unscaled logit |
| **Loss** | **BCEWithLogitsLoss** | Combines Sigmoid activation and Binary Cross Entropy | Numerically stable binary classification loss |

---

## 5. Dataset Architecture

- **Total Sequences:** 5,000 sequences
- **Timesteps per Sequence:** 20 time windows
- **Total Dataset Size:** 100,000 rows × 24 columns
- **Class Balance:** 2,500 Benign (50%), 2,500 Ransomware-like (50%)
- **Metadata Columns:** `sequence_id`, `time_step`, `label`, `host_id`

---

## 6. The 20 Behavioral Telemetry Features

| # | Feature Name | Measurement Unit | Description |
| :---: | :--- | :--- | :--- |
| 1 | `file_write_rate` | Operations / window | Frequency of file modification and write operations |
| 2 | `file_rename_rate` | Operations / window | Frequency of file rename requests |
| 3 | `file_delete_rate` | Operations / window | Frequency of file deletion requests |
| 4 | `entropy_delta` | Shannon entropy $\Delta$ | Change in data randomness of written files |
| 5 | `extension_change_rate` | Operations / window | Rate of files changing to new extensions (e.g. `.locked`) |
| 6 | `process_create_rate` | Processes / window | Number of child processes spawned |
| 7 | `shadow_copy_cmd_count` | Commands / window | Backup inhibition commands (`vssadmin`, `bcdedit`, `wmic`) |
| 8 | `backup_delete_count` | Deletions / window | Detected backup repository deletion attempts |
| 9 | `cpu_percent` | Percentage (0–100%) | Overall CPU utilization |
| 10 | `memory_percent` | Percentage (0–100%) | Overall host memory utilization |
| 11 | `network_conn_rate` | Connections / window | Outbound network connection attempts (C2 beaconing) |
| 12 | `file_open_rate` | Operations / window | Frequency of file handles opened |
| 13 | `directory_traversal_rate` | Traversal events | Rate of directory enumeration operations |
| 14 | `suspicious_api_rate` | Invocations / window | Hooked crypto and evasion API calls (`CryptEncrypt`, etc.) |
| 15 | `unique_extensions` | Count | Number of distinct file extensions accessed |
| 16 | `encrypted_file_ratio` | Ratio (0.0–1.0) | Fraction of accessed files exhibiting encryption characteristics |
| 17 | `failed_access_rate` | Operations / window | Number of access denied / permission violation errors |
| 18 | `admin_action_rate` | Operations / window | Administrative privilege escalation events |
| 19 | `bytes_written_mb` | Megabytes (MB) | Total volume of data written to disk |
| 20 | `new_unknown_process_rate` | Binaries / window | Number of newly spawned unverified binaries |

---

## 7. Synthetic Data Methodology & Kill-Chain Progression

The synthetic telemetry generator strictly simulates numerical patterns without executing harmful code.

### Benign Profile
Simulates desktop productivity, video rendering, compilation bursts, and web browsing. Features stationary baselines with Gaussian noise and occasional benign I/O bursts. Shadow copy deletions and high entropy deltas are virtually absent.

### Ransomware Progression
Follows realistic multi-stage kill-chain dynamics using normalized progression intensity:
$$\text{intensity} = \frac{\text{time\_step}}{\text{SEQ\_LEN} - 1}$$
- **Windows 0–4 (Discovery):** Subtle filesystem traversal, low file writes, initial unknown process check-in.
- **Windows 5–9 (Inhibition):** Execution of backup deletion commands (`vssadmin delete shadows`), elevated administrative actions, API hooking.
- **Windows 10–14 (Encryption Ramp):** Sudden escalation in `file_write_rate`, `file_rename_rate`, and `entropy_delta`.
- **Windows 15–19 (Impact):** Massive continuous file modifications, extension replacement, elevated CPU utilization.

---

## 8. Installation & Environment Setup

Ensure Python 3.11+ is installed.

```bash
# 1. Clone repository and navigate to root directory
cd "d:/anup final project"

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment (Windows)
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 9. Reproducible Execution Pipeline

Execute the complete end-to-end pipeline in sequence:

```bash
# 1. Generate 100,000-row synthetic telemetry dataset
python -m src.dataset

# 2. Run Exploratory Data Analysis & generate plots
python -m src.eda

# 3. Train Proposed Hybrid Model (1D-CNN + BiLSTM + Attention)
python -m src.train

# 4. Evaluate Proposed Model on held-out test data
python -m src.evaluate

# 5. Benchmark Baseline Models (LR, RF, CNN, BiLSTM, CNN+BiLSTM)
python -m src.baselines

# 6. Run Architectural Ablation Study (Configs A, B, C, D)
python -m src.ablation

# 7. Perform Decision Threshold Sensitivity Analysis
python -m src.threshold_analysis

# 8. Compute Permutation Feature Importance & Explainability
python -m src.explainability

# 9. Generate standalone sample telemetry CSV logs
python -m src.make_sample_logs

# 10. Run automated test suite
python -m pytest -q

# 11. Launch Streamlit Web Application
python -m streamlit run app.py
```

---

## 10. Empirical Results & Model Comparison

Evaluated on 750 completely held-out test sequences with zero data leakage:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Tabular Features) | 97.07% | 95.84% | 98.40% | 97.11% | 0.9727 |
| **Random Forest** (Tabular Features) | 97.07% | 95.84% | 98.40% | 97.11% | 0.9727 |
| **CNN Only** | 97.07% | 95.84% | 98.40% | 97.11% | 0.9724 |
| **BiLSTM Only** | 97.07% | 95.84% | 98.40% | 97.11% | 0.9786 |
| **CNN + BiLSTM** | 97.07% | 95.84% | 98.40% | 97.11% | 0.9724 |
| **Proposed Hybrid Model** (CNN + BiLSTM + Attention) | **97.07%** | **95.84%** | **98.40%** | **97.11%** | **0.9778** |

*All results reflect actual experimental runs logged in `reports/model_comparison.csv` (Confusion Matrix: 359 TN, 16 FP, 6 FN, 369 TP).*

---

## 11. Ablation Study

Quantifying the individual contribution of each deep-learning block:

| Configuration | Architecture | Accuracy | F1 Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: |
| **A** | 1D-CNN Only | 97.07% | 97.11% | 0.9724 |
| **B** | BiLSTM Only | 97.07% | 97.11% | 0.9786 |
| **C** | 1D-CNN + BiLSTM (No Attention) | 97.07% | 97.11% | 0.9750 |
| **D** | 1D-CNN + BiLSTM + Attention (**Proposed**) | **97.07%** | **97.11%** | **0.9778** |

---

## 12. Threshold Sensitivity Analysis

Performance across operational SOC decision thresholds ($0.10$ to $0.90$):

| Threshold | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 97.07% | 95.84% | 98.40% | 97.11% | 95.73% | 4.27% | 1.60% |
| **0.30** | 97.07% | 95.84% | 98.40% | 97.11% | 95.73% | 4.27% | 1.60% |
| **0.50** (Default) | 97.07% | 95.84% | 98.40% | 97.11% | 95.73% | 4.27% | 1.60% |
| **0.70** | 97.07% | 95.84% | 98.40% | 97.11% | 95.73% | 4.27% | 1.60% |
| **0.90** | 97.07% | 95.84% | 98.40% | 97.11% | 95.73% | 4.27% | 1.60% |

---

## 13. CLI Prediction

Run inference directly from the command line:

```bash
# Predict on benign sample
python -m src.predict data/sample_benign.csv

# Output:
========================================
RANSOMWARE BEHAVIORAL DETECTION
========================================

Classification:          BENIGN
Ransomware Probability:  3.46%
Confidence:              93.08%
Threshold:               50.0%
Risk Indicator:          LOW

# Predict on ransomware sample
python -m src.predict data/sample_ransomware_detected.csv

# Output:
========================================
RANSOMWARE BEHAVIORAL DETECTION
========================================

Classification:          RANSOMWARE
Ransomware Probability:  98.67%
Confidence:              97.34%
Threshold:               50.0%
Risk Indicator:          HIGH
```

---

## 14. Interactive Streamlit Dashboard

Run the full web interface:

```bash
python -m streamlit run app.py
```
*(Or `streamlit run app.py` if Python Scripts is added to your system PATH)*

### Dashboard Features:
1. **Executive Dashboard:** Live system status, metrics cards, and architectural diagrams.
2. **Dataset Explorer:** Visualizations of feature distributions, class balance, and correlation matrix.
3. **Live Detection:** Real-time CSV file drag-and-drop, predefined test sample loader, threshold slider, risk badge, interactive Plotly behavioral time-series, and timestep attention weight distribution.
4. **Model Performance:** Real confusion matrix, ROC curve, PR curve, and training history curves.
5. **Model Comparison:** Interactive comparison table and bar charts for all 6 models.
6. **Ablation Study:** Comparative evaluation of configurations A, B, C, D.
7. **Threshold Analysis:** Interactive trade-off curves across decision thresholds.
8. **Explainability:** Permutation feature importance charts and attention weight extraction.
9. **About & Viva Guide:** Comprehensive final-year academic viva presentation talking points.

---

## 15. Automated Testing Suite

The project includes 22 automated tests covering the entire pipeline:

```bash
python -m pytest -q
```

Output:
```text
......................                                                   [100%]
22 passed in 5.64s
```

Test coverage includes:
- `test_dataset.py`: Shape, sequence continuity, class balance, nulls.
- `test_preprocessing.py`: Sequence-level split, anti-leakage verification, 3D tensor conversion.
- `test_model.py`: Tensor output shapes, attention sum verification ($\sum \alpha = 1.0$).
- `test_prediction.py`: End-to-end inference on benign and ransomware files, threshold variations.
- `test_validation.py`: Missing columns, insufficient rows ($<20$), NaN detection, file existence.

---

## 16. Project Structure

```text
d:\anup final project\
├── app.py                      # Multi-page Streamlit Dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive academic guide
├── .gitignore                  # Git exclusions
│
├── data/
│   ├── behavior_sequences.csv  # 5,000 sequences × 20 steps = 100,000 rows
│   ├── sample_benign.csv       # Predefined test benign sequence
│   └── sample_ransomware_like.csv # Predefined test ransomware sequence
│
├── models/
│   ├── ransomware_hybrid.pt    # PyTorch model state_dict & hyperparameters
│   └── scaler.joblib           # Preprocessing StandardScaler fitted on train only
│
├── reports/
│   ├── eda/                    # EDA charts (distributions, boxplots, balance)
│   ├── experiments/            # JSON execution metadata logs
│   ├── training_history.csv    # Epoch loss & accuracy logs
│   ├── training_curve.png      # Loss & accuracy curves
│   ├── confusion_matrix.png    # Evaluation confusion matrix
│   ├── roc_curve.png           # ROC curve plot
│   ├── precision_recall_curve.png # Precision-Recall curve
│   ├── threshold_analysis.csv  # Metrics across thresholds 0.10 to 0.90
│   ├── model_comparison.csv    # Benchmark comparison across 6 models
│   ├── ablation_results.csv    # Ablation study metrics table
│   ├── ablation_comparison.png # Ablation study performance chart
│   ├── feature_importance.csv  # Permutation feature importance rankings
│   ├── feature_importance.png  # Feature importance bar chart
│   └── classification_report.txt # Text classification report
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration & hyperparameters
│   ├── dataset.py              # Synthetic behavioral telemetry generator
│   ├── preprocessing.py        # Leakage-free sequence scaling & tensor builder
│   ├── eda.py                  # Exploratory Data Analysis module
│   ├── model.py                # PyTorch architectures (Hybrid, CNN, BiLSTM)
│   ├── baselines.py            # Baseline benchmark engine (LR, RF, CNN, BiLSTM)
│   ├── train.py                # Model training, early stopping, and checkpointing
│   ├── evaluate.py             # Evaluation routines & curve generation
│   ├── ablation.py             # Architectural ablation study
│   ├── threshold_analysis.py   # Decision threshold sensitivity analysis
│   ├── explainability.py       # Attention weights & permutation feature importance
│   ├── predict.py              # CLI & programmatic inference API
│   └── make_sample_logs.py     # Standalone sample log generator
│
└── tests/
    ├── __init__.py
    ├── test_dataset.py
    ├── test_preprocessing.py
    ├── test_model.py
    ├── test_prediction.py
    └── test_validation.py
```

---

## 17. Dataset Limitations & Academic Safe Harbor

> **Important Academic Notice:**  
> The dataset used in this academic research prototype is synthetically generated to represent behavioral patterns associated with benign workloads and ransomware-like kill-chain escalation. Performance on this synthetic dataset demonstrates the validity of the temporal deep-learning pipeline and feature engineering methodology, but should not be interpreted as directly equivalent to real-world in-the-wild ransomware detection performance on physical hosts.

---

## 18. Security & Ethical Considerations

This project is strictly defensive:
- **No Malicious Code:** The codebase contains zero malware, ransomware executables, or payload droppers.
- **No File Encryption:** No real files are ever encrypted, locked, or modified destructively.
- **No Destructive Operations:** The system never alters shadow copies, backups, or registry keys on the host machine.
- **Numerical Telemetry Only:** All suspicious activity is represented exclusively as simulated numerical measurements.

---

## 19. Future Research & Extensions

1. **Real-world Telemetry Ingestion:** Integrate Windows Event Tracing (ETW) and Linux Syscall auditd logs mapped to the same 20-feature schema.
2. **Streaming Online Inference:** Implement sliding-window online sequence streaming for sub-second detection in SOC environments.
3. **Ransomware Family Multi-class Classification:** Expand label taxonomy from binary detection to classifying specific ransomware strains (e.g. LockBit, BlackCat, Conti).
4. **Federated Learning:** Enable decentralized training across multiple corporate networks without sharing sensitive endpoint telemetry.

---

## 20. Academic Viva Presentation Highlights

When presenting this project for viva examination, emphasize:
- **The Justification for Deep Learning:** Tabular models ignore the sequential order of events. The combination of CNN (local burst patterns) + BiLSTM (multi-phase kill-chain) + Attention (escalation weighting) models both local and long-range temporal dependencies.
- **Data Leakage Safeguards:** Telemetry is split by `sequence_id`, not by individual row. Standard scaling is fitted strictly on training data.
- **Explainability:** Attention weights provide cybersecurity analysts with visibility into *when* the network identified the critical escalation phase.
