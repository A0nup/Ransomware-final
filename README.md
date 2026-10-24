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
12. Deliver enterprise SOC architecture and technical documentation.

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
git clone https://github.com/anup-rawat/ransomware-detection.git
cd ransomware-detection

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment (Linux / macOS)
source .venv/bin/activate
# Or on Windows:
# .venv\Scripts\activate

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

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Tabular Aggregates) | 89.20% | 87.53% | 91.47% | 89.46% | 0.9418 |
| **Random Forest** (Tabular Aggregates) | 92.40% | 91.03% | 94.13% | 92.55% | 0.9652 |
| **CNN Only** | 94.53% | 93.78% | 95.47% | 94.62% | 0.9715 |
| **BiLSTM Only** | 95.60% | 94.97% | 96.27% | 95.62% | 0.9784 |
| **CNN + BiLSTM** | 96.80% | 96.10% | 97.60% | 96.84% | 0.9842 |
| **Proposed Hybrid Model** (CNN + BiLSTM + Attention) | **98.13%** | **97.87%** | **98.40%** | **98.13%** | **0.9912** |

*All results reflect actual experimental runs logged in `reports/model_comparison.csv` (Confusion Matrix at default $\tau=0.50$: 367 TN, 8 FP, 6 FN, 369 TP).*

---

## 11. Ablation Study

Quantifying the individual contribution of each deep-learning block:

| Configuration | Architecture | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | 1D-CNN Only | 94.53% | 93.78% | 95.47% | 94.62% | 0.9715 |
| **B** | BiLSTM Only | 95.60% | 94.97% | 96.27% | 95.62% | 0.9784 |
| **C** | 1D-CNN + BiLSTM (No Attention) | 96.80% | 96.10% | 97.60% | 96.84% | 0.9842 |
| **D** | 1D-CNN + BiLSTM + Attention (**Proposed**) | **98.13%** | **97.87%** | **98.40%** | **98.13%** | **0.9912** |

*Adding Additive Self-Attention yields an empirical +1.33% accuracy increase and lifts ROC-AUC to 0.9912 while providing per-timestep attention explainability.*

---

## 12. Threshold Sensitivity Analysis

Performance across operational SOC decision thresholds ($\tau \in [0.10, 0.90]$):

| Threshold | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.10** | 95.20% | 91.22% | 99.73% | 95.29% | 90.67% | 9.33% | 0.27% |
| **0.20** | 96.27% | 93.48% | 99.47% | 96.38% | 93.07% | 6.93% | 0.53% |
| **0.30** | 97.07% | 95.14% | 99.20% | 97.13% | 94.93% | 5.07% | 0.80% |
| **0.40** | 97.73% | 96.61% | 98.93% | 97.76% | 96.53% | 3.47% | 1.07% |
| **0.50** (Default) | 98.13% | 97.87% | 98.40% | 98.13% | 97.87% | 2.13% | 1.60% |
| **0.60** | 97.87% | 98.38% | 97.33% | 97.85% | 98.40% | 1.60% | 2.67% |
| **0.70** | 97.20% | 98.90% | 95.47% | 97.15% | 98.93% | 1.07% | 4.53% |
| **0.80** | 95.73% | 99.43% | 92.00% | 95.57% | 99.47% | 0.53% | 8.00% |
| **0.90** | 94.13% | 99.70% | 88.53% | 93.78% | 99.73% | 0.27% | 11.47% |

---

## 13. CLI Prediction

Run inference directly from the command line:

```bash
# Predict on benign sample
python -m src.predict data/sample_benign.csv

# Predict on multi-phase ransomware attack trajectory
python -m src.predict data/sample_ransomware_like.csv

# Predict on pre-recorded detected ransomware sample
python -m src.predict data/sample_ransomware_detected.csv
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
5. **Model Comparison:** Interactive comparison table and bar charts across all 6 benchmark models.
6. **Ablation Study:** Comparative evaluation of configurations A, B, C, D proving attention utility.
7. **Threshold Analysis:** Interactive trade-off curves across decision thresholds.
8. **Explainability:** Permutation feature importance charts and attention weight extraction.
9. **System Architecture & Technical Documentation:** Comprehensive technical reference covering temporal feature engineering, deep learning layers, mathematical formulations, and engineering FAQs.

---

## 15. Automated Testing Suite

The project includes **48 automated tests** across 9 test suites covering the entire defensive pipeline:

```bash
python -m pytest tests/
```

Output:
```text
============================= 48 passed in 11.26s ==============================
```

Test coverage includes:
- `test_advanced_soc_features.py`: Threat intel enrichment, Shannon entropy analysis, MITRE ATT&CK mapping, batch folder triage, hybrid YARA/Sigma rule scanning, and PDF incident reporting.
- `test_endpoint_monitor.py`: Real-time localhost process monitoring, safe telemetry ingestion, and disk rate calculations.
- `test_scientific_audit.py`: Multi-model metric differentiation, threshold sensitivity curves, non-saturated probability curves, and CPU inference profiling (<50ms).
- `test_static_scanner.py`: Zero-execution PE header parsing, section entropy calculation, and suspicious indicator extraction.
- `test_dataset.py`: Multi-phase sequence generation, temporal continuity, class balance, and null checks.
- `test_preprocessing.py`: Sequence-level train/val/test splitting, anti-leakage verification, and 3D tensor formatting.
- `test_model.py`: Tensor output shapes, forward pass validation, and attention sum verification ($\sum \alpha_t = 1.0$).
- `test_prediction.py`: End-to-end inference on benign, attack, and detected sequences.
- `test_validation.py`: Schema validation, missing feature handling, insufficient rows ($<20$), and NaN detection.

---

## 16. Project Structure

```text
ransomware-detection/
├── app.py                      # Multi-page Streamlit Dashboard & Threat Analysis Studio
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive technical documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git exclusions
│
├── data/
│   ├── behavior_sequences.csv      # 5,000 sequences × 20 steps = 100,000 rows
│   ├── sample_benign.csv           # Predefined benign multi-step workload
│   ├── sample_ransomware_like.csv  # Predefined multi-phase attack trajectory
│   ├── sample_ransomware_detected.csv # Pre-recorded high-confidence ransomware detection log
│   ├── live_threat_logs/           # Localhost real-time threat audit logs
│   └── model_monitoring/           # Prediction history, feedback, and drift logs
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
│   └── classification_report.txt # Text classification report
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration & hyperparameters
│   ├── dataset.py              # Realistic multi-phase synthetic telemetry generator
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
│   ├── endpoint_monitor.py     # Real-time localhost process & filesystem EDR shield
│   ├── scientific_validation.py# Statistical validation, saturation audit & benchmarking
│   ├── mitre_attack.py         # MITRE ATT&CK mapping & IOC extraction
│   ├── batch_analyzer.py       # Batch telemetry triage & folder ingestion
│   ├── threat_intel.py         # Threat intelligence enrichment (VT, MalwareBazaar, AbuseIPDB)
│   ├── entropy_analyzer.py     # Shannon entropy computation & sliding-window analysis
│   ├── report_generator.py     # One-click professional PDF incident report generator
│   ├── model_monitoring.py     # Real-time model drift tracking & analyst feedback loop
│   └── hybrid_rules.py         # Hybrid detection (YARA + Sigma rules + ML)
│
└── tests/
    ├── __init__.py
    ├── test_advanced_soc_features.py
    ├── test_dataset.py
    ├── test_endpoint_monitor.py
    ├── test_model.py
    ├── test_prediction.py
    ├── test_preprocessing.py
    ├── test_scientific_audit.py
    ├── test_static_scanner.py
    └── test_validation.py
```

---

## 17. Dataset Limitations & Defensive Scope

> **Defensive Engineering Notice:**  
> The dataset used in this research prototype simulates behavioral telemetry patterns associated with benign workloads and ransomware kill-chain escalation. Performance on this benchmark demonstrates the validity of the temporal deep-learning pipeline and feature engineering methodology, but physical production deployments must calibrate against host-specific ETW (Event Tracing for Windows) or auditd baselines.

---

## 18. Security & Ethical Considerations

This project is strictly defensive:
- **No Malicious Code:** The codebase contains zero malware, ransomware executables, or payload droppers.
- **No File Encryption:** No real files are ever encrypted, locked, or modified destructively.
- **No Destructive Operations:** The system never alters shadow copies, backups, or registry keys on the host machine.
- **Numerical Telemetry Only:** All suspicious activity is represented exclusively as simulated numerical measurements.
- **Localhost Privacy Guarantee:** All telemetry ingestion and model evaluations run 100% locally on localhost without transmitting endpoint data across external networks.

---

## 19. Future Research & Extensions

1. **Kernel Driver Telemetry Ingestion:** Integrate Windows Kernel Minifilter and Linux eBPF probes mapped to the 20-feature schema.
2. **Streaming Online Inference:** Implement sliding-window online sequence streaming for sub-second detection in enterprise SOC environments.
3. **Ransomware Family Multi-class Classification:** Expand label taxonomy from binary detection to classifying specific ransomware strains (e.g., LockBit 3.0, BlackCat/ALPHV, Conti).
4. **Federated Threat Intelligence:** Enable decentralized model retraining across enterprise enclaves without sharing sensitive endpoint telemetry.

---

## 20. Enterprise SOC Operations & Incident Playbooks

In a production Security Operations Center (SOC), this system acts as a behavioral decision-support layer integrated with SIEM/SOAR platforms.

### SOC Incident Response Workflow:
1. **Real-Time Behavioral Triage:**
   - The endpoint monitor continuously evaluates sliding 20-second behavioral windows.
   - If probability exceeds the SOC threshold ($\tau = 0.50$), an alert triggers with classified severity (**Low**, **Medium**, **High**, or **Critical**).
2. **MITRE ATT&CK Mapping & IOC Extraction:**
   - Active telemetry is automatically mapped to MITRE ATT&CK technique IDs (e.g., `T1486 Data Encrypted for Impact`, `T1490 Inhibit System Recovery`, `T1082 System Information Discovery`, `T1070.004 File Deletion`).
   - IOCs (process names, hashes, parent PIDs, extensions, target paths) are automatically extracted into CSV and JSON export formats.
3. **Hybrid Verification (YARA + Sigma + Attention):**
   - Signature checks (YARA), process command-line patterns (Sigma rules), and temporal deep-learning attention weights corroborate the alert, preventing single-point failure.
4. **Automated Incident Playbook Execution:**
   - **Critical / High Risk:** Immediately isolate the endpoint from the corporate network, suspend the offensive process tree, preserve memory telemetry, and alert the Incident Response on-call team.
   - **Medium Risk:** Trigger targeted forensic memory snapshot and prompt the administrator for verification.
   - **Low Risk / Benign:** Log telemetry locally for baseline calibration and model drift tracking.
5. **Continuous Governance & Analyst Feedback:**
   - Security analysts record feedback ("Correct Detection" or "False Positive") directly in the dashboard, updating the drift monitoring log to trigger periodic model fine-tuning.
