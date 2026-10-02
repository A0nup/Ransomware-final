"""
Sample Logs Generator
Generates standalone sample CSV files for benign and ransomware-like telemetry
for testing the CLI and Web Dashboard upload capabilities.
Contains no malicious code or executable payloads.
"""

import pandas as pd
import numpy as np

from src.config import (
    FEATURES,
    SEQ_LEN,
    SAMPLE_BENIGN_PATH,
    SAMPLE_RANSOMWARE_PATH,
)
from src.dataset import (
    generate_benign_sequence,
    generate_ransomware_sequence,
    generate_host_profiles,
)


def create_sample_csvs() -> None:
    """Generate sample_benign.csv and sample_ransomware_like.csv."""
    print("=" * 60)
    print("GENERATING SAMPLE TELEMETRY LOGS")
    print("=" * 60)

    hosts = generate_host_profiles(5)
    host_id = "HOST-001"
    profile = hosts[host_id]

    # 1. Benign Sample (20 steps)
    benign_data = generate_benign_sequence(seq_id=99991, host_id=host_id, host_profile=profile)
    benign_rows = []
    for t in range(SEQ_LEN):
        row = {
            "sequence_id": 99991,
            "time_step": t,
            "host_id": host_id,
        }
        for idx, feat in enumerate(FEATURES):
            row[feat] = round(float(benign_data[t, idx]), 4)
        benign_rows.append(row)

    benign_df = pd.DataFrame(benign_rows)
    benign_df.to_csv(SAMPLE_BENIGN_PATH, index=False)
    print(f"[+] Sample benign log saved to: {SAMPLE_BENIGN_PATH}")

    # 2. Ransomware Sample (20 steps)
    ransomware_data = generate_ransomware_sequence(seq_id=99992, host_id=host_id, host_profile=profile)
    ransomware_rows = []
    for t in range(SEQ_LEN):
        row = {
            "sequence_id": 99992,
            "time_step": t,
            "host_id": host_id,
        }
        for idx, feat in enumerate(FEATURES):
            row[feat] = round(float(ransomware_data[t, idx]), 4)
        ransomware_rows.append(row)

    ransomware_df = pd.DataFrame(ransomware_rows)
    ransomware_df.to_csv(SAMPLE_RANSOMWARE_PATH, index=False)
    print(f"[+] Sample ransomware log saved to: {SAMPLE_RANSOMWARE_PATH}")


if __name__ == "__main__":
    create_sample_csvs()
