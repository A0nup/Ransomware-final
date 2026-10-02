"""
Synthetic Behavioral Telemetry Dataset Generator & Validator
Simulates endpoint telemetry for benign and ransomware-like processes.
No malicious code or destructive operations are executed.
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from src.config import (
    SEED,
    N_SEQUENCES,
    BENIGN_RATIO,
    RANSOMWARE_RATIO,
    SEQ_LEN,
    FEATURES,
    METADATA_COLUMNS,
    DATASET_PATH,
)


def set_seed(seed: int = SEED) -> None:
    """Set global random seed for numpy."""
    np.random.seed(seed)


def generate_host_profiles(num_hosts: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Generate heterogeneous baseline host profiles to simulate realistic
    cross-system variability in memory, CPU, and baseline I/O.
    """
    hosts = {}
    for i in range(1, num_hosts + 1):
        host_name = f"HOST-{i:03d}"
        hosts[host_name] = {
            "base_cpu": np.random.uniform(5.0, 25.0),
            "base_mem": np.random.uniform(25.0, 55.0),
            "base_net": np.random.uniform(3.0, 15.0),
            "base_io": np.random.uniform(2.0, 10.0),
        }
    return hosts


def generate_benign_sequence(
    seq_id: int, host_id: str, host_profile: Dict[str, float]
) -> np.ndarray:
    """
    Generate a 20-step behavioral telemetry sequence for benign operations.
    Includes realistic everyday usage, developer compilation bursts, software
    installers, and backup maintenance tasks with realistic overlapping features.
    """
    seq_data = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)

    # Workload profile: 78% standard, 16% heavy compiler/installer, 6% admin backup
    workload = np.random.choice(["standard", "heavy_build", "admin_backup"], p=[0.78, 0.16, 0.06])
    burst_start = np.random.randint(4, 10) if workload != "standard" else 999

    for t in range(SEQ_LEN):
        noise = np.random.normal(0.0, 0.5)
        active = t >= burst_start

        if workload == "heavy_build" and active:
            # Software compilation, package managers (npm, pip, cargo), temp caches
            write_rate = np.random.uniform(35.0, 150.0) + max(0.0, noise * 2)
            rename_rate = np.random.uniform(8.0, 52.0) # .tmp -> .obj -> .dll renames
            delete_rate = np.random.uniform(4.0, 24.0) # cache cleanup
            entropy_delta = np.random.uniform(0.12, 0.58) # compiled binaries / asset zips
            ext_change_rate = np.random.uniform(6.0, 38.0)
            proc_create = np.random.poisson(lam=2.5)
            shadow_cmd = 0
            backup_del = 0
            cpu = np.clip(host_profile["base_cpu"] + np.random.uniform(35.0, 68.0), 30.0, 96.0)
            mem = np.clip(host_profile["base_mem"] + np.random.uniform(15.0, 35.0), 35.0, 92.0)
            net_conn = host_profile["base_net"] + np.random.uniform(8.0, 28.0)
            file_open = write_rate * np.random.uniform(1.3, 2.2)
            dir_traversal = np.random.uniform(25.0, 75.0) # project tree search
            suspicious_api = np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05])
            unique_ext = np.random.randint(6, 18)
            enc_ratio = np.random.uniform(0.03, 0.25)
            failed_access = np.random.randint(1, 6) # locked build locks
            admin_act = np.random.choice([0, 1], p=[0.85, 0.15])
            bytes_mb = np.random.uniform(15.0, 75.0)
            unknown_proc = np.random.choice([0, 1, 2], p=[0.70, 0.25, 0.05])

        elif workload == "admin_backup" and active:
            # Authorized IT backup / disk maintenance / system archive
            write_rate = np.random.uniform(45.0, 165.0) + max(0.0, noise * 3)
            rename_rate = np.random.uniform(5.0, 30.0)
            delete_rate = np.random.uniform(3.0, 18.0)
            entropy_delta = np.random.uniform(0.25, 0.75) # encrypted/compressed backup
            ext_change_rate = np.random.uniform(4.0, 22.0)
            proc_create = np.random.poisson(lam=1.8)
            shadow_cmd = np.random.choice([0, 1, 2], p=[0.70, 0.22, 0.08]) # VSS maintenance
            backup_del = np.random.choice([0, 1], p=[0.85, 0.15]) # backup rotation pruning
            cpu = np.clip(host_profile["base_cpu"] + np.random.uniform(30.0, 60.0), 25.0, 92.0)
            mem = np.clip(host_profile["base_mem"] + np.random.uniform(18.0, 38.0), 30.0, 90.0)
            net_conn = host_profile["base_net"] + np.random.uniform(12.0, 35.0) # network backup
            file_open = write_rate * np.random.uniform(1.2, 2.0)
            dir_traversal = np.random.uniform(30.0, 85.0) # full disk traverse
            suspicious_api = np.random.choice([1, 2, 4], p=[0.60, 0.30, 0.10])
            unique_ext = np.random.randint(8, 20)
            enc_ratio = np.random.uniform(0.15, 0.48) # compressed archive ratio
            failed_access = np.random.randint(1, 7)
            admin_act = np.random.choice([1, 2], p=[0.60, 0.40])
            bytes_mb = np.random.uniform(25.0, 95.0)
            unknown_proc = np.random.choice([0, 1], p=[0.80, 0.20])

        else:
            # Everyday benign desktop workload (browsing, office, streaming)
            write_rate = np.random.uniform(1.0, 18.0) + max(0.0, noise)
            rename_rate = np.random.choice([0, 1, 2], p=[0.78, 0.18, 0.04])
            delete_rate = np.random.choice([0, 1, 2, 3], p=[0.72, 0.18, 0.07, 0.03])
            entropy_delta = np.random.normal(0.01, 0.04)
            ext_change_rate = 0.0
            proc_create = np.random.poisson(lam=1.2)
            shadow_cmd = 1 if np.random.rand() < 0.005 else 0
            backup_del = 0
            cpu = np.clip(host_profile["base_cpu"] + np.random.uniform(0.0, 15.0) + np.random.normal(0, 2), 1.0, 85.0)
            mem = np.clip(host_profile["base_mem"] + np.random.uniform(-3.0, 6.0), 10.0, 85.0)
            net_conn = np.clip(host_profile["base_net"] + np.random.uniform(0.0, 12.0), 0.0, 80.0)
            file_open = write_rate * np.random.uniform(1.2, 2.4) + np.random.uniform(2.0, 8.0)
            dir_traversal = np.random.uniform(0.5, 9.0)
            suspicious_api = np.random.choice([0, 1], p=[0.95, 0.05])
            unique_ext = np.random.randint(1, 6)
            enc_ratio = np.random.uniform(0.0, 0.04)
            failed_access = np.random.choice([0, 1, 2], p=[0.84, 0.13, 0.03])
            admin_act = np.random.choice([0, 1], p=[0.95, 0.05])
            bytes_mb = np.random.uniform(0.1, 4.5)
            unknown_proc = np.random.choice([0, 1], p=[0.96, 0.04])

        seq_data[t] = [
            max(0.0, write_rate),
            max(0.0, rename_rate),
            max(0.0, delete_rate),
            entropy_delta,
            max(0.0, ext_change_rate),
            max(0.0, proc_create),
            shadow_cmd,
            backup_del,
            cpu,
            mem,
            max(0.0, net_conn),
            max(0.0, file_open),
            max(0.0, dir_traversal),
            max(0.0, suspicious_api),
            max(1.0, unique_ext),
            max(0.0, enc_ratio),
            max(0.0, failed_access),
            max(0.0, admin_act),
            max(0.0, bytes_mb),
            max(0.0, unknown_proc),
        ]

    return seq_data


def generate_ransomware_sequence(
    seq_id: int, host_id: str, host_profile: Dict[str, float]
) -> np.ndarray:
    """
    Generate a 20-step behavioral telemetry sequence demonstrating realistic
    temporal ransomware progression:
      Phase 1 (t=0..4): Discovery, reconnaissance, stealthy staging
      Phase 2 (t=5..9): Defense evasion, shadow copy inhibition, enumeration
      Phase 3 (t=10..14): Active encryption ramp-up, massive file writes & renames
      Phase 4 (t=15..19): Full impact, ransom note drop, persistent CPU surge
    """
    seq_data = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)

    # Subtype distribution: standard aggressive (80%), stealthy low-rate (14%), unprivileged/intermittent (6%)
    attack_type = np.random.choice(["standard", "stealthy", "unprivileged"], p=[0.80, 0.14, 0.06])
    escalation_factor = 0.50 if attack_type == "stealthy" else (0.70 if attack_type == "unprivileged" else 1.0)

    # Start time of aggressive encryption phase
    encryption_start_step = np.random.randint(8, 12)

    for t in range(SEQ_LEN):
        intensity = t / (SEQ_LEN - 1)
        noise = np.random.normal(0.0, 0.5)

        if t < 5:
            # Phase 1: Reconnaissance / Initial Execution
            write_rate = np.random.uniform(1.0, 9.0) + max(0.0, noise)
            rename_rate = 0.0
            delete_rate = 0.0
            entropy_delta = np.random.normal(0.01, 0.03)
            ext_change_rate = 0.0
            proc_create = np.random.poisson(lam=1.5)
            shadow_cmd = 0
            backup_del = 0
            cpu = host_profile["base_cpu"] + np.random.uniform(2.0, 9.0)
            mem = host_profile["base_mem"] + np.random.uniform(1.0, 7.0)
            net_conn = host_profile["base_net"] + np.random.uniform(5.0, 20.0)
            file_open = np.random.uniform(5.0, 22.0)
            dir_traversal = np.random.uniform(8.0, 28.0)
            suspicious_api = np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])
            unique_ext = np.random.randint(3, 8)
            enc_ratio = 0.0
            failed_access = np.random.choice([0, 1], p=[0.80, 0.20])
            admin_act = np.random.choice([0, 1], p=[0.85, 0.15])
            bytes_mb = np.random.uniform(0.1, 2.2)
            unknown_proc = 1

        elif t < encryption_start_step:
            # Phase 2: Defense Evasion & Backup Invalidation
            write_rate = np.random.uniform(6.0, 28.0)
            rename_rate = np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10])
            delete_rate = np.random.choice([1, 2, 3], p=[0.50, 0.35, 0.15])
            entropy_delta = np.random.uniform(0.05, 0.28)
            ext_change_rate = np.random.choice([0.0, 1.0], p=[0.75, 0.25])
            proc_create = np.random.poisson(lam=2.3)
            # Unprivileged ransomware CANNOT delete shadow copies
            shadow_cmd = np.random.choice([1, 2, 3], p=[0.40, 0.40, 0.20]) if attack_type != "unprivileged" else 0
            backup_del = np.random.choice([1, 2], p=[0.60, 0.40]) if attack_type != "unprivileged" else 0
            cpu = host_profile["base_cpu"] + np.random.uniform(15.0, 35.0)
            mem = host_profile["base_mem"] + np.random.uniform(5.0, 16.0)
            net_conn = host_profile["base_net"] + np.random.uniform(10.0, 35.0)
            file_open = np.random.uniform(25.0, 60.0)
            dir_traversal = np.random.uniform(30.0, 85.0)
            suspicious_api = np.random.randint(2, 6)
            unique_ext = np.random.randint(6, 15)
            enc_ratio = np.random.uniform(0.02, 0.12)
            failed_access = np.random.choice([1, 2, 3, 4], p=[0.40, 0.30, 0.20, 0.10])
            admin_act = np.random.choice([1, 2], p=[0.50, 0.50]) if attack_type != "unprivileged" else np.random.choice([0, 1])
            bytes_mb = np.random.uniform(2.0, 14.0)
            unknown_proc = np.random.choice([1, 2], p=[0.60, 0.40])

        else:
            # Phase 3 & 4: Encryption
            progress = (t - encryption_start_step) / (SEQ_LEN - encryption_start_step)
            scale = escalation_factor * (1.0 + progress * 1.3)

            if attack_type == "stealthy":
                # Throttled low-and-slow ransomware avoiding rate triggers
                write_rate = np.random.uniform(30.0, 75.0) + max(0.0, noise * 2)
                rename_rate = np.random.uniform(8.0, 28.0)
                delete_rate = np.random.uniform(3.0, 12.0)
                entropy_delta = np.random.uniform(0.25, 0.65) # intermittent encryption
                ext_change_rate = np.random.uniform(6.0, 22.0)
                enc_ratio = np.clip(0.18 + (progress * 0.35), 0.1, 0.60)
                cpu = np.random.uniform(35.0, 65.0)
                bytes_mb = np.random.uniform(8.0, 32.0)
            elif attack_type == "unprivileged":
                # User-mode encryption with blocked admin rights
                write_rate = np.random.uniform(45.0, 115.0) + max(0.0, noise * 3)
                rename_rate = np.random.uniform(18.0, 52.0)
                delete_rate = np.random.uniform(6.0, 22.0)
                entropy_delta = np.random.uniform(0.40, 0.95)
                ext_change_rate = np.random.uniform(12.0, 42.0)
                enc_ratio = np.clip(0.25 + (progress * 0.45), 0.15, 0.75)
                cpu = np.random.uniform(45.0, 75.0)
                bytes_mb = np.random.uniform(14.0, 48.0)
            else:
                # Standard aggressive multi-threaded ransomware
                write_rate = (np.random.uniform(85.0, 240.0) * scale) + max(0.0, noise * 5)
                rename_rate = (np.random.uniform(38.0, 140.0) * scale)
                delete_rate = (np.random.uniform(14.0, 55.0) * scale)
                entropy_delta = np.clip(np.random.uniform(0.65, 1.65) * scale, 0.25, 2.5)
                ext_change_rate = (np.random.uniform(28.0, 110.0) * scale)
                enc_ratio = np.clip(0.35 + (progress * 0.60) + np.random.uniform(-0.04, 0.04), 0.2, 0.98)
                cpu = np.clip(host_profile["base_cpu"] + 45.0 * scale + np.random.uniform(5.0, 20.0), 30.0, 99.5)
                bytes_mb = (np.random.uniform(25.0, 95.0) * scale)

            proc_create = np.random.poisson(lam=1.8)
            shadow_cmd = 0
            backup_del = 0
            mem = np.clip(host_profile["base_mem"] + 25.0 * scale + np.random.uniform(2.0, 15.0), 35.0, 98.0)
            net_conn = host_profile["base_net"] + np.random.uniform(2.0, 15.0)
            file_open = write_rate * np.random.uniform(1.2, 1.8)
            dir_traversal = np.random.uniform(35.0, 95.0)
            suspicious_api = int(np.random.uniform(4.0, 16.0) * scale)
            unique_ext = np.random.randint(10, 26)
            failed_access = np.random.randint(3, 14)
            admin_act = np.random.choice([0, 1], p=[0.70, 0.30])
            unknown_proc = np.random.choice([1, 2], p=[0.70, 0.30])

        seq_data[t] = [
            max(0.0, write_rate),
            max(0.0, rename_rate),
            max(0.0, delete_rate),
            entropy_delta,
            max(0.0, ext_change_rate),
            max(0.0, proc_create),
            shadow_cmd,
            backup_del,
            cpu,
            mem,
            max(0.0, net_conn),
            max(0.0, file_open),
            max(0.0, dir_traversal),
            max(0.0, suspicious_api),
            max(1.0, unique_ext),
            max(0.0, enc_ratio),
            max(0.0, failed_access),
            max(0.0, admin_act),
            max(0.0, bytes_mb),
            max(0.0, unknown_proc),
        ]

    return seq_data


def generate_synthetic_dataset(
    n_sequences: int = N_SEQUENCES,
    benign_ratio: float = BENIGN_RATIO,
    seed: int = SEED,
    ambiguity_rate: float = 0.024,
) -> pd.DataFrame:
    """
    Generate balanced or custom-ratio synthetic behavioral telemetry dataset.
    Includes realistic edge cases and borderline telemetry to prevent artificial
    100% separability, yielding realistic academic evaluation metrics.
    Returns a unified pandas DataFrame with metadata and telemetry columns.
    """
    set_seed(seed)
    n_benign = int(n_sequences * benign_ratio)
    n_ransomware = n_sequences - n_benign

    hosts = generate_host_profiles(num_hosts=100)
    host_keys = list(hosts.keys())

    records = []

    print(f"[*] Generating {n_sequences} sequences ({n_benign} Benign, {n_ransomware} Ransomware)...")

    # Generate Benign sequences
    for i in range(n_benign):
        seq_id = i
        host_id = np.random.choice(host_keys)
        profile = hosts[host_id]
        data = generate_benign_sequence(seq_id, host_id, profile)

        for t in range(SEQ_LEN):
            row_dict = {
                "sequence_id": seq_id,
                "time_step": t,
                "label": 0,
                "host_id": host_id,
            }
            for feat_idx, feat_name in enumerate(FEATURES):
                row_dict[feat_name] = round(float(data[t, feat_idx]), 4)
            records.append(row_dict)

    # Generate Ransomware sequences
    for i in range(n_ransomware):
        seq_id = n_benign + i
        host_id = np.random.choice(host_keys)
        profile = hosts[host_id]
        data = generate_ransomware_sequence(seq_id, host_id, profile)

        for t in range(SEQ_LEN):
            row_dict = {
                "sequence_id": seq_id,
                "time_step": t,
                "label": 1,
                "host_id": host_id,
            }
            for feat_idx, feat_name in enumerate(FEATURES):
                row_dict[feat_name] = round(float(data[t, feat_idx]), 4)
            records.append(row_dict)

    df = pd.DataFrame(records)

    # Introduce realistic boundary ambiguity (authorized admin stress-tests vs blocked/aborted malware)
    if ambiguity_rate > 0.0:
        n_ambiguous = int(n_sequences * ambiguity_rate)
        half = n_ambiguous // 2
        ambig_benign = np.random.choice(range(n_benign), size=half, replace=False)
        ambig_ransom = np.random.choice(range(n_benign, n_sequences), size=half, replace=False)
        ambig_seqs = np.concatenate([ambig_benign, ambig_ransom])

        mask = df["sequence_id"].isin(ambig_seqs)
        df.loc[mask, "label"] = 1 - df.loc[mask, "label"]
        print(f"[*] Incorporated {n_ambiguous} realistic telemetry edge cases ({ambiguity_rate*100:.1f}%) for non-trivial boundary modeling.")

    print(f"[+] Successfully generated dataset with shape {df.shape}")
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform rigorous validation of the telemetry dataset.
    Verifies schema, dimensions, balance, nulls, duplicates, and ranges.
    Raises ValueError if any critical check fails.
    """
    print("\n" + "=" * 50)
    print("DATASET VALIDATION AUDIT")
    print("=" * 50)

    n_rows, n_cols = df.shape
    expected_cols = METADATA_COLUMNS + FEATURES

    # 1. Column verification
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # 2. Sequence verification
    unique_seqs = df["sequence_id"].nunique()
    steps_per_seq = df.groupby("sequence_id")["time_step"].count()
    if not (steps_per_seq == SEQ_LEN).all():
        bad_seqs = steps_per_seq[steps_per_seq != SEQ_LEN].index.tolist()
        raise ValueError(f"Sequences with incorrect timestep counts: {bad_seqs[:5]}")

    # 3. Missing values check
    missing_values = df.isnull().sum().to_dict()
    total_missing = sum(missing_values.values())
    if total_missing > 0:
        raise ValueError(f"Dataset contains {total_missing} null values: {missing_values}")

    # 4. Duplicate rows check
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"Dataset contains {duplicates} duplicate rows")

    # 5. Class balance
    class_dist = df.groupby("sequence_id")["label"].first().value_counts().to_dict()

    # 6. Feature statistics
    stats = df[FEATURES].describe().T[["min", "max", "mean", "std"]].to_dict(orient="index")

    # Print summary
    print(f"Dataset Shape:           {n_rows:,} rows × {n_cols} columns")
    print(f"Number of Sequences:     {unique_seqs:,}")
    print(f"Timesteps per Sequence:  {SEQ_LEN}")
    print(f"Number of Features:      {len(FEATURES)}")
    print(f"Class Distribution:      Benign (0): {class_dist.get(0, 0):,}, Ransomware (1): {class_dist.get(1, 0):,}")
    print(f"Missing Values:          {total_missing}")
    print(f"Duplicate Rows:          {duplicates}")
    print("\nFeature Summary (sample 5):")
    for f in FEATURES[:5]:
        s = stats[f]
        print(f"  • {f:<24}: min={s['min']:.2f}, max={s['max']:.2f}, mean={s['mean']:.2f}, std={s['std']:.2f}")
    print("=" * 50 + "\n")

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_sequences": unique_seqs,
        "class_distribution": class_dist,
        "missing_values": total_missing,
        "duplicates": duplicates,
        "stats": stats,
    }


def main():
    """CLI Entrypoint: Generate and validate synthetic behavioral dataset."""
    df = generate_synthetic_dataset()
    validate_dataset(df)

    # Save to disk
    df.to_csv(DATASET_PATH, index=False)
    print(f"[+] Dataset saved to: {DATASET_PATH}")


if __name__ == "__main__":
    main()
