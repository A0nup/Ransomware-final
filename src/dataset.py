"""
Synthetic Behavioral Telemetry Dataset Generator & Validator
Simulates endpoint telemetry for benign and ransomware-like processes.
Features dynamic, non-deterministic phase transitions, heterogeneous host profiles,
and realistic enterprise workload telemetry with continuous distribution overlaps.
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
    installers, backup maintenance, and media processing with continuous
    feature distribution overlap against ransomware patterns.
    """
    seq_data = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)

    # Workload archetypes: 35% desktop, 30% compiler builds, 20% backup archives, 15% media renamers
    workload = np.random.choice(["desktop", "heavy_build", "backup_archive", "media_rename"], p=[0.35, 0.30, 0.20, 0.15])
    burst_start = np.random.randint(2, 9) if workload != "desktop" else 99
    burst_len = np.random.randint(6, 16)
    burst_end = burst_start + burst_len

    base_cpu = host_profile["base_cpu"]
    base_mem = host_profile["base_mem"]
    base_net = host_profile["base_net"]

    for t in range(SEQ_LEN):
        noise = np.random.normal(0.0, 0.5)
        active = (burst_start <= t < burst_end)

        if workload == "backup_archive":
            # Legitimate backup (7-Zip, Borg, Veeam, Windows Backup): high entropy compression, VSS snapshot
            write_rate = np.random.uniform(70.0, 195.0) if active else np.random.uniform(5.0, 20.0)
            rename_rate = np.random.uniform(10.0, 45.0) if active else np.random.uniform(0.0, 2.0)
            delete_rate = np.random.uniform(2.0, 12.0) if active else 0.0
            entropy_delta = np.random.uniform(0.85, 2.05) if active else np.random.normal(0.02, 0.04)
            ext_change_rate = np.random.uniform(8.0, 38.0) if active else 0.0
            proc_create = np.random.uniform(2.0, 5.0) if active else 1.0
            shadow_cmd = 1 if (active and t in [burst_start, burst_start + 1] and np.random.rand() < 0.35) else 0
            backup_del = 1 if (active and t in [burst_start + 1, burst_start + 2] and np.random.rand() < 0.20) else 0
            cpu = np.clip(base_cpu + (np.random.uniform(30.0, 60.0) if active else 4.0), 5.0, 95.0)
            mem = np.clip(base_mem + (np.random.uniform(20.0, 42.0) if active else 4.0), 10.0, 95.0)
            net_conn = base_net + np.random.uniform(10.0, 32.0)
            file_open = np.random.uniform(75.0, 240.0) if active else 18.0
            dir_traversal = np.random.uniform(35.0, 90.0) if active else 12.0
            suspicious_api = np.random.uniform(3.5, 9.5) if active else 1.5
            unique_ext = np.random.randint(10, 25) if active else 4
            enc_ratio = np.random.uniform(0.20, 0.70) if active else 0.0
            failed_access = np.random.choice([0, 1, 2, 4], p=[0.45, 0.35, 0.15, 0.05])
            admin_act = 1 if (active and np.random.rand() < 0.65) else 0
            bytes_mb = np.random.uniform(30.0, 115.0) if active else 2.0
            unknown_proc = np.random.choice([0, 1, 2], p=[0.45, 0.40, 0.15])

        elif workload == "heavy_build":
            # Software compilation (npm, cargo, gcc, maven): high renames, temp files, directory traversal
            write_rate = np.random.uniform(45.0, 155.0) if active else np.random.uniform(5.0, 18.0)
            rename_rate = np.random.uniform(20.0, 75.0) if active else 0.0
            delete_rate = np.random.uniform(4.0, 25.0) if active else 0.0
            entropy_delta = np.random.uniform(0.25, 0.85) if active else np.random.normal(0.02, 0.04)
            ext_change_rate = np.random.uniform(14.0, 65.0) if active else 0.0
            proc_create = np.random.uniform(3.0, 7.0) if active else 1.0
            shadow_cmd = 0
            backup_del = 0
            cpu = np.clip(base_cpu + (np.random.uniform(35.0, 70.0) if active else 5.0), 5.0, 96.0)
            mem = np.clip(base_mem + (np.random.uniform(18.0, 38.0) if active else 5.0), 10.0, 95.0)
            net_conn = base_net + np.random.uniform(5.0, 25.0)
            file_open = np.random.uniform(55.0, 210.0) if active else 15.0
            dir_traversal = np.random.uniform(35.0, 92.0) if active else 10.0
            suspicious_api = np.random.uniform(3.0, 8.5) if active else 1.4
            unique_ext = np.random.randint(8, 25) if active else 3
            enc_ratio = np.random.uniform(0.02, 0.22) if active else 0.0
            failed_access = np.random.choice([0, 1, 3, 5], p=[0.45, 0.35, 0.15, 0.05])
            admin_act = 1 if (active and np.random.rand() < 0.25) else 0
            bytes_mb = np.random.uniform(18.0, 80.0) if active else 1.8
            unknown_proc = np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20])

        elif workload == "media_rename":
            # Photo/music organizer, tagger: mass renames, ext changes, compressed media entropy
            write_rate = np.random.uniform(50.0, 135.0) if active else np.random.uniform(4.0, 18.0)
            rename_rate = np.random.uniform(35.0, 95.0) if active else np.random.uniform(0.0, 3.0)
            delete_rate = np.random.uniform(1.0, 8.0) if active else 0.0
            entropy_delta = np.random.uniform(0.65, 1.50) if active else np.random.normal(0.03, 0.04)
            ext_change_rate = np.random.uniform(28.0, 88.0) if active else 0.0
            proc_create = np.random.uniform(1.2, 3.5)
            shadow_cmd = 0
            backup_del = 0
            cpu = np.clip(base_cpu + (np.random.uniform(20.0, 50.0) if active else 4.0), 5.0, 90.0)
            mem = np.clip(base_mem + 15.0, 10.0, 90.0)
            net_conn = base_net + 5.0
            file_open = np.random.uniform(50.0, 165.0) if active else 12.0
            dir_traversal = np.random.uniform(28.0, 80.0) if active else 8.0
            suspicious_api = np.random.uniform(2.5, 6.8) if active else 1.3
            unique_ext = np.random.randint(10, 24) if active else 3
            enc_ratio = np.random.uniform(0.06, 0.32) if active else 0.0
            fail = np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05])
            admin_act = 0
            bytes_mb = np.random.uniform(22.0, 80.0) if active else 1.5
            unknown_proc = np.random.choice([0, 1], p=[0.75, 0.25])
            failed_access = fail

        else:
            # Everyday benign desktop workload (browsing, office, streaming)
            write_rate = np.random.uniform(2.0, 22.0) + max(0.0, noise)
            rename_rate = np.random.choice([0, 1, 2], p=[0.72, 0.22, 0.06])
            delete_rate = np.random.choice([0, 1], p=[0.82, 0.18])
            entropy_delta = np.random.normal(0.01, 0.04)
            ext_change_rate = 0.0
            proc_create = np.random.poisson(lam=1.2)
            shadow_cmd = 0
            backup_del = 0
            cpu = np.clip(base_cpu + np.random.uniform(0.0, 12.0), 2.0, 80.0)
            mem = np.clip(base_mem + np.random.uniform(-2.0, 4.0), 5.0, 80.0)
            net_conn = base_net + np.random.uniform(0.0, 10.0)
            file_open = np.random.uniform(5.0, 32.0)
            dir_traversal = np.random.uniform(2.0, 14.0)
            suspicious_api = np.random.uniform(0.6, 2.6)
            unique_ext = np.random.randint(1, 6)
            enc_ratio = 0.0
            failed_access = np.random.choice([0, 1], p=[0.86, 0.14])
            admin_act = 0
            bytes_mb = np.random.uniform(0.2, 3.8)
            unknown_proc = np.random.choice([0, 1], p=[0.82, 0.18])

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
            np.clip(enc_ratio, 0.0, 1.0),
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
      Phase 1 (Reconnaissance): Directory traversal, file enumeration, API probes
      Phase 2 (Defense Evasion): Volume shadow copy deletion, catalog pruning, privilege actions
      Phase 3 (Bulk Encryption): High entropy, mass file writes, extension changes
      Phase 4 (Extortion/Impact): Ransom notes dropped, CPU surge, lockups
    Transitions are dynamically paced with randomized durations.
    """
    seq_data = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)

    # Attack types: 50% standard kill-chain, 25% fast burst, 15% stealthy throttled, 10% interrupted/stalled
    attack_type = np.random.choice(["standard", "fast", "stealthy", "interrupted"], p=[0.50, 0.25, 0.15, 0.10])

    recon_len = np.random.randint(1, 6)
    inhibit_len = np.random.randint(1, 4)
    enc_start = recon_len + inhibit_len

    base_cpu = host_profile["base_cpu"]
    base_mem = host_profile["base_mem"]
    base_net = host_profile["base_net"]

    for t in range(SEQ_LEN):
        noise = np.random.normal(0.0, 0.5)

        if attack_type == "standard":
            if t < recon_len:
                # Phase 1: Reconnaissance
                write_rate = np.random.uniform(3.0, 18.0)
                rename_rate = 0.0
                delete_rate = 0.0
                entropy_delta = np.random.normal(0.02, 0.04)
                ext_change_rate = 0.0
                proc_create = np.random.uniform(1.4, 3.8)
                shadow_cmd = 0
                backup_del = 0
                cpu = base_cpu + np.random.uniform(2.0, 8.0)
                mem = base_mem + np.random.uniform(2.0, 7.0)
                net_conn = base_net + np.random.uniform(5.0, 18.0)
                file_open = np.random.uniform(15.0, 50.0)
                dir_traversal = np.random.uniform(25.0, 70.0)
                suspicious_api = np.random.uniform(2.0, 5.0)
                unique_ext = np.random.randint(4, 9)
                enc_ratio = 0.0
                failed_access = np.random.choice([0, 1], p=[0.75, 0.25])
                admin_act = 0
                bytes_mb = np.random.uniform(0.5, 3.0)
                unknown_proc = np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20])

            elif t < enc_start:
                # Phase 2: Defense Evasion & Inhibition
                write_rate = np.random.uniform(14.0, 36.0)
                rename_rate = np.random.uniform(1.0, 4.0)
                delete_rate = np.random.uniform(1.0, 4.0)
                entropy_delta = np.random.uniform(0.10, 0.35)
                ext_change_rate = np.random.uniform(1.0, 4.0)
                proc_create = np.random.uniform(3.0, 6.0)
                shadow_cmd = np.random.choice([1, 2, 3], p=[0.45, 0.40, 0.15])
                backup_del = np.random.choice([1, 2], p=[0.55, 0.45])
                cpu = base_cpu + np.random.uniform(15.0, 30.0)
                mem = base_mem + np.random.uniform(10.0, 20.0)
                net_conn = base_net + np.random.uniform(10.0, 24.0)
                file_open = np.random.uniform(30.0, 80.0)
                dir_traversal = np.random.uniform(38.0, 85.0)
                suspicious_api = np.random.uniform(5.5, 11.5)
                unique_ext = np.random.randint(6, 15)
                enc_ratio = 0.04
                failed_access = np.random.choice([1, 2, 4], p=[0.40, 0.40, 0.20])
                admin_act = 1
                bytes_mb = np.random.uniform(3.0, 12.0)
                unknown_proc = np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20])

            else:
                # Phase 3 & 4: Bulk Encryption & Note Dropping
                progress = (t - enc_start) / max(1, (SEQ_LEN - enc_start))
                write_rate = np.random.uniform(75.0, 195.0) * (1.0 + 0.30 * progress) + max(0.0, noise * 2)
                rename_rate = np.random.uniform(28.0, 88.0) * (1.0 + 0.40 * progress)
                delete_rate = np.random.uniform(8.0, 36.0)
                entropy_delta = np.random.uniform(0.95, 2.05)
                ext_change_rate = np.random.uniform(26.0, 88.0)
                proc_create = np.random.uniform(1.0, 4.0)
                shadow_cmd = 0
                backup_del = 0
                cpu = np.clip(base_cpu + 35.0 + progress * 25.0, 30.0, 98.0)
                mem = np.clip(base_mem + 20.0 + progress * 20.0, 30.0, 95.0)
                net_conn = base_net + np.random.uniform(5.0, 18.0)
                file_open = write_rate * np.random.uniform(1.2, 1.8)
                dir_traversal = np.random.uniform(38.0, 98.0)
                suspicious_api = np.random.uniform(6.5, 17.0)
                unique_ext = np.random.randint(14, 30)
                enc_ratio = np.clip(0.30 + 0.65 * progress, 0.18, 0.98)
                failed_access = np.random.choice([2, 5, 8, 14], p=[0.30, 0.40, 0.20, 0.10])
                admin_act = 0
                bytes_mb = np.random.uniform(30.0, 115.0)
                unknown_proc = np.random.choice([0, 1, 2], p=[0.30, 0.50, 0.20])

        elif attack_type == "fast":
            if t < 2:
                write_rate, rename_rate, delete_rate, entropy_delta = 28.0, 2.0, 2.0, 0.20
                ext_change_rate, proc_create, shadow_cmd, backup_del = 2.0, 4.0, 2, 1
                cpu, mem, net_conn = base_cpu + 25.0, base_mem + 15.0, base_net + 15.0
                file_open, dir_traversal, suspicious_api, unique_ext = 55.0, 50.0, 9.0, 8
                enc_ratio, failed_access, admin_act, bytes_mb, unknown_proc = 0.08, 3, 1, 12.0, 1
            else:
                write_rate = np.random.uniform(105.0, 230.0)
                rename_rate = np.random.uniform(45.0, 115.0)
                delete_rate = np.random.uniform(14.0, 48.0)
                entropy_delta = np.random.uniform(1.10, 2.20)
                ext_change_rate = np.random.uniform(45.0, 115.0)
                proc_create = 2.0
                shadow_cmd = 0
                backup_del = 0
                cpu = np.random.uniform(70.0, 99.0)
                mem = np.random.uniform(50.0, 92.0)
                net_conn = base_net + 10.0
                file_open = np.random.uniform(150.0, 310.0)
                dir_traversal = np.random.uniform(50.0, 98.0)
                suspicious_api = np.random.uniform(8.5, 19.5)
                unique_ext = np.random.randint(16, 32)
                enc_ratio = np.clip(0.38 + 0.03 * t, 0.28, 0.99)
                failed_access = np.random.randint(3, 14)
                admin_act = 0
                bytes_mb = np.random.uniform(55.0, 140.0)
                unknown_proc = np.random.choice([0, 1, 2], p=[0.20, 0.60, 0.20])

        elif attack_type == "stealthy":
            # Throttled low-and-slow: lower write rate, subtle partial encryption, LOLBins execution
            enc_active = (t >= 7)
            write_rate = np.random.uniform(22.0, 60.0) if enc_active else np.random.uniform(4.0, 18.0)
            rename_rate = np.random.uniform(8.0, 26.0) if enc_active else 0.0
            delete_rate = np.random.uniform(2.0, 7.0) if enc_active else 0.0
            entropy_delta = np.random.uniform(0.38, 0.85) if enc_active else 0.03
            ext_change_rate = np.random.uniform(8.0, 24.0) if enc_active else 0.0
            proc_create = np.random.uniform(1.5, 3.2)
            shadow_cmd = 1 if (t == 4 and np.random.rand() < 0.35) else 0
            backup_del = 0
            cpu = base_cpu + (np.random.uniform(16.0, 35.0) if enc_active else 5.0)
            mem = base_mem + 12.0
            net_conn = base_net + 8.0
            file_open = np.random.uniform(30.0, 75.0) if enc_active else 15.0
            dir_traversal = np.random.uniform(25.0, 55.0)
            suspicious_api = np.random.uniform(3.0, 7.0) if enc_active else 1.6
            unique_ext = np.random.randint(6, 16) if enc_active else 4
            enc_ratio = np.random.uniform(0.12, 0.42) if enc_active else 0.0
            failed_access = np.random.choice([0, 1, 2], p=[0.55, 0.30, 0.15])
            admin_act = 1 if (t == 4 and np.random.rand() < 0.35) else 0
            bytes_mb = np.random.uniform(10.0, 32.0) if enc_active else 2.0
            unknown_proc = np.random.choice([0, 1], p=[0.75, 0.25]) # LOLBins (known process binaries)

        else:
            # Interrupted / stalled ransomware
            write_rate = np.random.uniform(8.0, 35.0)
            rename_rate = np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])
            delete_rate = 0.0
            entropy_delta = np.random.uniform(0.08, 0.35)
            ext_change_rate = np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])
            proc_create = 2.0
            shadow_cmd = 1 if (t == 4 and np.random.rand() < 0.55) else 0
            backup_del = 0
            cpu = base_cpu + 15.0
            mem = base_mem + 10.0
            net_conn = base_net + 10.0
            file_open = np.random.uniform(20.0, 60.0)
            dir_traversal = np.random.uniform(25.0, 65.0)
            api = np.random.uniform(3.0, 7.5)
            suspicious_api = api
            unique_ext = np.random.randint(4, 10)
            enc_ratio = 0.08
            failed_access = np.random.choice([1, 2, 4], p=[0.40, 0.40, 0.20])
            admin_act = 1 if t == 4 else 0
            bytes_mb = 4.5
            unknown_proc = np.random.choice([0, 1, 2], p=[0.35, 0.50, 0.15])

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
            np.clip(enc_ratio, 0.0, 1.0),
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
) -> pd.DataFrame:
    """
    Generate balanced synthetic behavioral telemetry dataset with realistic,
    continuous, and heterogeneous endpoint dynamics.
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
