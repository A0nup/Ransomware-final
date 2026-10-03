"""
Unit Tests for Scientific Validation & Model Audit Module
Verifies diagnostic audit of synthetic baseline saturation, realistic benchmark trade-offs,
and resource profiling.
"""

import pytest
from pathlib import Path
import numpy as np

from src.scientific_validation import (
    audit_synthetic_dataset_and_saturation,
    generate_realistic_noisy_benchmark,
    profile_runtime_performance,
    AUDIT_REPORT_PATH,
)
from src.config import MODEL_CHECKPOINT_PATH, SCALER_PATH, DEVICE
from src.preprocessing import load_scaler
from src.evaluate import load_trained_model


def test_synthetic_saturation_audit():
    audit = audit_synthetic_dataset_and_saturation()
    assert audit["total_test_sequences"] == 750
    p_dist = audit["probability_distribution"]
    assert p_dist["count_between_0_10_and_0_90"] == 0, "Confirms zero test samples in mid probability range"
    assert len(audit["fp_sequence_ids"]) == 16, "Confirms exactly 16 overlapping synthetic false positives"
    assert len(audit["fn_sequence_ids"]) == 6, "Confirms exactly 6 overlapping synthetic false negatives"


def test_realistic_benchmark_generation():
    X_real, y_real = generate_realistic_noisy_benchmark(n_sequences=50)
    assert X_real.shape == (50, 20, 20)
    assert y_real.shape == (50,)
    assert set(np.unique(y_real)).issubset({0.0, 1.0})


def test_runtime_profiling():
    scaler = load_scaler(SCALER_PATH)
    model = load_trained_model(MODEL_CHECKPOINT_PATH, device=DEVICE)
    stats = profile_runtime_performance(model, scaler)
    assert stats["mean_latency_ms"] > 0
    assert stats["mean_latency_ms"] < 50.0, "CPU inference must be under 50ms per sequence"
    assert stats["resident_memory_mb"] > 0
