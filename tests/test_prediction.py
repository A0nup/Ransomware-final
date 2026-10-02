"""
Unit Tests: Prediction Pipeline & CLI Compatibility
"""

import pytest
import pandas as pd
from pathlib import Path

from src.config import SAMPLE_BENIGN_PATH, SAMPLE_RANSOMWARE_PATH, SEQ_LEN
from src.predict import predict_csv, determine_risk_level


def test_predict_sample_benign():
    """Verify inference on sample_benign.csv returns structured benign result."""
    assert SAMPLE_BENIGN_PATH.exists()
    res = predict_csv(SAMPLE_BENIGN_PATH, threshold=0.50)

    assert isinstance(res, dict)
    assert res["label"] in ["BENIGN", "RANSOMWARE"]
    assert 0.0 <= res["ransomware_probability"] <= 100.0
    assert 50.0 <= res["confidence"] <= 100.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(res["attention_weights"]) == SEQ_LEN
    assert res["label"] == "BENIGN"


def test_predict_sample_ransomware():
    """Verify inference on sample_ransomware_like.csv returns ransomware detection."""
    assert SAMPLE_RANSOMWARE_PATH.exists()
    res = predict_csv(SAMPLE_RANSOMWARE_PATH, threshold=0.50)

    assert isinstance(res, dict)
    assert res["label"] == "RANSOMWARE"
    assert res["ransomware_probability"] > 50.0
    assert res["risk_level"] in ["MEDIUM", "HIGH"]
    assert len(res["attention_weights"]) == SEQ_LEN


def test_predict_custom_threshold():
    """Verify prediction with conservative and aggressive thresholds."""
    # Ultra-high threshold
    res_high = predict_csv(SAMPLE_BENIGN_PATH, threshold=0.95)
    assert res_high["threshold"] == 95.0
    assert res_high["label"] == "BENIGN"

    # Ultra-low threshold
    res_low = predict_csv(SAMPLE_RANSOMWARE_PATH, threshold=0.10)
    assert res_low["threshold"] == 10.0
    assert res_low["label"] == "RANSOMWARE"


def test_determine_risk_level():
    """Verify risk badge categorization boundaries."""
    assert determine_risk_level(0.15) == "LOW"
    assert determine_risk_level(0.50) == "MEDIUM"
    assert determine_risk_level(0.85) == "HIGH"
