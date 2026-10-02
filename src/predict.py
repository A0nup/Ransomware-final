"""
Prediction & Inference API Module
Provides programmatic and command-line inference on behavioral telemetry CSV files.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import torch

from src.config import (
    MODEL_CHECKPOINT_PATH,
    SCALER_PATH,
    DEFAULT_THRESHOLD,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    DEVICE,
    SEQ_LEN,
    FEATURES,
)
from src.preprocessing import load_scaler, prepare_single_sequence_for_inference
from src.evaluate import load_trained_model
from src.explainability import extract_sample_attention


# Cached model & scaler
_MODEL = None
_SCALER = None


def get_inference_resources():
    """Load and cache model and scaler for fast repeated inference."""
    global _MODEL, _SCALER
    if _SCALER is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler artifact not found at {SCALER_PATH}. Run 'python -m src.train' first."
            )
        _SCALER = load_scaler(SCALER_PATH)

    if _MODEL is None:
        if not MODEL_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {MODEL_CHECKPOINT_PATH}. Run 'python -m src.train' first."
            )
        _MODEL = load_trained_model(MODEL_CHECKPOINT_PATH, device=DEVICE)

    return _MODEL, _SCALER


def determine_risk_level(prob: float) -> str:
    """Categorize application-level risk presentation category."""
    if prob < RISK_THRESHOLD_LOW:
        return "LOW"
    elif prob < RISK_THRESHOLD_HIGH:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_csv(
    data_source: Union[str, Path, pd.DataFrame],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Predict whether a behavioral telemetry sequence is BENIGN or RANSOMWARE.

    Parameters:
      data_source: File path to a CSV or an existing pandas DataFrame (minimum 20 rows).
      threshold: Decision threshold for positive classification (default=0.50).

    Returns:
      Dict with keys:
        - label: 'BENIGN' or 'RANSOMWARE'
        - ransomware_probability: float (0.0 to 100.0)
        - confidence: float (0.0 to 100.0)
        - threshold: float
        - risk_level: 'LOW', 'MEDIUM', 'HIGH'
        - attention_weights: list of 20 floats
        - sequence_df: formatted DataFrame of the evaluated sequence
    """
    model, scaler = get_inference_resources()

    if isinstance(data_source, (str, Path)):
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
    else:
        raise TypeError(f"Unsupported data source type: {type(data_source)}")

    # Preprocess sequence tensor (1, 20, 20)
    tensor, sequence_df = prepare_single_sequence_for_inference(df, scaler, seq_len=SEQ_LEN)
    tensor = tensor.to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits, attn = model(tensor)
        prob = torch.sigmoid(logits).item()

    is_ransomware = prob >= threshold
    label = "RANSOMWARE" if is_ransomware else "BENIGN"

    # Confidence: distance from threshold normalized
    if is_ransomware:
        confidence = (prob - threshold) / (1.0 - threshold) if threshold < 1.0 else 1.0
    else:
        confidence = (threshold - prob) / threshold if threshold > 0.0 else 1.0
    confidence = float(np.clip(confidence * 100.0, 50.0, 100.0))

    attn_weights = attn.squeeze(0).cpu().numpy().tolist() if attn is not None else [0.05] * SEQ_LEN
    risk_level = determine_risk_level(prob)

    return {
        "label": label,
        "ransomware_probability": round(prob * 100.0, 2),
        "raw_probability": float(prob),
        "confidence": round(confidence, 2),
        "threshold": round(threshold * 100.0, 1),
        "risk_level": risk_level,
        "attention_weights": attn_weights,
        "sequence_df": sequence_df,
    }


def main():
    """CLI Entrypoint for behavioral telemetry prediction."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_telemetry_csv> [threshold=0.5]")
        print("Example: python -m src.predict data/sample_benign.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD

    try:
        res = predict_csv(csv_path, threshold=threshold)
        print("\n" + "=" * 40)
        print("RANSOMWARE BEHAVIORAL DETECTION")
        print("=" * 40)
        print(f"\nClassification:          {res['label']}")
        print(f"Ransomware Probability:  {res['ransomware_probability']:.2f}%")
        print(f"Confidence:              {res['confidence']:.2f}%")
        print(f"Threshold:               {res['threshold']:.1f}%")
        print(f"Risk Indicator:          {res['risk_level']}")
        print("\nNotice:")
        print("This result is a machine-learning classification of behavioral telemetry")
        print("and is not, by itself, proof of malware infection.\n")
    except Exception as e:
        print(f"\n[!] Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
