"""
Classification Threshold Sensitivity Analysis Module
Evaluates detection thresholds from 0.10 to 0.90 to analyze precision-recall
and FPR/FNR trade-offs for operational SOC / cybersecurity deployment.
Saves results to reports/threshold_analysis.csv.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.config import (
    SEED,
    BATCH_SIZE,
    THRESHOLDS,
    THRESHOLD_ANALYSIS_PATH,
    MODEL_CHECKPOINT_PATH,
    SCALER_PATH,
    DEVICE,
)
from src.preprocessing import (
    load_raw_dataset,
    split_sequences,
    load_scaler,
    get_dataloaders,
)
from src.evaluate import load_trained_model, get_predictions, compute_metrics


def run_threshold_analysis() -> pd.DataFrame:
    """Analyze model behavior across variable classification thresholds."""
    print("=" * 60)
    print("CLASSIFICATION THRESHOLD SENSITIVITY ANALYSIS")
    print(f"Evaluating Thresholds: {THRESHOLDS}")
    print("=" * 60)

    # Load data
    df = load_raw_dataset()
    train_df, val_df, test_df = split_sequences(df, seed=SEED)
    scaler = load_scaler(SCALER_PATH)

    _, _, test_loader, _ = get_dataloaders(
        train_df, val_df, test_df, scaler, batch_size=BATCH_SIZE
    )

    # Load model & get test probabilities
    model = load_trained_model(MODEL_CHECKPOINT_PATH, device=DEVICE)
    probs, targets, _ = get_predictions(model, test_loader, device=DEVICE)

    records: List[Dict[str, Any]] = []

    for t in THRESHOLDS:
        m = compute_metrics(targets, probs, threshold=t)
        records.append({
            "Threshold": round(t, 2),
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4),
            "Specificity": round(m["specificity"], 4),
            "FPR": round(m["fpr"], 4),
            "FNR": round(m["fnr"], 4),
            "TP": m["tp"],
            "FP": m["fp"],
            "TN": m["tn"],
            "FN": m["fn"],
        })

    thresh_df = pd.DataFrame(records)
    thresh_df.to_csv(THRESHOLD_ANALYSIS_PATH, index=False)
    print(f"[+] Threshold analysis saved to: {THRESHOLD_ANALYSIS_PATH}\n")
    print(thresh_df.to_string(index=False))

    return thresh_df


if __name__ == "__main__":
    run_threshold_analysis()
