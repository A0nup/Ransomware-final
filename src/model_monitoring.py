"""
Model Monitoring, Concept Drift Tracking & Analyst Feedback Loop Module
Tracks live inference telemetry, calculates distribution shift (drift),
and captures human-in-the-loop analyst feedback for continuous retraining.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


MONITORING_DIR = Path("data/model_monitoring")
MONITORING_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_FILE = MONITORING_DIR / "prediction_audit.jsonl"
FEEDBACK_LOG_FILE = MONITORING_DIR / "analyst_feedback.json"


def record_prediction_audit(
    sample_id: str,
    probability: float,
    risk_level: str,
    top_feature: str,
    input_vector_summary: Optional[List[float]] = None,
) -> None:
    """Record a model prediction entry to local audit log on localhost."""
    record = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sample_id": sample_id,
        "probability": round(probability * 100.0, 2),
        "risk_level": risk_level,
        "top_feature": top_feature,
        "input_summary": input_vector_summary or [],
    }
    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def record_analyst_feedback(
    sample_id: str,
    predicted_probability: float,
    analyst_verdict: str,  # "CORRECT", "FALSE_POSITIVE", "FALSE_NEGATIVE"
    reason_tag: str = "Standard Review",
    notes: str = "",
) -> Dict[str, Any]:
    """Capture analyst human-in-the-loop feedback for retraining."""
    feedback_entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sample_id": sample_id,
        "predicted_probability": round(predicted_probability, 2),
        "analyst_verdict": analyst_verdict,
        "reason_tag": reason_tag,
        "notes": notes,
    }

    feedback_list = []
    if FEEDBACK_LOG_FILE.exists():
        try:
            with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
        except Exception:
            feedback_list = []

    feedback_list.append(feedback_entry)
    with open(FEEDBACK_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_list, f, indent=2)

    return {"status": "SUCCESS", "total_feedback_entries": len(feedback_list)}


def get_drift_and_monitoring_metrics() -> Dict[str, Any]:
    """Calculate rolling metrics, distribution shifts, and analyst feedback stats."""
    records = []
    if AUDIT_LOG_FILE.exists():
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception:
            pass

    feedback = []
    if FEEDBACK_LOG_FILE.exists():
        try:
            with open(FEEDBACK_LOG_FILE, "r", encoding="utf-8") as f:
                feedback = json.load(f)
        except Exception:
            pass

    total_preds = len(records)
    if total_preds == 0:
        return {
            "total_predictions": 0,
            "mean_probability": 0.0,
            "drift_status": "STABLE_INSUFFICIENT_DATA",
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "analyst_feedback_count": len(feedback),
            "false_positive_count": 0,
            "false_negative_count": 0,
            "retraining_queue_size": len(feedback),
            "timeline_df": pd.DataFrame(),
        }

    df = pd.DataFrame(records)
    probs = df["probability"].tolist()

    recent_probs = probs[-50:]
    baseline_mean = 4.5  # Expected nominal host baseline
    rolling_mean = float(np.mean(recent_probs))

    # Detect concept drift (if rolling mean shifts significantly from baseline)
    if rolling_mean > 25.0:
        drift_status = "DRIFT_ALERT_HIGH_THREAT_ACTIVITY"
    elif abs(rolling_mean - baseline_mean) > 10.0:
        drift_status = "MODERATE_SHIFT_DETECTED"
    else:
        drift_status = "STABLE_NOMINAL_BASELINE"

    # Feedback breakdown
    fp_count = sum(1 for fb in feedback if fb.get("analyst_verdict") == "FALSE_POSITIVE")
    fn_count = sum(1 for fb in feedback if fb.get("analyst_verdict") == "FALSE_NEGATIVE")
    correct_count = sum(1 for fb in feedback if fb.get("analyst_verdict") == "CORRECT")

    return {
        "total_predictions": total_preds,
        "rolling_mean_probability": round(rolling_mean, 2),
        "drift_status": drift_status,
        "critical_count": sum(1 for p in probs if p >= 90.0),
        "high_count": sum(1 for p in probs if 70.0 <= p < 90.0),
        "medium_count": sum(1 for p in probs if 30.0 <= p < 70.0),
        "low_count": sum(1 for p in probs if p < 30.0),
        "analyst_feedback_count": len(feedback),
        "false_positive_count": fp_count,
        "false_negative_count": fn_count,
        "correct_count": correct_count,
        "retraining_queue_size": len(feedback),
        "recent_audit_records": records[-10:],
    }
