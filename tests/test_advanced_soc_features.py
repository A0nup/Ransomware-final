"""
Unit & Integration Tests for Advanced SOC-Grade Capabilities:
Explainability, MITRE ATT&CK Mapping, IOC Extraction, Threat Intel,
Entropy Profiling, PDF Report Generation, Model Monitoring, Hybrid Rules, Batch Analysis.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.explainability import compute_sample_attribution, create_attribution_bar_chart
from src.mitre_attack import map_threat_to_mitre, MITRE_TECHNIQUES_DATABASE
from src.ioc_extractor import extract_iocs_from_data, export_iocs_to_stix, export_iocs_to_csv
from src.threat_intel import query_threat_intel_hash, query_ip_reputation
from src.entropy_analyzer import compute_sliding_block_entropy, create_entropy_profile_chart
from src.report_generator import generate_pdf_incident_report
from src.model_monitoring import record_prediction_audit, record_analyst_feedback, get_drift_and_monitoring_metrics
from src.hybrid_rules import scan_with_yara_rules, scan_with_sigma_rules, compute_hybrid_fused_score
from src.batch_analyzer import process_batch_files
from src.evaluate import load_trained_model
from src.preprocessing import load_scaler
from src.config import MODEL_CHECKPOINT_PATH, SCALER_PATH, FEATURES, SEQ_LEN


def test_explainability_attribution():
    model = load_trained_model(MODEL_CHECKPOINT_PATH, device="cpu")
    scaler = load_scaler(SCALER_PATH)

    dummy_seq = np.zeros((SEQ_LEN, len(FEATURES)), dtype=np.float32)
    dummy_seq[:, 0] = 120.0  # high write rate
    dummy_seq[:, 3] = 3.5    # high entropy delta
    dummy_seq[:, 6] = 1.0    # shadow copy deletion

    res = compute_sample_attribution(model, scaler, dummy_seq)
    assert "probability" in res
    assert "contributions" in res
    assert len(res["contributions"]) == len(FEATURES)
    assert "smoking_gun" in res
    assert len(res["smoking_gun"]) > 0

    fig = create_attribution_bar_chart(res)
    assert fig is not None


def test_mitre_attack_mapping():
    meta = {
        "entropy_delta": 3.8,
        "file_write_rate": 150.0,
        "shadow_cmd": 1.0,
        "culprit_process": {"cmd": "vssadmin delete shadows /all /quiet"},
    }
    mitre_hits = map_threat_to_mitre(telemetry_meta=meta, probability=0.98)
    tech_ids = [m["id"] for m in mitre_hits]
    assert "T1486" in tech_ids  # Data Encrypted for Impact
    assert "T1490" in tech_ids  # Inhibit System Recovery


def test_ioc_extraction_and_stix():
    sample_text = (
        "Malware dropped payload communicating with 185.220.101.5 and c2.darknet-onion.com. "
        "Files were renamed to .locked. Registry key modified: HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\Locker"
    )
    iocs = extract_iocs_from_data(sample_text, file_name="sample_ransom.exe")
    assert "185.220.101.5" in iocs["network"]["ipv4"]
    assert ".locked" in iocs["host"]["ransom_extensions"]
    assert len(iocs["host"]["registry_keys"]) > 0
    assert "sha256" in iocs["hashes"]

    stix_json = export_iocs_to_stix(iocs, threat_name="Test Threat")
    assert "bundle--" in stix_json
    assert "indicator--" in stix_json

    csv_data = export_iocs_to_csv(iocs)
    assert "indicator" in csv_data
    assert "185.220.101.5" in csv_data


def test_threat_intel_enrichment():
    # Test known cached hash (WannaCry)
    wc_hash = "ed01ebf83334a19373140995c13e6b1510b4f266041f0e77d33c4f131dafa37a"
    res = query_threat_intel_hash(wc_hash)
    assert "WannaCry" in res["family"]
    assert res["positives"] > 50

    # Test unknown hash
    unknown_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    res_unk = query_threat_intel_hash(unknown_hash)
    assert res_unk["category"] == "Zero-Day or Unindexed"

    # Test IP query
    ip_res = query_ip_reputation("192.168.1.1")
    assert ip_res["reputation"] == "LOCAL_PRIVATE"


def test_sliding_block_entropy_analyzer():
    # Low entropy data (repeating pattern)
    low_ent_bytes = b"AAAA" * 1000
    res_low = compute_sliding_block_entropy(low_ent_bytes)
    assert res_low["overall_entropy"] < 2.0
    assert res_low["packing_verdict"] == "STANDARD_UNCOMPRESSED_DATA"

    # High entropy data (random bytes simulating AES ciphertext)
    import os
    random_bytes = os.urandom(8192)
    res_high = compute_sliding_block_entropy(random_bytes)
    assert res_high["overall_entropy"] >= 7.5
    assert res_high["packing_verdict"] == "HEAVILY_ENCRYPTED_OR_PACKED"

    fig = create_entropy_profile_chart(res_high, file_name="random_test.bin")
    assert fig is not None


def test_pdf_report_generation():
    iocs = {
        "file_name": "malicious_ransomware.exe",
        "hashes": {"sha256": "abc1234567890", "md5": "def12345"},
        "network": {"ipv4": ["185.220.101.5"], "domains": ["evil.com"]},
    }
    mitre_techs = [
        {"id": "T1486", "name": "Data Encrypted for Impact", "tactic": "Impact", "confidence": "HIGH"},
        {"id": "T1490", "name": "Inhibit System Recovery", "tactic": "Impact", "confidence": "CRITICAL"},
    ]
    top_feats = [
        {"feature": "entropy_delta", "readable_name": "Entropy Delta", "raw_value": 3.8, "contribution_pct": 35.0, "direction": "MALICIOUS"},
    ]

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        pdf_path = tmp_pdf.name

    pdf_bytes = generate_pdf_incident_report(
        incident_id="INC-TEST-001",
        threat_verdict="RANSOMWARE_CONFIRMED",
        probability=98.7,
        mitre_techniques=mitre_techs,
        top_features=top_feats,
        iocs=iocs,
        culprit_process={"name": "test_ransom.exe", "pid": 4321, "cmd": "vssadmin delete shadows"},
        output_path=pdf_path,
    )
    assert Path(pdf_path).exists()
    assert Path(pdf_path).stat().st_size > 500
    assert len(pdf_bytes) > 500
    Path(pdf_path).unlink(missing_ok=True)


def test_model_monitoring_and_feedback():
    record_prediction_audit(
        sample_id="SAMPLE-TEST-99",
        probability=0.985,
        risk_level="CRITICAL",
        top_feature="entropy_delta",
    )
    fb_res = record_analyst_feedback(
        sample_id="SAMPLE-TEST-99",
        predicted_probability=98.5,
        analyst_verdict="CORRECT",
        reason_tag="Verified ransomware encryption",
    )
    assert fb_res["status"] == "SUCCESS"

    metrics = get_drift_and_monitoring_metrics()
    assert metrics["total_predictions"] >= 1
    assert "drift_status" in metrics


def test_hybrid_rules_engine():
    # YARA rules
    ransom_note = b"ALL YOUR FILES ARE ENCRYPTED! Send 0.5 bitcoin to our address to decrypt your files. Visit our tor browser onion link."
    yara_hits = scan_with_yara_rules(ransom_note)
    rule_ids = [r["rule_id"] for r in yara_hits]
    assert "YARA_RANSOM_NOTE_01" in rule_ids

    # Sigma rules
    sigma_meta = {
        "shadow_cmd": 1.0,
        "culprit_process": {"cmd": "vssadmin delete shadows /all /quiet"},
    }
    sigma_hits = scan_with_sigma_rules(sigma_meta)
    sig_ids = [s["rule_id"] for s in sigma_hits]
    assert "SIGMA_VSSADMIN_DELETE" in sig_ids

    # Hybrid Score Fusion
    fused = compute_hybrid_fused_score(ml_probability=0.60, yara_matches=yara_hits, sigma_matches=sigma_hits)
    assert fused["fused_score"] >= 90.0  # Critical rule fired, override floor applied
    assert fused["severity_tier"] == "CRITICAL"


def test_batch_analysis_engine():
    # 1. Benign small file
    clean_bytes = b"Hello, this is a normal test document."
    # 2. Ransom note file
    note_bytes = b"All your files have been encrypted! Send bitcoin to decrypt."

    batch_input = [
        ("clean_doc.txt", clean_bytes),
        ("ransom_instructions.txt", note_bytes),
    ]

    batch_res = process_batch_files(batch_input)
    assert batch_res["total_files"] == 2
    assert len(batch_res["items"]) == 2
    assert not batch_res["summary_table"].empty
