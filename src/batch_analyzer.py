"""
Batch Telemetry & Multi-File SOC Triage Analyzer
Processes multiple files or zipped telemetry dumps concurrently,
generates unified triage queues, and prioritizes critical threats for analysts.
"""

import io
import zipfile
from typing import Dict, Any, List, Tuple
import pandas as pd
import plotly.express as px

from src.predict import predict_csv
from src.static_scanner import inspect_executable_file
from src.mitre_attack import map_threat_to_mitre
from src.hybrid_rules import scan_with_yara_rules, compute_hybrid_fused_score


def process_batch_files(files_data: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    """
    Analyze a list of (filename, file_bytes) pairs in batch.
    Supports CSV telemetry logs, binary executables, scripts, and archives.
    """
    results = []

    # Unpack any ZIP archives in the list
    expanded_files: List[Tuple[str, bytes]] = []
    for name, b in files_data:
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(b)) as zf:
                    for member in zf.namelist():
                        if not member.endswith("/") and not member.startswith("__MACOSX"):
                            expanded_files.append((member, zf.read(member)))
            except Exception:
                expanded_files.append((name, b))
        else:
            expanded_files.append((name, b))

    for name, content in expanded_files:
        is_csv = name.lower().endswith(".csv")
        try:
            if is_csv:
                # Process behavioral CSV telemetry
                csv_df = pd.read_csv(io.BytesIO(content))
                pred = predict_csv(csv_df)
                ml_prob = pred["ransomware_probability"] / 100.0

                mitre_techs = map_threat_to_mitre(
                    telemetry_meta={"entropy_delta": 2.5 if pred["label"] == "RANSOMWARE" else 0.1, "file_write_rate": 80.0 if pred["label"] == "RANSOMWARE" else 1.0},
                    probability=ml_prob,
                )
                fused = compute_hybrid_fused_score(ml_prob, [], [])

                results.append({
                    "file_name": name,
                    "file_type": "CSV Behavioral Telemetry",
                    "file_size_kb": round(len(content) / 1024, 2),
                    "fused_score": fused["fused_score"],
                    "severity": fused["severity_tier"],
                    "ml_probability": round(ml_prob * 100.0, 2),
                    "primary_technique": mitre_techs[0]["name"] if mitre_techs else "N/A",
                    "mitre_id": mitre_techs[0]["id"] if mitre_techs else "N/A",
                    "content_bytes": content,
                })
            else:
                # Process binary / executable file statically
                static_res = inspect_executable_file(content, file_name=name)
                yara_matches = scan_with_yara_rules(content)
                mitre_techs = map_threat_to_mitre(static_indicators=static_res.get("indicators", []))

                ml_equiv_prob = (static_res["risk_score"] / 100.0) if static_res["verdict"] in ["KNOWN_MALICIOUS", "SUSPICIOUS"] else 0.05
                fused = compute_hybrid_fused_score(ml_equiv_prob, yara_matches, [])

                results.append({
                    "file_name": name,
                    "file_type": static_res.get("file_type", "Binary Data"),
                    "file_size_kb": round(len(content) / 1024, 2),
                    "fused_score": fused["fused_score"],
                    "severity": fused["severity_tier"],
                    "ml_probability": round(ml_equiv_prob * 100.0, 2),
                    "primary_technique": mitre_techs[0]["name"] if mitre_techs else "N/A",
                    "mitre_id": mitre_techs[0]["id"] if mitre_techs else "N/A",
                    "content_bytes": content,
                })
        except Exception as e:
            results.append({
                "file_name": name,
                "file_type": "Unparseable",
                "file_size_kb": round(len(content) / 1024, 2),
                "fused_score": 0.0,
                "severity": "ERROR",
                "ml_probability": 0.0,
                "primary_technique": "Error",
                "mitre_id": "N/A",
                "content_bytes": content,
            })

    # Sort triage queue: Critical first, then High, Medium, Low
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "ERROR": 4}
    results.sort(key=lambda r: (severity_order.get(r["severity"], 99), -r["fused_score"]))

    total = len(results)
    crit_count = sum(1 for r in results if r["severity"] == "CRITICAL")
    high_count = sum(1 for r in results if r["severity"] == "HIGH")
    med_count = sum(1 for r in results if r["severity"] == "MEDIUM")
    low_count = sum(1 for r in results if r["severity"] == "LOW")

    summary_df = pd.DataFrame([{
        "File Name": r["file_name"],
        "Type": r["file_type"],
        "Size (KB)": r["file_size_kb"],
        "Threat Score": f"{r['fused_score']:.1f}/100",
        "Severity Tier": r["severity"],
        "MITRE Technique": f"{r['mitre_id']}: {r['primary_technique']}",
    } for r in results])

    return {
        "total_files": total,
        "critical_count": crit_count,
        "high_count": high_count,
        "medium_count": med_count,
        "low_count": low_count,
        "items": results,
        "summary_table": summary_df,
    }

