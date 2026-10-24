"""
Hybrid Detection Engine: Machine Learning + YARA Rules + Sigma Rules
Combines deep neural network inference with signature-based (YARA) and
behavioral event log rules (Sigma) to prevent zero-day blind spots.
"""

import re
from typing import Dict, Any, List, Tuple, Optional


# Embedded YARA-Equivalent Rules
YARA_RULES = [
    {
        "id": "YARA_RANSOM_NOTE_01",
        "name": "Ransomware Note Indicators",
        "category": "Ransom Message",
        "severity": "CRITICAL",
        "weight": 35.0,
        "patterns": [
            r"your files have been encrypted",
            r"all your files are encrypted",
            r"how to decrypt your files",
            r"send \d+(\.\d+)? (bitcoin|btc|monero)",
            r"tor browser (link|address|\.onion)",
            r"restore-my-files",
            r"read_me_for_decrypt",
        ],
    },
    {
        "id": "YARA_CRYPTO_APIS_02",
        "name": "Cryptographic Encryption APIs",
        "category": "API Imports",
        "severity": "HIGH",
        "weight": 25.0,
        "patterns": [
            r"CryptEncrypt",
            r"BCryptEncrypt",
            r"CryptGenRandom",
            r"CryptAcquireContext",
            r"CryptImportKey",
            r"BCryptGenerateSymmetricKey",
        ],
    },
    {
        "id": "YARA_SHADOW_SABOTAGE_03",
        "name": "Volume Shadow Sabotage Strings",
        "category": "System Sabotage",
        "severity": "CRITICAL",
        "weight": 40.0,
        "patterns": [
            r"vssadmin\s+delete\s+shadows",
            r"bcdedit\s+/set\s+(\{[^}]+\}\s+)?recoveryenabled\s+no",
            r"wbadmin\s+delete\s+catalog",
            r"wmic\s+shadowcopy\s+delete",
        ],
    },
    {
        "id": "YARA_KNOWN_RANSOM_FAMILIES_04",
        "name": "Known Ransomware Brand Signatures",
        "category": "Threat Family",
        "severity": "CRITICAL",
        "weight": 40.0,
        "patterns": [
            r"LockBit 3\.0",
            r"LockBit Black",
            r"WanaCrypt0r",
            r"Wannacry",
            r"Conti Locker",
            r"BlackCat Ransomware",
            r"ALPHV",
        ],
    },
    {
        "id": "YARA_EICAR_STANDARD_05",
        "name": "EICAR Standard Test Pattern",
        "category": "Test Signature",
        "severity": "HIGH",
        "weight": 30.0,
        "patterns": [
            r"X5O!P%@AP\[4\\PZX54\(P\^\)7CC\)7\}\$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!\$H\+H\*",
        ],
    },
]

# Embedded Sigma Behavioral Rules
SIGMA_RULES = [
    {
        "id": "SIGMA_VSSADMIN_DELETE",
        "title": "Inhibit System Recovery via Vssadmin",
        "severity": "CRITICAL",
        "weight": 40.0,
        "mitre_id": "T1490",
        "condition": lambda m: (
            float(m.get("shadow_cmd", 0)) > 0
            or "vssadmin" in str(m.get("culprit_process", {}).get("cmd", "")).lower()
            or "bcdedit" in str(m.get("culprit_process", {}).get("cmd", "")).lower()
        ),
        "description": "Detects command execution attempting to delete local volume shadow copy snapshots.",
    },
    {
        "id": "SIGMA_RAPID_MASS_ENCRYPT",
        "title": "Anomalous Mass Encryption Burst",
        "severity": "HIGH",
        "weight": 35.0,
        "mitre_id": "T1486",
        "condition": lambda m: (
            float(m.get("entropy_delta", 0.0)) >= 1.5
            and float(m.get("file_write_rate", 0.0)) >= 25.0
        ),
        "description": "Detects concurrent spike in Shannon entropy with high file write operations per second.",
    },
    {
        "id": "SIGMA_MASS_EXTENSION_RENAME",
        "title": "Rapid Batch File Extension Tampering",
        "severity": "HIGH",
        "weight": 30.0,
        "mitre_id": "T1486",
        "condition": lambda m: float(m.get("extension_changes", 0.0)) >= 5.0,
        "description": "Detects multiple sequential file rename operations altering original extensions to ransomware extensions.",
    },
    {
        "id": "SIGMA_EVENT_LOG_CLEARING",
        "title": "Security Event Log Clearing",
        "severity": "HIGH",
        "weight": 25.0,
        "mitre_id": "T1070.001",
        "condition": lambda m: "wevtutil" in str(m.get("culprit_process", {}).get("cmd", "")).lower(),
        "description": "Detects execution of wevtutil attempting to clear security or system event logs.",
    },
]


def scan_with_yara_rules(data_or_text: Any) -> List[Dict[str, Any]]:
    """Scan raw bytes or text against YARA ransomware rules."""
    if isinstance(data_or_text, bytes):
        text = data_or_text.decode("latin-1", errors="ignore")
    else:
        text = str(data_or_text)

    matches = []
    for rule in YARA_RULES:
        matched_patterns = []
        for pat in rule["patterns"]:
            if re.search(pat, text, re.IGNORECASE):
                matched_patterns.append(pat)
        if matched_patterns:
            matches.append({
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "category": rule["category"],
                "severity": rule["severity"],
                "weight": rule["weight"],
                "matched_patterns": matched_patterns,
            })
    return matches


def scan_with_sigma_rules(telemetry_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate live host telemetry and process meta against Sigma behavioral rules."""
    matches = []
    for rule in SIGMA_RULES:
        try:
            if rule["condition"](telemetry_meta):
                matches.append({
                    "rule_id": rule["id"],
                    "title": rule["title"],
                    "severity": rule["severity"],
                    "weight": rule["weight"],
                    "mitre_id": rule["mitre_id"],
                    "description": rule["description"],
                })
        except Exception:
            pass
    return matches


def compute_hybrid_fused_score(
    ml_probability: float,  # 0.0 to 1.0
    yara_matches: List[Dict[str, Any]],
    sigma_matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fuse Machine Learning, YARA signatures, and Sigma behavioral rules into a composite SOC score.
    Ensures that clear rule matches override ML false-negatives, while ML catches novel zero-days.
    """
    ml_score = ml_probability * 100.0

    yara_score = min(100.0, sum(m.get("weight", 20.0) for m in yara_matches))
    sigma_score = min(100.0, sum(m.get("weight", 25.0) for m in sigma_matches))

    # Composite weighted fusion
    fused_score = 0.50 * ml_score + 0.30 * yara_score + 0.20 * sigma_score

    # Hard overrides: Critical rule matches set floor dynamically
    has_critical_yara = any(m.get("severity") == "CRITICAL" for m in yara_matches)
    has_critical_sigma = any(m.get("severity") == "CRITICAL" for m in sigma_matches)

    if has_critical_yara or has_critical_sigma:
        dynamic_floor = 90.0 + min(6.5, (yara_score + sigma_score) * 0.03 + ml_score * 0.03)
        fused_score = max(fused_score, dynamic_floor)

    fused_score = round(min(98.5, max(1.5, fused_score)), 2)

    tier = "CRITICAL" if fused_score >= 90.0 else ("HIGH" if fused_score >= 70.0 else ("MEDIUM" if fused_score >= 30.0 else "LOW"))

    return {
        "fused_score": fused_score,
        "severity_tier": tier,
        "ml_probability": round(ml_score, 2),
        "yara_score": round(yara_score, 2),
        "sigma_score": round(sigma_score, 2),
        "yara_match_count": len(yara_matches),
        "sigma_match_count": len(sigma_matches),
        "yara_matches": yara_matches,
        "sigma_matches": sigma_matches,
    }

