"""
MITRE ATT&CK Enterprise Matrix Mapping Module
Maps behavioral telemetry signals, static inspection indicators, and process events
to official MITRE ATT&CK tactics, techniques, and sub-techniques.
"""

from typing import Dict, Any, List, Optional


MITRE_TECHNIQUES_DATABASE = {
    "T1486": {
        "id": "T1486",
        "name": "Data Encrypted for Impact",
        "tactic": "Impact",
        "url": "https://attack.mitre.org/techniques/T1486/",
        "description": "Adversaries may encrypt data on target systems to interrupt availability to system and network resources.",
        "trigger_conditions": ["high_entropy", "mass_file_writes", "encrypted_ratio", "extension_changes"],
    },
    "T1490": {
        "id": "T1490",
        "name": "Inhibit System Recovery",
        "tactic": "Impact",
        "url": "https://attack.mitre.org/techniques/T1490/",
        "description": "Adversaries may delete or remove built-in operating system recovery features such as Volume Shadow Copies.",
        "trigger_conditions": ["shadow_copy_cmd", "backup_delete_cmd", "vssadmin", "bcdedit", "wbadmin"],
    },
    "T1059": {
        "id": "T1059",
        "name": "Command and Scripting Interpreter",
        "tactic": "Execution",
        "url": "https://attack.mitre.org/techniques/T1059/",
        "description": "Adversaries may abuse command and script interpreters (PowerShell, cmd.exe, wscript) to execute commands.",
        "trigger_conditions": ["powershell", "cmd.exe", "wscript", "cscript", "script_execution"],
    },
    "T1083": {
        "id": "T1083",
        "name": "File and Directory Discovery",
        "tactic": "Discovery",
        "url": "https://attack.mitre.org/techniques/T1083/",
        "description": "Adversaries may enumerate files and directories or search in specific locations to identify targets for encryption.",
        "trigger_conditions": ["directory_traversal", "rapid_file_enumeration"],
    },
    "T1027": {
        "id": "T1027",
        "name": "Obfuscated/Compressed Files and Information",
        "tactic": "Defense Evasion",
        "url": "https://attack.mitre.org/techniques/T1027/",
        "description": "Adversaries may pack or compress payload data to evade signature-based detection.",
        "trigger_conditions": ["high_pe_entropy", "packed_section", "upx", "anti_debug"],
    },
    "T1112": {
        "id": "T1112",
        "name": "Modify Registry",
        "tactic": "Defense Evasion",
        "url": "https://attack.mitre.org/techniques/T1112/",
        "description": "Adversaries may modify system registry keys to hide execution or disable security configurations.",
        "trigger_conditions": ["registry_modification", "uac_bypass_reg", "disable_defender"],
    },
    "T1489": {
        "id": "T1489",
        "name": "Service Stop",
        "tactic": "Impact",
        "url": "https://attack.mitre.org/techniques/T1489/",
        "description": "Adversaries may stop critical services to disrupt backups, databases, or antivirus defenses.",
        "trigger_conditions": ["net_stop_vss", "kill_antivirus", "service_kill"],
    },
    "T1070": {
        "id": "T1070",
        "name": "Indicator Removal on Host",
        "tactic": "Defense Evasion",
        "url": "https://attack.mitre.org/techniques/T1070/",
        "description": "Adversaries may delete event logs or system audit trails to prevent digital forensic analysis.",
        "trigger_conditions": ["wevtutil", "clear_event_logs"],
    },
    "T1204": {
        "id": "T1204",
        "name": "User Execution: Malicious File",
        "tactic": "Execution",
        "url": "https://attack.mitre.org/techniques/T1204/",
        "description": "An adversary may rely upon specific actions by a user to execute malicious code, such as opening an attachment.",
        "trigger_conditions": ["user_launch", "unverified_executable"],
    },
}


def map_threat_to_mitre(
    telemetry_meta: Optional[Dict[str, Any]] = None,
    static_indicators: Optional[List[Dict[str, Any]]] = None,
    probability: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Correlate observed operational signals and static indicators to MITRE ATT&CK techniques.
    Returns prioritized list of matched techniques with evidence.
    """
    matched: Dict[str, Dict[str, Any]] = {}
    telemetry_meta = telemetry_meta or {}
    static_indicators = static_indicators or []

    # 1. Check Data Encrypted for Impact (T1486)
    ent_delta = float(telemetry_meta.get("entropy_delta", 0.0))
    write_rate = float(telemetry_meta.get("file_write_rate", 0.0))
    ext_rate = float(telemetry_meta.get("extension_changes", 0.0))
    if ent_delta > 1.5 or write_rate > 30.0 or ext_rate > 3.0 or probability >= 0.70:
        tech = dict(MITRE_TECHNIQUES_DATABASE["T1486"])
        tech["confidence"] = "HIGH" if (ent_delta > 2.0 or probability >= 0.85) else "MEDIUM"
        tech["evidence"] = f"Measured entropy delta: {ent_delta:.3f} | Write rate: {write_rate:.0f} ops/s | Renames: {ext_rate:.0f}"
        matched["T1486"] = tech

    # 2. Check Inhibit System Recovery (T1490)
    shadow_cmd = float(telemetry_meta.get("shadow_cmd", 0.0))
    culprit = telemetry_meta.get("culprit_process") or {}
    cmd_str = str(culprit.get("cmd", "")).lower()

    if shadow_cmd > 0 or "vssadmin" in cmd_str or "bcdedit" in cmd_str or "wbadmin" in cmd_str:
        tech = dict(MITRE_TECHNIQUES_DATABASE["T1490"])
        tech["confidence"] = "CRITICAL"
        tech["evidence"] = f"Tampering command detected: '{cmd_str or 'vssadmin delete shadows'}'"
        matched["T1490"] = tech

    # 3. Check Obfuscated/Compressed Files (T1027) from static indicators
    for ind in static_indicators:
        cat = ind.get("category", "")
        detail = ind.get("detail", "").lower()
        if "entropy" in cat.lower() or "packing" in cat.lower() or "packed" in detail:
            tech = dict(MITRE_TECHNIQUES_DATABASE["T1027"])
            tech["confidence"] = "HIGH"
            tech["evidence"] = f"Static analysis: {ind.get('indicator', '')} - {ind.get('detail', '')}"
            matched["T1027"] = tech

        if "command" in cat.lower() and ("vssadmin" in detail or "shadow" in detail):
            tech = dict(MITRE_TECHNIQUES_DATABASE["T1490"])
            tech["confidence"] = "CRITICAL"
            tech["evidence"] = f"Embedded command signature: {ind.get('indicator', '')}"
            matched["T1490"] = tech

        if "powershell" in detail or "cmd.exe" in detail or "script" in detail:
            tech = dict(MITRE_TECHNIQUES_DATABASE["T1059"])
            tech["confidence"] = "MEDIUM"
            tech["evidence"] = f"Script execution indicator: {ind.get('indicator', '')}"
            matched["T1059"] = tech

    # 4. Check File & Directory Discovery (T1083)
    if write_rate > 50.0:
        tech = dict(MITRE_TECHNIQUES_DATABASE["T1083"])
        tech["confidence"] = "MEDIUM"
        tech["evidence"] = f"Rapid recursive traversal and file access ({write_rate:.0f} ops/s)"
        matched["T1083"] = tech

    return list(matched.values())

