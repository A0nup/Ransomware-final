"""
Static Executable File Scanner Module
Provides safe, zero-execution inspection of suspicious binary executables (.exe, PE files).

Features:
- Cryptographic SHA-256 and MD5 hashing
- PE header inspection (machine type, compile timestamp, subsystem)
- Section entropy calculation (Shannon entropy to detect packing/encryption)
- Suspicious API import detection (cryptography, process injection, evasion)
- High-risk string pattern matching (shadow copy deletion, ransomware indicators)
- Threat classification: KNOWN_MALICIOUS, SUSPICIOUS, INCONCLUSIVE, or LOW_RISK_BENIGN
- Transparent disclaimers and recommended next steps
- STRICT GUARANTEE: Never loads, launches, or executes the submitted file
"""

import os
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False


# Known threat indicators and test hashes (EICAR, test signatures, known ransomware indicators)
LOCAL_THREAT_SIGNATURES = {
    # Standard EICAR test string SHA-256 for harmless validation testing
    "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f": {
        "name": "EICAR-Standard-Antivirus-Test-File",
        "category": "TEST_INDICATOR",
        "severity": "HIGH",
        "description": "Standard EICAR antivirus verification signature used for scanner validation.",
    },
    # Common test hashes for controlled verification
    "44d88612fea8a8f36de82e1278abb02f": {
        "name": "EICAR-MD5",
        "category": "TEST_INDICATOR",
        "severity": "HIGH",
        "description": "Standard EICAR MD5 hash.",
    },
}

# Suspicious APIs frequently abused by ransomware during the kill-chain
SUSPICIOUS_APIS = {
    "CryptEncrypt": "Used for encrypting file buffers in-place",
    "CryptDecrypt": "Cryptographic routine identifier",
    "CryptGenKey": "Runtime cryptographic key generation",
    "CryptAcquireContext": "Initialization of cryptographic service provider",
    "VirtualAllocEx": "Memory allocation in remote process (often used for injection)",
    "WriteProcessMemory": "Writing code into remote process (process hollowing/injection)",
    "CreateRemoteThread": "Executing injected code in another process",
    "AdjustTokenPrivileges": "Privilege escalation (e.g. enabling SeDebugPrivilege)",
    "DeleteVolumeMountPoint": "Inhibiting recovery by unmounting volumes",
    "GetLogicalDrives": "Enumerating available disk partitions for encryption",
    "FindFirstFileW": "Filesystem traversal and file discovery",
    "FindNextFileW": "Filesystem traversal and file discovery",
    "MoveFileExW": "Bulk file renaming (appending encrypted extensions)",
    "SetFileAttributesW": "Hiding files or marking attributes",
}

# Suspicious command strings embedded in binaries (e.g., backup inhibition)
SUSPICIOUS_STRINGS = [
    (b"vssadmin", "Volume Shadow Copy administrative tool reference"),
    (b"delete shadows", "Explicit command to purge volume backup snapshots"),
    (b"bcdedit", "Boot configuration editor (often used to disable recovery)"),
    (b"recoveryenabled no", "Disabling Windows boot automated recovery"),
    (b"wbadmin", "Windows Backup administration utility"),
    (b"delete catalog", "Deletion of Windows backup catalog"),
    (b".locked", "Common generic ransomware encrypted extension marker"),
    (b"ransom", "Explicit reference to ransom terminology"),
    (b"decrypt", "Explicit reference to decryption instructions"),
    (b"bitcoin", "Cryptocurrency payment reference"),
    (b"tor browser", "Onion routing / darknet payment portal reference"),
]


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte data (0.0 to 8.0)."""
    if not data:
        return 0.0
    occ = [0] * 256
    for b in data:
        occ[b] += 1
    length = len(data)
    entropy = 0.0
    for count in occ:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def extract_strings(data: bytes, min_len: int = 4, max_strings: int = 2000) -> List[str]:
    """Safely extract ASCII and printable strings from binary without execution."""
    result = []
    current = bytearray()
    for b in data:
        if 32 <= b <= 126:
            current.append(b)
        else:
            if len(current) >= min_len:
                try:
                    result.append(current.decode("ascii", errors="ignore"))
                except Exception:
                    pass
                if len(result) >= max_strings:
                    break
            current = bytearray()
    if len(current) >= min_len and len(result) < max_strings:
        try:
            result.append(current.decode("ascii", errors="ignore"))
        except Exception:
            pass
    return result


def inspect_executable_file(file_path_or_bytes, file_name: str = "uploaded_file.exe") -> Dict[str, Any]:
    """
    Perform safe static analysis on an executable file.
    Does NOT execute, load, or launch the file under any circumstance.

    Parameters:
        file_path_or_bytes: Path to file or raw bytes of uploaded file.
        file_name: Name of the file for reporting.

    Returns:
        Structured dictionary containing hashes, PE metadata, section entropy,
        indicators found, risk classification, plain English findings, and next steps.
    """
    if isinstance(file_path_or_bytes, (str, Path)):
        p = Path(file_path_or_bytes)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        file_name = p.name
        with open(p, "rb") as f:
            data = f.read()
    elif isinstance(file_path_or_bytes, (bytes, bytearray)):
        data = bytes(file_path_or_bytes)
    else:
        # File-like object (e.g., Streamlit UploadedFile)
        data = file_path_or_bytes.read()

    file_size = len(data)
    if file_size == 0:
        return {
            "file_name": file_name,
            "status": "ERROR",
            "error_message": "Submitted file is completely empty (0 bytes).",
            "risk_level": "UNKNOWN",
        }

    # 1. Cryptographic Hashes
    sha256 = hashlib.sha256(data).hexdigest()
    md5 = hashlib.md5(data).hexdigest()
    overall_entropy = calculate_entropy(data)

    # 2. Check Magic Numbers
    is_pe = data.startswith(b"MZ")
    is_elf = data.startswith(b"\x7fELF")
    is_macho = data.startswith(b"\xfe\xed\xfa\xce") or data.startswith(b"\xcf\xfa\xed\xfe") or data.startswith(b"\xca\xfe\xba\xbe")

    if is_pe:
        detected_format = "Windows Portable Executable (PE / .exe / .dll)"
    elif is_elf:
        detected_format = "Linux Executable and Linkable Format (ELF)"
    elif is_macho:
        detected_format = "macOS Mach-O Binary"
    else:
        detected_format = "Non-PE / Generic Binary Data"

    findings: List[str] = []
    indicators: List[Dict[str, str]] = []
    risk_score = 0  # 0 to 100

    # 3. Known Signature / Hash Check
    if sha256 in LOCAL_THREAT_SIGNATURES:
        sig = LOCAL_THREAT_SIGNATURES[sha256]
        indicators.append({
            "category": "KNOWN_THREAT_SIGNATURE",
            "indicator": sig["name"],
            "severity": sig["severity"],
            "detail": sig["description"],
        })
        findings.append(f"Known indicator detected in threat intelligence: {sig['name']}.")
        risk_score += 90

    if md5 in LOCAL_THREAT_SIGNATURES:
        sig = LOCAL_THREAT_SIGNATURES[md5]
        indicators.append({
            "category": "KNOWN_THREAT_SIGNATURE",
            "indicator": sig["name"],
            "severity": sig["severity"],
            "detail": sig["description"],
        })
        findings.append(f"Known MD5 signature match: {sig['name']}.")
        risk_score += 90

    # 4. Overall Entropy Analysis
    # Normal uncompressed code is usually 4.5 - 6.5. Packed/Encrypted code is > 7.0
    if overall_entropy >= 7.2:
        findings.append(f"Very high overall Shannon entropy ({overall_entropy:.2f} / 8.0). Strongly indicates compression, runtime packing, or encryption.")
        indicators.append({
            "category": "HIGH_ENTROPY",
            "indicator": f"Overall Entropy = {overall_entropy}",
            "severity": "HIGH",
            "detail": "Data distribution resembles random ciphertext or heavily packed payload.",
        })
        risk_score += 35
    elif overall_entropy >= 6.8:
        findings.append(f"Elevated overall entropy ({overall_entropy:.2f} / 8.0). Potential packing or embedded compressed assets.")
        indicators.append({
            "category": "ELEVATED_ENTROPY",
            "indicator": f"Overall Entropy = {overall_entropy}",
            "severity": "MEDIUM",
            "detail": "Entropy exceeds normal executable code thresholds.",
        })
        risk_score += 15

    # 5. String-Level Threat Inspection
    data_lower = data.lower()
    matched_suspicious_strings = []
    for pattern, desc in SUSPICIOUS_STRINGS:
        if pattern in data_lower:
            matched_suspicious_strings.append(pattern.decode("ascii", errors="ignore"))
            indicators.append({
                "category": "SUSPICIOUS_STRING",
                "indicator": pattern.decode("ascii", errors="ignore"),
                "severity": "HIGH" if b"delete" in pattern or b"vssadmin" in pattern else "MEDIUM",
                "detail": desc,
            })
            if b"vssadmin" in pattern or b"delete shadows" in pattern or b"recoveryenabled" in pattern:
                risk_score += 40
            else:
                risk_score += 15

    if matched_suspicious_strings:
        findings.append(f"Embedded string inspection found threat-related command keywords: {', '.join(matched_suspicious_strings)}.")

    # 6. Detailed PE Header Parsing (if PE and pefile available)
    pe_details: Dict[str, Any] = {}
    sections_info: List[Dict[str, Any]] = []
    suspicious_apis_found: List[str] = []

    if is_pe and PEFILE_AVAILABLE:
        try:
            pe = pefile.PE(data=data, fast_load=False)

            # Basic Header Info
            compile_time = None
            try:
                ts = pe.FILE_HEADER.TimeDateStamp
                compile_time = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                pass

            machine_type = hex(pe.FILE_HEADER.Machine)
            subsystem = hex(pe.OPTIONAL_HEADER.Subsystem) if hasattr(pe, "OPTIONAL_HEADER") else "Unknown"

            pe_details["machine"] = machine_type
            pe_details["compile_timestamp"] = compile_time
            pe_details["num_sections"] = pe.FILE_HEADER.NumberOfSections
            pe_details["entry_point"] = hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint) if hasattr(pe, "OPTIONAL_HEADER") else "Unknown"

            # Check Section Entropies
            for sec in pe.sections:
                sec_name = sec.Name.decode("ascii", errors="ignore").strip("\x00")
                sec_entropy = sec.get_entropy()
                sec_size = sec.SizeOfRawData
                sec_data = {
                    "name": sec_name,
                    "entropy": round(sec_entropy, 4),
                    "raw_size_bytes": sec_size,
                    "is_executable": bool(sec.Characteristics & 0x20000000),
                    "is_writable": bool(sec.Characteristics & 0x80000000),
                }
                sections_info.append(sec_data)

                # High section entropy (> 7.2) is a classic indicator of packed malicious code
                if sec_entropy > 7.2:
                    indicators.append({
                        "category": "PACKED_SECTION",
                        "indicator": f"Section {sec_name} Entropy = {sec_entropy:.2f}",
                        "severity": "HIGH",
                        "detail": f"Section '{sec_name}' has entropy resembling encrypted/compressed code.",
                    })
                    findings.append(f"Section '{sec_name}' is heavily packed or encrypted (Entropy: {sec_entropy:.2f}).")
                    risk_score += 25

            # Inspect Imports
            if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    for imp in entry.imports:
                        if imp.name:
                            name_str = imp.name.decode("ascii", errors="ignore")
                            if name_str in SUSPICIOUS_APIS:
                                suspicious_apis_found.append(name_str)
                                indicators.append({
                                    "category": "SUSPICIOUS_API_IMPORT",
                                    "indicator": name_str,
                                    "severity": "MEDIUM",
                                    "detail": SUSPICIOUS_APIS[name_str],
                                })
                                risk_score += 10

            if suspicious_apis_found:
                findings.append(f"Imported Windows APIs include sensitive functions: {', '.join(suspicious_apis_found)}.")

            pe.close()

        except Exception as pe_err:
            findings.append(f"PE header analysis encountered non-fatal parsing warning: {pe_err}")

    # Fallback / Non-PE analysis if pefile wasn't used
    elif is_pe and not PEFILE_AVAILABLE:
        findings.append("Executable recognized as Windows PE, but 'pefile' package is not installed for deep section inspection.")

    # 7. Final Classification Decision
    # Strict rule: DO NOT label a file as SAFE merely because no indicators are found.
    risk_score = min(100, risk_score)

    if any(ind["category"] == "KNOWN_THREAT_SIGNATURE" for ind in indicators) or risk_score >= 80:
        verdict = "KNOWN_MALICIOUS"
        verdict_title = "MALICIOUS / HIGH-CONFIDENCE THREAT"
        verdict_badge = "🔴 KNOWN MALICIOUS / HIGH RISK"
        plain_english_summary = (
            "This file matches known threat signatures or exhibits critical ransomware behavior patterns "
            "(such as volume shadow copy destruction commands and packed executable code). Do not execute this file."
        )
    elif risk_score >= 40:
        verdict = "SUSPICIOUS"
        verdict_title = "SUSPICIOUS EXECUTABLE"
        verdict_badge = "🟠 SUSPICIOUS INDICATORS DETECTED"
        plain_english_summary = (
            "This file contains multiple anomalies, such as high entropy sections (possible packing/encryption) "
            "or sensitive cryptographic API imports. While not conclusively identified as known malware, it warrants extreme caution."
        )
    elif not is_pe:
        verdict = "INCONCLUSIVE"
        verdict_title = "INCONCLUSIVE / NON-PE FILE"
        verdict_badge = "⚪ INCONCLUSIVE FILE TYPE"
        plain_english_summary = (
            f"The uploaded file is formatted as '{detected_format}'. The static scanner specializes in Windows PE executables. "
            "Evidence is insufficient to determine whether this file is benign or harmful without dynamic analysis."
        )
    else:
        # Valid PE with low risk score
        verdict = "INCONCLUSIVE"
        verdict_title = "INCONCLUSIVE / NO OBVIOUS STATIC INDICATORS"
        verdict_badge = "🟡 INCONCLUSIVE (NO KNOWN SIGNATURES)"
        plain_english_summary = (
            "No known malicious signatures or overt packing anomalies were identified in static metadata. "
            "IMPORTANT: Absence of static indicators does NOT guarantee safety. Modern zero-day ransomware "
            "frequently passes static checks. Dynamic behavioral monitoring or sandbox execution is strongly recommended."
        )

    # Recommendations
    recommendations = [
        "DO NOT execute this file on production workstations or critical servers.",
        "Submit the file to a secure, isolated sandbox (e.g. Cuckoo, Any.Run) if runtime execution is necessary.",
        "Check endpoint antivirus/EDR definitions to confirm signature updates.",
        "If this file arrived unexpectedly as an email attachment, verify the sender's identity through an out-of-band channel.",
    ]
    if verdict in ["KNOWN_MALICIOUS", "SUSPICIOUS"]:
        recommendations.insert(0, "IMMEDIATELY QUARANTINE or delete this file from downloads.")

    return {
        "file_name": file_name,
        "file_size_bytes": file_size,
        "file_size_formatted": f"{file_size / (1024 * 1024):.2f} MB" if file_size >= 1048576 else f"{file_size / 1024:.1f} KB",
        "sha256": sha256,
        "md5": md5,
        "format": detected_format,
        "overall_entropy": overall_entropy,
        "is_pe": is_pe,
        "pe_details": pe_details,
        "sections": sections_info,
        "suspicious_apis": suspicious_apis_found,
        "matched_strings": matched_suspicious_strings,
        "indicators": indicators,
        "indicator_count": len(indicators),
        "risk_score": risk_score,
        "verdict": verdict,
        "verdict_title": verdict_title,
        "verdict_badge": verdict_badge,
        "plain_english_summary": plain_english_summary,
        "findings": findings,
        "recommendations": recommendations,
        "analysis_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "analysis_type": "Static Binary & Metadata Inspection (Zero-Execution)",
    }
