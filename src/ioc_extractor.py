"""
Automated IOC (Indicators of Compromise) Extractor & SOC Exporter
Extracts hashes, IP addresses, C2 domains/URLs, registry keys, and ransomware extensions.
Exports into JSON, CSV, and STIX 2.1 standard format.
"""

import re
import json
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Optional


# Regular Expressions for IOC Extraction
IPV4_REGEX = re.compile(r"\b(?!(?:10|127|192\.168|172\.(?:1[6-9]|2[0-9]|3[0-1]))\b)(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
DOMAIN_REGEX = re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|ru|xyz|top|biz|info|onion|cc|pw|live)\b", re.IGNORECASE)
URL_REGEX = re.compile(r"https?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::[0-9]+)?(?:/[^\s\"'<>)]*)?", re.IGNORECASE)
REGISTRY_REGEX = re.compile(r"\b(?:HKLM|HKCU|HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER)\\[a-zA-Z0-9_\\]+\b", re.IGNORECASE)
RANSOM_EXT_REGEX = re.compile(r"\.(?:locked|enc|crypto|crypted|ransom|lockbit|blackcat|wncry|mallox|conti|stop)\b", re.IGNORECASE)


def extract_iocs_from_data(
    raw_bytes_or_text: Any,
    file_name: str = "sample_file",
    process_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract forensic Indicators of Compromise (IOCs) from binary bytes, strings, or process context.
    """
    if isinstance(raw_bytes_or_text, bytes):
        raw_bytes = raw_bytes_or_text
        text = raw_bytes.decode("latin-1", errors="ignore")
    else:
        text = str(raw_bytes_or_text)
        raw_bytes = text.encode("utf-8", errors="ignore")

    # Hashes
    md5 = hashlib.md5(raw_bytes).hexdigest()
    sha1 = hashlib.sha1(raw_bytes).hexdigest()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # Network IOCs
    ips: Set[str] = set(IPV4_REGEX.findall(text))
    domains: Set[str] = set(DOMAIN_REGEX.findall(text))
    urls: Set[str] = set(URL_REGEX.findall(text))
    regs: Set[str] = set(REGISTRY_REGEX.findall(text))
    extensions: Set[str] = set(RANSOM_EXT_REGEX.findall(text))

    # Clean out obvious false positives
    filtered_domains = {d for d in domains if not d.endswith((".dll", ".exe", ".manifest", ".cat", ".sys")) and len(d) > 4}

    iocs = {
        "file_name": file_name,
        "hashes": {
            "md5": md5,
            "sha1": sha1,
            "sha256": sha256,
        },
        "network": {
            "ipv4": sorted(list(ips))[:10],
            "domains": sorted(list(filtered_domains))[:10],
            "urls": sorted(list(urls))[:10],
        },
        "host": {
            "registry_keys": sorted(list(regs))[:10],
            "ransom_extensions": sorted(list(extensions))[:10],
        },
        "process": process_info or {},
        "extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    return iocs


def export_iocs_to_stix(iocs: Dict[str, Any], threat_name: str = "Behavioral Ransomware Detection") -> str:
    """
    Convert extracted IOCs into a compliant STIX 2.1 JSON bundle.
    """
    bundle_id = f"bundle--{uuid.uuid4()}"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    objects = []

    # 1. Malware Object
    malware_id = f"malware--{uuid.uuid4()}"
    objects.append({
        "type": "malware",
        "spec_version": "2.1",
        "id": malware_id,
        "created": timestamp,
        "modified": timestamp,
        "name": threat_name,
        "is_family": False,
        "malware_types": ["ransomware"],
        "description": "Suspected ransomware detected via hybrid deep learning and behavioral telemetry.",
    })

    # 2. File Hash Indicator
    sha256_val = iocs.get("hashes", {}).get("sha256")
    if sha256_val:
        ind_id = f"indicator--{uuid.uuid4()}"
        objects.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": ind_id,
            "created": timestamp,
            "modified": timestamp,
            "name": f"File Hash: {sha256_val}",
            "pattern_type": "stix",
            "pattern": f"[file:hashes.'SHA-256' = '{sha256_val}']",
            "valid_from": timestamp,
        })

    # 3. IP Indicators
    for ip in iocs.get("network", {}).get("ipv4", []):
        objects.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": f"indicator--{uuid.uuid4()}",
            "created": timestamp,
            "modified": timestamp,
            "name": f"Malicious IP: {ip}",
            "pattern_type": "stix",
            "pattern": f"[ipv4-addr:value = '{ip}']",
            "valid_from": timestamp,
        })

    # 4. Domain Indicators
    for domain in iocs.get("network", {}).get("domains", []):
        objects.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": f"indicator--{uuid.uuid4()}",
            "created": timestamp,
            "modified": timestamp,
            "name": f"C2 Domain: {domain}",
            "pattern_type": "stix",
            "pattern": f"[domain-name:value = '{domain}']",
            "valid_from": timestamp,
        })

    stix_bundle = {
        "type": "bundle",
        "id": bundle_id,
        "objects": objects,
    }
    return json.dumps(stix_bundle, indent=2)


def export_iocs_to_csv(iocs: Dict[str, Any]) -> str:
    """Export flattened IOC list as CSV string."""
    rows = ["type,indicator,context,extracted_at"]
    ts = iocs.get("extracted_at", "")

    for h_type, h_val in iocs.get("hashes", {}).items():
        rows.append(f"hash_{h_type},{h_val},file_hash,{ts}")

    for ip in iocs.get("network", {}).get("ipv4", []):
        rows.append(f"ipv4,{ip},network_socket,{ts}")

    for dom in iocs.get("network", {}).get("domains", []):
        rows.append(f"domain,{dom},c2_candidate,{ts}")

    for reg in iocs.get("host", {}).get("registry_keys", []):
        rows.append(f"registry_key,{reg},persistence_attempt,{ts}")

    for ext in iocs.get("host", {}).get("ransom_extensions", []):
        rows.append(f"extension,{ext},ransom_renaming,{ts}")

    return "\n".join(rows)
