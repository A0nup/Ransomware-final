"""
Unit & Integration Tests for Static Executable Scanner Module
Verifies hashing, PE inspection, signature matching, entropy calculation,
and zero-execution guarantees.
"""

import pytest
from pathlib import Path
from src.static_scanner import (
    calculate_entropy,
    extract_strings,
    inspect_executable_file,
    LOCAL_THREAT_SIGNATURES,
)


def test_calculate_entropy():
    # Empty bytes has 0.0 entropy
    assert calculate_entropy(b"") == 0.0
    # Repeating single byte has 0.0 entropy
    assert calculate_entropy(b"\x00" * 100) == 0.0
    # Uniformly distributed random-like bytes approaches 8.0
    uniform_bytes = bytes(range(256)) * 10
    ent = calculate_entropy(uniform_bytes)
    assert 7.9 <= ent <= 8.0


def test_extract_strings():
    data = b"SomeGarbage\x00\x01\x02TargetStringHere\x00\x00AnotherOne"
    extracted = extract_strings(data, min_len=4)
    assert "SomeGarbage" in extracted
    assert "TargetStringHere" in extracted
    assert "AnotherOne" in extracted


def test_eicar_test_signature_detection():
    eicar_bytes = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
    res = inspect_executable_file(eicar_bytes, file_name="eicar.com")
    assert res["verdict"] == "KNOWN_MALICIOUS"
    assert "KNOWN_THREAT_SIGNATURE" in [ind["category"] for ind in res["indicators"]]
    assert res["risk_score"] >= 80


def test_simulated_ransomware_strings_detection():
    # Build a simulated PE containing backup deletion strings
    fake_pe = (
        b"MZ" + b"\x00" * 58 + b"\x80\x00\x00\x00" + b"\x00" * 60 + b"PE\x00\x00"
        + b"vssadmin.exe delete shadows /all /quiet bcdedit /set recoveryenabled no"
        + b"\x00" * 200
    )
    res = inspect_executable_file(fake_pe, file_name="sample_ransomware.exe")
    assert res["verdict"] in ["KNOWN_MALICIOUS", "SUSPICIOUS"]
    assert "vssadmin" in res["matched_strings"]
    assert "delete shadows" in res["matched_strings"]
    assert res["risk_score"] >= 40


def test_benign_file_inconclusive_disclaimer():
    # A clean text/binary file should never be marked as "SAFE"
    clean_data = b"This is a standard developer document. Everything is nominal."
    res = inspect_executable_file(clean_data, file_name="document.txt")
    assert res["verdict"] == "INCONCLUSIVE"
    # Verify presence of safety disclaimer
    assert any(term in res["plain_english_summary"].lower() for term in ["guarantee", "inconclusive", "insufficient"])


def test_empty_file_handling():
    res = inspect_executable_file(b"", file_name="empty.exe")
    assert res["status"] == "ERROR"
    assert "empty" in res["error_message"].lower()


def test_zero_execution_guarantee():
    # Verify that inspection of binary data never spawns a process or runs code
    import psutil
    initial_pids = set(psutil.pids())
    fake_binary = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00" + b"\x90" * 1000
    _ = inspect_executable_file(fake_binary, file_name="dummy_calc.exe")
    final_pids = set(psutil.pids())
    # No new lingering processes should exist
    assert len(final_pids - initial_pids) == 0
