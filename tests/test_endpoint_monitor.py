"""
Unit & Integration Tests for Endpoint Behavioral Monitor Module
Verifies telemetry extraction, feature schema compliance, quarantine mechanisms,
controlled termination, and incident reporting.
"""

import os
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.endpoint_monitor import EndpointMonitor
from src.config import FEATURES, SEQ_LEN


def test_monitor_initialization_and_sampling():
    with tempfile.TemporaryDirectory() as tmpdir:
        mon = EndpointMonitor(watch_path=tmpdir, sampling_interval_sec=0.5)
        assert mon.watch_path == Path(tmpdir).resolve()
        assert not mon.is_active()

        # Generate a test file in monitored zone
        test_file = Path(tmpdir) / "workload_sample.dat"
        test_file.write_bytes(b"A" * 1024)

        vec, meta = mon._sample_current_telemetry_vector()
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (len(FEATURES),), f"Vector shape mismatch: {vec.shape} vs {len(FEATURES)}"
        assert meta["file_write_rate"] >= 1.0


def test_sliding_window_buffer_management():
    with tempfile.TemporaryDirectory() as tmpdir:
        mon = EndpointMonitor(watch_path=tmpdir, sampling_interval_sec=0.5)

        # Simulate 25 sequential timesteps
        for i in range(25):
            dummy_vec = np.zeros(len(FEATURES), dtype=np.float32)
            dummy_vec[0] = float(i)
            mon.history_buffer.append(dummy_vec)

        # Deque must strictly retain exactly SEQ_LEN (20) timesteps
        assert len(mon.history_buffer) == SEQ_LEN
        # Most recent sample must be 24
        assert mon.history_buffer[-1][0] == 24.0


def test_controlled_quarantine_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        q_dir = Path(tmpdir) / ".quarantine_test"
        mon = EndpointMonitor(watch_path=tmpdir, quarantine_dir=q_dir)

        # Create sample file to be quarantined
        sus_file = Path(tmpdir) / "threat_payload.exe"
        sus_file.write_bytes(b"Simulated malicious payload bytes 12345")

        q_res = mon.quarantine_file(str(sus_file))
        assert q_res["status"] == "SUCCESS"
        assert not sus_file.exists(), "Original file must no longer exist in source directory"

        quarantined_path = Path(q_res["quarantine_path"])
        assert quarantined_path.exists(), "Quarantined file must exist in isolated quarantine folder"

        # Check permissions: must be read-only (0400 or mode 0o100400)
        mode = oct(os.stat(quarantined_path).st_mode)
        assert mode.endswith("400"), f"Expected 0400 permission, got {mode}"


def test_process_termination_guardrail():
    with tempfile.TemporaryDirectory() as tmpdir:
        mon = EndpointMonitor(watch_path=tmpdir)

        # Must refuse termination without confirmation flag
        unconfirmed = mon.terminate_process_safely(pid=99999, confirm=False)
        assert unconfirmed["status"] == "CONFIRMATION_REQUIRED"

        # Non-existent PID should gracefully return error without crashing
        non_existent = mon.terminate_process_safely(pid=9999999, confirm=True)
        assert non_existent["status"] == "ERROR"


def test_incident_report_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        mon = EndpointMonitor(watch_path=tmpdir)
        mon.alerts.append({
            "timestamp": "2026-10-03 12:00:00 UTC",
            "probability": 96.5,
            "severity": "CRITICAL",
            "culprit_process": {"name": "ransom.exe", "pid": 1234},
            "affected_files": ["/tmp/file1.locked"],
        })

        json_out = mon.export_incident_report(out_format="json")
        assert Path(json_out).exists()
        assert Path(json_out).stat().st_size > 0

        csv_out = mon.export_incident_report(out_format="csv")
        assert Path(csv_out).exists()
        assert Path(csv_out).stat().st_size > 0
