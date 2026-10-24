"""
SOC Incident Response PDF Report Generator
Generates publication-quality forensic incident reports using ReportLab.
Includes executive summary, MITRE ATT&CK mapping, explainability attributions,
IOC tables, and prescriptive response playbooks.
"""

import io
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


def generate_pdf_incident_report(
    incident_id: str,
    threat_verdict: str,
    probability: float,
    mitre_techniques: List[Dict[str, Any]],
    top_features: List[Dict[str, Any]],
    iocs: Dict[str, Any],
    culprit_process: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate a professional multi-page PDF SOC Incident Response Report.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer if output_path is None else output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    bold_label = ParagraphStyle(
        "BoldLabel",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a"),
    )

    elements = []

    # 1. Header Banner
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    elements.append(Paragraph("AEROSHIELD AI — SOC INCIDENT RESPONSE REPORT", title_style))
    elements.append(Paragraph(f"Forensic Audit Document | Incident ID: <b>{incident_id}</b> | Generated: {now_str}", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#3b82f6"), spaceAfter=12))

    # 2. Executive Threat Verdict Card
    risk_color = colors.HexColor("#ef4444") if probability >= 70.0 else (colors.HexColor("#f59e0b") if probability >= 30.0 else colors.HexColor("#10b981"))
    verdict_text = f"THREAT VERDICT: {threat_verdict} ({probability:.1f}% Probability)"

    verdict_table_data = [
        [Paragraph(f"<b>{verdict_text}</b>", ParagraphStyle("VText", parent=bold_label, textColor=colors.white, fontSize=11)),
         Paragraph(f"Status: <b>CONTAINMENT PENDING</b>", ParagraphStyle("VStat", parent=body_style, textColor=colors.white, alignment=2))]
    ]
    verdict_table = Table(verdict_table_data, colWidths=[360, 180])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), risk_color),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 14))

    # 3. Host Telemetry & Culprit Context
    elements.append(Paragraph("1. Host Context & Incident Summary", heading_style))
    culprit_proc_name = (culprit_process or {}).get("name", "Unknown/Unattributed")
    culprit_pid = str((culprit_process or {}).get("pid", "N/A"))
    culprit_cmd = str((culprit_process or {}).get("cmd", "N/A"))

    ctx_data = [
        [Paragraph("Target Host:", bold_label), Paragraph(os.uname().nodename if hasattr(os, "uname") else "Endpoint Workstation", body_style)],
        [Paragraph("Offending Process:", bold_label), Paragraph(f"<b>{culprit_proc_name}</b> (PID: {culprit_pid})", body_style)],
        [Paragraph("Command Line:", bold_label), Paragraph(f"<code>{culprit_cmd[:90]}</code>", body_style)],
        [Paragraph("Primary File Target:", bold_label), Paragraph(f"<code>{iocs.get('file_name', 'System Filesystem')}</code>", body_style)],
    ]
    t_ctx = Table(ctx_data, colWidths=[130, 410])
    t_ctx.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_ctx)
    elements.append(Spacer(1, 14))

    # 4. MITRE ATT&CK Mapping
    elements.append(Paragraph("2. MITRE ATT&CK Matrix Correlation", heading_style))
    mitre_rows = [[Paragraph("<b>Technique ID</b>", bold_label), Paragraph("<b>Technique Name</b>", bold_label), Paragraph("<b>Tactic</b>", bold_label), Paragraph("<b>Confidence</b>", bold_label)]]
    if mitre_techniques:
        for m in mitre_techniques[:6]:
            mitre_rows.append([
                Paragraph(f"<b>{m.get('id', 'N/A')}</b>", body_style),
                Paragraph(m.get("name", "N/A"), body_style),
                Paragraph(m.get("tactic", "Impact"), body_style),
                Paragraph(m.get("confidence", "HIGH"), body_style),
            ])
    else:
        mitre_rows.append([Paragraph("T1486", body_style), Paragraph("Data Encrypted for Impact", body_style), Paragraph("Impact", body_style), Paragraph("HEURISTIC", body_style)])

    t_mitre = Table(mitre_rows, colWidths=[90, 230, 120, 100])
    t_mitre.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_mitre)
    elements.append(Spacer(1, 14))

    # 5. Explainability (Feature Attribution)
    elements.append(Paragraph("3. Model Explainability & Feature Attribution", heading_style))
    feat_rows = [[Paragraph("<b>Feature Indicator</b>", bold_label), Paragraph("<b>Measured Value</b>", bold_label), Paragraph("<b>Contribution</b>", bold_label), Paragraph("<b>Direction</b>", bold_label)]]
    for feat in top_features[:5]:
        val = feat.get("raw_value", 0.0)
        contrib = feat.get("contribution_pct", 0.0)
        direction = feat.get("direction", "MALICIOUS")
        feat_rows.append([
            Paragraph(feat.get("readable_name", feat.get("feature", "N/A")), body_style),
            Paragraph(f"{val:.2f}", body_style),
            Paragraph(f"{contrib:+.1f}%", body_style),
            Paragraph(direction, body_style),
        ])
    t_feat = Table(feat_rows, colWidths=[220, 100, 110, 110])
    t_feat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_feat)
    elements.append(Spacer(1, 14))

    # 6. Indicators of Compromise (IOCs)
    elements.append(Paragraph("4. Extracted Indicators of Compromise (IOCs)", heading_style))
    h_sha256 = iocs.get("hashes", {}).get("sha256", "N/A")
    h_md5 = iocs.get("hashes", {}).get("md5", "N/A")
    net_ips = ", ".join(iocs.get("network", {}).get("ipv4", [])) or "None detected"
    doms = ", ".join(iocs.get("network", {}).get("domains", [])) or "None detected"

    ioc_rows = [
        [Paragraph("SHA-256 Hash:", bold_label), Paragraph(f"<code>{h_sha256}</code>", body_style)],
        [Paragraph("MD5 Hash:", bold_label), Paragraph(f"<code>{h_md5}</code>", body_style)],
        [Paragraph("Observed Network IPs:", bold_label), Paragraph(net_ips, body_style)],
        [Paragraph("C2 Domains:", bold_label), Paragraph(doms, body_style)],
    ]
    t_ioc = Table(ioc_rows, colWidths=[140, 400])
    t_ioc.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_ioc)
    elements.append(Spacer(1, 14))

    # 7. Prescriptive Response Playbook
    elements.append(Paragraph("5. Prescriptive SOC Response Playbook", heading_style))
    playbook_steps = [
        "1. <b>Host Isolation:</b> Sever target endpoint from local subnets to inhibit lateral movement.",
        "2. <b>Process Neutralization:</b> Terminate rogue process PID immediately via SIGTERM / SIGKILL.",
        "3. <b>Forensic Quarantine:</b> Relocate encrypted or suspicious binaries to .quarantine/ with chmod 0400.",
        "4. <b>Volume Shadow Audit:</b> Execute <code>vssadmin list shadows</code> to verify backup catalog integrity.",
        "5. <b>Remediation & Recovery:</b> Restore validated files from off-site immutable backup repository.",
    ]
    for step in playbook_steps:
        elements.append(Paragraph(step, body_style))
        elements.append(Spacer(1, 3))

    doc.build(elements)

    if output_path is not None:
        with open(output_path, "rb") as f:
            return f.read()

    return buffer.getvalue()
