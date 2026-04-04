"""
Human-readable PDF reports from the same list shape as GET /api/vulnerabilities.json.

Built with ReportLab: a SimpleDocTemplate consumes a list of flowables (Paragraph,
Preformatted, Spacer, etc.) — the usual ReportLab "story" pattern.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape


def _format_vuln_type(t: str) -> str:
    if not t:
        return "Unclassified"
    return " ".join(w.capitalize() for w in t.replace("_", " ").split())


def _sanitize_line_for_pdf(line: str, tab_width: int = 4) -> str:
    """
    Courier / built-in PDF fonts often draw U+FFFD or black boxes for tab (0x09).
    Expand tabs to spaces and drop other C0 control characters (except normal space).
    """
    if not line:
        return ""
    s = line.replace("\r", "")
    parts: List[str] = []
    for ch in s:
        if ch == "\t":
            parts.append(" " * tab_width)
        else:
            o = ord(ch)
            if o < 32:
                parts.append(" ")
            else:
                parts.append(ch)
    return "".join(parts)


def _code_context_as_text(ctx: List[Dict[str, Any]], max_lines: int = 80) -> str:
    lines_out: List[str] = []
    for row in ctx:
        if not isinstance(row, dict):
            continue
        if row.get("ellipsis"):
            lab = row.get("label") or "…"
            lines_out.append(f"        {lab}")
            continue
        num = row.get("num")
        gutter = str(num) if num is not None else "·"
        raw = row.get("text")
        s = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
        s = s.rstrip("\n")
        s = _sanitize_line_for_pdf(s)
        mark = "*" if row.get("in_vuln_range") else " "
        lines_out.append(f"{mark} {gutter:>5} | {s}")
        if len(lines_out) >= max_lines:
            lines_out.append("… (source context truncated for PDF size)")
            break
    return "\n".join(lines_out)


def _count_findings(payload: List[Dict[str, Any]]) -> int:
    n = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        fd = item.get("findings")
        if isinstance(fd, list):
            n += len(fd)
            continue
        ln = item.get("line_numbers")
        if isinstance(ln, list):
            n += len(ln)
    return n


def build_vulnerability_pdf_bytes(
    payload: List[Dict[str, Any]],
    *,
    github_url: Optional[str] = None,
    generated_at_utc: Optional[str] = None,
    scan_completed_at: Optional[str] = None,
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="HarnoldsEye vulnerability report",
    )
    styles = getSampleStyleSheet()
    story: List[Any] = []

    title_style = ParagraphStyle(
        "PdfTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=10,
        textColor=colors.HexColor("#1a1a2e"),
    )
    h_file = ParagraphStyle(
        "PdfFile",
        parent=styles["Heading2"],
        fontSize=11.5,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#16213e"),
    )
    h_section = ParagraphStyle(
        "PdfSection",
        parent=styles["Heading3"],
        fontSize=9.5,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#0f3460"),
    )
    body = ParagraphStyle(
        "PdfBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        alignment=TA_JUSTIFY,
    )
    small = ParagraphStyle(
        "PdfSmall",
        parent=styles["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#555555"),
    )
    code_style = ParagraphStyle(
        "PdfCode",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=7,
        leading=8.5,
        leftIndent=6,
    )

    story.append(Paragraph(escape("HarnoldsEye — Vulnerability report"), title_style))
    meta_lines: List[str] = []
    if github_url:
        meta_lines.append(f"<b>Repository:</b> {escape(github_url)}")
    if scan_completed_at:
        meta_lines.append(f"<b>Scan completed (UTC):</b> {escape(scan_completed_at)}")
    if generated_at_utc:
        meta_lines.append(f"<b>PDF generated (UTC):</b> {escape(generated_at_utc)}")
    if meta_lines:
        story.append(Paragraph("<br/>".join(meta_lines), body))
    story.append(Spacer(1, 0.12 * inch))

    n_find = _count_findings(payload)
    story.append(
        Paragraph(
            escape(
                f"Summary: {n_find} finding(s) across {len(payload)} file record(s)."
            ),
            body,
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(
        Paragraph(
            escape(
                "Scope and limitations: Findings come from static pattern matching plus "
                "ML (embeddings or classifier) on C/C++ snippets, optionally refined by "
                "Google Gemini. Related CVEs are NVD keyword matches for context only, "
                "not confirmed exploit mapping. This document does not replace manual "
                "code review or a professional security assessment."
            ),
            small,
        )
    )
    story.append(Spacer(1, 0.16 * inch))

    if not payload:
        story.append(
            Paragraph(
                escape("No vulnerability records are present in the current result set."),
                body,
            )
        )
        doc.build(story)
        return buf.getvalue()

    finding_idx = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        fn = item.get("file_name")
        if not isinstance(fn, str) or not fn.strip():
            continue
        findings = item.get("findings")
        if not isinstance(findings, list):
            ranges = item.get("line_numbers")
            if not isinstance(ranges, list):
                continue
            findings = []
            for pair in ranges:
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and all(isinstance(x, (int, float)) for x in pair)
                ):
                    findings.append(
                        {
                            "line_numbers": [int(pair[0]), int(pair[1])],
                            "vulnerability_type": "Unclassified",
                        }
                    )

        story.append(Paragraph(escape(f"File: {fn}"), h_file))

        for f in findings:
            if not isinstance(f, dict):
                continue
            pair = f.get("line_numbers")
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            finding_idx += 1
            a, b = int(pair[0]), int(pair[1])
            line_part = f"line {a}" if a == b else f"lines {a}–{b}"
            vtype = _format_vuln_type(str(f.get("vulnerability_type") or ""))

            story.append(
                Paragraph(
                    escape(f"{finding_idx}. {fn} — {line_part}"),
                    h_section,
                )
            )
            story.append(Paragraph(escape(f"Type: {vtype}"), body))
            tex = f.get("type_explanation")
            if isinstance(tex, str) and tex.strip():
                story.append(
                    Paragraph(escape(tex.strip()[:900]), small),
                )

            if f.get("finding_role") == "call_site" and f.get("sink_function"):
                sk = str(f.get("sink_function"))
                sln = f.get("sink_line_numbers")
                if isinstance(sln, list) and len(sln) == 2:
                    sa, sb = int(sln[0]), int(sln[1])
                    sp = f"line {sa}" if sa == sb else f"lines {sa}–{sb}"
                    story.append(
                        Paragraph(
                            escape(
                                f"Call-site note: invokes sink «{sk}» ({sp})."
                            ),
                            body,
                        )
                    )

            ctx = f.get("code_context")
            if isinstance(ctx, list) and ctx:
                story.append(Paragraph(escape("Source context"), h_section))
                block = _code_context_as_text(ctx)
                # Preformatted is literal text (not mini-HTML like Paragraph).
                story.append(Preformatted(block, code_style))

            cves = f.get("related_cves")
            if isinstance(cves, list) and cves:
                story.append(
                    Paragraph(escape("Related CVEs (NVD keyword search)"), h_section)
                )
                shown = 0
                for c in cves:
                    if shown >= 20:
                        story.append(
                            Paragraph(
                                escape(
                                    "(Additional CVE rows omitted from this PDF for length.)"
                                ),
                                small,
                            )
                        )
                        break
                    if not isinstance(c, dict):
                        continue
                    cid = str(c.get("cve_id") or "").strip()
                    if not cid.startswith("CVE-"):
                        continue
                    desc = str(c.get("description") or "")
                    desc_short = desc[:380] + ("…" if len(desc) > 380 else "")
                    pub = str(c.get("published") or "").strip()[:16]
                    pub_bit = f", published {pub}" if pub else ""
                    story.append(
                        Paragraph(
                            escape(f"• {cid}{pub_bit} — {desc_short}"),
                            body,
                        )
                    )
                    shown += 1

            story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    return buf.getvalue()
