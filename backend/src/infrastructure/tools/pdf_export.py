"""
PDF export tool.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

def _resolve_output_path(output_path: str) -> Path:
    base_dir = Path(os.getenv("EXPORT_BASE_DIR", "data/exports")).expanduser()
    base_dir = base_dir.resolve()

    candidate = Path(output_path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate

    resolved = candidate.resolve()
    if resolved != base_dir and base_dir not in resolved.parents:
        raise ValueError("Output path must be within EXPORT_BASE_DIR.")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@tool
def export_pdf(title: str, content: str, output_path: str) -> dict[str, str]:
    """Export content to a PDF file and return its path."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    path = _resolve_output_path(output_path)

    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    _, height = LETTER
    y = height - 72

    font_name = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
        active_font = font_name
    except Exception:
        logger.debug("Unicode font unavailable, fallback to Helvetica.")
        active_font = "Helvetica"

    if title:
        pdf.setFont(active_font, 16)
        pdf.drawString(72, y, title)
        y -= 24

    pdf.setFont(active_font, 12)
    for line in content.splitlines() or [""]:
        if y < 72:
            pdf.showPage()
            pdf.setFont(active_font, 12)
            y = height - 72
        pdf.drawString(72, y, line)
        y -= 18

    pdf.save()
    return {"path": str(path)}
