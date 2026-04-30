"""
PDF export tool.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Callable

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

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Cannot create export directory '{resolved.parent}': {exc!r}") from exc
    return resolved


def _max_prefix_that_fits(text: str, max_width: float, measure: Callable[[str], float]) -> int:
    if not text:
        return 0
    lo, hi = 1, len(text)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if measure(text[:mid]) <= max_width:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(1, best)


def _wrap_text_lines(text: str, max_width: float, measure: Callable[[str], float]) -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized.split("\n")
    if not raw_lines:
        return [""]

    wrapped: list[str] = []
    for raw in raw_lines:
        remaining = raw.expandtabs(4)
        if not remaining:
            wrapped.append("")
            continue
        while remaining:
            if measure(remaining) <= max_width:
                wrapped.append(remaining)
                break
            cut = _max_prefix_that_fits(remaining, max_width, measure)
            prefix = remaining[:cut]
            # Prefer breaking on whitespace for Latin text; fall back to hard wrap.
            break_pos = prefix.rfind(" ")
            if break_pos > 0 and break_pos >= cut // 2:
                line = remaining[:break_pos].rstrip()
                next_start = break_pos + 1
            else:
                line = prefix.rstrip()
                next_start = cut
            if not line:
                line = remaining[:cut]
            wrapped.append(line)
            remaining = remaining[next_start:].lstrip()
    return wrapped


@tool
def export_pdf(title: str, content: str, output_path: str) -> dict[str, str]:
    """Export content to a PDF file and return its path."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    path = _resolve_output_path(output_path)

    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    margin_left = 72
    margin_right = 72
    margin_top = 72
    margin_bottom = 72
    content_width = width - margin_left - margin_right
    y = height - margin_top

    font_name = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
        active_font = font_name
    except Exception:
        logger.debug("Unicode font unavailable, fallback to Helvetica.")
        active_font = "Helvetica"

    if title:
        title_font_size = 16
        title_line_height = 24

        def title_measure(s: str) -> float:
            return pdf.stringWidth(s, active_font, title_font_size)

        title_lines = _wrap_text_lines(title, content_width, title_measure)
        pdf.setFont(active_font, title_font_size)
        for line in title_lines:
            if y < margin_bottom:
                pdf.showPage()
                pdf.setFont(active_font, title_font_size)
                y = height - margin_top
            pdf.drawString(margin_left, y, line)
            y -= title_line_height
        y -= 4

    body_font_size = 12
    body_line_height = 18

    def body_measure(s: str) -> float:
        return pdf.stringWidth(s, active_font, body_font_size)

    body_lines = _wrap_text_lines(content, content_width, body_measure)
    pdf.setFont(active_font, body_font_size)
    for line in body_lines:
        if y < margin_bottom:
            pdf.showPage()
            pdf.setFont(active_font, body_font_size)
            y = height - margin_top
        pdf.drawString(margin_left, y, line)
        y -= body_line_height

    try:
        pdf.save()
    except OSError as exc:
        raise ValueError(f"Cannot write PDF file '{path}': {exc!r}") from exc
    return {"path": str(path)}
