"""
PDF export tool.
"""
from __future__ import annotations

from pathlib import Path
from langchain_core.tools import tool
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


@tool
def export_pdf(title: str, content: str, output_path: str) -> dict:
    """Export content to a PDF file and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    _, height = LETTER
    y = height - 72

    if title:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(72, y, title)
        y -= 24

    pdf.setFont("Helvetica", 12)
    for line in content.splitlines() or [""]:
        if y < 72:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = height - 72
        pdf.drawString(72, y, line)
        y -= 18

    pdf.save()
    return {"path": str(path)}
