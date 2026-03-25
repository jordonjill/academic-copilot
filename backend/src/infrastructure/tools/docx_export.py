"""
DOCX export tool.
"""
from __future__ import annotations

from pathlib import Path
from langchain_core.tools import tool


@tool
def export_docx(title: str, content: str, output_path: str) -> dict:
    """Export content to a DOCX file and return its path."""
    from docx import Document

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    document = Document()
    if title:
        document.add_heading(title, level=1)
    for line in content.splitlines() or [""]:
        document.add_paragraph(line)

    document.save(path)
    return {"path": str(path)}
