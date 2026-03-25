from src.infrastructure.tools import export_docx, export_pdf


def test_export_docx_creates_file(tmp_path):
    output_path = tmp_path / "sample.docx"
    result = export_docx.invoke(
        {
            "title": "Sample Title",
            "content": "First line\nSecond line",
            "output_path": str(output_path),
        }
    )
    assert result == {"path": str(output_path)}
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_export_pdf_creates_file(tmp_path):
    output_path = tmp_path / "sample.pdf"
    result = export_pdf.invoke(
        {
            "title": "Sample Title",
            "content": "First line\nSecond line",
            "output_path": str(output_path),
        }
    )
    assert result == {"path": str(output_path)}
    assert output_path.exists()
    assert output_path.stat().st_size > 0
