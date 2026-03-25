import pytest

from src.infrastructure.tools import export_docx, export_pdf


def test_export_docx_creates_file_in_base_dir(tmp_path, monkeypatch):
    base_dir = tmp_path / "exports"
    monkeypatch.setenv("EXPORT_BASE_DIR", str(base_dir))

    result = export_docx.invoke(
        {
            "title": "Sample Title",
            "content": "First line\nSecond line",
            "output_path": "sample.docx",
        }
    )

    expected_path = base_dir / "sample.docx"
    assert result == {"path": str(expected_path)}
    assert expected_path.exists()
    assert expected_path.stat().st_size > 0


def test_export_pdf_creates_file_in_base_dir(tmp_path, monkeypatch):
    base_dir = tmp_path / "exports"
    monkeypatch.setenv("EXPORT_BASE_DIR", str(base_dir))

    result = export_pdf.invoke(
        {
            "title": "示例标题",
            "content": "第一行\n第二行",
            "output_path": "sample.pdf",
        }
    )

    expected_path = base_dir / "sample.pdf"
    assert result == {"path": str(expected_path)}
    assert expected_path.exists()
    assert expected_path.stat().st_size > 0


def test_export_rejects_paths_outside_base_dir(tmp_path, monkeypatch):
    base_dir = tmp_path / "exports"
    monkeypatch.setenv("EXPORT_BASE_DIR", str(base_dir))

    outside_path = tmp_path / "outside.docx"
    with pytest.raises(ValueError):
        export_docx.invoke(
            {
                "title": "Nope",
                "content": "Blocked",
                "output_path": str(outside_path),
            }
        )
