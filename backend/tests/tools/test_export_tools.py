import pytest

from src.infrastructure.tools import academic_export, academic_read, export_docx, export_pdf


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


def test_academic_read_returns_normalized_payload(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "note.txt").write_text("transformer baseline details", encoding="utf-8")
    monkeypatch.setenv("LOCAL_DOC_ROOT", str(docs_dir))

    result = academic_read.invoke({"query": "transformer"})

    assert result["ok"] is True
    assert result["operation"] == "academic_read"
    assert "artifacts" in result
    assert result["artifacts"]["matched_files"] >= 1
    assert isinstance(result["artifacts"]["files"], list)
    assert result["artifacts"]["files"][0]["path"] == "note.txt"


def test_academic_export_writes_docx_and_pdf(tmp_path, monkeypatch):
    base_dir = tmp_path / "exports"
    monkeypatch.setenv("EXPORT_BASE_DIR", str(base_dir))

    result = academic_export.invoke(
        {
            "title": "Proposal",
            "content": "line1\nline2",
            "base_filename": "proposal_v1",
            "output_subdir": "academic",
        }
    )

    assert result["ok"] is True
    exports = result["artifacts"]["report_exports"]
    assert "docx_path" in exports
    assert "pdf_path" in exports
    assert (base_dir / "academic" / "proposal_v1.docx").exists()
    assert (base_dir / "academic" / "proposal_v1.pdf").exists()


def test_academic_export_ignores_env_style_output_subdir(tmp_path, monkeypatch):
    base_dir = tmp_path / "exports"
    monkeypatch.setenv("EXPORT_BASE_DIR", str(base_dir))

    result = academic_export.invoke(
        {
            "title": "Proposal",
            "content": "line1\nline2",
            "base_filename": "report_output",
            "output_subdir": "EXPORT_BASE_DIR",
        }
    )

    assert result["ok"] is True
    assert isinstance(result.get("warnings"), list)
    assert any("output_subdir looked like an env var token" in msg for msg in result["warnings"])
    exports = result["artifacts"]["report_exports"]
    assert (base_dir / "report_output.docx").exists()
    assert (base_dir / "report_output.pdf").exists()
    assert "EXPORT_BASE_DIR" not in exports["docx_path"]
    assert "EXPORT_BASE_DIR" not in exports["pdf_path"]


def test_academic_export_replaces_unsafe_base_filename(tmp_path, monkeypatch):
    base_dir = tmp_path / "exports"
    monkeypatch.setenv("EXPORT_BASE_DIR", str(base_dir))

    result = academic_export.invoke(
        {
            "title": "Proposal",
            "content": "line1\nline2",
            "base_filename": "${EXPORT_BASE_DIR}",
        }
    )

    assert result["ok"] is True
    assert isinstance(result.get("warnings"), list)
    assert any("base_filename looked like an env var token" in msg for msg in result["warnings"])
    exports = result["artifacts"]["report_exports"]
    assert (base_dir / "report_output.docx").exists()
    assert (base_dir / "report_output.pdf").exists()
    assert exports["docx_path"].endswith("report_output.docx")
    assert exports["pdf_path"].endswith("report_output.pdf")
