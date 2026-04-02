from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

from .docx_export import export_docx
from .local_filesystem import filesystem
from .pdf_export import export_pdf

_SAFE_FILENAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SAFE_SUBDIR_PATTERN = re.compile(r"^[a-z0-9][a-z0-9/_-]{0,127}$")
_ENV_STYLE_PATTERN = re.compile(r"^\$?\{?[A-Z][A-Z0-9_]*\}?$")


def _safe_stem(name: str) -> tuple[str, str | None]:
    raw = (name or "").strip()
    if not raw:
        return "report_output", None
    if _ENV_STYLE_PATTERN.fullmatch(raw) or "${" in raw or "$" in raw:
        return "report_output", "base_filename looked like an env var token and was replaced with 'report_output'"
    if "/" in raw or "\\" in raw or ".." in raw:
        return "report_output", "base_filename contained path characters and was replaced with 'report_output'"
    if not _SAFE_FILENAME_PATTERN.fullmatch(raw):
        return "report_output", "base_filename violated safe pattern and was replaced with 'report_output'"
    return raw, None


def _safe_subdir(path: str) -> tuple[str, str | None]:
    raw = (path or "").strip().replace("\\", "/")
    if not raw:
        return "", None
    if raw.startswith("/") or ".." in raw.split("/"):
        return "", "output_subdir must be a safe relative path; fallback to root export directory"
    normalized = raw.strip("/")
    if _ENV_STYLE_PATTERN.fullmatch(normalized) or "${" in normalized or "$" in normalized:
        return "", "output_subdir looked like an env var token and was ignored"
    if not _SAFE_SUBDIR_PATTERN.fullmatch(normalized):
        return "", "output_subdir violated safe pattern and was ignored"
    return normalized, None


@tool
def academic_read(
    query: str = "",
    subdir: str = "",
    glob_pattern: str = "**/*",
    max_files: int = 12,
    max_chars_per_file: int = 4000,
) -> dict[str, Any]:
    """Unified local academic document reading with normalized output."""
    try:
        raw = filesystem.invoke(
            {
                "query": query,
                "subdir": subdir,
                "glob_pattern": glob_pattern,
                "max_files": max_files,
                "max_chars_per_file": max_chars_per_file,
            }
        )
    except Exception as exc:
        return {
            "ok": False,
            "operation": "academic_read",
            "final_text": "academic_read failed to execute filesystem tool",
            "errors": [repr(exc)],
            "artifacts": {"files": []},
        }

    if not isinstance(raw, dict):
        return {
            "ok": False,
            "operation": "academic_read",
            "final_text": "filesystem tool returned non-dict payload",
            "errors": ["FILESYSTEM_INVALID_PAYLOAD"],
            "artifacts": {"files": []},
        }

    ok = bool(raw.get("ok"))
    files = raw.get("files") if isinstance(raw.get("files"), list) else []
    error = str(raw.get("error") or "").strip()

    return {
        "ok": ok,
        "operation": "academic_read",
        "final_text": (
            f"Read {len(files)} file(s) under local academic storage."
            if ok
            else "academic_read finished with errors"
        ),
        "errors": [error] if error else [],
        "artifacts": {
            "root": raw.get("root"),
            "searched_dir": raw.get("searched_dir"),
            "query": raw.get("query"),
            "glob_pattern": raw.get("glob_pattern"),
            "matched_files": raw.get("matched_files"),
            "files": files,
        },
    }


@tool
def academic_export(
    title: str,
    content: str,
    base_filename: str = "report_output",
    output_subdir: str = "",
    export_docx_enabled: bool = True,
    export_pdf_enabled: bool = True,
) -> dict[str, Any]:
    """Unified academic report export to DOCX/PDF with normalized output."""
    stem, stem_warning = _safe_stem(base_filename)
    subdir, subdir_warning = _safe_subdir(output_subdir)
    warnings: list[str] = []
    if stem_warning:
        warnings.append(stem_warning)
    if subdir_warning:
        warnings.append(subdir_warning)

    if not export_docx_enabled and not export_pdf_enabled:
        return {
            "ok": False,
            "operation": "academic_export",
            "final_text": "No export format enabled",
            "errors": ["At least one of export_docx_enabled/export_pdf_enabled must be true"],
            "warnings": warnings,
            "artifacts": {"report_exports": {}},
        }

    def _join(rel_name: str) -> str:
        if not subdir:
            return rel_name
        return f"{subdir}/{rel_name}"

    errors: list[str] = []
    exports: dict[str, str] = {}

    if export_docx_enabled:
        try:
            docx_res = export_docx.invoke(
                {
                    "title": title,
                    "content": content,
                    "output_path": _join(f"{stem}.docx"),
                }
            )
            if isinstance(docx_res, dict) and isinstance(docx_res.get("path"), str):
                exports["docx_path"] = docx_res["path"]
            else:
                errors.append("DOCX export returned invalid payload")
        except Exception as exc:
            errors.append(f"DOCX export failed: {exc!r}")

    if export_pdf_enabled:
        try:
            pdf_res = export_pdf.invoke(
                {
                    "title": title,
                    "content": content,
                    "output_path": _join(f"{stem}.pdf"),
                }
            )
            if isinstance(pdf_res, dict) and isinstance(pdf_res.get("path"), str):
                exports["pdf_path"] = pdf_res["path"]
            else:
                errors.append("PDF export returned invalid payload")
        except Exception as exc:
            errors.append(f"PDF export failed: {exc!r}")

    ok = bool(exports) and not errors
    if exports and errors:
        final_text = "Export partially succeeded"
    elif exports:
        final_text = "Export succeeded"
    else:
        final_text = "Export failed"
    if warnings:
        final_text = f"{final_text} (with validation fallback)"

    return {
        "ok": ok,
        "operation": "academic_export",
        "final_text": final_text,
        "errors": errors,
        "warnings": warnings,
        "artifacts": {"report_exports": exports},
    }
