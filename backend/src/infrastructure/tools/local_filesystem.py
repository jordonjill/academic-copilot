from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_DEFAULT_MAX_FILES = 12
_DEFAULT_MAX_CHARS_PER_FILE = 4000
_PDF_SUFFIX = ".pdf"
_TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sh",
    ".sql",
}


def _resolve_root() -> Path:
    raw = os.getenv("LOCAL_DOC_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    # backend/src/infrastructure/tools/local_filesystem.py -> backend/
    return Path(__file__).resolve().parents[3]


def _safe_dir(root: Path, subdir: str) -> Path:
    if not subdir.strip():
        return root
    candidate = (root / subdir).resolve()
    candidate.relative_to(root)
    return candidate


def _is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return True
    try:
        with path.open("rb") as handle:
            chunk = handle.read(1024)
        return b"\x00" not in chunk
    except Exception:
        return False


def _read_pdf_text(path: Path, max_chars: int) -> tuple[str, bool, bool, str | None]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", False, False, "PDF support requires dependency: pypdf"

    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        used = 0
        truncated_by_chars = False
        has_more_pages = False
        total_pages = len(reader.pages)

        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text:
                continue
            if used >= max_chars:
                truncated_by_chars = True
                has_more_pages = idx < total_pages
                break
            remain = max_chars - used
            if len(text) > remain:
                parts.append(text[:remain])
                used += remain
                truncated_by_chars = True
                has_more_pages = True
                break
            parts.append(text)
            used += len(text)

        return "".join(parts), truncated_by_chars, has_more_pages, None
    except Exception as exc:
        return "", False, False, f"PDF read failed: {exc!r}"


@tool("filesystem")
def filesystem(
    query: str = "",
    subdir: str = "",
    glob_pattern: str = "**/*",
    max_files: int = _DEFAULT_MAX_FILES,
    max_chars_per_file: int = _DEFAULT_MAX_CHARS_PER_FILE,
) -> dict[str, Any]:
    """
    Search and read local text/PDF files under LOCAL_DOC_ROOT.

    - query: optional keyword filter (matched against path/content, case-insensitive)
    - subdir: optional subdirectory under LOCAL_DOC_ROOT
    - glob_pattern: file pattern, default '**/*'
    - max_files: max returned files
    - max_chars_per_file: max chars per file content
    """
    root = _resolve_root()
    if not root.exists():
        return {
            "ok": False,
            "error": f"LOCAL_DOC_ROOT does not exist: {root}",
            "root": str(root),
            "files": [],
        }

    try:
        search_dir = _safe_dir(root, subdir)
    except Exception:
        return {
            "ok": False,
            "error": f"Invalid subdir outside root: {subdir}",
            "root": str(root),
            "files": [],
        }

    if not search_dir.exists():
        return {
            "ok": False,
            "error": f"Search directory does not exist: {search_dir}",
            "root": str(root),
            "files": [],
        }

    pattern = (glob_pattern or "**/*").strip() or "**/*"
    needle = (query or "").strip().casefold()
    max_files = max(1, min(int(max_files), 100))
    max_chars_per_file = max(200, min(int(max_chars_per_file), 12000))

    matched: list[dict[str, Any]] = []
    scanned_files = 0

    for path in search_dir.glob(pattern):
        if not path.is_file():
            continue
        scanned_files += 1

        rel_path = str(path.relative_to(root))
        suffix = path.suffix.lower()

        if suffix == _PDF_SUFFIX:
            text, truncated_by_chars, has_more_pages, error = _read_pdf_text(path, max_chars_per_file)
            if needle and needle not in rel_path.casefold() and needle not in text.casefold():
                continue
            truncated = truncated_by_chars or has_more_pages
            item: dict[str, Any] = {
                "path": rel_path,
                "chars": len(text),
                "truncated": truncated,
                "truncated_by_chars": truncated_by_chars,
                "has_more_pages": has_more_pages,
                "content": text,
            }
            if error:
                item["error"] = error
            matched.append(item)
            if len(matched) >= max_files:
                break
            continue

        if not _is_probably_text(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if needle and needle not in rel_path.casefold() and needle not in text.casefold():
            continue

        content = text[:max_chars_per_file]
        matched.append(
            {
                "path": rel_path,
                "chars": len(text),
                "truncated": len(text) > len(content),
                "content": content,
            }
        )
        if len(matched) >= max_files:
            break

    return {
        "ok": True,
        "root": str(root),
        "searched_dir": str(search_dir),
        "glob_pattern": pattern,
        "query": query,
        "scanned_files": scanned_files,
        "matched_files": len(matched),
        "files": matched,
    }
