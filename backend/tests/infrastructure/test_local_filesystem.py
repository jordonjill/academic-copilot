from __future__ import annotations

from pathlib import Path

from src.infrastructure.tools.local_filesystem import filesystem


def test_filesystem_rejects_subdir_outside_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LOCAL_DOC_ROOT", str(tmp_path))
    result = filesystem.invoke(
        {
            "query": "",
            "subdir": "../outside",
            "glob_pattern": "**/*",
            "max_files": 5,
            "max_chars_per_file": 1000,
        }
    )
    assert result["ok"] is False
    assert "Invalid subdir outside root" in result["error"]


def test_filesystem_skips_symlink_target_outside_root(monkeypatch, tmp_path: Path):
    root = tmp_path / "docs"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir(parents=True)

    (root / "inside.txt").write_text("safe content", encoding="utf-8")
    (outside / "secret.txt").write_text("TOP_SECRET", encoding="utf-8")
    (root / "leak.txt").symlink_to(outside / "secret.txt")

    monkeypatch.setenv("LOCAL_DOC_ROOT", str(root))
    result = filesystem.invoke(
        {
            "query": "",
            "subdir": "",
            "glob_pattern": "**/*",
            "max_files": 10,
            "max_chars_per_file": 2000,
        }
    )

    assert result["ok"] is True
    paths = {entry["path"] for entry in result["files"]}
    assert "inside.txt" in paths
    assert "leak.txt" not in paths
