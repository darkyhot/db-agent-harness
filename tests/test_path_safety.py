"""Тесты безопасной нормализации путей внутри workspace."""

from pathlib import Path

import pytest

from tools.path_safety import resolve_workspace_path


def test_relative_path_inside_workspace(tmp_path: Path):
    resolved = resolve_workspace_path(tmp_path, "reports/out.csv")
    assert resolved == (tmp_path / "reports" / "out.csv").resolve()


def test_parent_traversal_blocked(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_workspace_path(tmp_path, "../outside.txt")


def test_sibling_prefix_attack_blocked(tmp_path: Path):
    sibling = tmp_path.parent / f"{tmp_path.name}_evil" / "file.csv"
    with pytest.raises(ValueError):
        resolve_workspace_path(tmp_path, sibling)


def test_windows_absolute_path_blocked(tmp_path: Path):
    outside = Path("C:/Windows/System32/drivers/etc/hosts")
    with pytest.raises(ValueError):
        resolve_workspace_path(tmp_path, outside)
