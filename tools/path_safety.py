"""Безопасная работа с путями внутри workspace."""

from pathlib import Path


def resolve_workspace_path(workspace_dir: Path, path: str | Path) -> Path:
    """Нормализовать путь и гарантировать, что он внутри workspace.

    Args:
        workspace_dir: Корень workspace.
        path: Пользовательский путь (относительный или абсолютный).

    Returns:
        Абсолютный Path внутри workspace.

    Raises:
        ValueError: Если путь выходит за пределы workspace.
    """
    workspace_root = workspace_dir.resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (workspace_root / candidate).resolve()

    try:
        resolved.relative_to(workspace_root)
    except ValueError as e:
        raise ValueError(f"Путь выходит за пределы workspace: {path}") from e

    return resolved
