"""Инструменты файловой системы для агента. Все операции внутри workspace/."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_path(path: str) -> Path:
    """Получить безопасный путь внутри workspace.

    Args:
        path: Относительный путь.

    Returns:
        Абсолютный путь внутри workspace.

    Raises:
        ValueError: Если путь выходит за пределы workspace.
    """
    resolved = (WORKSPACE_DIR / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE_DIR.resolve())):
        raise ValueError(f"Путь выходит за пределы workspace: {path}")
    return resolved


@tool
def create_file(path: str, content: str) -> str:
    """Создать файл в рабочей директории workspace/.

    Args:
        path: Относительный путь файла внутри workspace.
        content: Содержимое файла.

    Returns:
        Сообщение об успехе или ошибке.
    """
    try:
        file_path = _safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        logger.info("Создан файл: %s", file_path)
        return f"Файл создан: {path}"
    except Exception as e:
        logger.error("Ошибка создания файла %s: %s", path, e)
        return f"Ошибка: {e}"


@tool
def read_file(path: str) -> str:
    """Прочитать файл из рабочей директории workspace/.

    Args:
        path: Относительный путь файла внутри workspace.

    Returns:
        Содержимое файла или сообщение об ошибке.
    """
    try:
        file_path = _safe_path(path)
        if not file_path.exists():
            return f"Файл не найден: {path}"
        content = file_path.read_text(encoding="utf-8")
        logger.info("Прочитан файл: %s (%d символов)", file_path, len(content))
        return content
    except Exception as e:
        logger.error("Ошибка чтения файла %s: %s", path, e)
        return f"Ошибка: {e}"


@tool
def edit_file(path: str, content: str) -> str:
    """Перезаписать файл в рабочей директории workspace/.

    Args:
        path: Относительный путь файла внутри workspace.
        content: Новое содержимое файла.

    Returns:
        Сообщение об успехе или ошибке.
    """
    try:
        file_path = _safe_path(path)
        if not file_path.exists():
            return f"Файл не найден: {path}. Используйте create_file для создания."
        file_path.write_text(content, encoding="utf-8")
        logger.info("Файл перезаписан: %s", file_path)
        return f"Файл обновлён: {path}"
    except Exception as e:
        logger.error("Ошибка редактирования файла %s: %s", path, e)
        return f"Ошибка: {e}"


@tool
def delete_file(path: str) -> str:
    """Удалить файл из рабочей директории workspace/.

    Args:
        path: Относительный путь файла внутри workspace.

    Returns:
        Сообщение об успехе или ошибке.
    """
    try:
        file_path = _safe_path(path)
        if not file_path.exists():
            return f"Файл не найден: {path}"
        file_path.unlink()
        logger.info("Удалён файл: %s", file_path)
        return f"Файл удалён: {path}"
    except Exception as e:
        logger.error("Ошибка удаления файла %s: %s", path, e)
        return f"Ошибка: {e}"


@tool
def list_files(subdir: str = "") -> str:
    """Показать список файлов в рабочей директории workspace/.

    Args:
        subdir: Поддиректория внутри workspace (по умолчанию — корень workspace).

    Returns:
        Список файлов и папок.
    """
    try:
        dir_path = _safe_path(subdir) if subdir else WORKSPACE_DIR
        if not dir_path.exists():
            return f"Директория не найдена: {subdir}"

        items = sorted(dir_path.iterdir())
        if not items:
            return "Директория пуста."

        lines = []
        for item in items:
            rel = item.relative_to(WORKSPACE_DIR)
            if item.is_dir():
                lines.append(f"📁 {rel}/")
            else:
                size = item.stat().st_size
                lines.append(f"📄 {rel} ({size} bytes)")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Ошибка листинга %s: %s", subdir, e)
        return f"Ошибка: {e}"


@tool
def create_directory(path: str) -> str:
    """Создать папку в рабочей директории workspace/.

    Args:
        path: Относительный путь папки внутри workspace.

    Returns:
        Сообщение об успехе или ошибке.
    """
    try:
        dir_path = _safe_path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info("Создана директория: %s", dir_path)
        return f"Директория создана: {path}"
    except Exception as e:
        logger.error("Ошибка создания директории %s: %s", path, e)
        return f"Ошибка: {e}"


@tool
def save_dataframe(data_json: str, filename: str, format: str = "csv") -> str:
    """Сохранить данные (JSON-строку DataFrame) в файл в workspace/.

    Args:
        data_json: JSON-строка с данными (формат pandas orient='records').
        filename: Имя файла для сохранения.
        format: Формат файла — 'csv' или 'excel'.

    Returns:
        Сообщение об успехе или ошибке.
    """
    try:
        file_path = _safe_path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_json(data_json, orient="records")

        if format == "excel":
            df.to_excel(file_path, index=False)
        else:
            df.to_csv(file_path, index=False, encoding="utf-8")

        logger.info("DataFrame сохранён: %s (%d строк)", file_path, len(df))
        return f"Данные сохранены в {filename} ({len(df)} строк)"
    except Exception as e:
        logger.error("Ошибка сохранения DataFrame %s: %s", filename, e)
        return f"Ошибка: {e}"


# Список всех инструментов для регистрации в агенте
FS_TOOLS = [
    create_file,
    read_file,
    edit_file,
    delete_file,
    list_files,
    create_directory,
    save_dataframe,
]
