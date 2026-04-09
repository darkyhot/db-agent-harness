"""Кэш для идентичных/похожих запросов пользователя.

Хранит результат (sql + final_answer) с TTL 1 час в JSON-файле.
Ключ — SHA-256 от нормализованного user_input.
Интегрируется в CLIInterface до запуска графа.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory import MemoryManager

from core.memory import _load_json, _write_json_atomic

logger = logging.getLogger(__name__)

# TTL кэша — 1 час
_CACHE_TTL_SECONDS = 3600


def _normalize(user_input: str) -> str:
    """Нормализовать запрос: lowercase, убрать лишние пробелы и пунктуацию."""
    text = user_input.lower().strip()
    text = re.sub(r'[^\w\sа-яёА-ЯЁ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _cache_key(user_input: str) -> str:
    """Вычислить SHA-256 ключ по нормализованному запросу."""
    normalized = _normalize(user_input)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class QueryCache:
    """JSON-backed кэш с TTL для повторных запросов."""

    def __init__(self, memory: "MemoryManager") -> None:
        self._cache_path: Path = memory._memory_dir / "query_cache.json"

    def _load(self) -> dict:
        return _load_json(self._cache_path, {})

    def _save(self, data: dict) -> None:
        _write_json_atomic(self._cache_path, data)

    def get(self, user_input: str) -> dict | None:
        """Найти кэшированный результат.

        Returns:
            Словарь {"sql": str, "final_answer": str, "created_at": str}
            или None если кэш промах / устарел.
        """
        key = _cache_key(user_input)
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=_CACHE_TTL_SECONDS)
        ).isoformat()

        try:
            cache = self._load()
            entry = cache.get(key)
            if entry and entry.get("created_at", "") >= cutoff:
                logger.info("QueryCache: попадание в кэш (key=%s)", key[:12])
                return {
                    "sql": entry.get("sql", ""),
                    "final_answer": entry["final_answer"],
                    "created_at": entry["created_at"],
                }
        except Exception as e:
            logger.warning("QueryCache: ошибка чтения: %s", e)

        return None

    def put(self, user_input: str, final_answer: str, sql: str | None = None) -> None:
        """Сохранить результат в кэш.

        Args:
            user_input: Оригинальный запрос пользователя.
            final_answer: Финальный ответ агента.
            sql: Выполненный SQL (необязательно).
        """
        key = _cache_key(user_input)
        created_at = datetime.now(timezone.utc).isoformat()

        try:
            cache = self._load()
            cache[key] = {
                "user_input": user_input,
                "sql": sql or "",
                "final_answer": final_answer,
                "created_at": created_at,
            }
            self._save(cache)
            logger.info("QueryCache: сохранён результат (key=%s)", key[:12])
        except Exception as e:
            logger.warning("QueryCache: ошибка записи: %s", e)

    def invalidate(self, user_input: str) -> None:
        """Удалить конкретный запрос из кэша."""
        key = _cache_key(user_input)
        try:
            cache = self._load()
            if key in cache:
                del cache[key]
                self._save(cache)
        except Exception as e:
            logger.warning("QueryCache: ошибка удаления: %s", e)

    def clear_expired(self) -> int:
        """Очистить устаревшие записи. Возвращает количество удалённых."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(seconds=_CACHE_TTL_SECONDS)
        ).isoformat()
        try:
            cache = self._load()
            expired = [k for k, v in cache.items() if v.get("created_at", "") < cutoff]
            for k in expired:
                del cache[k]
            if expired:
                self._save(cache)
            return len(expired)
        except Exception as e:
            logger.warning("QueryCache: ошибка очистки: %s", e)
            return 0

    def clear_all(self) -> None:
        """Полностью очистить кэш."""
        try:
            self._save({})
            logger.info("QueryCache: кэш очищен")
        except Exception as e:
            logger.warning("QueryCache: ошибка очистки: %s", e)
