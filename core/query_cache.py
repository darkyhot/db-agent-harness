"""Кэш для идентичных/похожих запросов пользователя.

Хранит результат (sql + final_answer) с TTL 1 час в SQLite.
Ключ — SHA-256 от нормализованного user_input.
Интегрируется в CLIInterface до запуска графа.
"""

import hashlib
import logging
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory import MemoryManager

logger = logging.getLogger(__name__)

# TTL кэша — 1 час
_CACHE_TTL_SECONDS = 3600

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_cache (
    key TEXT PRIMARY KEY,
    user_input TEXT NOT NULL,
    sql TEXT,
    final_answer TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


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
    """SQLite-backed кэш с TTL для повторных запросов."""

    def __init__(self, memory: "MemoryManager") -> None:
        self._memory = memory
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Создать таблицу кэша если не существует."""
        try:
            with self._memory._connect() as conn:
                conn.execute(_CREATE_TABLE_SQL)
                conn.commit()
        except Exception as e:
            logger.warning("QueryCache: ошибка создания таблицы: %s", e)

    def get(self, user_input: str) -> dict | None:
        """Найти кэшированный результат.

        Returns:
            Словарь {"sql": str, "final_answer": str, "created_at": str}
            или None если кэш промах / устарел.
        """
        key = _cache_key(user_input)
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=_CACHE_TTL_SECONDS)).isoformat()

        try:
            with self._memory._connect() as conn:
                row = conn.execute(
                    "SELECT sql, final_answer, created_at FROM query_cache "
                    "WHERE key = ? AND created_at >= ?",
                    (key, cutoff),
                ).fetchone()

            if row:
                logger.info("QueryCache: попадание в кэш (key=%s)", key[:12])
                return {"sql": row[0], "final_answer": row[1], "created_at": row[2]}
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
            with self._memory._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO query_cache (key, user_input, sql, final_answer, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, user_input, sql or "", final_answer, created_at),
                )
                conn.commit()
            logger.info("QueryCache: сохранён результат (key=%s)", key[:12])
        except Exception as e:
            logger.warning("QueryCache: ошибка записи: %s", e)

    def invalidate(self, user_input: str) -> None:
        """Удалить конкретный запрос из кэша."""
        key = _cache_key(user_input)
        try:
            with self._memory._connect() as conn:
                conn.execute("DELETE FROM query_cache WHERE key = ?", (key,))
                conn.commit()
        except Exception as e:
            logger.warning("QueryCache: ошибка удаления: %s", e)

    def clear_expired(self) -> int:
        """Очистить устаревшие записи. Возвращает количество удалённых."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=_CACHE_TTL_SECONDS)).isoformat()
        try:
            with self._memory._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM query_cache WHERE created_at < ?", (cutoff,)
                )
                conn.commit()
                return cur.rowcount
        except Exception as e:
            logger.warning("QueryCache: ошибка очистки: %s", e)
            return 0

    def clear_all(self) -> None:
        """Полностью очистить кэш."""
        try:
            with self._memory._connect() as conn:
                conn.execute("DELETE FROM query_cache")
                conn.commit()
            logger.info("QueryCache: кэш очищен")
        except Exception as e:
            logger.warning("QueryCache: ошибка очистки: %s", e)
