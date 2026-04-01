"""Персистентная память агента на SQLite."""

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory"
DB_PATH = MEMORY_DIR / "agent_memory.db"


class MemoryManager:
    """Управление персистентной памятью агента через SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Инициализация менеджера памяти.

        Args:
            db_path: Путь к файлу SQLite. По умолчанию — memory/agent_memory.db.
        """
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id: str | None = None
        self._init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Открыть соединение, выполнить операцию, закрыть."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30,
            check_same_thread=False,
        )
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        """Совместимость с прежним API (соединения теперь закрываются автоматически)."""

    def _init_db(self) -> None:
        """Создать таблицы если не существуют."""
        # WAL mode персистентен в файле — ставим один раз при инициализации.
        # Используем отдельное короткое соединение, чтобы не блокировать надолго.
        try:
            wal_conn = sqlite3.connect(str(self._db_path), timeout=5)
            wal_conn.execute("PRAGMA busy_timeout=5000")
            mode = wal_conn.execute("PRAGMA journal_mode").fetchone()
            if mode and mode[0].lower() != "wal":
                wal_conn.execute("PRAGMA journal_mode=WAL")
                logger.info("SQLite journal_mode переключён на WAL")
            wal_conn.close()
        except Exception as e:
            logger.warning("Не удалось установить WAL mode: %s (продолжаем)", e)

        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    summary TEXT,
                    user_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id),
                    role TEXT,
                    content TEXT,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sql_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp TEXT,
                    user_input TEXT,
                    sql TEXT,
                    row_count INTEGER,
                    status TEXT,
                    duration_ms INTEGER,
                    retry_count INTEGER DEFAULT 0,
                    error_type TEXT DEFAULT ''
                )
            """)
            # Миграция: добавить новые колонки для существующих БД
            try:
                conn.execute("ALTER TABLE sql_audit ADD COLUMN retry_count INTEGER DEFAULT 0")
            except Exception:
                pass  # колонка уже существует
            try:
                conn.execute("ALTER TABLE sql_audit ADD COLUMN error_type TEXT DEFAULT ''")
            except Exception:
                pass  # колонка уже существует

            # Индексы для ускорения частых запросов
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session "
                "ON messages(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sql_audit_session "
                "ON sql_audit(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sql_audit_timestamp "
                "ON sql_audit(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_timestamp "
                "ON sessions(timestamp)"
            )
        logger.info("SQLite память инициализирована: %s", self._db_path)

    def start_session(self, user_id: str = "") -> str:
        """Начать новую сессию.

        Args:
            user_id: Идентификатор пользователя.

        Returns:
            ID новой сессии.
        """
        self._session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, timestamp, summary, user_id) VALUES (?, ?, ?, ?)",
                (self._session_id, now, "", user_id),
            )
        logger.info("Начата сессия: %s", self._session_id)
        return self._session_id

    @property
    def session_id(self) -> str | None:
        """ID текущей сессии."""
        return self._session_id

    def add_message(self, role: str, content: str) -> None:
        """Записать сообщение в текущую сессию.

        Args:
            role: Роль отправителя ('user', 'assistant', 'tool').
            content: Текст сообщения.
        """
        if not self._session_id:
            logger.warning("Нет активной сессии, сообщение не записано")
            return

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (self._session_id, role, content, now),
            )

    def save_session_summary(self, summary: str) -> None:
        """Сохранить резюме текущей сессии.

        Args:
            summary: Текст резюме, сгенерированный LLM.
        """
        if not self._session_id:
            logger.warning("Нет активной сессии для сохранения резюме")
            return

        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET summary = ? WHERE id = ?",
                (summary, self._session_id),
            )
        logger.info("Резюме сессии сохранено: %s", self._session_id)

    def get_recent_sessions(self, n: int = 5) -> list[dict[str, str]]:
        """Получить резюме последних N сессий.

        Args:
            n: Количество сессий.

        Returns:
            Список словарей с id, timestamp, summary.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT id, timestamp, summary FROM sessions "
                "WHERE summary != '' ORDER BY timestamp DESC LIMIT ?",
                (n,),
            )
            rows = cursor.fetchall()

        return [
            {"id": r[0], "timestamp": r[1], "summary": r[2]}
            for r in rows
        ]

    def get_sessions_context(self, n: int = 5) -> str:
        """Сформировать контекст из последних сессий для системного промпта.

        Args:
            n: Количество сессий.

        Returns:
            Форматированная строка с резюме.
        """
        sessions = self.get_recent_sessions(n)
        if not sessions:
            return "Предыдущих сессий нет."

        lines = ["Резюме предыдущих сессий:"]
        for s in reversed(sessions):
            lines.append(f"  [{s['timestamp'][:10]}] {s['summary']}")
        return "\n".join(lines)

    def get_session_messages(self, session_id: str | None = None) -> list[dict[str, str]]:
        """Получить сообщения сессии.

        Args:
            session_id: ID сессии. По умолчанию — текущая.

        Returns:
            Список сообщений с role и content.
        """
        sid = session_id or self._session_id
        if not sid:
            return []

        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT role, content, timestamp FROM messages "
                "WHERE session_id = ? ORDER BY id",
                (sid,),
            )
            rows = cursor.fetchall()

        return [
            {"role": r[0], "content": r[1], "timestamp": r[2]}
            for r in rows
        ]

    def get_unsummarized_sessions(self) -> list[str]:
        """Получить ID сессий, у которых есть сообщения, но нет резюме.

        Используется для восстановления памяти после аварийного завершения.

        Returns:
            Список ID незавершённых сессий (исключая текущую).
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT s.id FROM sessions s "
                "WHERE (s.summary IS NULL OR s.summary = '') "
                "AND s.id != ? "
                "AND EXISTS (SELECT 1 FROM messages m WHERE m.session_id = s.id) "
                "ORDER BY s.timestamp DESC LIMIT 5",
                (self._session_id or "",),
            )
            return [row[0] for row in cursor.fetchall()]

    # --- SQL audit ---

    def log_sql_execution(
        self,
        user_input: str,
        sql: str,
        row_count: int,
        status: str,
        duration_ms: int,
        retry_count: int = 0,
        error_type: str = "",
    ) -> None:
        """Записать выполненный SQL в аудит-лог.

        Args:
            user_input: Исходный запрос пользователя.
            sql: Выполненный SQL-запрос.
            row_count: Количество строк в результате.
            status: Статус выполнения ('success', 'empty', 'error', 'row_explosion').
            duration_ms: Время выполнения в миллисекундах.
            retry_count: Количество попыток коррекции до успешного выполнения.
            error_type: Тип ошибки (например, 'syntax', 'join_explosion', 'missing_column').
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO sql_audit "
                    "(session_id, timestamp, user_input, sql, row_count, status, "
                    "duration_ms, retry_count, error_type) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (self._session_id or "", now, user_input, sql, row_count, status,
                     duration_ms, retry_count, error_type),
                )
        except Exception as e:
            logger.warning("Ошибка записи SQL-аудита: %s", e)

    # --- Long-term memory ---

    def set_memory(self, key: str, value: str) -> None:
        """Записать или обновить значение в долгосрочной памяти.

        Args:
            key: Ключ (например, 'favorite_tables', 'user_preferences').
            value: Значение.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO long_term_memory (key, value, updated_at) "
                "VALUES (?, ?, ?)",
                (key, value, now),
            )
        logger.info("Долгосрочная память обновлена: key=%s", key)

    def get_memory(self, key: str) -> str | None:
        """Получить значение из долгосрочной памяти.

        Args:
            key: Ключ.

        Returns:
            Значение или None если не найдено.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT value FROM long_term_memory WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
        return row[0] if row else None

    def get_all_memory(self) -> dict[str, str]:
        """Получить всю долгосрочную память.

        Returns:
            Словарь key -> value.
        """
        with self._connect() as conn:
            cursor = conn.execute("SELECT key, value FROM long_term_memory")
            rows = cursor.fetchall()
        return {r[0]: r[1] for r in rows}

    def get_memory_list(self, key: str) -> list[str]:
        """Получить значение из долгосрочной памяти как JSON-список.

        Args:
            key: Ключ.

        Returns:
            Список строк или пустой список если не найдено / ошибка парсинга.
        """
        raw = self.get_memory(key)
        if not raw:
            return []
        try:
            val = json.loads(raw)
            return val if isinstance(val, list) else []
        except (json.JSONDecodeError, ValueError):
            return []

    def delete_memory(self, key: str) -> None:
        """Удалить запись из долгосрочной памяти.

        Args:
            key: Ключ для удаления.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM long_term_memory WHERE key = ?", (key,))
        logger.info("Долгосрочная память удалена: key=%s", key)

    def cleanup_old_sessions(self, keep_days: int = 90) -> int:
        """Удалить сессии и их сообщения старше keep_days дней.

        Args:
            keep_days: Сколько дней хранить (по умолчанию 90).

        Returns:
            Количество удалённых сессий.
        """
        cutoff = datetime.now(timezone.utc).isoformat()
        # Вычисляем дату отсечения (ISO формат сортируется лексикографически)
        from datetime import timedelta
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=keep_days)
        cutoff = cutoff_dt.isoformat()

        with self._connect() as conn:
            # Сначала считаем сколько удалим
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE timestamp < ?", (cutoff,)
            )
            count = cursor.fetchone()[0]

            if count > 0:
                # Удаляем сообщения старых сессий
                conn.execute(
                    "DELETE FROM messages WHERE session_id IN "
                    "(SELECT id FROM sessions WHERE timestamp < ?)",
                    (cutoff,),
                )
                # Удаляем записи аудита старых сессий
                conn.execute(
                    "DELETE FROM sql_audit WHERE session_id IN "
                    "(SELECT id FROM sessions WHERE timestamp < ?)",
                    (cutoff,),
                )
                # Удаляем сессии
                conn.execute("DELETE FROM sessions WHERE timestamp < ?", (cutoff,))
                logger.info("Очищено %d старых сессий (старше %d дней)", count, keep_days)

        return count

    def get_sql_quality_metrics(self, days: int = 30) -> dict[str, Any]:
        """Получить метрики качества генерации SQL за указанный период.

        Args:
            days: Период в днях для анализа.

        Returns:
            Словарь с метриками качества.
        """
        from datetime import timedelta
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff = cutoff_dt.isoformat()

        with self._connect() as conn:
            # Общее количество SQL-запросов
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sql_audit WHERE timestamp >= ?", (cutoff,)
            )
            total = cursor.fetchone()[0]

            if total == 0:
                return {"total_queries": 0, "period_days": days}

            # Распределение по статусам
            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM sql_audit WHERE timestamp >= ? GROUP BY status",
                (cutoff,),
            )
            status_dist = dict(cursor.fetchall())

            # Успешность с первой попытки (retry_count = 0 и status = 'success')
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sql_audit "
                "WHERE timestamp >= ? AND status = 'success' AND retry_count = 0",
                (cutoff,),
            )
            first_try_success = cursor.fetchone()[0]

            # Среднее количество retry
            cursor = conn.execute(
                "SELECT AVG(retry_count), MAX(retry_count) FROM sql_audit WHERE timestamp >= ?",
                (cutoff,),
            )
            avg_retry, max_retry = cursor.fetchone()

            # Среднее время выполнения
            cursor = conn.execute(
                "SELECT AVG(duration_ms), MAX(duration_ms) FROM sql_audit WHERE timestamp >= ?",
                (cutoff,),
            )
            avg_duration, max_duration = cursor.fetchone()

            # Распределение типов ошибок
            cursor = conn.execute(
                "SELECT error_type, COUNT(*) FROM sql_audit "
                "WHERE timestamp >= ? AND error_type != '' GROUP BY error_type",
                (cutoff,),
            )
            error_dist = dict(cursor.fetchall())

            success_count = status_dist.get("success", 0)
            return {
                "total_queries": total,
                "period_days": days,
                "success_rate": round(success_count / total * 100, 1) if total > 0 else 0,
                "first_try_success_rate": round(first_try_success / total * 100, 1) if total > 0 else 0,
                "status_distribution": status_dist,
                "avg_retries": round(float(avg_retry or 0), 2),
                "max_retries": int(max_retry or 0),
                "avg_duration_ms": round(float(avg_duration or 0), 0),
                "max_duration_ms": int(max_duration or 0),
                "error_distribution": error_dist,
            }

    @property
    def session_count(self) -> int:
        """Количество сессий с резюме."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE summary != ''"
            )
            return cursor.fetchone()[0]
