"""Персистентная память агента на SQLite."""

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    def _get_conn(self) -> sqlite3.Connection:
        """Получить соединение с SQLite."""
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        """Создать таблицы если не существуют."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    summary TEXT,
                    user_id TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT REFERENCES sessions(id),
                    role TEXT,
                    content TEXT,
                    timestamp TEXT
                );

                CREATE TABLE IF NOT EXISTS long_term_memory (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
            """)
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
        with self._get_conn() as conn:
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
        with self._get_conn() as conn:
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

        with self._get_conn() as conn:
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
        with self._get_conn() as conn:
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

        with self._get_conn() as conn:
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

    # --- Long-term memory ---

    def set_memory(self, key: str, value: str) -> None:
        """Записать или обновить значение в долгосрочной памяти.

        Args:
            key: Ключ (например, 'favorite_tables', 'user_preferences').
            value: Значение.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
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
        with self._get_conn() as conn:
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
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT key, value FROM long_term_memory")
            rows = cursor.fetchall()
        return {r[0]: r[1] for r in rows}

    def delete_memory(self, key: str) -> None:
        """Удалить запись из долгосрочной памяти.

        Args:
            key: Ключ для удаления.
        """
        with self._get_conn() as conn:
            conn.execute("DELETE FROM long_term_memory WHERE key = ?", (key,))
        logger.info("Долгосрочная память удалена: key=%s", key)

    @property
    def session_count(self) -> int:
        """Количество сессий с резюме."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE summary != ''"
            )
            return cursor.fetchone()[0]
