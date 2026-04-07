"""Тесты MemoryManager: сессии, сообщения, долгосрочная память, очистка."""

import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from core.memory import MemoryManager


@pytest.fixture
def memory(tmp_path):
    """Создать MemoryManager с временной БД."""
    db_path = tmp_path / "test_memory.db"
    return MemoryManager(db_path=db_path)


class TestSessions:
    def test_start_session(self, memory):
        sid = memory.start_session("user1")
        assert sid is not None
        assert memory.session_id == sid

    def test_add_and_get_messages(self, memory):
        memory.start_session("user1")
        memory.add_message("user", "Привет")
        memory.add_message("assistant", "Здравствуйте!")

        messages = memory.get_session_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "Здравствуйте!"

    def test_no_session_no_message(self, memory):
        # Без старта сессии сообщение не записывается
        memory.add_message("user", "test")
        assert memory.get_session_messages() == []

    def test_session_summary(self, memory):
        memory.start_session("user1")
        memory.save_session_summary("Тестовое резюме")

        sessions = memory.get_recent_sessions(5)
        assert len(sessions) == 1
        assert sessions[0]["summary"] == "Тестовое резюме"


class TestLongTermMemory:
    def test_set_and_get(self, memory):
        memory.set_memory("key1", "value1")
        assert memory.get_memory("key1") == "value1"

    def test_get_nonexistent(self, memory):
        assert memory.get_memory("missing") is None

    def test_get_memory_list(self, memory):
        memory.set_memory("facts", json.dumps(["fact1", "fact2"]))
        result = memory.get_memory_list("facts")
        assert result == ["fact1", "fact2"]

    def test_get_memory_list_invalid_json(self, memory):
        memory.set_memory("bad", "not json")
        assert memory.get_memory_list("bad") == []

    def test_delete_memory(self, memory):
        memory.set_memory("key1", "value1")
        memory.delete_memory("key1")
        assert memory.get_memory("key1") is None

    def test_get_all_memory(self, memory):
        memory.set_memory("a", "1")
        memory.set_memory("b", "2")
        all_mem = memory.get_all_memory()
        assert all_mem == {"a": "1", "b": "2"}


class TestCleanup:
    def test_cleanup_old_sessions(self, memory):
        # Создаём «старую» сессию вручную
        from core.memory import sqlite3
        old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()

        with memory._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, timestamp, summary, user_id) VALUES (?, ?, ?, ?)",
                ("old-session", old_time, "old summary", "user1"),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                ("old-session", "user", "old message", old_time),
            )

        # Создаём свежую сессию
        memory.start_session("user1")
        memory.add_message("user", "fresh message")

        # Очищаем старше 90 дней
        deleted = memory.cleanup_old_sessions(keep_days=90)
        assert deleted == 1

        # Свежая сессия осталась
        messages = memory.get_session_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == "fresh message"

    def test_cleanup_nothing_to_delete(self, memory):
        memory.start_session("user1")
        memory.add_message("user", "test")
        deleted = memory.cleanup_old_sessions(keep_days=90)
        assert deleted == 0
