"""Тесты для core/query_cache.py."""

import time
from unittest.mock import MagicMock, patch
import sqlite3
import pytest

from core.query_cache import QueryCache, _normalize, _cache_key


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases(self):
        assert _normalize("ПОКАЖИ КЛИЕНТОВ") == "покажи клиентов"

    def test_strips_whitespace(self):
        assert _normalize("  покажи  клиентов  ") == "покажи клиентов"

    def test_removes_punctuation(self):
        result = _normalize("покажи, клиентов!")
        assert "," not in result
        assert "!" not in result

    def test_collapses_spaces(self):
        result = _normalize("покажи    клиентов")
        assert "  " not in result


class TestCacheKey:
    def test_same_input_same_key(self):
        assert _cache_key("покажи клиентов") == _cache_key("покажи клиентов")

    def test_different_input_different_key(self):
        assert _cache_key("покажи клиентов") != _cache_key("покажи заказы")

    def test_normalized_same_key(self):
        # Пробелы и регистр не влияют на ключ
        assert _cache_key("покажи клиентов") == _cache_key("  Покажи  Клиентов  ")

    def test_key_is_hex_string(self):
        key = _cache_key("test")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# QueryCache через in-memory SQLite
# ---------------------------------------------------------------------------

def _make_cache():
    """Создать QueryCache с in-memory SQLite."""
    conn = sqlite3.connect(":memory:")

    memory_mock = MagicMock()
    memory_mock._connect.return_value.__enter__ = lambda s: conn
    memory_mock._connect.return_value.__exit__ = MagicMock(return_value=False)

    cache = QueryCache(memory_mock)
    return cache, conn


class TestQueryCachePutGet:
    def setup_method(self):
        self.cache, self.conn = _make_cache()

    def test_put_and_get(self):
        self.cache.put("покажи клиентов", "Клиентов: 100", sql="SELECT COUNT(*) FROM dm.clients")
        result = self.cache.get("покажи клиентов")
        assert result is not None
        assert result["final_answer"] == "Клиентов: 100"
        assert result["sql"] == "SELECT COUNT(*) FROM dm.clients"

    def test_get_miss_returns_none(self):
        result = self.cache.get("несуществующий запрос xyz123")
        assert result is None

    def test_normalized_key_hit(self):
        self.cache.put("покажи клиентов", "answer")
        # Тот же запрос с иным регистром/пробелами
        result = self.cache.get("  ПОКАЖИ  КЛИЕНТОВ  ")
        assert result is not None

    def test_put_overwrites(self):
        self.cache.put("запрос", "первый ответ")
        self.cache.put("запрос", "второй ответ")
        result = self.cache.get("запрос")
        assert result["final_answer"] == "второй ответ"

    def test_put_without_sql(self):
        self.cache.put("запрос без sql", "answer")
        result = self.cache.get("запрос без sql")
        assert result is not None
        assert result["sql"] == ""


class TestQueryCacheInvalidate:
    def setup_method(self):
        self.cache, self.conn = _make_cache()

    def test_invalidate_removes_entry(self):
        self.cache.put("запрос", "answer")
        self.cache.invalidate("запрос")
        assert self.cache.get("запрос") is None

    def test_invalidate_nonexistent_no_error(self):
        self.cache.invalidate("несуществующий запрос")  # не должно бросать исключение


class TestQueryCacheClear:
    def setup_method(self):
        self.cache, self.conn = _make_cache()

    def test_clear_all(self):
        self.cache.put("запрос 1", "answer 1")
        self.cache.put("запрос 2", "answer 2")
        self.cache.clear_all()
        assert self.cache.get("запрос 1") is None
        assert self.cache.get("запрос 2") is None


class TestQueryCacheTTL:
    def setup_method(self):
        self.cache, self.conn = _make_cache()

    def test_expired_entry_not_returned(self):
        """Запись с истёкшим TTL не должна возвращаться."""
        # Вставляем запись вручную с очень старым timestamp
        old_ts = "2020-01-01T00:00:00+00:00"
        key = _cache_key("старый запрос")
        self.conn.execute(
            "INSERT OR REPLACE INTO query_cache (key, user_input, sql, final_answer, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (key, "старый запрос", "", "old answer", old_ts),
        )
        self.conn.commit()
        result = self.cache.get("старый запрос")
        assert result is None

    def test_fresh_entry_returned(self):
        """Свежая запись возвращается."""
        self.cache.put("свежий запрос", "fresh answer")
        result = self.cache.get("свежий запрос")
        assert result is not None
        assert result["final_answer"] == "fresh answer"
