"""Тесты для core/query_cache.py."""

import pytest

from core.memory import MemoryManager
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
# QueryCache через реальные JSON-файлы во временной директории
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    """Создать QueryCache с временной директорией."""
    memory = MemoryManager(memory_dir=tmp_path)
    return QueryCache(memory)


class TestQueryCachePutGet:
    def test_put_and_get(self, cache):
        cache.put("покажи клиентов", "Клиентов: 100", sql="SELECT COUNT(*) FROM dm.clients")
        result = cache.get("покажи клиентов")
        assert result is not None
        assert result["final_answer"] == "Клиентов: 100"
        assert result["sql"] == "SELECT COUNT(*) FROM dm.clients"

    def test_get_miss_returns_none(self, cache):
        result = cache.get("несуществующий запрос xyz123")
        assert result is None

    def test_normalized_key_hit(self, cache):
        cache.put("покажи клиентов", "answer")
        # Тот же запрос с иным регистром/пробелами
        result = cache.get("  ПОКАЖИ  КЛИЕНТОВ  ")
        assert result is not None

    def test_put_overwrites(self, cache):
        cache.put("запрос", "первый ответ")
        cache.put("запрос", "второй ответ")
        result = cache.get("запрос")
        assert result["final_answer"] == "второй ответ"

    def test_put_without_sql(self, cache):
        cache.put("запрос без sql", "answer")
        result = cache.get("запрос без sql")
        assert result is not None
        assert result["sql"] == ""


class TestQueryCacheInvalidate:
    def test_invalidate_removes_entry(self, cache):
        cache.put("запрос", "answer")
        cache.invalidate("запрос")
        assert cache.get("запрос") is None

    def test_invalidate_nonexistent_no_error(self, cache):
        cache.invalidate("несуществующий запрос")  # не должно бросать исключение


class TestQueryCacheClear:
    def test_clear_all(self, cache):
        cache.put("запрос 1", "answer 1")
        cache.put("запрос 2", "answer 2")
        cache.clear_all()
        assert cache.get("запрос 1") is None
        assert cache.get("запрос 2") is None


class TestQueryCacheTTL:
    def test_expired_entry_not_returned(self, cache):
        """Запись с истёкшим TTL не должна возвращаться."""
        import json
        from core.memory import _write_json_atomic, _load_json

        # Вставляем запись вручную с очень старым timestamp
        old_ts = "2020-01-01T00:00:00+00:00"
        key = _cache_key("старый запрос")
        data = _load_json(cache._cache_path, {})
        data[key] = {
            "user_input": "старый запрос",
            "sql": "",
            "final_answer": "old answer",
            "created_at": old_ts,
        }
        _write_json_atomic(cache._cache_path, data)

        result = cache.get("старый запрос")
        assert result is None

    def test_fresh_entry_returned(self, cache):
        """Свежая запись возвращается."""
        cache.put("свежий запрос", "fresh answer")
        result = cache.get("свежий запрос")
        assert result is not None
        assert result["final_answer"] == "fresh answer"
