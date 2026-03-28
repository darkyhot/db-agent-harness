"""Тесты helper-функций DatabaseManager."""

from core.database import DatabaseManager, _has_top_level_limit


def test_top_level_limit_detected():
    assert _has_top_level_limit("SELECT * FROM t LIMIT 10") is True


def test_nested_limit_not_detected():
    sql = "SELECT * FROM (SELECT * FROM t LIMIT 10) sub"
    assert _has_top_level_limit(sql) is False


def test_limit_in_identifier_not_detected():
    sql = "SELECT credit_limit FROM t"
    assert _has_top_level_limit(sql) is False


def test_limit_in_string_not_detected():
    sql = "SELECT 'limit' AS label FROM t"
    assert _has_top_level_limit(sql) is False


def test_execute_query_delegates_to_preview_query():
    db = DatabaseManager()
    calls = []

    def _fake_preview(sql: str, limit: int = 1000):
        calls.append((sql, limit))
        return "ok"

    db.preview_query = _fake_preview  # type: ignore[method-assign]
    result = db.execute_query("SELECT 1", limit=77)

    assert result == "ok"
    assert calls == [("SELECT 1", 77)]


def test_estimate_affected_rows_aliases_count_readonly():
    db = DatabaseManager()
    calls = []

    def _fake_count(where_clause: str, schema: str, table: str):
        calls.append((where_clause, schema, table))
        return 42

    db.count_affected_rows_readonly = _fake_count  # type: ignore[method-assign]
    result = db.estimate_affected_rows("id = 1", "hr", "emp")

    assert result == 42
    assert calls == [("id = 1", "hr", "emp")]
