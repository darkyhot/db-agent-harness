"""Тесты helper-функций DatabaseManager."""

import json

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


def test_missing_fields_detect_incomplete_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"user_id": "alice"}), encoding="utf-8")

    db = DatabaseManager(config_path=config_path)

    assert db.missing_connection_fields() == ["host", "port", "database"]
    assert db.missing_runtime_fields() == ["debug_prompt", "show_plan"]
    assert db.has_complete_config is False


def test_save_connection_and_runtime_params_preserve_full_config(tmp_path):
    config_path = tmp_path / "config.json"
    db = DatabaseManager(config_path=config_path)

    db.save_connection_config("alice", "db.local", 5433, "analytics")
    db.save_runtime_params(debug_prompt=True, show_plan=False)

    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved == {
        "user_id": "alice",
        "host": "db.local",
        "port": 5433,
        "database": "analytics",
        "debug_prompt": True,
        "show_plan": False,
    }
    assert db.has_complete_config is True
