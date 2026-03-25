"""Тесты валидации SQL: определение режима, извлечение JOIN, проверка WHERE/LIMIT."""

import pytest

from core.sql_validator import (
    SQLMode,
    detect_mode,
    _extract_join_conditions,
    _has_where_or_limit,
)


class TestDetectMode:
    def test_select(self):
        assert detect_mode("SELECT * FROM t") == SQLMode.READ

    def test_select_with_comment(self):
        assert detect_mode("-- comment\nSELECT * FROM t") == SQLMode.READ

    def test_insert(self):
        assert detect_mode("INSERT INTO t VALUES (1)") == SQLMode.WRITE

    def test_update(self):
        assert detect_mode("UPDATE t SET a = 1") == SQLMode.WRITE

    def test_delete(self):
        assert detect_mode("DELETE FROM t WHERE id = 1") == SQLMode.WRITE

    def test_create(self):
        assert detect_mode("CREATE TABLE s.t (id INT)") == SQLMode.DDL

    def test_drop(self):
        assert detect_mode("DROP TABLE s.t") == SQLMode.DDL

    def test_truncate(self):
        assert detect_mode("TRUNCATE TABLE s.t") == SQLMode.DDL

    def test_alter(self):
        assert detect_mode("ALTER TABLE s.t ADD COLUMN x INT") == SQLMode.DDL


class TestHasWhereOrLimit:
    def test_with_where(self):
        assert _has_where_or_limit("SELECT * FROM t WHERE id = 1") is True

    def test_with_limit(self):
        assert _has_where_or_limit("SELECT * FROM t LIMIT 10") is True

    def test_without_both(self):
        assert _has_where_or_limit("SELECT * FROM t") is False

    def test_where_in_subquery_only(self):
        # WHERE в подзапросе не считается
        sql = "SELECT * FROM (SELECT * FROM t WHERE id = 1) sub"
        assert _has_where_or_limit(sql) is False

    def test_where_in_string_literal(self):
        sql = "SELECT 'WHERE is a keyword' FROM t"
        assert _has_where_or_limit(sql) is False


class TestExtractJoinConditions:
    def test_simple_join(self):
        sql = """
            SELECT * FROM hr.emp e
            JOIN hr.dept d ON d.id = e.dept_id
        """
        joins = _extract_join_conditions(sql)
        assert len(joins) >= 1
        assert joins[0]["schema"] == "hr"
        assert joins[0]["table"] == "dept"

    def test_no_joins(self):
        sql = "SELECT * FROM hr.emp"
        joins = _extract_join_conditions(sql)
        assert joins == []
