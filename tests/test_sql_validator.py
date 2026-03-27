"""Тесты валидации SQL: определение режима, извлечение JOIN, проверка WHERE/LIMIT."""

import pytest

from core.sql_validator import (
    SQLMode,
    SQLValidator,
    ValidationResult,
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


class TestMultiplicationFactor:
    """Тесты оценки multiplication factor для row explosion."""

    def test_all_unique_joins(self):
        checks = [
            {"is_unique": True, "duplicate_pct": 0.0, "table": "t1", "columns": "id"},
            {"is_unique": True, "duplicate_pct": 0.0, "table": "t2", "columns": "id"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        assert factor == 1.0

    def test_single_non_unique_join(self):
        checks = [
            {"is_unique": False, "duplicate_pct": 80.0, "table": "t1", "columns": "key"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        # unique_perc = 20%, factor = 100/20 = 5.0
        assert factor == 5.0

    def test_multiple_non_unique_joins(self):
        checks = [
            {"is_unique": False, "duplicate_pct": 50.0, "table": "t1", "columns": "k1"},
            {"is_unique": False, "duplicate_pct": 80.0, "table": "t2", "columns": "k2"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        # 100/50 * 100/20 = 2 * 5 = 10
        assert factor == 10.0

    def test_mixed_unique_and_non_unique(self):
        checks = [
            {"is_unique": True, "duplicate_pct": 0.0, "table": "t1", "columns": "pk"},
            {"is_unique": False, "duplicate_pct": 75.0, "table": "t2", "columns": "fk"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        # unique * 100/25 = 1 * 4 = 4.0
        assert factor == 4.0

    def test_fully_non_unique(self):
        checks = [
            {"is_unique": False, "duplicate_pct": 100.0, "table": "t1", "columns": "k"},
        ]
        factor = SQLValidator._estimate_multiplication_factor(checks)
        assert factor == 100.0

    def test_empty_checks(self):
        assert SQLValidator._estimate_multiplication_factor([]) == 1.0


class TestRewriteSuggestion:
    """Тесты генерации шаблонов переписывания."""

    def test_suggestion_format(self):
        join = {"schema": "dm", "table": "sales", "column": "manager_id"}
        suggestion = SQLValidator._generate_rewrite_suggestion(join)
        assert "ROW EXPLOSION" in suggestion
        assert "dm.sales" in suggestion
        assert "manager_id" in suggestion
        assert "DISTINCT" in suggestion
        assert "ЗАПРЕЩЕНО" in suggestion

    def test_suggestion_contains_template(self):
        join = {"schema": "hr", "table": "emp", "column": "dept_id"}
        suggestion = SQLValidator._generate_rewrite_suggestion(join)
        assert "SELECT DISTINCT" in suggestion
        assert "подзапрос" in suggestion


class TestValidationResultExplosion:
    """Тесты ValidationResult с данными row explosion."""

    def test_summary_with_multiplication_factor(self):
        result = ValidationResult(is_valid=True, mode=SQLMode.READ)
        result.join_checks = [
            {"table": "dm.t", "columns": "k", "is_unique": False, "duplicate_pct": 80.0}
        ]
        result.multiplication_factor = 5.0
        summary = result.summary()
        assert "5.0x" in summary

    def test_summary_with_rewrite_suggestions(self):
        result = ValidationResult(is_valid=False, mode=SQLMode.READ)
        result.rewrite_suggestions = ["Use DISTINCT subquery"]
        summary = result.summary()
        assert "Use DISTINCT subquery" in summary
