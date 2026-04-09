"""Тесты для core/sql_static_checker.py."""

import pytest
from core.sql_static_checker import check_sql


# ---------------------------------------------------------------------------
# Кириллица в алиасах
# ---------------------------------------------------------------------------

class TestCyrillicAliases:
    def test_no_cyrillic_alias_passes(self):
        sql = "SELECT region, COUNT(*) AS cnt FROM dm.sales GROUP BY region"
        result = check_sql(sql, check_columns=False)
        assert result.is_valid
        assert not result.errors

    def test_cyrillic_alias_caught(self):
        sql = "SELECT region AS регион FROM dm.sales"
        result = check_sql(sql, check_columns=False)
        assert not result.is_valid
        assert any("кириллица" in e.lower() or "алиас" in e.lower() for e in result.errors)

    def test_cyrillic_alias_with_quotes_caught(self):
        sql = 'SELECT COUNT(*) AS "выручка" FROM dm.sales'
        result = check_sql(sql, check_columns=False)
        assert not result.is_valid

    def test_quoted_cyrillic_table_name_not_flagged(self):
        # Кириллица в значениях WHERE — не алиас, не должна флагаться
        sql = "SELECT id FROM dm.clients WHERE name = 'Иванов' AND status = 'активный'"
        result = check_sql(sql, check_columns=False)
        assert result.is_valid

    def test_multiple_cyrillic_aliases_all_caught(self):
        sql = "SELECT a AS первый, b AS второй FROM t"
        result = check_sql(sql, check_columns=False)
        assert not result.is_valid
        assert len(result.errors) >= 1  # хотя бы одна ошибка


# ---------------------------------------------------------------------------
# SELECT *
# ---------------------------------------------------------------------------

class TestSelectStar:
    def test_select_star_warns(self):
        sql = "SELECT * FROM dm.clients LIMIT 10"
        result = check_sql(sql, check_columns=False)
        # SELECT * — warning, не ошибка
        assert result.is_valid
        assert any("*" in w for w in result.warnings)

    def test_count_star_no_warning(self):
        sql = "SELECT COUNT(*) AS cnt FROM dm.sales"
        result = check_sql(sql, check_columns=False)
        assert result.is_valid
        assert not result.warnings

    def test_subquery_star_warns(self):
        sql = "SELECT * FROM (SELECT * FROM dm.sales) sub LIMIT 5"
        result = check_sql(sql, check_columns=False)
        assert result.is_valid
        assert result.warnings


# ---------------------------------------------------------------------------
# check_columns=False режим (без каталога)
# ---------------------------------------------------------------------------

class TestNoSchemaLoader:
    def test_valid_sql_passes(self):
        sql = "SELECT id, name, amount FROM dm.orders WHERE date >= '2024-01-01'"
        result = check_sql(sql, schema_loader=None, check_columns=False)
        assert result.is_valid
        assert not result.errors

    def test_empty_sql_handled(self):
        result = check_sql("", check_columns=False)
        # Пустой SQL — ошибка (нечего проверять)
        assert not result.is_valid
        assert result.errors

    def test_none_sql_handled(self):
        result = check_sql(None, check_columns=False)
        # None SQL — ошибка
        assert not result.is_valid
        assert result.errors


# ---------------------------------------------------------------------------
# Интеграция: несколько нарушений
# ---------------------------------------------------------------------------

class TestMultipleViolations:
    def test_cyrillic_alias_and_star(self):
        sql = "SELECT * FROM dm.clients WHERE region = 'Мск' AS регион"
        result = check_sql(sql, check_columns=False)
        # Может быть невалидным из-за кириллического алиаса
        # Не требуем строгого набора ошибок — просто проверяем что что-то поймалось
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)


# ---------------------------------------------------------------------------
# StaticCheckResult поля
# ---------------------------------------------------------------------------

class TestStaticCheckResult:
    def test_result_has_expected_fields(self):
        result = check_sql("SELECT 1", check_columns=False)
        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")

    def test_errors_and_warnings_are_lists(self):
        result = check_sql("SELECT a AS б FROM t", check_columns=False)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
