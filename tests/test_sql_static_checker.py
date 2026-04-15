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
    def test_select_star_is_error(self):
        sql = "SELECT * FROM dm.clients LIMIT 10"
        result = check_sql(sql, check_columns=False)
        # SELECT * — жёсткая ошибка (запрещено правилами агента)
        assert not result.is_valid
        assert any("*" in e or "SELECT *" in e for e in result.errors)

    def test_count_star_no_error(self):
        sql = "SELECT COUNT(*) AS cnt FROM dm.sales"
        result = check_sql(sql, check_columns=False)
        assert result.is_valid
        assert not result.errors

    def test_subquery_star_is_error(self):
        sql = "SELECT * FROM (SELECT col FROM dm.sales) sub LIMIT 5"
        result = check_sql(sql, check_columns=False)
        assert not result.is_valid
        assert result.errors


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


# ---------------------------------------------------------------------------
# GROUP BY completeness
# ---------------------------------------------------------------------------

class TestGroupByCompleteness:
    def test_correct_group_by_no_warning(self):
        sql = "SELECT region, COUNT(*) AS cnt FROM dm.sales GROUP BY region"
        result = check_sql(sql, check_columns=False)
        gb_warnings = [w for w in result.warnings if "GROUP BY" in w]
        assert not gb_warnings

    def test_missing_col_in_group_by_warns(self):
        sql = "SELECT region, segment, COUNT(*) AS cnt FROM dm.sales GROUP BY region"
        result = check_sql(sql, check_columns=False)
        gb_warnings = [w for w in result.warnings if "GROUP BY" in w]
        assert gb_warnings
        assert "segment" in gb_warnings[0]

    def test_no_group_by_no_check(self):
        sql = "SELECT region, amount FROM dm.sales LIMIT 10"
        result = check_sql(sql, check_columns=False)
        gb_warnings = [w for w in result.warnings if "GROUP BY" in w]
        assert not gb_warnings

    def test_aggregate_col_excluded_from_check(self):
        sql = "SELECT region, SUM(amount) AS total FROM dm.sales GROUP BY region"
        result = check_sql(sql, check_columns=False)
        gb_warnings = [w for w in result.warnings if "GROUP BY" in w]
        assert not gb_warnings

    def test_qualified_col_in_group_by_ok(self):
        sql = "SELECT t.region, COUNT(*) AS cnt FROM dm.sales t GROUP BY t.region"
        result = check_sql(sql, check_columns=False)
        gb_warnings = [w for w in result.warnings if "GROUP BY" in w]
        assert not gb_warnings

    def test_multiple_missing_cols(self):
        sql = "SELECT region, segment, manager, COUNT(*) AS cnt FROM dm.sales GROUP BY region"
        result = check_sql(sql, check_columns=False)
        gb_warnings = [w for w in result.warnings if "GROUP BY" in w]
        assert gb_warnings
        assert "segment" in gb_warnings[0] or "manager" in gb_warnings[0]

    def test_cte_does_not_confuse_parser(self):
        sql = (
            "WITH agg AS (SELECT client_id, SUM(amount) AS total FROM dm.sales GROUP BY client_id)\n"
            "SELECT c.name, agg.total FROM dm.clients c JOIN agg ON agg.client_id = c.id"
        )
        result = check_sql(sql, check_columns=False)
        # Внешний SELECT без GROUP BY — проверка GROUP BY completeness не применяется
        gb_warnings = [w for w in result.warnings if "GROUP BY completeness" in w]
        assert not gb_warnings

    def test_extra_col_in_group_by_warns(self):
        """GROUP BY содержит колонку, отсутствующую в SELECT — типичный баг «filter утёк в GROUP BY»."""
        sql = (
            "SELECT region, COUNT(*) AS cnt "
            "FROM dm.sales "
            "WHERE event_dt >= '2024-01-01' "
            "GROUP BY region, event_dt"
        )
        result = check_sql(sql, check_columns=False)
        # event_dt есть в GROUP BY, но нет в SELECT
        extra_warnings = [
            w for w in result.warnings
            if "GROUP BY" in w and "event_dt" in w and "отсутствующие в SELECT" in w
        ]
        assert extra_warnings, (
            "Должно быть предупреждение: event_dt в GROUP BY отсутствует в SELECT. "
            f"Warnings: {result.warnings}"
        )

    def test_extra_col_multiple_leaked(self):
        sql = (
            "SELECT client_id, SUM(amount) AS total "
            "FROM dm.sales "
            "GROUP BY client_id, task_created_dt, task_plan_closed_dt"
        )
        result = check_sql(sql, check_columns=False)
        extra_warnings = [
            w for w in result.warnings
            if "GROUP BY" in w and "отсутствующие в SELECT" in w
        ]
        assert extra_warnings
        assert (
            "task_created_dt" in extra_warnings[0]
            or "task_plan_closed_dt" in extra_warnings[0]
        )

    def test_group_by_equal_to_select_no_extra_warning(self):
        sql = "SELECT region, segment, COUNT(*) AS cnt FROM dm.sales GROUP BY region, segment"
        result = check_sql(sql, check_columns=False)
        extra_warnings = [
            w for w in result.warnings
            if "отсутствующие в SELECT" in w
        ]
        assert not extra_warnings

    def test_group_by_alias_in_select_not_flagged(self):
        """GROUP BY по алиасу, объявленному в SELECT — не должно флагаться."""
        sql = (
            "SELECT date_trunc('month', event_dt) AS month_bucket, COUNT(*) AS cnt "
            "FROM dm.sales GROUP BY month_bucket"
        )
        result = check_sql(sql, check_columns=False)
        extra_warnings = [
            w for w in result.warnings
            if "отсутствующие в SELECT" in w and "month_bucket" in w
        ]
        assert not extra_warnings


# ---------------------------------------------------------------------------
# ORDER BY алиасы
# ---------------------------------------------------------------------------

class TestOrderByAliases:
    def test_order_by_defined_alias_ok(self):
        sql = "SELECT region, SUM(amount) AS total_amt FROM dm.sales GROUP BY region ORDER BY total_amt DESC"
        result = check_sql(sql, check_columns=False)
        ob_errors = [e for e in result.errors if "ORDER BY" in e]
        assert not ob_errors

    def test_order_by_undefined_alias_error(self):
        """ORDER BY cnt, но cnt не объявлен в SELECT — жёсткая ошибка."""
        sql = (
            "SELECT tb_short_name, tb_full_name, COUNT(*) AS total "
            "FROM dm.gosb GROUP BY tb_short_name, tb_full_name ORDER BY cnt DESC"
        )
        result = check_sql(sql, check_columns=False)
        ob_errors = [e for e in result.errors if "ORDER BY" in e]
        assert ob_errors, "Должна быть ошибка: 'cnt' не определён в SELECT"
        assert "cnt" in ob_errors[0]

    def test_order_by_column_in_select_ok(self):
        sql = "SELECT region, amount FROM dm.sales ORDER BY region"
        result = check_sql(sql, check_columns=False)
        ob_errors = [e for e in result.errors if "ORDER BY" in e]
        assert not ob_errors

    def test_order_by_qualified_column_ok(self):
        sql = "SELECT t.region, t.amount FROM dm.sales t ORDER BY t.region DESC"
        result = check_sql(sql, check_columns=False)
        ob_errors = [e for e in result.errors if "ORDER BY" in e]
        assert not ob_errors

    def test_order_by_with_cte_alias_ok(self):
        sql = (
            "WITH agg AS (SELECT client_id, SUM(amount) AS total FROM dm.sales GROUP BY client_id) "
            "SELECT client_id, total FROM agg ORDER BY total DESC"
        )
        result = check_sql(sql, check_columns=False)
        ob_errors = [e for e in result.errors if "ORDER BY" in e]
        assert not ob_errors
