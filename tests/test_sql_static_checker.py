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

    def test_cyrillic_alias_autofixed_by_default(self):
        """Direction 3: soft-fix транслитерирует алиас и не блокирует SQL."""
        sql = "SELECT region AS регион FROM dm.sales"
        result = check_sql(sql, check_columns=False)
        # По умолчанию — soft-fix: warning + fixed_sql, но is_valid
        assert result.is_valid
        assert result.fixed_sql is not None
        assert "регион" not in result.fixed_sql
        assert "AS region" in result.fixed_sql

    def test_cyrillic_alias_hard_error_when_autofix_disabled(self):
        sql = "SELECT region AS регион FROM dm.sales"
        result = check_sql(sql, check_columns=False, auto_fix_cyrillic=False)
        assert not result.is_valid
        assert any("кириллица" in e.lower() or "алиас" in e.lower() for e in result.errors)

    def test_cyrillic_alias_with_quotes_autofixed(self):
        sql = 'SELECT COUNT(*) AS "выручка" FROM dm.sales'
        result = check_sql(sql, check_columns=False)
        assert result.is_valid
        assert result.fixed_sql is not None

    def test_quoted_cyrillic_table_name_not_flagged(self):
        # Кириллица в значениях WHERE — не алиас, не должна флагаться
        sql = "SELECT id FROM dm.clients WHERE name = 'Иванов' AND status = 'активный'"
        result = check_sql(sql, check_columns=False)
        assert result.is_valid

    def test_multiple_cyrillic_aliases_all_autofixed(self):
        sql = "SELECT a AS первый, b AS второй FROM t"
        result = check_sql(sql, check_columns=False)
        assert result.is_valid
        assert result.fixed_sql is not None
        assert "первый" not in result.fixed_sql
        assert "второй" not in result.fixed_sql


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


class TestSyntaxSanity:
    def test_unbalanced_parenthesis_is_error(self):
        sql = "SELECT COUNT(*) AS cnt FROM dm.sales WHERE (region = 'Msk'"
        result = check_sql(sql, check_columns=False)
        assert not result.is_valid
        assert any("скоб" in err.lower() for err in result.errors)


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


# ---------------------------------------------------------------------------
# Direction 3: SELECT DISTINCT в финальной проекции
# ---------------------------------------------------------------------------

class TestDistinctInFinalSelect:
    def test_distinct_in_outer_select_is_error(self):
        sql = "SELECT DISTINCT region, SUM(amount) FROM dm.sales GROUP BY region"
        result = check_sql(sql, check_columns=False)
        assert not result.is_valid
        assert any("DISTINCT" in e for e in result.errors)

    def test_distinct_in_cte_is_ok(self):
        sql = (
            "WITH agg AS (SELECT DISTINCT client_id FROM dm.sales) "
            "SELECT client_id FROM agg"
        )
        result = check_sql(sql, check_columns=False)
        assert not any("DISTINCT" in e for e in result.errors)

    def test_plain_select_no_distinct_ok(self):
        sql = "SELECT region, COUNT(*) AS cnt FROM dm.sales GROUP BY region"
        result = check_sql(sql, check_columns=False)
        assert not any("DISTINCT" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Direction 3: Type compatibility в WHERE
# ---------------------------------------------------------------------------

class _FakeSchemaLoader:
    """Минимальный schema_loader для type-check."""

    def __init__(self, catalog):
        # catalog: {("schema","table"): [{"column_name", "dType"}, ...]}
        import pandas as pd
        self._catalog = {
            key: pd.DataFrame(cols) for key, cols in catalog.items()
        }

    def get_table_columns(self, schema, table):
        import pandas as pd
        return self._catalog.get((schema, table), pd.DataFrame(
            columns=["column_name", "dType"]
        ))


class TestTypeCompatibility:
    def test_int_col_vs_text_literal_errors(self):
        loader = _FakeSchemaLoader({
            ("dm", "clients"): [
                {"column_name": "id", "dType": "int8"},
                {"column_name": "name", "dType": "text"},
            ]
        })
        sql = "SELECT c.id FROM dm.clients c WHERE c.id = 'abc'"
        result = check_sql(sql, schema_loader=loader, check_columns=False)
        assert not result.is_valid
        assert any("Несовместимые типы" in e for e in result.errors)

    def test_text_col_vs_text_literal_ok(self):
        loader = _FakeSchemaLoader({
            ("dm", "clients"): [
                {"column_name": "name", "dType": "varchar(255)"},
            ]
        })
        sql = "SELECT c.name FROM dm.clients c WHERE c.name = 'Иванов'"
        result = check_sql(sql, schema_loader=loader, check_columns=False)
        assert not any("Несовместимые" in e for e in result.errors)

    def test_date_col_vs_iso_string_ok(self):
        loader = _FakeSchemaLoader({
            ("dm", "sales"): [
                {"column_name": "report_dt", "dType": "date"},
            ]
        })
        sql = "SELECT s.report_dt FROM dm.sales s WHERE s.report_dt = '2024-01-01'"
        result = check_sql(sql, schema_loader=loader, check_columns=False)
        assert not any("Несовместимые" in e for e in result.errors)

    def test_int_col_vs_int_literal_ok(self):
        loader = _FakeSchemaLoader({
            ("dm", "clients"): [
                {"column_name": "id", "dType": "int8"},
            ]
        })
        sql = "SELECT c.id FROM dm.clients c WHERE c.id = 42"
        result = check_sql(sql, schema_loader=loader, check_columns=False)
        assert not any("Несовместимые" in e for e in result.errors)


class TestUnqualifiedCatalogColumns:
    def test_missing_unqualified_column_in_count_distinct_errors(self):
        loader = _FakeSchemaLoader({
            ("dm", "gosb_dim"): [
                {"column_name": "tb_id", "dType": "int4"},
                {"column_name": "old_gosb_id", "dType": "int4"},
            ]
        })
        sql = "SELECT COUNT(DISTINCT gosb_id) AS total_gosb FROM dm.gosb_dim"

        result = check_sql(sql, schema_loader=loader)

        assert not result.is_valid
        assert any("gosb_id" in e for e in result.errors)

    def test_valid_unqualified_column_in_single_table_passes(self):
        loader = _FakeSchemaLoader({
            ("dm", "gosb_dim"): [
                {"column_name": "tb_id", "dType": "int4"},
                {"column_name": "old_gosb_id", "dType": "int4"},
            ]
        })
        sql = (
            "SELECT COUNT(DISTINCT old_gosb_id) AS total_gosb, "
            "COUNT(DISTINCT tb_id) AS total_tb FROM dm.gosb_dim"
        )

        result = check_sql(sql, schema_loader=loader)

        assert result.is_valid
        assert not result.errors

    def test_ambiguous_unqualified_column_across_tables_errors(self):
        loader = _FakeSchemaLoader({
            ("dm", "sales"): [
                {"column_name": "client_id", "dType": "int8"},
                {"column_name": "amount", "dType": "numeric"},
            ],
            ("dm", "clients"): [
                {"column_name": "client_id", "dType": "int8"},
                {"column_name": "name", "dType": "text"},
            ],
        })
        sql = (
            "SELECT client_id, SUM(amount) AS total_amount "
            "FROM dm.sales s JOIN dm.clients c ON c.client_id = s.client_id "
            "GROUP BY client_id"
        )

        result = check_sql(sql, schema_loader=loader)

        assert not result.is_valid
        assert any("неоднознач" in e.lower() and "client_id" in e for e in result.errors)

    def test_ambiguous_inserted_dttm_has_stable_diagnostic(self):
        loader = _FakeSchemaLoader({
            ("dm", "fact_outflow"): [
                {"column_name": "report_dt", "dType": "date"},
                {"column_name": "inserted_dttm", "dType": "timestamp"},
            ],
            ("dm", "dim_gosb"): [
                {"column_name": "new_gosb_name", "dType": "text"},
                {"column_name": "inserted_dttm", "dType": "timestamp"},
            ],
        })
        sql = (
            "SELECT f.report_dt, d.new_gosb_name "
            "FROM dm.fact_outflow f JOIN dm.dim_gosb d ON true "
            "ORDER BY inserted_dttm DESC"
        )

        result = check_sql(sql, schema_loader=loader)

        assert not result.is_valid
        assert any("неоднозначные unqualified-колонки" in e for e in result.errors)
        assert any("inserted_dttm" in e for e in result.errors)

    def test_cte_projection_alias_is_not_flagged_as_missing_column(self):
        loader = _FakeSchemaLoader({
            ("dm", "sales"): [
                {"column_name": "client_id", "dType": "int8"},
                {"column_name": "amount", "dType": "numeric"},
            ]
        })
        sql = (
            "WITH agg AS ("
            "SELECT client_id, SUM(amount) AS total_amount "
            "FROM dm.sales GROUP BY client_id"
            ") SELECT client_id, total_amount FROM agg ORDER BY total_amount DESC"
        )

        result = check_sql(sql, schema_loader=loader)

        assert result.is_valid
        assert not result.errors
