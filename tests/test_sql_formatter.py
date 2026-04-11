"""Тесты детерминированного SQL-форматирования."""

import pytest
from core.sql_formatter import format_sql, fix_cyrillic_aliases, format_sql_safe


# ---------------------------------------------------------------------------
# fix_cyrillic_aliases
# ---------------------------------------------------------------------------

class TestFixCyrillicAliases:
    def test_bare_cyrillic_alias(self):
        sql = "SELECT SUM(amount) AS итого FROM dm.sales"
        result = fix_cyrillic_aliases(sql)
        assert "AS итого" not in result
        assert "итого" not in result
        assert "AS itogo" in result

    def test_quoted_cyrillic_alias(self):
        sql = 'SELECT region AS "Регион" FROM dm.sales'
        result = fix_cyrillic_aliases(sql)
        assert '"Регион"' not in result
        assert "AS region" in result

    def test_single_quoted_cyrillic_alias(self):
        sql = "SELECT amount AS 'Сумма оттока' FROM dm.fact"
        result = fix_cyrillic_aliases(sql)
        assert "'Сумма оттока'" not in result
        # Пробелы → подчёркивания
        assert "_" in result

    def test_latin_alias_untouched(self):
        sql = "SELECT amount AS total_sum FROM dm.sales"
        result = fix_cyrillic_aliases(sql)
        assert result == sql

    def test_multiple_cyrillic_aliases(self):
        sql = "SELECT region AS Регион, SUM(a) AS итого FROM t"
        result = fix_cyrillic_aliases(sql)
        assert "Регион" not in result
        assert "итого" not in result

    def test_empty_sql(self):
        assert fix_cyrillic_aliases("") == ""
        assert fix_cyrillic_aliases("   ") == "   "


# ---------------------------------------------------------------------------
# format_sql
# ---------------------------------------------------------------------------

class TestFormatSql:
    def test_keywords_uppercased(self):
        sql = "select region, count(*) from dm.sales group by region"
        result = format_sql(sql)
        # SQL keywords (DML, clauses) are uppercased by sqlparse
        assert "SELECT" in result
        assert "FROM" in result
        assert "GROUP BY" in result
        # count() is a function name — treated as identifier (lowercase) by sqlparse
        assert "count(" in result.lower()

    def test_identifiers_lowercased(self):
        sql = "SELECT Region, SUM(Amount) AS Total FROM DM.Sales"
        result = format_sql(sql)
        # Identifiers should be lowercase after formatting
        assert "DM.Sales" not in result
        assert "Region" not in result

    def test_indentation_applied(self):
        sql = "SELECT a, b FROM t WHERE x=1 GROUP BY a, b ORDER BY a"
        result = format_sql(sql)
        # Should have newlines between major clauses
        assert "\n" in result

    def test_cyrillic_alias_normalized(self):
        sql = "SELECT SUM(amount) AS итого FROM dm.sales"
        result = format_sql(sql)
        assert "итого" not in result

    def test_trailing_whitespace_removed(self):
        sql = "SELECT a   FROM t  "
        result = format_sql(sql)
        lines = result.split("\n")
        for line in lines:
            assert line == line.rstrip(), f"Trailing whitespace in: {repr(line)}"

    def test_deterministic(self):
        """Один и тот же входной SQL всегда даёт одинаковый результат."""
        sql = "select a,b,count(*) from t where x=1 group by a,b order by a desc"
        r1 = format_sql(sql)
        r2 = format_sql(sql)
        assert r1 == r2

    def test_empty_input(self):
        assert format_sql("") == ""
        assert format_sql("   ") == "   "

    def test_cte_preserved(self):
        sql = (
            "with cte as (select a from t) "
            "select a, sum(b) from cte group by a"
        )
        result = format_sql(sql)
        assert "WITH" in result
        assert "SELECT" in result

    def test_no_triple_blank_lines(self):
        sql = "SELECT a FROM t\n\n\n\nWHERE x=1"
        result = format_sql(sql)
        assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# format_sql_safe
# ---------------------------------------------------------------------------

class TestFormatSqlSafe:
    def test_valid_sql_formatted(self):
        sql = "select a from t"
        result = format_sql_safe(sql)
        assert "SELECT" in result

    def test_empty_returns_empty(self):
        assert format_sql_safe("") == ""

    def test_cyrillic_normalized_on_error(self):
        """Даже при ошибке форматирования кириллица нормализуется."""
        sql = "SELECT x AS итого"
        result = format_sql_safe(sql)
        assert "итого" not in result
