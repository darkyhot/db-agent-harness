"""Тесты SQL-генераторов и интерпретаторов join_validator."""

import pytest
from core.join_validator import (
    build_uniqueness_check_sql,
    build_join_fanout_check_sql,
    build_fk_coverage_check_sql,
    build_null_check_sql,
    build_group_by_cardinality_sql,
    build_validation_plan,
    interpret_uniqueness,
    interpret_fanout,
    interpret_fk_coverage,
    interpret_null_check,
)


# ---------------------------------------------------------------------------
# SQL генераторы — структурные проверки
# ---------------------------------------------------------------------------

class TestBuildUniquenessSql:
    def test_contains_schema_table(self):
        sql = build_uniqueness_check_sql("dm", "clients", "client_id")
        assert "dm.clients" in sql
        assert "client_id" in sql

    def test_contains_required_columns(self):
        sql = build_uniqueness_check_sql("dm", "clients", "client_id")
        assert "total_rows" in sql
        assert "unique_vals" in sql
        assert "dup_pct" in sql

    def test_select_only(self):
        sql = build_uniqueness_check_sql("dm", "clients", "client_id").upper()
        assert "INSERT" not in sql
        assert "UPDATE" not in sql
        assert "DELETE" not in sql
        assert "DROP" not in sql

    def test_deterministic(self):
        s1 = build_uniqueness_check_sql("dm", "clients", "client_id")
        s2 = build_uniqueness_check_sql("dm", "clients", "client_id")
        assert s1 == s2


class TestBuildFanoutSql:
    def test_contains_both_tables(self):
        sql = build_join_fanout_check_sql(
            "dm", "fact_sales", "client_id",
            "dm", "clients", "client_id",
        )
        assert "dm.fact_sales" in sql
        assert "dm.clients" in sql

    def test_contains_join(self):
        sql = build_join_fanout_check_sql(
            "dm", "fact_sales", "client_id",
            "dm", "clients", "client_id",
        ).upper()
        assert "JOIN" in sql

    def test_contains_fanout_column(self):
        sql = build_join_fanout_check_sql(
            "dm", "fact_sales", "client_id",
            "dm", "clients", "client_id",
        )
        assert "fanout" in sql
        assert "fact_rows" in sql
        assert "joined_rows" in sql

    def test_select_only(self):
        sql = build_join_fanout_check_sql(
            "dm", "fact_sales", "client_id",
            "dm", "clients", "client_id",
        ).upper()
        for kw in ("INSERT", "UPDATE", "DELETE", "DROP"):
            assert kw not in sql


class TestBuildFkCoverageSql:
    def test_contains_left_join(self):
        sql = build_fk_coverage_check_sql(
            "dm", "fact_sales", "client_id",
            "dm", "clients", "client_id",
        ).upper()
        assert "LEFT JOIN" in sql

    def test_contains_coverage_column(self):
        sql = build_fk_coverage_check_sql(
            "dm", "fact_sales", "client_id",
            "dm", "clients", "client_id",
        )
        assert "coverage_pct" in sql


class TestBuildNullCheckSql:
    def test_contains_null_pct(self):
        sql = build_null_check_sql("dm", "sales", "amount")
        assert "null_pct" in sql
        assert "dm.sales" in sql
        assert "amount" in sql


class TestBuildGroupByCardinalitySql:
    def test_contains_group_by(self):
        sql = build_group_by_cardinality_sql("dm", "sales", ["region", "segment"])
        assert "GROUP BY" in sql.upper()
        assert "region" in sql
        assert "segment" in sql

    def test_contains_limit(self):
        sql = build_group_by_cardinality_sql("dm", "sales", ["region"], limit=3)
        # limit применяется к внутреннему запросу
        assert "3000" in sql or "LIMIT" in sql.upper()


# ---------------------------------------------------------------------------
# Интерпретаторы
# ---------------------------------------------------------------------------

class TestInterpretUniqueness:
    def test_unique_column(self):
        row = {"total_rows": 1000, "unique_vals": 1000, "dup_pct": 0.0}
        result = interpret_uniqueness(row)
        assert result["is_unique"] is True
        assert result["warning"] is None
        assert result["dup_pct"] == 0.0

    def test_non_unique_column(self):
        row = {"total_rows": 1000, "unique_vals": 800, "dup_pct": 20.0}
        result = interpret_uniqueness(row)
        assert result["is_unique"] is False
        assert result["warning"] is not None
        assert "20" in result["warning"]

    def test_empty_table(self):
        row = {"total_rows": 0, "unique_vals": 0, "dup_pct": 0.0}
        result = interpret_uniqueness(row)
        # 0 == 0 but 0 rows → не уникален (нет данных)
        assert result["total_rows"] == 0


class TestInterpretFanout:
    def test_safe_join(self):
        row = {"fact_rows": 1000, "joined_rows": 1000, "fanout": 1.000}
        result = interpret_fanout(row)
        assert result["is_safe"] is True
        assert result["warning"] is None

    def test_unsafe_join(self):
        row = {"fact_rows": 1000, "joined_rows": 1500, "fanout": 1.500}
        result = interpret_fanout(row)
        assert result["is_safe"] is False
        assert result["warning"] is not None
        assert "1500" in result["warning"] or "500" in result["warning"]

    def test_fanout_float_tolerance(self):
        # fanout=1.001 должен считаться безопасным (погрешность float)
        row = {"fact_rows": 1000, "joined_rows": 1001, "fanout": 1.001}
        result = interpret_fanout(row)
        assert result["is_safe"] is True


class TestInterpretFkCoverage:
    def test_full_coverage(self):
        row = {"fact_rows": 1000, "matched_rows": 1000, "coverage_pct": 100.0}
        result = interpret_fk_coverage(row)
        assert result["is_full"] is True
        assert result["warning"] is None

    def test_partial_coverage(self):
        row = {"fact_rows": 1000, "matched_rows": 900, "coverage_pct": 90.0}
        result = interpret_fk_coverage(row)
        assert result["is_full"] is False
        assert result["warning"] is not None
        assert "90" in result["warning"]


class TestInterpretNullCheck:
    def test_no_nulls(self):
        row = {"total_rows": 500, "not_null_rows": 500, "null_pct": 0.0}
        result = interpret_null_check(row)
        assert result["has_nulls"] is False
        assert result["warning"] is None

    def test_has_nulls(self):
        row = {"total_rows": 500, "not_null_rows": 400, "null_pct": 20.0}
        result = interpret_null_check(row)
        assert result["has_nulls"] is True
        assert result["warning"] is not None
        assert "20" in result["warning"]


# ---------------------------------------------------------------------------
# build_validation_plan
# ---------------------------------------------------------------------------

class TestBuildValidationPlan:
    def test_safe_join_plan(self):
        join_spec = [
            {
                "left": "dm.fact_sales.client_id",
                "right": "dm.clients.client_id",
                "safe": True,
            }
        ]
        plan = build_validation_plan(join_spec)
        # Безопасный JOIN → только uniqueness + null_check (без fanout/coverage)
        check_types = [p["check_type"] for p in plan]
        assert "uniqueness" in check_types
        assert "null_check" in check_types
        assert "fanout" not in check_types
        assert "fk_coverage" not in check_types

    def test_unsafe_join_plan(self):
        join_spec = [
            {
                "left": "dm.fact_sales.client_id",
                "right": "dm.clients.client_id",
                "safe": False,
            }
        ]
        plan = build_validation_plan(join_spec)
        check_types = [p["check_type"] for p in plan]
        assert "fanout" in check_types
        assert "fk_coverage" in check_types

    def test_all_sqls_are_select_only(self):
        join_spec = [
            {
                "left": "dm.fact_sales.client_id",
                "right": "dm.clients.client_id",
                "safe": False,
            }
        ]
        plan = build_validation_plan(join_spec)
        for item in plan:
            sql_upper = item["sql"].upper()
            for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE"):
                assert forbidden not in sql_upper, (
                    f"Forbidden keyword {forbidden} in {item['check_type']} SQL"
                )

    def test_empty_join_spec(self):
        plan = build_validation_plan([])
        assert plan == []

    def test_invalid_join_spec_skipped(self):
        join_spec = [{"left": "nope", "right": "also_nope", "safe": False}]
        plan = build_validation_plan(join_spec)
        assert plan == []
