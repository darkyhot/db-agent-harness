"""Тесты для core/sql_builder.py."""

import re
import pytest
from core.sql_builder import SqlBuilder, _short_alias, _resolve_join_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_join_spec(left_table, left_col, right_table, right_col, safe=True):
    return [{
        "left": f"{left_table}.{left_col}",
        "right": f"{right_table}.{right_col}",
        "safe": safe,
    }]


def norm(sql: str) -> str:
    """Нормализовать SQL для сравнения: убрать лишние пробелы/переносы."""
    return re.sub(r"\s+", " ", sql.strip().upper())


builder = SqlBuilder()


# ---------------------------------------------------------------------------
# _short_alias
# ---------------------------------------------------------------------------

class TestShortAlias:
    def test_single_word(self):
        used: set[str] = set()
        assert _short_alias("dm.sales", used) == "s"

    def test_snake_case(self):
        used: set[str] = set()
        assert _short_alias("dm.fact_outflow", used) == "fo"

    def test_unique_aliases(self):
        used: set[str] = set()
        a1 = _short_alias("dm.clients", used)
        a2 = _short_alias("dm.contracts", used)
        assert a1 != a2

    def test_collision_resolved(self):
        used: set[str] = {"c"}
        a = _short_alias("dm.clients", used)
        assert a != "c"
        assert a.startswith("c")


# ---------------------------------------------------------------------------
# _resolve_join_key
# ---------------------------------------------------------------------------

class TestResolveJoinKey:
    def test_direct_match(self):
        spec = _simple_join_spec("dm.sales", "client_id", "dm.clients", "id")
        left_col, right_col = _resolve_join_key(spec, "dm.sales", "dm.clients")
        assert left_col == "client_id"
        assert right_col == "id"

    def test_reversed_match(self):
        spec = _simple_join_spec("dm.sales", "client_id", "dm.clients", "id")
        left_col, right_col = _resolve_join_key(spec, "dm.clients", "dm.sales")
        assert left_col == "id"
        assert right_col == "client_id"

    def test_no_match(self):
        spec = _simple_join_spec("dm.sales", "client_id", "dm.clients", "id")
        l, r = _resolve_join_key(spec, "dm.other", "dm.clients")
        assert l == "" and r == ""

    def test_empty_spec(self):
        l, r = _resolve_join_key([], "dm.sales", "dm.clients")
        assert l == "" and r == ""


# ---------------------------------------------------------------------------
# simple_select
# ---------------------------------------------------------------------------

class TestSimpleSelect:
    def _cols(self, agg=None, group_by=None):
        return {"dm.sales": {
            "select": ["region", "amount"] if not agg else ["region"],
            "filter": [],
            "aggregate": [agg] if agg else [],
            "group_by": group_by or (["region"] if agg else []),
        }}

    def _blueprint(self, group_by=None, agg=None, where=None, order=None, limit=100):
        return {
            "strategy": "simple_select",
            "main_table": "dm.sales",
            "aggregation": agg,
            "group_by": group_by or [],
            "where_conditions": where or [],
            "order_by": order,
            "limit": limit,
        }

    def test_basic_select(self):
        sql = builder.build("simple_select", self._cols(), [], self._blueprint(), {})
        assert sql is not None
        n = norm(sql)
        assert "SELECT" in n
        assert "FROM DM.SALES" in n
        assert "LIMIT 100" in n

    def test_aggregation_in_select(self):
        agg = {"function": "SUM", "column": "amount", "alias": "sum_amount"}
        bp = self._blueprint(group_by=["region"], agg=agg, order="sum_amount DESC")
        sql = builder.build("simple_select", self._cols(agg="amount"), [], bp, {})
        assert sql is not None
        n = norm(sql)
        assert "SUM(" in n
        assert "SUM_AMOUNT" in n
        assert "GROUP BY" in n
        assert "REGION" in n

    def test_where_clause(self):
        bp = self._blueprint(where=["sale_date >= '2024-01-01'::date"])
        sql = builder.build("simple_select", self._cols(), [], bp, {})
        assert sql is not None
        assert "WHERE" in sql
        assert "2024-01-01" in sql

    def test_no_limit_when_none(self):
        bp = self._blueprint(limit=None)
        sql = builder.build("simple_select", self._cols(), [], bp, {})
        assert sql is not None
        assert "LIMIT" not in sql.upper()

    def test_empty_selected_columns_returns_none(self):
        sql = builder.build("simple_select", {}, [], self._blueprint(), {})
        assert sql is None


# ---------------------------------------------------------------------------
# fact_dim_join
# ---------------------------------------------------------------------------

class TestFactDimJoin:
    def _cols(self):
        return {
            "dm.sales": {
                "select": ["region", "amount"],
                "filter": [],
                "aggregate": ["amount"],
                "group_by": ["region"],
            },
            "dm.clients": {
                "select": ["name"],
                "filter": [],
                "aggregate": [],
                "group_by": [],
            },
        }

    def _bp(self, safe=True):
        agg = {"function": "SUM", "column": "amount", "alias": "sum_amount"}
        return {
            "strategy": "fact_dim_join",
            "main_table": "dm.sales",
            "aggregation": agg,
            "group_by": ["region", "name"],
            "where_conditions": [],
            "order_by": "sum_amount DESC",
            "limit": 100,
        }

    def _spec(self, safe=True):
        return _simple_join_spec("dm.sales", "client_id", "dm.clients", "id", safe=safe)

    def test_produces_sql(self):
        sql = builder.build("fact_dim_join", self._cols(), self._spec(), self._bp(), {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is not None

    def test_join_keyword_present(self):
        sql = builder.build("fact_dim_join", self._cols(), self._spec(), self._bp(), {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is not None
        assert "JOIN" in sql.upper()

    def test_unsafe_join_uses_distinct_on(self):
        sql = builder.build("fact_dim_join", self._cols(), self._spec(safe=False), self._bp(), {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is not None
        assert "DISTINCT ON" in sql.upper()

    def test_safe_join_no_distinct_on(self):
        sql = builder.build("fact_dim_join", self._cols(), self._spec(safe=True), self._bp(), {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is not None
        # Safe → нет DISTINCT ON в подзапросе, прямой JOIN
        n = norm(sql)
        assert "DISTINCT ON" not in n

    def test_no_join_spec_returns_none(self):
        sql = builder.build("fact_dim_join", self._cols(), [], self._bp(), {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is None

    def test_aggregation_in_sql(self):
        sql = builder.build("fact_dim_join", self._cols(), self._spec(), self._bp(), {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is not None
        assert "SUM" in sql.upper()
        assert "sum_amount" in sql

    def test_order_by_present(self):
        sql = builder.build("fact_dim_join", self._cols(), self._spec(), self._bp(), {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is not None
        assert "ORDER BY" in sql.upper()

    def test_group_by_present(self):
        sql = builder.build("fact_dim_join", self._cols(), self._spec(), self._bp(), {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is not None
        assert "GROUP BY" in sql.upper()


# ---------------------------------------------------------------------------
# fact_fact_join
# ---------------------------------------------------------------------------

class TestFactFactJoin:
    def _cols(self):
        return {
            "dm.sales": {
                "select": ["client_id"],
                "filter": [],
                "aggregate": ["amount"],
                "group_by": ["client_id"],
            },
            "dm.payments": {
                "select": ["client_id"],
                "filter": [],
                "aggregate": ["payment"],
                "group_by": ["client_id"],
            },
        }

    def _spec(self):
        return _simple_join_spec("dm.sales", "client_id", "dm.payments", "client_id")

    def _bp(self):
        agg = {"function": "SUM", "column": "amount", "alias": "sum_amount"}
        return {
            "strategy": "fact_fact_join",
            "main_table": "dm.sales",
            "aggregation": agg,
            "group_by": ["client_id"],
            "where_conditions": [],
            "order_by": None,
            "limit": 100,
        }

    def test_has_cte(self):
        sql = builder.build("fact_fact_join", self._cols(), self._spec(), self._bp(), {})
        assert sql is not None
        assert "WITH" in sql.upper()

    def test_two_ctes(self):
        sql = builder.build("fact_fact_join", self._cols(), self._spec(), self._bp(), {})
        assert sql is not None
        n = norm(sql)
        # Два AS (...) в WITH
        assert n.count(" AS (") >= 2

    def test_join_between_ctes(self):
        sql = builder.build("fact_fact_join", self._cols(), self._spec(), self._bp(), {})
        assert sql is not None
        assert "JOIN" in sql.upper()

    def test_no_join_spec_returns_none(self):
        sql = builder.build("fact_fact_join", self._cols(), [], self._bp(), {})
        assert sql is None

    def test_aggregation_in_cte(self):
        sql = builder.build("fact_fact_join", self._cols(), self._spec(), self._bp(), {})
        assert sql is not None
        assert "SUM(" in sql.upper()

    def test_limit_present(self):
        sql = builder.build("fact_fact_join", self._cols(), self._spec(), self._bp(), {})
        assert sql is not None
        assert "LIMIT 100" in sql.upper()


# ---------------------------------------------------------------------------
# dim_dim_join
# ---------------------------------------------------------------------------

class TestDimDimJoin:
    def _cols(self):
        return {
            "dm.orgs": {
                "select": ["org_id", "org_name"],
                "filter": [],
                "aggregate": [],
                "group_by": ["org_id"],
            },
            "dm.regions": {
                "select": ["org_id", "region"],
                "filter": [],
                "aggregate": [],
                "group_by": [],
            },
        }

    def _spec(self):
        return _simple_join_spec("dm.orgs", "org_id", "dm.regions", "org_id")

    def _bp(self):
        return {
            "strategy": "dim_dim_join",
            "main_table": "dm.orgs",
            "aggregation": None,
            "group_by": [],
            "where_conditions": [],
            "order_by": None,
            "limit": 100,
        }

    def test_has_distinct_on_in_cte(self):
        sql = builder.build("dim_dim_join", self._cols(), self._spec(), self._bp(), {})
        assert sql is not None
        assert "DISTINCT ON" in sql.upper()

    def test_has_with(self):
        sql = builder.build("dim_dim_join", self._cols(), self._spec(), self._bp(), {})
        assert sql is not None
        assert "WITH" in sql.upper()

    def test_no_spec_returns_none(self):
        sql = builder.build("dim_dim_join", self._cols(), [], self._bp(), {})
        assert sql is None


# ---------------------------------------------------------------------------
# Fallback conditions
# ---------------------------------------------------------------------------

class TestFallbackConditions:
    def test_filter_without_where_returns_none(self):
        """Если есть filter-колонки но нет where_conditions — нужны литеральные значения."""
        cols = {"dm.sales": {
            "select": ["region"],
            "filter": ["region"],  # есть filter-колонка
            "aggregate": [],
            "group_by": [],
        }}
        bp = {
            "strategy": "simple_select",
            "main_table": "dm.sales",
            "aggregation": None,
            "group_by": [],
            "where_conditions": [],  # но нет WHERE-условий
            "order_by": None,
            "limit": 100,
        }
        sql = builder.build("simple_select", cols, [], bp, {})
        assert sql is None

    def test_filter_with_date_where_not_fallback(self):
        """Если filter-колонка есть + WHERE-условие с ::date — шаблон работает."""
        cols = {"dm.sales": {
            "select": ["region"],
            "filter": ["sale_date"],
            "aggregate": [],
            "group_by": [],
        }}
        bp = {
            "strategy": "simple_select",
            "main_table": "dm.sales",
            "aggregation": None,
            "group_by": [],
            "where_conditions": ["sale_date >= '2024-01-01'::date"],
            "order_by": None,
            "limit": 100,
        }
        sql = builder.build("simple_select", cols, [], bp, {})
        assert sql is not None
        assert "2024-01-01" in sql

    def test_unknown_strategy_returns_none(self):
        cols = {"dm.x": {"select": ["a"], "filter": [], "aggregate": [], "group_by": []}}
        bp = {"strategy": "custom_magic", "main_table": "dm.x", "aggregation": None,
              "group_by": [], "where_conditions": [], "order_by": None, "limit": 100}
        sql = builder.build("custom_magic", cols, [], bp, {})
        assert sql is None

    def test_no_join_spec_for_join_strategy_returns_none(self):
        cols = {
            "dm.sales": {"select": ["a"], "filter": [], "aggregate": [], "group_by": []},
            "dm.clients": {"select": ["b"], "filter": [], "aggregate": [], "group_by": []},
        }
        bp = {"strategy": "fact_dim_join", "main_table": "dm.sales",
              "aggregation": None, "group_by": [], "where_conditions": [], "order_by": None, "limit": 100}
        sql = builder.build("fact_dim_join", cols, [], bp, {"dm.sales": "fact", "dm.clients": "dim"})
        assert sql is None
