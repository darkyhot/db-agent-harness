"""Тесты для core/sql_planner_deterministic.py."""

import pandas as pd
import pytest
from unittest.mock import MagicMock
from core.sql_planner_deterministic import (
    build_blueprint,
    _determine_strategy,
    _compute_group_by,
    _compute_aggregation,
    _compute_aggregations,
    _choose_count_identifier_column,
    _compute_where_from_intent,
    _find_date_column,
    _apply_time_granularity,
)


# ---------------------------------------------------------------------------
# _determine_strategy
# ---------------------------------------------------------------------------

class TestDetermineStrategy:
    def test_single_table(self):
        strategy, main = _determine_strategy({"dm.sales": "fact"}, [])
        assert strategy == "simple_select"
        assert main == "dm.sales"

    def test_no_tables(self):
        strategy, main = _determine_strategy({}, [])
        assert strategy == "simple_select"
        assert main == ""

    def test_fact_dim(self):
        types = {"dm.sales": "fact", "dm.clients": "dim"}
        strategy, main = _determine_strategy(types, [])
        assert strategy == "fact_dim_join"
        assert main == "dm.sales"

    def test_fact_ref(self):
        types = {"dm.outflow": "fact", "dm.gosb": "ref"}
        strategy, main = _determine_strategy(types, [])
        assert strategy == "fact_dim_join"
        assert main == "dm.outflow"

    def test_fact_fact(self):
        types = {"dm.sales": "fact", "dm.payments": "fact"}
        strategy, main = _determine_strategy(types, [])
        assert strategy == "fact_fact_join"
        assert main == "dm.sales"

    def test_dim_dim(self):
        types = {"dm.orgs": "dim", "dm.regions": "dim"}
        strategy, main = _determine_strategy(types, [])
        assert strategy == "dim_dim_join"

    def test_fact_unknown_treated_as_fact_dim(self):
        types = {"dm.fact_sales": "fact", "dm.some_table": "unknown"}
        strategy, main = _determine_strategy(types, [])
        assert strategy == "fact_dim_join"
        assert main == "dm.fact_sales"

    def test_two_unknowns_treated_as_fact_fact(self):
        types = {"dm.a": "unknown", "dm.b": "unknown"}
        strategy, main = _determine_strategy(types, [])
        assert strategy == "fact_fact_join"


# ---------------------------------------------------------------------------
# _compute_aggregation
# ---------------------------------------------------------------------------

class TestComputeAggregation:
    def _cols(self, agg_cols):
        return {"dm.sales": {"select": [], "filter": [], "aggregate": agg_cols, "group_by": []}}

    def test_sum(self):
        intent = {"aggregation_hint": "sum"}
        agg = _compute_aggregation(intent, self._cols(["amount"]))
        assert agg == {"function": "SUM", "column": "amount", "alias": "sum_amount"}

    def test_multiple_distinct_aggregations_from_user_hints(self):
        aggs = _compute_aggregations(
            {"aggregation_hint": "count"},
            {"dm.sales": {"select": [], "filter": [], "aggregate": ["tb_id", "gosb_id"], "group_by": []}},
            user_hints={
                "aggregation_preferences_list": [
                    {"function": "count", "column": "tb_id", "distinct": True},
                    {"function": "count", "column": "gosb_id", "distinct": True},
                ]
            },
            main_table="dm.sales",
        )
        assert aggs == [
            {"function": "COUNT", "column": "tb_id", "alias": "count_tb_id", "distinct": True, "source_table": "dm.sales"},
            {"function": "COUNT", "column": "gosb_id", "alias": "count_gosb_id", "distinct": True, "source_table": "dm.sales"},
        ]

    def test_build_blueprint_keeps_multiple_aggregations(self):
        bp = build_blueprint(
            intent={"aggregation_hint": "count", "required_output": []},
            selected_columns={"dm.sales": {"select": [], "filter": [], "aggregate": ["tb_id", "gosb_id"], "group_by": []}},
            join_spec=[],
            table_types={"dm.sales": "fact"},
            join_analysis_data={},
            user_hints={
                "aggregation_preferences_list": [
                    {"function": "count", "column": "tb_id", "distinct": True},
                    {"function": "count", "column": "gosb_id", "distinct": True},
                ]
            },
        )
        assert bp["aggregation"] == {"function": "COUNT", "column": "tb_id", "alias": "count_tb_id", "distinct": True}
        assert bp["aggregations"] == [
            {"function": "COUNT", "column": "tb_id", "alias": "count_tb_id", "distinct": True, "source_table": "dm.sales"},
            {"function": "COUNT", "column": "gosb_id", "alias": "count_gosb_id", "distinct": True, "source_table": "dm.sales"},
        ]

    def test_count(self):
        intent = {"aggregation_hint": "count"}
        agg = _compute_aggregation(intent, self._cols(["client_id"]))
        assert agg["function"] == "COUNT"
        assert agg["column"] == "client_id"

    def test_count_distinct_override_from_user_hints(self):
        intent = {"aggregation_hint": "count"}
        agg = _compute_aggregation(
            intent,
            self._cols(["task_code"]),
            user_hints={
                "aggregation_preferences": {
                    "function": "count",
                    "column": "task_code",
                    "distinct": True,
                }
            },
        )
        assert agg["function"] == "COUNT"
        assert agg["column"] == "task_code"
        assert agg["distinct"] is True

    def test_count_no_agg_col_fallback(self):
        intent = {"aggregation_hint": "count"}
        agg = _compute_aggregation(intent, self._cols([]))
        assert agg == {"function": "COUNT", "column": "*", "alias": "count_all"}

    def test_avg(self):
        intent = {"aggregation_hint": "avg"}
        agg = _compute_aggregation(intent, self._cols(["salary"]))
        assert agg["function"] == "AVG"

    def test_no_hint(self):
        intent = {"aggregation_hint": None}
        agg = _compute_aggregation(intent, self._cols(["amount"]))
        assert agg is None

    def test_list_hint_no_agg(self):
        intent = {"aggregation_hint": "list"}
        agg = _compute_aggregation(intent, self._cols(["name"]))
        assert agg is None

    def test_max(self):
        intent = {"aggregation_hint": "max"}
        agg = _compute_aggregation(intent, self._cols(["score"]))
        assert agg == {"function": "MAX", "column": "score", "alias": "max_score"}

    def test_count_on_primary_identifier_becomes_distinct(self, tmp_path):
        from core.schema_loader import SchemaLoader

        tables_df = pd.DataFrame({
            "schema_name": ["dm"],
            "table_name": ["sales"],
            "description": ["Продажи по задачам"],
        })
        attrs_df = pd.DataFrame({
            "schema_name": ["dm"] * 2,
            "table_name": ["sales"] * 2,
            "column_name": ["task_code", "report_dt"],
            "dType": ["text", "date"],
            "description": ["Код задачи", "Отчетная дата"],
            "is_primary_key": [True, False],
            "unique_perc": [100.0, 0.2],
            "not_null_perc": [100.0, 100.0],
        })
        tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
        attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

        loader = SchemaLoader(data_dir=tmp_path)
        agg = _compute_aggregation(
            {"aggregation_hint": "count"},
            self._cols(["task_code"]),
            schema_loader=loader,
            semantic_frame={"requires_single_entity_count": True},
        )
        assert agg["function"] == "COUNT"
        assert agg["column"] == "task_code"
        assert agg["distinct"] is True

    def test_simple_select_no_distinct_even_when_aggregate_role_set(self, tmp_path):
        from core.schema_loader import SchemaLoader

        tables_df = pd.DataFrame({
            "schema_name": ["dm"],
            "table_name": ["snapshot_x"],
            "description": ["Снимок задач"],
            "grain": ["snapshot"],
        })
        attrs_df = pd.DataFrame({
            "schema_name": ["dm"] * 2,
            "table_name": ["snapshot_x"] * 2,
            "column_name": ["task_code", "report_dt"],
            "dType": ["text", "date"],
            "description": ["Код задачи", "Отчетная дата"],
            "is_primary_key": [True, True],
            "unique_perc": [100.0, 100.0],
            "not_null_perc": [100.0, 100.0],
        })
        tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
        attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
        loader = SchemaLoader(data_dir=tmp_path)

        agg = _compute_aggregation(
            {"aggregation_hint": "count"},
            self._cols(["task_code"]),
            schema_loader=loader,
            semantic_frame={"requires_single_entity_count": True},
            strategy="simple_select",
        )
        assert agg["column"] == "task_code"
        assert "distinct" not in agg

    def test_aggregation_preferences_distinct_false_drops_flag(self):
        intent = {"aggregation_hint": "count"}
        agg = _compute_aggregation(
            intent,
            self._cols(["task_code"]),
            user_hints={
                "aggregation_preferences": {
                    "function": "count",
                    "column": "task_code",
                    "distinct": False,
                }
            },
            semantic_frame={"requires_single_entity_count": True},
            strategy="fact_dim_join",
        )
        assert agg["column"] == "task_code"
        assert "distinct" not in agg

    def test_force_count_star_disables_safety_net_preference(self):
        intent = {"aggregation_hint": "count"}
        agg = _compute_aggregation(
            intent,
            self._cols(["task_code"]),
            user_hints={
                "aggregation_preferences": {
                    "function": "count",
                    "column": "*",
                    "distinct": False,
                    "force_count_star": True,
                }
            },
            semantic_frame={"requires_single_entity_count": True},
            strategy="fact_dim_join",
        )
        assert agg == {"function": "COUNT", "column": "*", "alias": "count_all"}


def _build_snapshot_loader(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["outflow_snapshot", "gosb_dim"],
        "description": ["Снимок фактического оттока", "Справочник ГОСБ"],
        "grain": ["snapshot", "organization"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": [
            "outflow_snapshot", "outflow_snapshot", "outflow_snapshot", "outflow_snapshot",
            "gosb_dim", "gosb_dim",
        ],
        "column_name": ["report_dt", "gosb_id", "inn", "task_code", "gosb_id", "gosb_name"],
        "dType": ["date", "text", "text", "text", "text", "text"],
        "description": [
            "Отчетная дата", "Код ГОСБ", "ИНН клиента", "Код задачи",
            "Код ГОСБ", "Название ГОСБ",
        ],
        "is_primary_key": [True, True, True, False, True, False],
        "unique_perc": [100.0, 100.0, 100.0, 100.0, 100.0, 20.0],
        "not_null_perc": [100.0] * 6,
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_simple_select_count_skips_single_entity_safety(tmp_path):
    loader = _build_snapshot_loader(tmp_path)
    blueprint = build_blueprint(
        {"aggregation_hint": "count", "date_filters": {"from": "2026-02-01", "to": "2026-03-01"}},
        {"dm.outflow_snapshot": {"select": [], "filter": ["report_dt"], "aggregate": ["*"], "group_by": []}},
        [],
        {"dm.outflow_snapshot": "fact"},
        {},
        user_input="Сколько задач по фактическому оттоку поставили в феврале 2026",
        schema_loader=loader,
        semantic_frame={"requires_single_entity_count": True, "subject": "task"},
    )
    assert blueprint["strategy"] == "simple_select"
    assert blueprint["aggregation"] == {"function": "COUNT", "column": "*", "alias": "count_all"}


def test_join_strategy_still_uses_distinct(tmp_path):
    loader = _build_snapshot_loader(tmp_path)
    blueprint = build_blueprint(
        {"aggregation_hint": "count"},
        {
            "dm.outflow_snapshot": {
                "select": ["report_dt"],
                "filter": ["report_dt"],
                "aggregate": ["*"],
                "group_by": ["report_dt"],
            },
            "dm.gosb_dim": {
                "select": ["gosb_name"],
                "filter": [],
                "aggregate": [],
                "group_by": ["gosb_name"],
            },
        },
        [{"left": "dm.outflow_snapshot.gosb_id", "right": "dm.gosb_dim.gosb_id"}],
        {"dm.outflow_snapshot": "fact", "dm.gosb_dim": "dim"},
        {},
        user_input="Сколько задач по ГОСБ",
        schema_loader=loader,
        semantic_frame={"requires_single_entity_count": True, "subject": "task"},
    )
    assert blueprint["strategy"] == "fact_dim_join"
    assert blueprint["aggregation"]["function"] == "COUNT"
    assert blueprint["aggregation"]["column"] in {"gosb_id", "inn", "task_code"}
    assert blueprint["aggregation"]["column"] != "report_dt"
    assert blueprint["aggregation"]["distinct"] is True


def test_choose_count_identifier_avoids_date_in_composite_pk(tmp_path):
    loader = _build_snapshot_loader(tmp_path)
    chosen = _choose_count_identifier_column(
        "dm.outflow_snapshot",
        loader,
        semantic_frame={"subject": "organization"},
        user_input="Сколько организаций по оттоку",
    )
    assert chosen in {"gosb_id", "inn"}
    assert chosen != "report_dt"


def test_choose_count_identifier_uses_subject_stems(tmp_path):
    loader = _build_snapshot_loader(tmp_path)
    chosen = _choose_count_identifier_column(
        "dm.outflow_snapshot",
        loader,
        semantic_frame={"subject": "task"},
        user_input="Сколько задач по оттоку",
    )
    assert chosen == "task_code"


def test_force_count_star_disables_safety_net(tmp_path):
    loader = _build_snapshot_loader(tmp_path)
    blueprint = build_blueprint(
        {"aggregation_hint": "count"},
        {
            "dm.outflow_snapshot": {"select": [], "filter": ["report_dt"], "aggregate": ["*"], "group_by": ["report_dt"]},
            "dm.gosb_dim": {"select": ["gosb_name"], "filter": [], "aggregate": [], "group_by": ["gosb_name"]},
        },
        [{"left": "dm.outflow_snapshot.gosb_id", "right": "dm.gosb_dim.gosb_id"}],
        {"dm.outflow_snapshot": "fact", "dm.gosb_dim": "dim"},
        {},
        user_input="Посчитай просто количество строк",
        schema_loader=loader,
        semantic_frame={"requires_single_entity_count": True, "subject": "task"},
        user_hints={"aggregation_preferences": {"function": "count", "column": "*", "distinct": False, "force_count_star": True}},
    )
    assert blueprint["aggregation"] == {"function": "COUNT", "column": "*", "alias": "count_all"}


# ---------------------------------------------------------------------------
# _compute_group_by
# ---------------------------------------------------------------------------

class TestComputeGroupBy:
    def _cols(self, select=None, aggregate=None, group_by=None):
        return {"dm.sales": {
            "select": select or [],
            "filter": [],
            "aggregate": aggregate or [],
            "group_by": group_by or [],
        }}

    def test_group_by_from_explicit_role(self):
        cols = self._cols(select=["region", "amount"], aggregate=["amount"], group_by=["region"])
        gb = _compute_group_by(cols, {"column": "amount"})
        assert "region" in gb
        assert "amount" not in gb

    def test_non_agg_select_goes_to_group_by(self):
        cols = self._cols(select=["region", "segment"], aggregate=[])
        gb = _compute_group_by(cols, None)
        assert "region" in gb
        assert "segment" in gb

    def test_aggregate_col_excluded(self):
        cols = self._cols(select=["region", "amount"], aggregate=["amount"])
        gb = _compute_group_by(cols, {"column": "amount"})
        assert "amount" not in gb
        assert "region" in gb

    def test_no_duplicates(self):
        cols = self._cols(select=["region"], aggregate=[], group_by=["region"])
        gb = _compute_group_by(cols, None)
        assert gb.count("region") == 1

    def test_multiple_tables(self):
        cols = {
            "dm.sales": {"select": ["amount"], "filter": [], "aggregate": ["amount"], "group_by": []},
            "dm.clients": {"select": ["region"], "filter": [], "aggregate": [], "group_by": ["region"]},
        }
        gb = _compute_group_by(cols, {"column": "amount"})
        assert "region" in gb
        assert "amount" not in gb


# ---------------------------------------------------------------------------
# _find_date_column
# ---------------------------------------------------------------------------

class TestFindDateColumn:
    def test_filter_date_suffix(self):
        cols = {"dm.sales": {"select": [], "filter": ["sale_date"], "aggregate": [], "group_by": []}}
        col = _find_date_column(cols)
        assert col == "sale_date"

    def test_dttm_suffix(self):
        cols = {"dm.t": {"select": [], "filter": ["report_dttm"], "aggregate": [], "group_by": []}}
        col = _find_date_column(cols)
        assert col == "report_dttm"

    def test_fallback_to_select(self):
        cols = {"dm.t": {"select": ["created_dt", "amount"], "filter": [], "aggregate": [], "group_by": []}}
        col = _find_date_column(cols)
        assert col == "created_dt"

    def test_no_date_column(self):
        cols = {"dm.t": {"select": ["region", "amount"], "filter": [], "aggregate": [], "group_by": []}}
        col = _find_date_column(cols)
        assert col is None


# ---------------------------------------------------------------------------
# _compute_where_from_intent
# ---------------------------------------------------------------------------

class TestComputeWhereFromIntent:
    def _cols_with_date_filter(self):
        return {"dm.sales": {"select": [], "filter": ["sale_date"], "aggregate": [], "group_by": []}}

    def _cols_with_region(self):
        return {"dm.sales": {"select": ["region"], "filter": ["region", "sale_date"], "aggregate": [], "group_by": []}}

    def test_date_from_only(self):
        intent = {"date_filters": {"from": "2024-01-01", "to": None}}
        where = _compute_where_from_intent(intent, self._cols_with_date_filter())
        assert len(where) == 1
        assert "sale_date >= '2024-01-01'::date" in where[0]

    def test_date_range(self):
        intent = {"date_filters": {"from": "2024-01-01", "to": "2024-02-01"}}
        where = _compute_where_from_intent(intent, self._cols_with_date_filter())
        assert len(where) == 2
        assert any(">=" in w for w in where)
        assert any("<" in w for w in where)

    def test_no_date_filter(self):
        intent = {"date_filters": {"from": None, "to": None}}
        where = _compute_where_from_intent(intent, self._cols_with_date_filter())
        assert where == []

    def test_no_date_column_returns_empty(self):
        intent = {"date_filters": {"from": "2024-01-01", "to": None}}
        cols = {"dm.t": {"select": ["region"], "filter": [], "aggregate": [], "group_by": []}}
        where = _compute_where_from_intent(intent, cols)
        assert where == []

    def test_filter_conditions_exact_match(self):
        intent = {
            "date_filters": {"from": None, "to": None},
            "filter_conditions": [{"column_hint": "region", "operator": "=", "value": "Москва"}],
        }
        where = _compute_where_from_intent(intent, self._cols_with_region())
        assert len(where) == 1
        assert "region = 'Москва'" in where[0]

    def test_filter_conditions_numeric_value(self):
        cols = {"dm.t": {"select": ["amount"], "filter": ["amount"], "aggregate": [], "group_by": []}}
        intent = {
            "date_filters": {"from": None, "to": None},
            "filter_conditions": [{"column_hint": "amount", "operator": ">", "value": "1000"}],
        }
        where = _compute_where_from_intent(intent, cols)
        assert len(where) == 1
        assert "amount > 1000" in where[0]

    def test_filter_conditions_partial_hint_match(self):
        cols = {"dm.t": {"select": ["gosb_code"], "filter": ["gosb_code"], "aggregate": [], "group_by": []}}
        intent = {
            "date_filters": {"from": None, "to": None},
            "filter_conditions": [{"column_hint": "gosb", "operator": "=", "value": "0770"}],
        }
        where = _compute_where_from_intent(intent, cols)
        assert len(where) == 1
        assert "gosb_code = '0770'" in where[0]

    def test_filter_conditions_invalid_operator_skipped(self):
        intent = {
            "date_filters": {"from": None, "to": None},
            "filter_conditions": [{"column_hint": "region", "operator": "DROP TABLE", "value": "x"}],
        }
        where = _compute_where_from_intent(intent, self._cols_with_region())
        # Недопустимый оператор — пропускаем
        assert not any("DROP" in w for w in where)

    def test_filter_conditions_combined_with_date(self):
        intent = {
            "date_filters": {"from": "2024-01-01", "to": None},
            "filter_conditions": [{"column_hint": "region", "operator": "=", "value": "Урал"}],
        }
        where = _compute_where_from_intent(intent, self._cols_with_region())
        assert len(where) == 2
        assert any("2024-01-01" in w for w in where)
        assert any("region" in w for w in where)


# ---------------------------------------------------------------------------
# build_blueprint (интеграционные тесты)
# ---------------------------------------------------------------------------

class TestBuildBlueprint:
    def _intent(self, agg="sum", date_from=None, date_to=None):
        return {
            "aggregation_hint": agg,
            "date_filters": {"from": date_from, "to": date_to},
            "complexity": "single_table",
        }

    def _cols(self, tables=None, aggregate=None):
        if tables is None:
            tables = ["dm.sales"]
        result = {}
        for t in tables:
            result[t] = {
                "select": ["region", "amount"],
                "filter": ["sale_date"],
                "aggregate": aggregate or ["amount"],
                "group_by": ["region"],
            }
        return result

    def test_simple_select_strategy(self):
        bp = build_blueprint(
            self._intent(agg=None),
            self._cols(),
            [],
            {"dm.sales": "fact"},
            {},
        )
        assert bp["strategy"] == "simple_select"
        assert bp["main_table"] == "dm.sales"
        assert bp["cte_needed"] is False

    def test_fact_dim_strategy(self):
        types = {"dm.sales": "fact", "dm.clients": "dim"}
        cols = {
            "dm.sales": {"select": ["region", "amount"], "filter": [], "aggregate": ["amount"], "group_by": ["region"]},
            "dm.clients": {"select": ["name"], "filter": [], "aggregate": [], "group_by": []},
        }
        bp = build_blueprint(self._intent(), cols, [], types, {})
        assert bp["strategy"] == "fact_dim_join"
        assert bp["main_table"] == "dm.sales"

    def test_fact_fact_cte_needed(self):
        types = {"dm.sales": "fact", "dm.payments": "fact"}
        bp = build_blueprint(self._intent(), self._cols(["dm.sales", "dm.payments"]), [], types, {})
        assert bp["strategy"] == "fact_fact_join"
        assert bp["cte_needed"] is True

    def test_unsafe_join_forces_cte(self):
        types = {"dm.sales": "fact", "dm.clients": "dim"}
        join_spec = [{"left": "dm.sales.client_id", "right": "dm.clients.id", "safe": False}]
        bp = build_blueprint(self._intent(), self._cols(["dm.sales", "dm.clients"]), join_spec, types, {})
        assert bp["cte_needed"] is True

    def test_safe_join_no_cte(self):
        types = {"dm.sales": "fact", "dm.clients": "dim"}
        join_spec = [{"left": "dm.sales.client_id", "right": "dm.clients.id", "safe": True}]
        cols = {
            "dm.sales": {"select": ["region", "client_id"], "filter": [], "aggregate": ["region"], "group_by": ["client_id"]},
            "dm.clients": {"select": ["name"], "filter": [], "aggregate": [], "group_by": []},
        }
        bp = build_blueprint(self._intent(agg="count"), cols, join_spec, types, {})
        # dim+dim или fact+dim с safe join → не требует CTE
        assert bp["strategy"] == "fact_dim_join"

    def test_aggregation_built(self):
        bp = build_blueprint(
            self._intent(agg="sum"),
            self._cols(),
            [],
            {"dm.sales": "fact"},
            {},
        )
        assert bp["aggregation"] is not None
        assert bp["aggregation"]["function"] == "SUM"
        assert bp["aggregation"]["column"] == "amount"
        assert bp["aggregation"]["alias"] == "sum_amount"

    def test_group_by_excludes_agg_col(self):
        bp = build_blueprint(
            self._intent(agg="sum"),
            self._cols(),
            [],
            {"dm.sales": "fact"},
            {},
        )
        assert "amount" not in bp["group_by"]
        assert "region" in bp["group_by"]

    def test_date_filter_in_where(self):
        bp = build_blueprint(
            self._intent(agg="sum", date_from="2024-01-01"),
            self._cols(),
            [],
            {"dm.sales": "fact"},
            {},
        )
        assert any("2024-01-01" in w for w in bp["where_conditions"])

    def test_order_by_agg_desc(self):
        bp = build_blueprint(
            self._intent(agg="sum"),
            self._cols(),
            [],
            {"dm.sales": "fact"},
            {},
        )
        assert bp["order_by"] == "sum_amount DESC"

    def test_aggregation_prefers_main_table_metric_when_dim_listed_first(self):
        cols = {
            "dm.dim_gosb": {
                "select": ["new_gosb_name"],
                "filter": [],
                "aggregate": [],
                "group_by": ["new_gosb_name"],
            },
            "dm.fact_outflow": {
                "select": ["report_dt", "outflow_qty"],
                "filter": [],
                "aggregate": ["outflow_qty"],
                "group_by": ["report_dt"],
            },
        }

        bp = build_blueprint(
            self._intent(agg="sum"),
            cols,
            [
                {"left": "dm.fact_outflow.gosb_id", "right": "dm.dim_gosb.old_gosb_id"},
                {"left": "dm.fact_outflow.tb_id", "right": "dm.dim_gosb.tb_id"},
            ],
            {"dm.fact_outflow": "fact", "dm.dim_gosb": "dim"},
            {},
        )

        assert bp["main_table"] == "dm.fact_outflow"
        assert bp["aggregation"]["column"] == "outflow_qty"
        assert bp["aggregation"]["alias"] == "sum_outflow_qty"

    def test_join_keys_do_not_leak_into_aggregate_group_by(self):
        cols = {
            "dm.fact_outflow": {
                "select": ["report_dt", "gosb_id", "tb_id"],
                "filter": [],
                "aggregate": ["outflow_qty"],
                "group_by": ["report_dt"],
            },
            "dm.dim_gosb": {
                "select": ["new_gosb_name", "old_gosb_id", "tb_id"],
                "filter": [],
                "aggregate": [],
                "group_by": ["new_gosb_name"],
            },
        }

        bp = build_blueprint(
            self._intent(agg="sum"),
            cols,
            [
                {"left": "dm.fact_outflow.gosb_id", "right": "dm.dim_gosb.old_gosb_id"},
                {"left": "dm.fact_outflow.tb_id", "right": "dm.dim_gosb.tb_id"},
            ],
            {"dm.fact_outflow": "fact", "dm.dim_gosb": "dim"},
            {},
        )

        assert bp["group_by"] == ["report_dt", "new_gosb_name"]
        assert cols["dm.fact_outflow"]["select"] == ["report_dt"]
        assert cols["dm.dim_gosb"]["select"] == ["new_gosb_name"]

    def test_no_aggregation_no_order_by(self):
        bp = build_blueprint(
            self._intent(agg=None),
            self._cols(),
            [],
            {"dm.sales": "fact"},
            {},
        )
        assert bp["aggregation"] is None
        assert bp["order_by"] is None

    def test_no_default_limit(self):
        """Без явного intent.limit не должен ставиться дефолтный LIMIT."""
        bp = build_blueprint(self._intent(), self._cols(), [], {"dm.sales": "fact"}, {})
        assert bp["limit"] is None

    def test_notes_contain_strategy(self):
        bp = build_blueprint(self._intent(), self._cols(), [], {"dm.sales": "fact"}, {})
        assert "[deterministic]" in bp["notes"]
        assert "simple_select" in bp["notes"]

    def test_dim_dim_cte_needed(self):
        types = {"dm.orgs": "dim", "dm.regions": "dim"}
        cols = {
            "dm.orgs": {"select": ["org_id", "org_name"], "filter": [], "aggregate": [], "group_by": ["org_id"]},
            "dm.regions": {"select": ["region"], "filter": [], "aggregate": [], "group_by": []},
        }
        bp = build_blueprint(self._intent(agg=None), cols, [], types, {})
        assert bp["strategy"] == "dim_dim_join"
        assert bp["cte_needed"] is True

    def test_filter_col_promoted_to_group_by_only_when_dimension_requested(self):
        """Filter-only дата попадает в GROUP BY только если пользователь просил измерение."""
        intent = {
            "aggregation_hint": "sum",
            "date_filters": {"from": None, "to": None},
            "complexity": "join",
        }
        cols = {
            "dm.fact_outflow": {
                "select": ["outflow_qty"],
                "filter": ["report_dt"],   # ← column_selector положил в filter
                "aggregate": ["outflow_qty"],
                "group_by": [],
            },
            "dm.dim_gosb": {
                "select": ["new_gosb_name"],
                "filter": [],
                "aggregate": [],
                "group_by": [],
            },
        }
        bp = build_blueprint(
            intent, cols, [],
            {"dm.fact_outflow": "fact", "dm.dim_gosb": "dim"},
            {},
            semantic_frame={"output_dimensions": ["date"]},
        )
        assert "report_dt" in bp["group_by"]

    def test_filter_col_not_promoted_when_where_exists(self):
        """Если для filter-колонки есть WHERE-условие, в group_by она НЕ добавляется."""
        intent = {
            "aggregation_hint": "sum",
            "date_filters": {"from": "2024-01-01", "to": None},
            "complexity": "single_table",
        }
        cols = {
            "dm.sales": {
                "select": ["region", "amount"],
                "filter": ["sale_date"],
                "aggregate": ["amount"],
                "group_by": ["region"],
            },
        }
        bp = build_blueprint(intent, cols, [], {"dm.sales": "fact"}, {})
        # sale_date используется в WHERE (date_filters.from) → не должна быть в group_by
        assert "sale_date" not in bp["group_by"]


# ---------------------------------------------------------------------------
# Проверки: нет дефолтного LIMIT и поддержка COUNT DISTINCT
# ---------------------------------------------------------------------------

class TestNoDefaultLimit:
    """LIMIT не должен добавляться если пользователь его не запрашивал."""

    def test_no_limit_by_default(self):
        intent = {
            "aggregation_hint": "sum",
            "date_filters": {"from": None, "to": None},
            "complexity": "single_table",
        }
        cols = {"dm.sales": {"select": ["region", "amount"], "aggregate": ["amount"], "group_by": ["region"]}}
        bp = build_blueprint(intent, cols, [], {"dm.sales": "fact"}, {})
        assert bp.get("limit") is None, f"Ожидали limit=None, получили {bp.get('limit')}"

    def test_explicit_limit_from_intent(self):
        intent = {
            "aggregation_hint": "sum",
            "limit": 50,
            "date_filters": {"from": None, "to": None},
            "complexity": "single_table",
        }
        cols = {"dm.sales": {"select": ["amount"], "aggregate": ["amount"], "group_by": []}}
        bp = build_blueprint(intent, cols, [], {"dm.sales": "fact"}, {})
        assert bp.get("limit") == 50


class TestCountAggregation:
    """COUNT не должен становиться DISTINCT без явного сигнала."""

    def test_count_with_pk_col_has_no_distinct_by_default(self):
        intent = {"aggregation_hint": "count"}
        cols = {"dm.gosb": {"select": ["gosb_id"], "aggregate": ["gosb_id"], "group_by": []}}
        agg = _compute_aggregation(intent, cols)
        assert agg is not None
        assert agg["function"] == "COUNT"
        assert not agg.get("distinct"), "COUNT по колонке не должен становиться DISTINCT по умолчанию"

    def test_count_star_fallback_no_distinct(self):
        intent = {"aggregation_hint": "count"}
        cols = {"dm.gosb": {"select": ["gosb_name"], "group_by": ["gosb_name"]}}
        agg = _compute_aggregation(intent, cols)
        assert agg is not None
        assert agg["column"] == "*"
        assert not agg.get("distinct"), "COUNT(*) не должен иметь distinct"


# ---------------------------------------------------------------------------
# Task 1.2: _apply_time_granularity
# ---------------------------------------------------------------------------

class TestApplyTimeGranularity:
    def _make_loader(self, date_cols: list[str]):
        """Сделать мок schema_loader с date-колонками."""
        loader = MagicMock()
        rows = [{"column_name": c, "dType": "timestamp"} for c in date_cols]
        loader.get_table_columns.return_value = pd.DataFrame(rows)
        return loader

    def test_wraps_date_col_in_group_by(self):
        """Если date-колонка уже в group_by — оборачивает в DATE_TRUNC."""
        loader = self._make_loader(["report_dt"])
        selected = {"dm.fact": {"select": ["report_dt", "amount"], "filter": [], "group_by": ["report_dt"]}}
        result = _apply_time_granularity(["report_dt", "amount"], selected, "month", loader)
        assert "DATE_TRUNC('month', report_dt)" in result
        assert "amount" in result

    def test_adds_date_col_not_in_group_by(self):
        """Если date-колонка только в filter — добавляет DATE_TRUNC в group_by."""
        loader = self._make_loader(["event_date"])
        selected = {"dm.fact": {"select": ["amount"], "filter": ["event_date"], "group_by": []}}
        result = _apply_time_granularity(["amount"], selected, "quarter", loader)
        assert any("DATE_TRUNC('quarter'" in r for r in result)

    def test_no_granularity_returns_unchanged(self):
        """Без gранулярности group_by не меняется."""
        loader = self._make_loader(["report_dt"])
        selected = {"dm.fact": {"select": ["report_dt"], "filter": [], "group_by": []}}
        original = ["report_dt"]
        result = _apply_time_granularity(original, selected, None, loader)
        assert result == original

    def test_no_date_cols_returns_unchanged(self):
        """Если нет date-колонок — group_by не меняется."""
        loader = self._make_loader([])
        selected = {"dm.fact": {"select": ["name"], "filter": [], "group_by": []}}
        result = _apply_time_granularity(["name"], selected, "month", loader)
        assert result == ["name"]

    def test_build_blueprint_applies_date_trunc(self):
        """build_blueprint с time_granularity='month' и date-колонкой → DATE_TRUNC в group_by."""
        loader = self._make_loader(["report_dt"])
        intent = {
            "aggregation_hint": "count",
            "date_filters": {"from": None, "to": None},
            "complexity": "single_table",
        }
        cols = {
            "dm.fact": {
                "select": ["report_dt", "client_id"],
                "aggregate": ["client_id"],
                "group_by": ["report_dt"],
                "filter": [],
            }
        }
        user_hints = {"time_granularity": "month"}
        bp = build_blueprint(intent, cols, [], {"dm.fact": "fact"}, {}, user_hints=user_hints, schema_loader=loader)
        group_by_str = " ".join(bp.get("group_by", []))
        assert "DATE_TRUNC('month'" in group_by_str


def test_build_blueprint_does_not_leak_filter_only_keys_into_group_by():
    intent = {"aggregation_hint": "count", "required_output": []}
    cols = {
        "dm.sales": {
            "select": ["task_code"],
            "filter": ["inn", "report_dt"],
            "aggregate": ["task_code"],
            "group_by": [],
        }
    }
    bp = build_blueprint(
        intent,
        cols,
        [],
        {"dm.sales": "fact"},
        {},
        semantic_frame={"output_dimensions": []},
    )
    assert "inn" not in bp["group_by"]
    assert "report_dt" not in bp["group_by"]


def test_build_blueprint_single_entity_count_clears_spurious_group_by():
    intent = {"aggregation_hint": "count", "required_output": []}
    cols = {
        "dm.sales": {
            "select": ["uzp_task_code", "task_code"],
            "filter": ["report_dt"],
            "aggregate": ["task_code"],
            "group_by": ["uzp_task_code"],
        }
    }
    bp = build_blueprint(
        intent,
        cols,
        [],
        {"dm.sales": "fact"},
        {},
        semantic_frame={
            "requires_single_entity_count": True,
            "output_dimensions": [],
        },
    )
    assert bp["aggregation"]["column"] == "task_code"
    assert "distinct" not in bp["aggregation"]
    assert bp["group_by"] == []


def test_build_blueprint_for_outflow_tasks_uses_count_task_code_without_distinct(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_data_split_mzp_sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": ["uzp_data_split_mzp_sale_funnel"] * 5,
        "column_name": ["report_dt", "task_code", "uzp_task_code", "task_subtype", "task_category"],
        "dType": ["date", "text", "text", "text", "text"],
        "description": [
            "Отчетная дата",
            "Код задачи",
            "Код задачи УЗП",
            "Подтип задачи",
            "Категория задачи",
        ],
        "is_primary_key": [False, True, False, False, False],
        "unique_perc": [0.5, 100.0, 36.68, 2.62, 0.02],
        "not_null_perc": [99.0, 100.0, 36.68, 37.06, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    bp = build_blueprint(
        intent={
            "aggregation_hint": "count",
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
            "required_output": [],
        },
        selected_columns={
            "dm.uzp_data_split_mzp_sale_funnel": {
                "select": ["task_code"],
                "filter": ["report_dt", "task_subtype", "task_category"],
                "aggregate": ["task_code"],
                "group_by": [],
            }
        },
        join_spec=[],
        table_types={"dm.uzp_data_split_mzp_sale_funnel": "fact"},
        join_analysis_data={},
        user_input="Сколько задач по фактическому оттоку поставили в феврале 26",
        schema_loader=loader,
        semantic_frame={
            "subject": "task",
            "requires_single_entity_count": True,
            "output_dimensions": [],
        },
    )
    assert bp["aggregation"]["column"] == "task_code"
    assert "distinct" not in bp["aggregation"]
    assert bp["group_by"] == []


def test_build_blueprint_simple_select_keeps_count_star_without_safety_net(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sales"],
        "description": ["Продажи по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["sales"] * 3,
        "column_name": ["task_code", "uzp_task_code", "report_dt"],
        "dType": ["text", "text", "date"],
        "description": ["Код задачи", "Код задачи УЗП", "Отчетная дата"],
        "is_primary_key": [True, False, False],
        "unique_perc": [100.0, 36.68, 0.5],
        "not_null_perc": [100.0, 36.68, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    bp = build_blueprint(
        intent={"aggregation_hint": "count", "required_output": []},
        selected_columns={
            "dm.sales": {
                "select": ["uzp_task_code"],
                "filter": ["report_dt"],
                "aggregate": ["*"],
                "group_by": ["uzp_task_code"],
            }
        },
        join_spec=[],
        table_types={"dm.sales": "fact"},
        join_analysis_data={},
        user_input="Сколько задач по фактическому оттоку",
        schema_loader=loader,
        semantic_frame={
            "subject": "task",
            "requires_single_entity_count": True,
            "output_dimensions": [],
        },
    )
    assert bp["aggregation"]["column"] == "*"
    assert "distinct" not in bp["aggregation"]
    assert bp["group_by"] == []


def _build_outflow_event_loader_for_count(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_dwh_fact_outflow"],
        "description": ["Информация по фактическим оттокам"],
        "grain": ["event"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": ["uzp_dwh_fact_outflow"] * 5,
        "column_name": ["report_dt", "inn", "gosb_id", "is_task", "inserted_dttm"],
        "dType": ["date", "int8", "int4", "boolean", "timestamp"],
        "description": [
            "Отчетная дата",
            "ИНН",
            "Идентификатор ГОСБ",
            "Признак выставленной задачи",
            "Дата и время загрузки",
        ],
        "is_primary_key": [False, True, False, False, False],
        "unique_perc": [0.5, 98.75, 0.93, 0.02, 0.02],
        "not_null_perc": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_build_blueprint_rewrites_report_dt_count_to_count_star_for_simple_select(tmp_path):
    loader = _build_outflow_event_loader_for_count(tmp_path)

    bp = build_blueprint(
        intent={
            "aggregation_hint": "count",
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
            "required_output": [],
        },
        selected_columns={
            "dm.uzp_dwh_fact_outflow": {
                "select": ["report_dt"],
                "filter": ["report_dt", "is_task"],
                "aggregate": ["report_dt"],
                "group_by": [],
            }
        },
        join_spec=[],
        table_types={"dm.uzp_dwh_fact_outflow": "fact"},
        join_analysis_data={},
        user_input="Сколько задач по фактическому оттоку поставили в феврале 26",
        schema_loader=loader,
        semantic_frame={"subject": "task", "requires_single_entity_count": True, "output_dimensions": []},
    )

    assert bp["aggregation"] == {"function": "COUNT", "column": "*", "alias": "count_all"}
    assert bp["group_by"] == []


def test_build_blueprint_employee_subject_still_rewrites_report_dt_count(tmp_path):
    loader = _build_outflow_event_loader_for_count(tmp_path)

    bp = build_blueprint(
        intent={
            "aggregation_hint": "count",
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
            "required_output": [],
        },
        selected_columns={
            "dm.uzp_dwh_fact_outflow": {
                "select": ["report_dt"],
                "filter": ["report_dt", "is_task"],
                "aggregate": ["report_dt"],
                "group_by": [],
            }
        },
        join_spec=[],
        table_types={"dm.uzp_dwh_fact_outflow": "fact"},
        join_analysis_data={},
        user_input="Сколько задач по фактическому оттоку поставили в феврале 26",
        schema_loader=loader,
        semantic_frame={"subject": "employee", "requires_single_entity_count": True, "output_dimensions": []},
    )

    assert bp["aggregation"] == {"function": "COUNT", "column": "*", "alias": "count_all"}


# ---------------------------------------------------------------------------
# A.4 / C.3: _find_date_column priority + alternatives
# ---------------------------------------------------------------------------

class TestFindDateColumnPriority:
    def test_prefers_report_dt_over_inserted_dttm(self):
        cols = {
            "dm.t": {
                "select": [],
                "filter": ["inserted_dttm", "report_dt"],
                "aggregate": [],
                "group_by": [],
            }
        }
        assert _find_date_column(cols) == "report_dt"

    def test_prefers_dt_suffix_over_inserted(self):
        cols = {
            "dm.t": {
                "filter": ["inserted_dttm", "event_dt"], "select": [], "aggregate": [], "group_by": [],
            }
        }
        assert _find_date_column(cols) == "event_dt"

    def test_only_inserted_picks_inserted(self):
        cols = {
            "dm.t": {
                "filter": ["inserted_dttm"], "select": [], "aggregate": [], "group_by": [],
            }
        }
        assert _find_date_column(cols) == "inserted_dttm"


# ---------------------------------------------------------------------------
# C.4: type validation in _compute_where_from_intent
# ---------------------------------------------------------------------------

def _build_loader_with_bool_and_date(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame(
        [{"schema_name": "dm", "table_name": "fact", "description": "t"}]
    )
    attrs_df = pd.DataFrame(
        [
            {
                "schema_name": "dm", "table_name": "fact",
                "column_name": "is_task", "dType": "bool",
                "description": "Признак задачи",
                "is_primary_key": False, "unique_perc": 50.0, "not_null_perc": 100.0,
            },
            {
                "schema_name": "dm", "table_name": "fact",
                "column_name": "report_dt", "dType": "date",
                "description": "Отчётная дата",
                "is_primary_key": False, "unique_perc": 1.0, "not_null_perc": 100.0,
            },
        ]
    )
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_compute_where_drops_date_literal_for_bool_column(tmp_path):
    """C.4: дата-литерал не должен попадать в WHERE для bool-колонки."""
    loader = _build_loader_with_bool_and_date(tmp_path)
    selected_columns = {
        "dm.fact": {
            "select": [], "filter": ["is_task", "report_dt"],
            "aggregate": [], "group_by": [],
        }
    }
    intent = {
        "filter_conditions": [
            {"column_hint": "is_task", "operator": "=", "value": "2026-03-01"},
        ],
        "date_filters": {"from": None, "to": None},
    }
    where = _compute_where_from_intent(
        intent, selected_columns, user_input="", schema_loader=loader,
    )
    joined = " | ".join(where)
    assert "is_task = '2026-03-01'" not in joined
    assert "is_task = \"2026-03-01\"" not in joined


def test_compute_where_keeps_compatible_filter(tmp_path):
    """C.4: совместимый литерал — сохраняется."""
    loader = _build_loader_with_bool_and_date(tmp_path)
    selected_columns = {
        "dm.fact": {
            "select": [], "filter": ["is_task"],
            "aggregate": [], "group_by": [],
        }
    }
    intent = {
        "filter_conditions": [
            {"column_hint": "is_task", "operator": "=", "value": "true"},
        ],
        "date_filters": {"from": None, "to": None},
    }
    where = _compute_where_from_intent(
        intent, selected_columns, user_input="", schema_loader=loader,
    )
    assert any("is_task" in c for c in where)
