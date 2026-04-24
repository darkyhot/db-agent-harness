import json
import sys
from unittest.mock import MagicMock

import pandas as pd

from core.column_selector_deterministic import normalize_join_spec
from core.schema_loader import SchemaLoader
from core.sql_builder import SqlBuilder
from core.sql_formatter import format_sql_safe
from core.sql_planner_deterministic import build_blueprint


def _ensure_mock_modules():
    for mod_name in (
        "langchain_gigachat",
        "langchain_gigachat.chat_models",
        "langchain_core",
        "langchain_core.messages",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_ensure_mock_modules()

from graph.nodes import GraphNodes


FACT = "schema.uzp_dwh_fact_outflow"
GOSB = "schema.uzp_dim_gosb"
EPK = "schema.uzp_data_epk_consolidation"


def _loader(tmp_path):
    pd.DataFrame(
        {
            "schema_name": ["schema", "schema", "schema"],
            "table_name": [
                "uzp_dwh_fact_outflow",
                "uzp_dim_gosb",
                "uzp_data_epk_consolidation",
            ],
            "description": ["outflow fact", "gosb dim", "epk dim"],
        }
    ).to_csv(tmp_path / "tables_list.csv", index=False)

    pd.DataFrame(
        {
            "schema_name": ["schema"] * 9,
            "table_name": [
                "uzp_dwh_fact_outflow",
                "uzp_dwh_fact_outflow",
                "uzp_dwh_fact_outflow",
                "uzp_dwh_fact_outflow",
                "uzp_dim_gosb",
                "uzp_dim_gosb",
                "uzp_dim_gosb",
                "uzp_data_epk_consolidation",
                "uzp_data_epk_consolidation",
            ],
            "column_name": [
                "report_dt",
                "gosb_id",
                "tb_id",
                "outflow_qty",
                "old_gosb_id",
                "tb_id",
                "new_gosb_name",
                "inn",
                "segment_name",
            ],
            "dType": ["date", "int", "int", "numeric", "int", "int", "text", "text", "text"],
            "description": [""] * 9,
            "is_primary_key": [False, False, False, False, True, True, False, False, False],
            "unique_perc": [10.0, 40.0, 5.0, 1.0, 70.0, 10.0, 20.0, 40.0, 10.0],
        }
    ).to_csv(tmp_path / "attr_list.csv", index=False)

    return SchemaLoader(data_dir=tmp_path)


def test_normalize_join_spec_completes_llm_single_pair_and_recomputes_composite_safety(tmp_path):
    loader = _loader(tmp_path)
    join_spec = [
        {
            "left": f"{FACT}.gosb_id",
            "right": f"{GOSB}.old_gosb_id",
            "safe": True,
            "strategy": "direct",
        }
    ]

    normalized = normalize_join_spec(
        join_spec,
        loader,
        {FACT: "fact", GOSB: "dim"},
    )

    pairs = {(j["left"], j["right"]) for j in normalized}
    assert (f"{FACT}.gosb_id", f"{GOSB}.old_gosb_id") in pairs
    assert (f"{FACT}.tb_id", f"{GOSB}.tb_id") in pairs
    assert all(j["safe"] is True for j in normalized)


def test_outflow_by_date_gosb_name_uses_full_composite_join_and_ignores_unused_join_analysis(tmp_path):
    loader = _loader(tmp_path)
    selected_columns = {
        FACT: {
            "select": ["report_dt", "outflow_qty"],
            "aggregate": ["outflow_qty"],
            "group_by": ["report_dt"],
        },
        GOSB: {"select": ["new_gosb_name"], "group_by": ["new_gosb_name"]},
    }
    join_spec = normalize_join_spec(
        [{"left": f"{FACT}.gosb_id", "right": f"{GOSB}.old_gosb_id", "safe": True}],
        loader,
        {FACT: "fact", GOSB: "dim", EPK: "dim"},
    )
    blueprint = build_blueprint(
        intent={"aggregation_hint": "sum"},
        selected_columns=selected_columns,
        join_spec=join_spec,
        table_types={FACT: "fact", GOSB: "dim", EPK: "dim"},
        join_analysis_data={
            "unused": {"table1": FACT, "table2": EPK, "text": "candidate inn = inn"}
        },
        user_input="Посчитай сумму оттока по дате и названию ГОСБ",
        schema_loader=loader,
    )
    sql = SqlBuilder().build(
        "fact_dim_join",
        selected_columns,
        join_spec,
        blueprint,
        {FACT: "fact", GOSB: "dim", EPK: "dim"},
    )

    assert sql is not None
    sql_l = sql.lower()
    assert "gosb_id" in sql_l and "old_gosb_id" in sql_l
    assert "tb_id" in sql_l
    assert " and " in sql_l
    assert EPK not in {j["left"].rsplit(".", 1)[0] for j in join_spec}


class _WriterLLM:
    def __init__(self, sql: str) -> None:
        self.sql = sql

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        return json.dumps({"tool": "execute_query", "args": {"sql": self.sql}}, ensure_ascii=False)


def _nodes(sql: str):
    memory = MagicMock()
    memory.get_memory_list.return_value = []
    return GraphNodes(
        _WriterLLM(sql),
        MagicMock(),
        MagicMock(),
        memory,
        MagicMock(),
        [],
        debug_prompt=False,
    )


def test_sql_writer_rejects_llm_sql_missing_composite_join_pair():
    bad_sql = (
        f"SELECT f.report_dt, g.new_gosb_name, SUM(f.outflow_qty) AS total_outflow "
        f"FROM {FACT} f JOIN {GOSB} g ON f.gosb_id = g.old_gosb_id "
        f"GROUP BY f.report_dt, g.new_gosb_name"
    )
    state = {
        "messages": [],
        "graph_iterations": 0,
        "current_step": 0,
        "user_input": "Посчитай сумму оттока по дате и названию ГОСБ",
        "sql_blueprint": {"strategy": "unsupported", "main_table": FACT},
        "selected_columns": {},
        "join_spec": [
            {"left": f"{FACT}.gosb_id", "right": f"{GOSB}.old_gosb_id", "safe": True},
            {"left": f"{FACT}.tb_id", "right": f"{GOSB}.tb_id", "safe": True},
        ],
        "table_types": {FACT: "fact", GOSB: "dim"},
        "allowed_tables": [],
        "planning_confidence": {"level": "high", "action": "execute"},
        "evidence_trace": {},
        "tool_calls": [],
    }

    result = _nodes(bad_sql).sql_writer(state)

    assert result["sql_to_validate"] is None
    assert "неполным составным JOIN" in result["last_error"]
    assert "tb_id = tb_id" in result["last_error"]


def test_sql_writer_normalizes_cyrillic_alias_from_llm_output():
    sql = 'SELECT SUM(outflow_qty) AS "Сумма оттока" FROM schema.uzp_dwh_fact_outflow'
    state = {
        "messages": [],
        "graph_iterations": 0,
        "current_step": 0,
        "user_input": "Посчитай сумму оттока",
        "sql_blueprint": {"strategy": "unsupported", "main_table": FACT},
        "selected_columns": {},
        "join_spec": [],
        "table_types": {},
        "allowed_tables": [],
        "planning_confidence": {"level": "high", "action": "execute"},
        "evidence_trace": {},
        "tool_calls": [],
    }

    result = _nodes(sql).sql_writer(state)

    assert result["sql_to_validate"] == format_sql_safe(sql)
    assert "Сумма" not in result["sql_to_validate"]
    assert "summa_ottoka" in result["sql_to_validate"]
