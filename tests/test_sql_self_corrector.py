import json
import sys
from unittest.mock import MagicMock


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


class ReviewLLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        self.calls.append((system_prompt, user_prompt))
        return json.dumps(self.response, ensure_ascii=False)


def _nodes(response: dict) -> GraphNodes:
    memory = MagicMock()
    memory.get_memory_list.return_value = []
    return GraphNodes(
        ReviewLLM(response),
        MagicMock(),
        None,
        memory,
        MagicMock(),
        [],
        debug_prompt=False,
    )


def _state(sql: str):
    return {
        "messages": [],
        "graph_iterations": 0,
        "current_step": 0,
        "user_input": "Покажи сумму продаж по региону",
        "sql_to_validate": sql,
        "pending_sql_tool_call": {
            "tool": "execute_query",
            "args": {"sql": sql},
            "step_idx": 0,
        },
        "tool_calls": [
            {"tool": "execute_query", "args": {"sql": sql}, "result": "awaiting_validation"}
        ],
        "query_spec": {},
        "query_grounding": {},
        "join_spec": [],
        "evidence_trace": {},
        "sql_blueprint": {
            "required_output": ["region"],
            "aggregation": {"function": "sum", "column": "amount"},
        },
        "selected_columns": {
            "dm.sales": {"select": ["region"], "aggregate": ["amount"]},
        },
        "allowed_tables": ["dm.sales"],
    }


def test_sql_self_corrector_passes_valid_review():
    sql = "SELECT region, SUM(amount) AS total_amount FROM dm.sales GROUP BY region"
    nodes = _nodes({"verdict": "pass", "issues": [], "rationale": "ok"})

    result = nodes.sql_self_corrector(_state(sql))

    assert result["sql_to_validate"] == sql
    assert result["sql_self_correction"]["verdict"] == "pass"
    assert "last_error" not in result


def test_sql_self_corrector_applies_corrected_sql_to_pending_call_and_tool_call():
    original = "SELECT SUM(amount) AS total_amount FROM dm.sales"
    corrected = "SELECT region, SUM(amount) AS total_amount FROM dm.sales GROUP BY region"
    nodes = _nodes(
        {
            "verdict": "fix",
            "issues": ["region requested but missing"],
            "corrected_sql": corrected,
            "rationale": "added region",
        }
    )

    result = nodes.sql_self_corrector(_state(original))

    assert "region" in result["sql_to_validate"].lower()
    assert result["pending_sql_tool_call"]["args"]["sql"] == result["sql_to_validate"]
    assert result["tool_calls"][-1]["args"]["sql"] == result["sql_to_validate"]
    assert result["sql_self_correction"]["corrected"] is True


def test_sql_self_corrector_rejects_semantic_mismatch():
    sql = "SELECT COUNT(*) AS total_rows FROM dm.sales"
    nodes = _nodes(
        {
            "verdict": "reject",
            "issues": ["asked for sum by region, SQL returns only count"],
            "rationale": "wrong metric and missing dimension",
        }
    )

    result = nodes.sql_self_corrector(_state(sql))

    assert result["sql_to_validate"] is None
    assert "self-correction" in result["last_error"]
    assert result["sql_self_correction"]["verdict"] == "reject"
