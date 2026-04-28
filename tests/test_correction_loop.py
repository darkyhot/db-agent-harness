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


class _NoCallLLM:
    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        raise AssertionError("deterministic ambiguous-column path should not call LLM")


def _nodes():
    memory = MagicMock()
    memory.get_memory_list.return_value = []
    return GraphNodes(
        _NoCallLLM(),
        MagicMock(),
        MagicMock(),
        memory,
        MagicMock(),
        [],
        debug_prompt=False,
    )


def _state(sql: str, error: str):
    return {
        "messages": [],
        "plan": ["execute"],
        "current_step": 0,
        "graph_iterations": 0,
        "retry_count": 0,
        "last_error": error,
        "tool_calls": [
            {"tool": "execute_query", "args": {"sql": sql}, "result": "awaiting_validation"}
        ],
        "sql_blueprint": {
            "aggregation": {"function": "SUM", "column": "outflow_qty", "alias": "sum_outflow_qty"},
            "group_by": ["report_dt", "new_gosb_name"],
            "order_by": "sum_outflow_qty DESC",
        },
        "selected_columns": {
            "dm.fact_outflow": {
                "select": ["report_dt", "outflow_qty"],
                "aggregate": ["outflow_qty"],
                "group_by": ["report_dt"],
            },
            "dm.dim_gosb": {
                "select": ["new_gosb_name"],
                "group_by": ["new_gosb_name"],
            },
        },
        "correction_examples": [],
        "replan_count": 0,
        "start_time": 0,
    }


def test_error_diagnoser_removes_unneeded_ambiguous_order_by_column():
    sql = (
        "SELECT f.report_dt, d.new_gosb_name, SUM(f.outflow_qty) AS sum_outflow_qty "
        "FROM dm.fact_outflow f JOIN dm.dim_gosb d ON true "
        "GROUP BY f.report_dt, d.new_gosb_name "
        "ORDER BY sum_outflow_qty DESC, inserted_dttm DESC"
    )
    error = (
        "SQL не выполнен: ошибка статической проверки.\n"
        "[sql_static_checker] Статические ошибки:\n"
        "  ✗ Найдены неоднозначные unqualified-колонки: inserted_dttm "
        "(dm.fact_outflow, dm.dim_gosb). Укажи алиас таблицы для каждой такой колонки."
    )

    result = _nodes().error_diagnoser(_state(sql, error))

    assert result["error_diagnosis"]["error_type"] == "ambiguous_column"
    assert "inserted_dttm" not in result["sql_to_validate"].lower()
    assert "sum_outflow_qty DESC" in result["sql_to_validate"]


def test_error_diagnoser_stops_repeated_same_sql_error():
    sql = "SELECT inserted_dttm FROM dm.fact_outflow f JOIN dm.dim_gosb d ON true"
    error = (
        "Найдены неоднозначные unqualified-колонки: inserted_dttm "
        "(dm.fact_outflow, dm.dim_gosb). Укажи алиас таблицы для каждой такой колонки."
    )
    state = _state(sql, error)
    first = _nodes().error_diagnoser(state)
    state.update(first)
    state["last_error"] = error
    state["tool_calls"] = [{"tool": "execute_query", "args": {"sql": sql}, "result": "awaiting_validation"}]

    second = _nodes().error_diagnoser(state)

    assert second["final_answer"].startswith("Не удалось исправить SQL")
