"""Регрессии summarizer для показа preview-таблицы."""

from unittest.mock import MagicMock


class StubLLM:
    def __init__(self, answer: str):
        self.answer = answer

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        return self.answer


def _make_nodes(answer: str):
    from graph.nodes import GraphNodes

    llm = StubLLM(answer)
    db = MagicMock()
    schema = MagicMock()
    memory = MagicMock()
    memory.get_memory_list.return_value = []
    memory.get_all_memory.return_value = {}
    memory.get_sessions_context.return_value = ""
    memory.get_session_messages.return_value = []
    validator = MagicMock()
    return GraphNodes(llm, db, schema, memory, validator, [], debug_prompt=False)


def test_summarizer_appends_preview_markdown_from_execute_query():
    from graph.graph import create_initial_state

    nodes = _make_nodes("Готово. Ниже результат запроса.")
    state = create_initial_state("Покажи клиентов")
    state["plan"] = ["Выполнить SQL"]
    state["tool_calls"] = [
        {
            "tool": "execute_query",
            "args": {"sql": "SELECT id, name FROM dm.clients"},
            "result": (
                '{"message": "Preview выполнен. Показано 2 строки.", '
                '"preview_markdown": "| id | name |\\n| --- | --- |\\n| 1 | Alice |\\n| 2 | Bob |", '
                '"rows_returned": 2, "rows_saved": 2, "is_empty": false, '
                '"is_truncated": false, "saved_file": "last_query_result.csv", "mode": "preview"}'
            ),
        }
    ]

    result = nodes.summarizer(state)
    answer = result["final_answer"]

    assert "Предварительный результат:" in answer
    assert "| id | name |" in answer
    assert "| 1 | Alice |" in answer
