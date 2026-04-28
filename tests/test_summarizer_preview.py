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


def _state(user_input: str):
    return {
        "messages": [],
        "plan": [],
        "tool_calls": [],
        "query_spec": {},
        "table_structures": {},
        "tables_context": "",
        "user_input": user_input,
        "correction_examples": [],
    }


def test_summarizer_system_prompt_contains_is_empty_guidance():
    """Системный промпт должен содержать инструкции по обработке 0 строк."""
    nodes = _make_nodes("answer")
    prompt = nodes._get_summarizer_system_prompt()
    assert "0 строк" in prompt or "is_empty" in prompt
    # Добавление не должно раздувать промпт более чем на 30% от базовой длины.
    base_len = len(
        "Ты — аналитический агент для Greenplum. Формируешь финальный ответ пользователю.\n\n"
        "Правила ответа:\n"
        "- Отвечай на русском языке\n"
        "- SQL-алиасы только на английском\n"
        "- Табличные данные оформляй в markdown-таблицу\n"
        "- SQL-код оборачивай в ```sql блок\n"
        "- Не пересказывай шаги плана — только результат\n"
        "- Не повторяй вопрос пользователя\n"
        "- Если были предупреждения — упомяни кратко в конце\n"
        "- Если был выполнен SQL-запрос — покажи его в блоке ```sql и кратко объясни логику\n"
        "- Интерпретируй результат в бизнес-терминах, если это возможно\n"
        "- Если данные обрезаны — укажи это и покажи общее количество строк\n"
        "- Если результат большой — покажи топ-10 строк и общую статистику\n"
        "- КРИТИЧНО: в блоке ```sql РАЗРЕШЁН ТОЛЬКО тот запрос, который дословно "
        "находится в разделе «Результаты инструментов». Запрещено писать ЛЮБОЙ другой "
        "SQL — даже если считаешь, что результат не отвечает на вопрос. "
        "Если данных недостаточно — напиши ТОЛЬКО текст: "
        "«Данных недостаточно, требуется дополнительный запрос» — без SQL-блока.\n"
    )
    assert len(prompt) <= base_len * 1.3


def test_summarizer_is_empty_prompt_passed_to_llm():
    """При is_empty: true LLM получает системный промпт с инструкцией про 0 строк."""
    captured = {}

    class CapturingLLM:
        def invoke_with_system(self, system_prompt, user_prompt, temperature=None):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            return "нет данных за февраль 2026 по фильтру reason_code='actual_churn'"

    from graph.nodes import GraphNodes
    from unittest.mock import MagicMock

    llm = CapturingLLM()
    db = MagicMock()
    schema = MagicMock()
    memory = MagicMock()
    memory.get_memory_list.return_value = []
    memory.get_all_memory.return_value = {}
    memory.get_sessions_context.return_value = ""
    memory.get_session_messages.return_value = []
    validator = MagicMock()
    nodes = GraphNodes(llm, db, schema, memory, validator, [], debug_prompt=False)

    state = _state("покажи отток за февраль 2026")
    state["plan"] = ["Выполнить SQL"]
    state["tool_calls"] = [
        {
            "tool": "execute_query",
            "args": {"sql": "SELECT * FROM dm.fact_churn WHERE month='2026-02'"},
            "result": (
                '{"message": "Запрос выполнен. Результат пуст.", '
                '"rows_returned": 0, "is_empty": true, '
                '"is_truncated": false, "mode": "preview"}'
            ),
        }
    ]
    nodes.summarizer(state)

    assert "0 строк" in captured["system"] or "is_empty" in captured["system"]
    assert "февраль" in captured["user"] or "2026-02" in captured["user"]


def test_summarizer_appends_preview_markdown_from_execute_query():
    nodes = _make_nodes("Готово. Ниже результат запроса.")
    state = _state("Покажи клиентов")
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


def test_summarizer_blocks_answer_data_without_successful_execute_query():
    nodes = _make_nodes("```sql\nSELECT COUNT(*) FROM dm.clients\n```")
    state = _state("Сколько клиентов?")
    state["query_spec"] = {
        "task": "answer_data",
        "metrics": [{"operation": "count", "target": "client_id"}],
    }
    state["tool_calls"] = [
        {
            "tool": "execute_query",
            "args": {"sql": "SELECT COUNT(*) AS cnt FROM dm.clients"},
            "result": "awaiting_validation",
        }
    ]

    result = nodes.summarizer(state)

    assert "SQL не был выполнен" in result["final_answer"]
    assert "```sql" not in result["final_answer"]


def test_summarizer_prepends_scalar_summary_from_single_row_preview():
    nodes = _make_nodes("Запрос выполнен.")
    state = _state("Сколько всего есть ТБ и ГОСБ?")
    state["query_spec"] = {
        "task": "answer_data",
        "metrics": [
            {"operation": "count", "target": "tb_id"},
            {"operation": "count", "target": "old_gosb_id"},
        ],
    }
    state["tool_calls"] = [
        {
            "tool": "execute_query",
            "args": {"sql": "SELECT COUNT(DISTINCT tb_id) AS total_tb FROM dm.gosb_dim"},
            "result": (
                '{"message": "Preview выполнен.", '
                '"preview_markdown": "| total_tb | total_gosb |\\n| --- | --- |\\n| 5 | 11 |", '
                '"rows_returned": 1, "rows_saved": 1, "is_empty": false, '
                '"is_truncated": false, "saved_file": null, "mode": "preview"}'
            ),
        }
    ]

    result = nodes.summarizer(state)

    assert result["final_answer"].startswith("Итог: total_tb = 5, total_gosb = 11.")
