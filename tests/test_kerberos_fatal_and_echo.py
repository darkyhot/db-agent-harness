"""Регрессии живого прогона:

1. Протухший Kerberos-тикет должен сразу показываться пользователю (fatal_error),
   а не уходить в 15-итерационный цикл исправлений с погребённым сообщением.
2. На новый вопрос оркестратор не должен переиспользовать прошлый ответ
   (summarize без работы в текущем ходе → run_analytics).
"""

from unittest.mock import MagicMock

from core.exceptions import KERBEROS_USER_MESSAGE
from graph.graph import _check_limits, _route_after_validator


# ---------------------------------------------------------------------------
# Маршрутизация при fatal_error
# ---------------------------------------------------------------------------
class TestFatalErrorRouting:
    def test_check_limits_short_circuits_to_summarizer(self):
        assert _check_limits({"fatal_error": KERBEROS_USER_MESSAGE}) == "summarizer"

    def test_check_limits_no_fatal_returns_none(self):
        assert _check_limits({"fatal_error": "", "graph_iterations": 0}) is None

    def test_route_after_validator_fatal_goes_to_summarizer(self):
        state = {
            "fatal_error": KERBEROS_USER_MESSAGE,
            "last_error": KERBEROS_USER_MESSAGE,
            "current_step": 0,
            "plan": [],
        }
        # Фатальная ошибка должна победить ветку last_error → error_diagnoser.
        assert _route_after_validator(state) == "summarizer"

    def test_route_after_validator_non_fatal_error_goes_to_diagnoser(self):
        state = {
            "fatal_error": "",
            "last_error": "some sql error",
            "current_step": 0,
            "plan": ["step1"],
        }
        assert _route_after_validator(state) == "error_diagnoser"


# ---------------------------------------------------------------------------
# Summarizer показывает fatal_error как основной ответ
# ---------------------------------------------------------------------------
def _make_nodes():
    from graph.nodes import GraphNodes

    llm = MagicMock()
    db = MagicMock()
    schema = MagicMock()
    memory = MagicMock()
    memory.get_memory_list.return_value = []
    memory.get_session_messages.return_value = []
    validator = MagicMock()
    return GraphNodes(llm, db, schema, memory, validator, [], debug_prompt=False)


class TestSummarizerFatalError:
    def test_fatal_error_returned_verbatim(self):
        nodes = _make_nodes()
        state = {
            "messages": [],
            "tool_calls": [],
            "correction_examples": [],
            "fatal_error": KERBEROS_USER_MESSAGE,
            "user_input": "посчитай отток",
        }
        result = nodes.summarizer(state)
        assert result["final_answer"] == KERBEROS_USER_MESSAGE
        # Не должно происходить обращения к LLM для доформулирования.
        nodes.llm.invoke_with_system.assert_not_called()


# ---------------------------------------------------------------------------
# Оркестратор: summarize без работы в текущем ходе → run_analytics
# ---------------------------------------------------------------------------
def _base_orch_state(user_input: str, orch_history=None):
    return {
        "user_input": user_input,
        "graph_iterations": 0,
        "orch_step_count": 0,
        "orch_history": list(orch_history or []),
        "orch_plan": [],
        "messages": [],
        "final_answer": None,
    }


class TestOrchestratorEchoGuardrail:
    def test_summarize_with_echo_answer_and_empty_history_degrades(self):
        nodes = _make_nodes()
        nodes._llm_json_with_retry = MagicMock(return_value={
            "step": "summarize",
            "answer": "Сумма оттока по дате и сегменту: <прошлая таблица>",
            "reason": "переиспользую прошлый результат",
        })
        result = nodes.orchestrator(_base_orch_state("а покажи теперь топ менеджеров"))
        assert result["orch_next_step"] == "run_analytics"
        # Эхо прошлого ответа НЕ должно утечь в final_answer.
        assert not result.get("final_answer")

    def test_summarize_with_real_work_in_history_is_kept(self):
        nodes = _make_nodes()
        nodes._llm_json_with_retry = MagicMock(return_value={
            "step": "summarize",
            "answer": "Готовый ответ по выполненной аналитике",
            "reason": "данные получены в этом ходе",
        })
        history = [{"step": "run_analytics", "reason": "посчитал", "ok": True}]
        result = nodes.orchestrator(_base_orch_state("посчитай отток", orch_history=history))
        assert result["orch_next_step"] == "summarize"
        assert result.get("final_answer") == "Готовый ответ по выполненной аналитике"

    def test_finish_with_answer_without_work_is_kept(self):
        """finish с answer (ответ «вне возможностей») не должен деградировать."""
        nodes = _make_nodes()
        nodes._llm_json_with_retry = MagicMock(return_value={
            "step": "finish",
            "answer": "Это вне моих возможностей — я отвечаю на вопросы по данным каталога.",
            "reason": "запрос вне возможностей",
        })
        result = nodes.orchestrator(_base_orch_state("отправь письмо коллеге"))
        assert result["orch_next_step"] == "finish"
        assert result.get("final_answer", "").startswith("Это вне моих возможностей")
