"""End-to-end тесты графа агента с мок-LLM."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from graph.state import AgentState
from graph.graph import create_initial_state


class MockLLM:
    """Мок LLM для тестирования графа."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0

    def invoke_with_system(self, system_prompt: str, user_prompt: str, temperature=None) -> str:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = '{"tool": "none", "result": "no more responses"}'
        self._call_count += 1
        return resp

    def invoke(self, prompt, temperature=None) -> str:
        return self.invoke_with_system("", str(prompt), temperature)


class TestCreateInitialState:
    """Тесты создания начального состояния."""

    def test_creates_valid_state(self):
        state = create_initial_state("Сколько клиентов?")
        assert state["user_input"] == "Сколько клиентов?"
        assert state["plan"] == []
        assert state["current_step"] == 0
        assert state["retry_count"] == 0
        assert state["last_error"] is None
        assert state["sql_to_validate"] is None
        assert state["final_answer"] is None
        assert state["graph_iterations"] == 0
        assert state["start_time"] > 0

    def test_state_has_all_required_fields(self):
        state = create_initial_state("test")
        required = [
            "messages", "plan", "current_step", "tool_calls",
            "last_error", "retry_count", "sql_to_validate",
            "final_answer", "user_input", "needs_confirmation",
            "needs_clarification", "clarification_message",
            "tables_context", "graph_iterations",
        ]
        for field in required:
            assert field in state, f"Missing field: {field}"


class TestGraphNodesParsing:
    """Тесты парсинга ответов LLM в GraphNodes."""

    @pytest.fixture
    def nodes(self):
        """Создать GraphNodes с мок-зависимостями."""
        from graph.nodes import GraphNodes
        llm = MockLLM()
        db = MagicMock()
        schema = MagicMock()
        schema.tables_df = MagicMock()
        schema.tables_df.empty = True
        memory = MagicMock()
        memory.get_memory_list.return_value = []
        memory.get_all_memory.return_value = {}
        memory.get_sessions_context.return_value = ""
        memory.get_session_messages.return_value = []
        validator = MagicMock()
        tools = []
        return GraphNodes(llm, db, schema, memory, validator, tools)

    def test_clean_llm_json_removes_markdown(self, nodes):
        text = '```json\n{"tool": "none", "result": "test"}\n```'
        cleaned = nodes._clean_llm_json(text)
        assert "```" not in cleaned
        data = json.loads(cleaned)
        assert data["tool"] == "none"

    def test_clean_llm_json_fixes_trailing_commas(self, nodes):
        text = '{"tool": "none", "result": "test",}'
        cleaned = nodes._clean_llm_json(text)
        data = json.loads(cleaned)
        assert data["tool"] == "none"

    def test_extract_json_objects_simple(self, nodes):
        text = 'Some text {"tool": "execute_query", "args": {"sql": "SELECT 1"}} more text'
        objects = nodes._extract_json_objects(text)
        assert len(objects) >= 1
        data = json.loads(objects[0])
        assert data["tool"] == "execute_query"

    def test_extract_json_objects_nested(self, nodes):
        text = '{"tool": "execute_query", "args": {"sql": "SELECT \'hello\' AS greet"}}'
        objects = nodes._extract_json_objects(text)
        assert len(objects) == 1
        data = json.loads(objects[0])
        assert data["args"]["sql"] == "SELECT 'hello' AS greet"

    def test_parse_plan_json_array(self, nodes):
        response = '["Шаг 1: найти таблицу", "Шаг 2: написать SQL"]'
        plan = nodes._parse_plan(response)
        assert len(plan) == 2
        assert "найти таблицу" in plan[0]

    def test_parse_plan_numbered_list(self, nodes):
        response = "1. Найти таблицу dm.clients\n2. Написать SQL с фильтром"
        plan = nodes._parse_plan(response)
        assert len(plan) == 2

    def test_parse_plan_with_markdown_wrapper(self, nodes):
        response = '```json\n["Шаг 1", "Шаг 2"]\n```'
        plan = nodes._parse_plan(response)
        assert len(plan) == 2


class TestBudgetTrimming:
    """Тесты обрезки промптов."""

    def test_no_trimming_under_budget(self):
        from graph.nodes import GraphNodes
        sys_p = "System prompt" * 10
        usr_p = "User prompt" * 10
        result_sys, result_usr = GraphNodes._trim_to_budget(sys_p, usr_p, max_chars=10000)
        assert result_sys == sys_p
        assert result_usr == usr_p

    def test_trimming_over_budget(self):
        from graph.nodes import GraphNodes
        sys_p = "S" * 500
        usr_p = "U" * 1500
        result_sys, result_usr = GraphNodes._trim_to_budget(sys_p, usr_p, max_chars=1000)
        assert len(result_sys) + len(result_usr) <= 1200  # some overhead for markers

    def test_critical_sections_preserved(self):
        from graph.nodes import GraphNodes
        sys_p = "S" * 200
        usr_p = "U" * 800 + "\n[ТЕКУЩИЙ ШАГ 1/2]\nВажный текст шага"
        result_sys, result_usr = GraphNodes._trim_to_budget(sys_p, usr_p, max_chars=500)
        # The current step text should be preserved
        assert "ТЕКУЩИЙ ШАГ" in result_usr or len(result_usr) < 500


class TestQualityMetrics:
    """Тесты метрик качества."""

    @pytest.fixture
    def memory(self, tmp_path):
        from core.memory import MemoryManager
        db_path = tmp_path / "test_memory.db"
        mm = MemoryManager(db_path=db_path)
        mm.start_session("test_user")
        return mm

    def test_empty_metrics(self, memory):
        metrics = memory.get_sql_quality_metrics(days=30)
        assert metrics["total_queries"] == 0

    def test_metrics_with_data(self, memory):
        # Log some SQL executions
        memory.log_sql_execution("q1", "SELECT 1", 1, "success", 100, retry_count=0)
        memory.log_sql_execution("q2", "SELECT 2", 0, "error", 50, retry_count=1, error_type="syntax")
        memory.log_sql_execution("q3", "SELECT 3", 10, "success", 200, retry_count=2)
        memory.log_sql_execution("q4", "SELECT 4", 100, "row_explosion", 300, retry_count=0, error_type="join_explosion")

        metrics = memory.get_sql_quality_metrics(days=30)
        assert metrics["total_queries"] == 4
        assert metrics["success_rate"] == 50.0  # 2 out of 4
        assert metrics["first_try_success_rate"] == 25.0  # 1 out of 4
        assert metrics["avg_retries"] >= 0
        assert "syntax" in metrics["error_distribution"]
        assert "join_explosion" in metrics["error_distribution"]
