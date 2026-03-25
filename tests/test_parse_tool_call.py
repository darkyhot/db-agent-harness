"""Тесты парсинга JSON tool-call из ответов LLM.

Тестирует _extract_json_objects и _parse_tool_call без полного импорта graph.nodes,
чтобы избежать зависимости от langchain_gigachat в тестовом окружении.
"""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


# Мокаем langchain_gigachat до импорта graph.nodes
def _ensure_mock_modules():
    """Подставить моки для модулей, недоступных в тестовом окружении."""
    for mod_name in (
        "langchain_gigachat",
        "langchain_gigachat.chat_models",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_ensure_mock_modules()

from graph.nodes import GraphNodes  # noqa: E402


class TestExtractJsonObjects:
    """Тесты статического метода _extract_json_objects."""

    def test_simple_object(self):
        text = '{"tool": "search_tables", "args": {"query": "зарплата"}}'
        result = GraphNodes._extract_json_objects(text)
        assert len(result) == 1
        assert '"tool"' in result[0]

    def test_nested_object(self):
        text = '{"tool": "execute_query", "args": {"sql": "SELECT * FROM hr.salary"}}'
        result = GraphNodes._extract_json_objects(text)
        assert len(result) == 1
        parsed = json.loads(result[0])
        assert parsed["tool"] == "execute_query"
        assert "hr.salary" in parsed["args"]["sql"]

    def test_json_with_surrounding_text(self):
        text = 'Вот мой ответ: {"tool": "get_sample", "args": {"schema": "hr", "table": "emp"}} — готово'
        result = GraphNodes._extract_json_objects(text)
        assert len(result) == 1
        assert "get_sample" in result[0]

    def test_braces_inside_strings(self):
        text = '{"tool": "execute_query", "args": {"sql": "SELECT \'{json}\' FROM t"}}'
        result = GraphNodes._extract_json_objects(text)
        assert len(result) == 1
        assert "execute_query" in result[0]

    def test_escaped_quotes_in_strings(self):
        text = r'{"tool": "create_file", "args": {"content": "He said \"hello\""}}'
        result = GraphNodes._extract_json_objects(text)
        assert len(result) >= 1

    def test_no_json(self):
        text = "Просто текстовый ответ без JSON"
        result = GraphNodes._extract_json_objects(text)
        assert result == []

    def test_multiple_objects(self):
        text = '{"tool": "a"} some text {"tool": "b"}'
        result = GraphNodes._extract_json_objects(text)
        assert len(result) == 2

    def test_deeply_nested(self):
        text = '{"tool": "x", "args": {"a": {"b": {"c": 1}}}}'
        result = GraphNodes._extract_json_objects(text)
        assert len(result) == 1
        parsed = json.loads(result[0])
        assert parsed["args"]["a"]["b"]["c"] == 1


class TestParseToolCall:
    """Тесты метода _parse_tool_call."""

    @pytest.fixture
    def nodes(self):
        """Создать GraphNodes с минимальными моками."""
        return GraphNodes(
            llm=MagicMock(),
            db_manager=None,
            schema_loader=None,
            memory=None,
            sql_validator=None,
            tools=[],
        )

    def test_simple_tool_call(self, nodes):
        response = '{"tool": "search_tables", "args": {"query": "test"}}'
        result = nodes._parse_tool_call(response)
        assert result["tool"] == "search_tables"
        assert result["args"]["query"] == "test"

    def test_nested_args(self, nodes):
        response = '{"tool": "execute_query", "args": {"sql": "SELECT * FROM s.t WHERE id = 1"}}'
        result = nodes._parse_tool_call(response)
        assert result["tool"] == "execute_query"
        assert "SELECT" in result["args"]["sql"]

    def test_no_tool_in_json(self, nodes):
        response = '{"result": "просто ответ"}'
        result = nodes._parse_tool_call(response)
        assert result["tool"] == "none"

    def test_plain_text(self, nodes):
        response = "Я не знаю как ответить на этот вопрос."
        result = nodes._parse_tool_call(response)
        assert result["tool"] == "none"
        assert result["result"] == response

    def test_json_with_preamble(self, nodes):
        response = 'Для этого вызываю: {"tool": "get_row_count", "args": {"schema": "hr", "table": "emp"}}'
        result = nodes._parse_tool_call(response)
        assert result["tool"] == "get_row_count"
        assert result["args"]["schema"] == "hr"
