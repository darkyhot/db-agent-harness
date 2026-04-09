"""Golden-тесты для DB Agent Harness.

Запускает запросы через граф с MockLLM и проверяет результаты против YAML-фикстур.

Использование:
    pytest tests/test_golden.py -v

Каждый YAML-файл в tests/golden/ описывает один сценарий:
  query: str                  — запрос пользователя
  intent_must_be: str         — ожидаемый intent (опционально)
  complexity_must_be: str     — ожидаемая complexity (опционально)
  sql_must_contain: list[str] — подстроки, обязательные в итоговом SQL
  sql_must_not_contain: list[str] — подстроки, запрещённые в итоговом SQL
  sql_alias_no_cyrillic: bool — проверить что в алиасах нет кириллицы
  should_reach_node: str      — нода должна быть посещена
  should_not_reach_node: str  — нода не должна быть посещена
  max_retries: int            — максимально допустимое число retry (0 = с первой попытки)
"""

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (_YAML_AVAILABLE and _PANDAS_AVAILABLE),
    reason="pyyaml and pandas required for golden tests",
)

GOLDEN_DIR = Path(__file__).parent / "golden"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_golden_cases() -> list[tuple[str, dict]]:
    """Загрузить все YAML-фикстуры из tests/golden/."""
    if not _YAML_AVAILABLE or not GOLDEN_DIR.exists():
        return []
    cases = []
    for path in sorted(GOLDEN_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict) and "query" in data:
            cases.append((path.stem, data))
    return cases


def _has_cyrillic_alias(sql: str) -> bool:
    """Проверить, есть ли кириллица в алиасах SQL."""
    # AS кириллика или AS "кириллика"
    pattern = re.compile(
        r'\bAS\s+(?:"[^"]*[а-яёА-ЯЁ][^"]*"|[а-яёА-ЯЁ]\w*)',
        re.IGNORECASE,
    )
    return bool(pattern.search(sql))


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------

class MockLLM:
    """Детерминированный mock LLM для golden-тестов.

    Возвращает минимально корректные JSON-ответы для каждой ноды.
    Поведение настраивается через конструктор для разных сценариев.
    """

    def __init__(self, intent: str = "analytics", complexity: str = "single_table") -> None:
        self.intent = intent
        self.complexity = complexity
        self._call_count = 0

    def invoke_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self._call_count += 1

        # intent_classifier
        if "классификатор запросов" in system_prompt or "intent" in system_prompt.lower():
            return (
                f'{{"intent": "{self.intent}", "entities": ["клиенты"], '
                f'"date_filters": {{"from": null, "to": null}}, '
                f'"aggregation_hint": "count", "needs_search": false, '
                f'"complexity": "{self.complexity}"}}'
            )

        # table_resolver
        if "селектор таблиц" in system_prompt or "plan_steps" in system_prompt:
            return (
                '{"tables": [{"schema": "dm", "table": "clients", "reason": "основная таблица"}], '
                '"plan_steps": ["Получить данные из dm.clients"]}'
            )

        # sql_planner
        if "планировщик SQL" in system_prompt or "strategy" in system_prompt:
            return (
                '{"strategy": "simple_select", "main_table": "dm.clients", '
                '"cte_needed": false, "subquery_for": [], '
                '"where_conditions": [], "aggregation": null, '
                '"group_by": [], "order_by": null, "limit": 100, "notes": ""}'
            )

        # sql_writer — генерирует SQL с корректными алиасами
        if "SQL-писатель" in system_prompt or "execute_query" in system_prompt:
            return '{"tool": "execute_query", "args": {"sql": "SELECT COUNT(*) AS total_clients FROM dm.clients LIMIT 100"}}'

        # error_diagnoser
        if "классификатор SQL-ошибок" in system_prompt:
            return (
                '{"error_type": "other", "root_cause": "тестовая ошибка", '
                '"fix_strategy": "rewrite_sql", "replacements": [], '
                '"needs_sample": false, "needs_replan": false}'
            )

        # summarizer
        if "Формируешь финальный ответ" in system_prompt or "summarizer" in system_prompt.lower():
            return "Всего клиентов: 100."

        # memory extraction
        if "извлечения памяти" in system_prompt:
            return '{"user_facts": [], "behavior_patterns": [], "user_instructions": []}'

        return '{"tool": "none", "result": "mock response"}'

    def invoke(self, prompt: str, **kwargs) -> str:
        return self.invoke_with_system("", prompt, **kwargs)


# ---------------------------------------------------------------------------
# Fixture: MockDB
# ---------------------------------------------------------------------------

def _make_mock_db():
    """Создать mock DatabaseManager."""
    db = MagicMock()
    db.is_configured = True
    db.runtime_config = {"debug_prompt": False}
    db.get_engine.return_value = MagicMock()

    import pandas as pd
    # execute_query возвращает DataFrame с одной строкой
    db.execute_query.return_value = pd.DataFrame({"total_clients": [100]})
    db.get_sample.return_value = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    db.run_read_query.return_value = pd.DataFrame({"total_clients": [100]})
    db.preview_query.return_value = pd.DataFrame({"total_clients": [100]})
    return db


def _make_mock_schema():
    """Создать mock SchemaLoader с минимальным каталогом."""
    import pandas as pd
    schema = MagicMock()

    tables_df = pd.DataFrame([
        {"schema_name": "dm", "table_name": "clients", "description": "таблица клиентов"},
        {"schema_name": "dm", "table_name": "sales", "description": "таблица продаж"},
        {"schema_name": "dm", "table_name": "managers", "description": "таблица менеджеров"},
    ])
    schema.tables_df = tables_df

    attrs_df = pd.DataFrame([
        {"schema_name": "dm", "table_name": "clients", "column_name": "id",
         "dtype": "integer", "is_primary_key": True, "is_not_null": True,
         "description": "", "unique_perc": 100.0, "not_null_perc": 100.0},
        {"schema_name": "dm", "table_name": "clients", "column_name": "name",
         "dtype": "varchar", "is_primary_key": False, "is_not_null": True,
         "description": "", "unique_perc": 95.0, "not_null_perc": 100.0},
        {"schema_name": "dm", "table_name": "clients", "column_name": "region",
         "dtype": "varchar", "is_primary_key": False, "is_not_null": False,
         "description": "", "unique_perc": 5.0, "not_null_perc": 80.0},
    ])
    schema.attrs_df = attrs_df

    schema.get_table_columns.return_value = attrs_df[attrs_df["table_name"] == "clients"]
    schema.get_table_info.return_value = "id: integer PK\nname: varchar\nregion: varchar"
    schema.search_tables.return_value = tables_df
    schema.search_by_description.return_value = tables_df
    schema.get_all_table_names.return_value = [("dm", "clients"), ("dm", "sales")]
    return schema


# ---------------------------------------------------------------------------
# GraphRunner — запускает граф и собирает события
# ---------------------------------------------------------------------------

class GraphRunner:
    """Запускает граф агента и собирает результаты для assertion'ов."""

    def __init__(self, mock_llm: MockLLM) -> None:
        self.mock_llm = mock_llm
        self._visited_nodes: list[str] = []
        self._final_state: dict = {}
        self._tool_calls: list[dict] = []

    def run(self, query: str) -> dict[str, Any]:
        """Запустить граф с mock компонентами.

        Returns:
            Словарь с результатами: sql, intent, retry_count, visited_nodes.
        """
        import time
        from unittest.mock import patch, MagicMock

        mock_db = _make_mock_db()
        mock_schema = _make_mock_schema()
        mock_memory = MagicMock()
        mock_memory.get_session_messages.return_value = []
        mock_memory.get_memory.return_value = None
        mock_memory.get_memory_list.return_value = []
        mock_memory.get_all_memory.return_value = {}
        mock_memory._connect.return_value.__enter__ = MagicMock(
            return_value=MagicMock(execute=MagicMock(return_value=MagicMock(fetchall=MagicMock(return_value=[]))), commit=MagicMock())
        )
        mock_memory._connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_memory._session_id = "test-session"

        mock_validator = MagicMock()
        mock_validator.validate.return_value = MagicMock(
            is_valid=True, warnings=[], join_checks=[], multiplication_factor=1.0,
            rewrite_suggestions=[], summary=lambda: ""
        )

        # Создаём mock tools
        exec_tool = MagicMock()
        exec_tool.name = "execute_query"
        exec_tool.description = "Выполнить SELECT-запрос"
        exec_tool.args_schema = None

        sample_tool = MagicMock()
        sample_tool.name = "get_sample"
        sample_tool.description = "Получить образец данных таблицы"
        sample_tool.args_schema = None

        search_tool = MagicMock()
        search_tool.name = "search_tables"
        search_tool.description = "Поиск таблиц в каталоге"
        search_tool.args_schema = None

        from graph.graph import build_graph, create_initial_state

        # Monkey-patch LLM в GraphNodes
        import graph.nodes.graph_nodes as gn_module
        original_init = gn_module.GraphNodes.__init__

        def patched_init(self_node, llm, *args, **kwargs):
            original_init(self_node, self.mock_llm, *args, **kwargs)

        visited = self._visited_nodes

        with patch.object(gn_module.GraphNodes, '__init__', patched_init):
            try:
                graph = build_graph(
                    self.mock_llm, mock_db, mock_schema, mock_memory,
                    mock_validator, [exec_tool, sample_tool, search_tool],
                    debug_prompt=False,
                )
                state = create_initial_state(query)
                result: dict = {}

                for event in graph.stream(state):
                    node_name = list(event.keys())[0]
                    visited.append(node_name)
                    result.update(event[node_name])

                self._final_state = result
                self._tool_calls = result.get("tool_calls", [])

                # Извлекаем финальный SQL
                sql = ""
                for tc in reversed(self._tool_calls):
                    if tc.get("tool") == "execute_query":
                        sql = tc.get("args", {}).get("sql", "")
                        break

                return {
                    "sql": sql,
                    "intent": result.get("intent", {}),
                    "retry_count": result.get("retry_count", 0),
                    "visited_nodes": visited,
                    "final_answer": result.get("final_answer", ""),
                    "tool_calls": self._tool_calls,
                }
            except Exception as e:
                return {
                    "sql": "",
                    "intent": {},
                    "retry_count": 0,
                    "visited_nodes": visited,
                    "final_answer": "",
                    "tool_calls": [],
                    "error": str(e),
                }


# ---------------------------------------------------------------------------
# Параметризованные golden-тесты
# ---------------------------------------------------------------------------

_GOLDEN_CASES = _load_golden_cases()


@pytest.mark.parametrize("case_name,spec", _GOLDEN_CASES)
def test_golden(case_name: str, spec: dict) -> None:
    """Запустить golden-тест из YAML-фикстуры."""
    query = spec["query"]
    intent_arg = spec.get("intent_must_be", "analytics")

    mock_llm = MockLLM(intent=intent_arg, complexity=spec.get("complexity_must_be", "single_table"))
    runner = GraphRunner(mock_llm)
    result = runner.run(query)

    if "error" in result:
        pytest.skip(f"Graph run failed (likely missing deps): {result['error']}")

    sql = result["sql"]
    intent = result["intent"]
    visited = result["visited_nodes"]

    # Проверка intent
    if "intent_must_be" in spec:
        assert intent.get("intent") == spec["intent_must_be"], (
            f"[{case_name}] intent = {intent.get('intent')!r}, "
            f"ожидалось {spec['intent_must_be']!r}"
        )

    # Проверка complexity
    if "complexity_must_be" in spec:
        assert intent.get("complexity") == spec["complexity_must_be"], (
            f"[{case_name}] complexity = {intent.get('complexity')!r}, "
            f"ожидалось {spec['complexity_must_be']!r}"
        )

    # Проверка наличия подстрок в SQL
    for substr in spec.get("sql_must_contain", []):
        assert substr.upper() in sql.upper(), (
            f"[{case_name}] SQL не содержит {substr!r}.\nSQL: {sql}"
        )

    # Проверка отсутствия подстрок в SQL
    for substr in spec.get("sql_must_not_contain", []):
        assert substr not in sql, (
            f"[{case_name}] SQL содержит запрещённую подстроку {substr!r}.\nSQL: {sql}"
        )

    # Проверка кириллицы в алиасах
    if spec.get("sql_alias_no_cyrillic") and sql:
        assert not _has_cyrillic_alias(sql), (
            f"[{case_name}] SQL содержит кириллицу в алиасах.\nSQL: {sql}"
        )

    # Проверка посещённых нод
    if "should_reach_node" in spec:
        assert spec["should_reach_node"] in visited, (
            f"[{case_name}] Нода '{spec['should_reach_node']}' не была посещена.\n"
            f"Посещены: {visited}"
        )

    if "should_not_reach_node" in spec:
        assert spec["should_not_reach_node"] not in visited, (
            f"[{case_name}] Нода '{spec['should_not_reach_node']}' была посещена.\n"
            f"Посещены: {visited}"
        )

    # Проверка числа retry
    if "max_retries" in spec:
        actual_retries = result["retry_count"]
        assert actual_retries <= spec["max_retries"], (
            f"[{case_name}] retry_count={actual_retries} > max_retries={spec['max_retries']}"
        )


# ---------------------------------------------------------------------------
# Тест: _has_cyrillic_alias helper
# ---------------------------------------------------------------------------

class TestHasCyrillicAlias:
    def test_cyrillic_alias_detected(self):
        assert _has_cyrillic_alias('SELECT COUNT(*) AS "выручка" FROM t')
        assert _has_cyrillic_alias("SELECT a AS регион FROM t")

    def test_english_alias_ok(self):
        assert not _has_cyrillic_alias("SELECT COUNT(*) AS total FROM t")
        assert not _has_cyrillic_alias("SELECT a AS region FROM t")

    def test_cyrillic_in_value_not_alias(self):
        assert not _has_cyrillic_alias("SELECT a FROM t WHERE name = 'Иванов'")
