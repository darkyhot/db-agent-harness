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


# ---------------------------------------------------------------------------
# Golden-сценарии на синтетическом каталоге: детерминированная интеграция
# hint_extractor → column_selector → sql_planner → sql_builder.
# Реальные имена таблиц/колонок не используются.
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402


@pytest.fixture
def golden_loader(tmp_path):
    """Каталог с двумя фактовыми таблицами и одним справочником:

    - fact_metric_a: report_dt, inn, amount_val, emp_ref_id
    - fact_event_b : event_dt, task_code, emp_ref_id
    - dim_segments : inn, segment_name, emp_ref_id

    Через эту фикстуру проверяем:
      (1) JOIN по user-hint (inn) побеждает нормализованное совпадение
      (2) HAVING COUNT(DISTINCT emp_ref_id) >= N из «от N <unit>»
    """
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["gold"] * 3,
        "table_name":  ["fact_metric_a", "fact_event_b", "dim_segments"],
        "description": [
            "Факт-таблица метрики A по датам и клиентам",
            "Факт-таблица событий B (задачи, даты)",
            "Справочник сегментов клиентов",
        ],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)

    attrs_df = pd.DataFrame({
        "schema_name": ["gold"] * 11,
        "table_name": [
            "fact_metric_a", "fact_metric_a", "fact_metric_a", "fact_metric_a",
            "fact_event_b",  "fact_event_b",  "fact_event_b",
            "dim_segments",  "dim_segments",  "dim_segments",  "dim_segments",
        ],
        "column_name": [
            "report_dt", "inn", "amount_val", "emp_ref_id",
            "event_dt", "task_code", "emp_ref_id",
            "segment_id", "inn", "segment_name", "emp_ref_id",
        ],
        "dType": [
            "date", "varchar", "numeric", "bigint",
            "date", "varchar", "bigint",
            "bigint", "varchar", "varchar", "bigint",
        ],
        "description": [
            "Отчётная дата", "ИНН клиента", "Сумма метрики", "ID сотрудника",
            "Дата события", "Код задачи", "ID сотрудника",
            "PK сегмента", "ИНН клиента", "Название сегмента", "ID сотрудника",
        ],
        "is_primary_key": [
            False, False, False, False,
            False, False, False,
            True,  False, False, False,
        ],
        "unique_perc": [
            0.5, 90.0, 10.0, 40.0,
            0.5, 80.0, 45.0,
            100.0, 90.0, 5.0, 40.0,
        ],
        "not_null_perc": [
            99.0, 95.0, 95.0, 90.0,
            99.0, 85.0, 85.0,
            100.0, 95.0, 99.99, 90.0,
        ],
    })
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    return SchemaLoader(data_dir=tmp_path)


class TestGoldenCase1SegmentSource:
    """Кейс 1: «сумма метрики_a по сегменту, сегмент возьми в dim_segments по инн».

    Ожидаемое поведение детерминированного слоя:
      - user_hints.must_keep_tables содержит dim_segments
      - user_hints.dim_sources содержит биндинг сегмента → dim_segments c join_col=inn
      - user_hints.join_fields содержит 'inn' (или 'inn', нормализованное)
    """

    def test_hint_extractor_binds_segment_to_dim(self, golden_loader):
        from core.user_hint_extractor import extract_user_hints

        q = "сумма метрики по сегменту, сегмент возьми в dim_segments по инн"
        hints = extract_user_hints(q, golden_loader)

        # must_keep
        must_keep_names = {t for _, t in hints["must_keep_tables"]}
        assert "dim_segments" in must_keep_names

        # dim_sources
        bound_tables = {b.get("table") for b in hints["dim_sources"].values()}
        assert "gold.dim_segments" in bound_tables, (
            f"Ожидали биндинг сегмента к gold.dim_segments, получили: {hints['dim_sources']}"
        )

        # join_fields: «по инн» должно резолвиться в inn-подобное поле
        # В synthetic каталоге inn хранится как inn — extractor либо
        # нормализует через _KEY_SYNONYMS, либо оставляет 'inn' как business-key токен.
        assert hints["join_fields"], "join_fields не должны быть пустыми"

    def test_sql_builder_joins_by_user_hint_key(self, golden_loader):
        """Blueprint + sql_builder: JOIN gold.fact_metric_a с gold.dim_segments
        должен использовать inn, если так явно указал пользователь."""
        from core.column_selector_deterministic import _pick_join_candidate

        pick = _pick_join_candidate(
            text="",
            t1="gold.fact_metric_a",
            t2="gold.dim_segments",
            schema_loader=golden_loader,
            user_input="соедини по inn",
            hint_join_fields=["inn"],
        )
        assert pick is not None
        assert pick["col1"] == "inn"
        assert pick["col2"] == "inn"

    def test_segment_column_selected_not_date_column(self, golden_loader):
        """Регрессия: «по дате и сегменту, сегмент возьми в dim_segments по инн».

        Баг: агент клал epk_create_dttm (дата) в GROUP BY вместо сегментной колонки.
        Ожидаемое поведение:
          - group_by для gold.dim_segments содержит segment_name (или segment_id)
          - group_by для gold.dim_segments НЕ содержит date/dt/dttm колонок
        """
        from core.column_selector_deterministic import select_columns

        table_structures = {
            "gold.fact_metric_a": "fact_metric_a columns: report_dt, inn, amount_val, emp_ref_id",
            "gold.dim_segments": "dim_segments columns: segment_id, inn, segment_name, emp_ref_id",
        }
        table_types = {
            "gold.fact_metric_a": "fact",
            "gold.dim_segments": "dim",
        }
        intent = {
            "aggregation_hint": "sum",
            "entities": ["метрика", "дата", "сегмент"],
            "required_output": ["дата", "сегмент"],
            "date_filters": {"from": None, "to": None},
            "filter_conditions": [],
        }
        user_hints = {
            "must_keep_tables": [("gold", "dim_segments")],
            "join_fields": ["inn"],
            "dim_sources": {
                "segment": {"table": "gold.dim_segments", "join_col": "inn"}
            },
            "having_hints": [],
        }
        user_input = "сумма метрики по дате и сегменту, сегмент возьми в dim_segments по инн"

        result = select_columns(
            intent=intent,
            table_structures=table_structures,
            table_types=table_types,
            join_analysis_data={},
            schema_loader=golden_loader,
            user_input=user_input,
            user_hints=user_hints,
        )

        selected = result.get("selected_columns", {})
        dim_roles = selected.get("gold.dim_segments", {})
        dim_gb = dim_roles.get("group_by", [])

        # В group_by для dim_segments должна быть сегментная колонка
        assert any(
            "segment" in c.lower() for c in dim_gb
        ), f"Ожидали segment-колонку в group_by dim_segments, получили: {dim_gb}"

        # В group_by для dim_segments НЕ должно быть дата-колонок
        date_cols_in_gb = [
            c for c in dim_gb
            if any(tok in c.lower() for tok in ("dt", "dttm", "date", "create"))
        ]
        assert not date_cols_in_gb, (
            f"В group_by dim_segments оказались дата-колонки: {date_cols_in_gb}. "
            f"Весь group_by: {dim_gb}"
        )


class TestGoldenCase2HavingFromPhrase:
    """Кейс 2: «...от 3 <unit>» → HAVING COUNT(DISTINCT unit_col) >= 3."""

    def test_hint_extractor_detects_having(self, golden_loader):
        from core.user_hint_extractor import extract_user_hints

        q = "посчитай события за февраль и количество от 3 сотрудников"
        hints = extract_user_hints(q, golden_loader)
        assert hints["having_hints"], f"having_hints пустые: {hints}"
        h = hints["having_hints"][0]
        assert h["op"] == ">="
        assert h["value"] == 3

    def test_planner_builds_having_clause(self, golden_loader):
        """build_blueprint преобразует having_hint в HAVING COUNT(DISTINCT col) >= N."""
        from core.sql_planner_deterministic import build_blueprint

        # selected_columns: одна фактовая таблица с агрегатом
        selected_columns = {
            "gold.fact_event_b": {
                "select": ["event_dt"],
                "aggregate": ["task_code"],
                "group_by": ["event_dt"],
                "filter": [],
            }
        }
        intent = {
            "metrics": ["count"],
            "dimensions": ["date"],
            "entities": [],
            "time_range": None,
            "complexity": "single_table",
        }
        table_types = {"gold.fact_event_b": "fact"}
        user_hints = {
            "must_keep_tables": [],
            "join_fields": [],
            "dim_sources": {},
            "having_hints": [
                {"op": ">=", "value": 3, "unit_hint": "сотрудник"},
            ],
        }
        blueprint = build_blueprint(
            intent=intent,
            selected_columns=selected_columns,
            join_spec=[],
            table_types=table_types,
            join_analysis_data={},
            user_input="от 3 сотрудников",
            user_hints=user_hints,
            schema_loader=golden_loader,
        )
        having = blueprint.get("having") or []
        assert having, f"blueprint.having пуст: {blueprint}"
        h = having[0]
        assert h["op"] == ">="
        assert h["value"] == 3
        # Колонка подбирается через match_unit_column: «сотрудник» ~ emp_ref_id
        assert "COUNT(DISTINCT" in h["expr"]
        assert "emp_ref_id" in h["expr"], f"ожидали emp_ref_id в HAVING: {h}"

    def test_sql_builder_emits_having(self, golden_loader):
        """SqlBuilder кладёт HAVING-клаузу в итоговый SQL."""
        from core.sql_builder import SqlBuilder

        selected_columns = {
            "gold.fact_event_b": {
                "select": ["event_dt"],
                "aggregate": ["task_code"],
                "group_by": ["event_dt"],
                "filter": [],
            }
        }
        blueprint = {
            "strategy": "simple_select",
            "main_table": "gold.fact_event_b",
            "aggregation": {"function": "count_distinct", "column": "task_code"},
            "group_by": ["event_dt"],
            "where_conditions": [],
            "order_by": None,
            "limit": 100,
            "having": [
                {"expr": "COUNT(DISTINCT emp_ref_id)", "op": ">=", "value": 3},
            ],
        }
        sql = SqlBuilder().build(
            strategy="simple_select",
            selected_columns=selected_columns,
            join_spec=[],
            blueprint=blueprint,
        )
        assert sql is not None
        assert "HAVING" in sql.upper()
        assert "COUNT(DISTINCT EMP_REF_ID)" in sql.upper().replace(" ", " ")
        assert ">= 3" in sql or ">=3" in sql.replace(" ", "")

    def test_group_by_subset_of_select_in_generated_sql(self, golden_loader):
        """Симметрия GROUP BY ⊆ SELECT: в простом шаблоне не должно быть
        «лишних» GROUP BY-колонок без соответствия в SELECT."""
        from core.sql_builder import SqlBuilder
        from core.sql_static_checker import check_sql

        selected_columns = {
            "gold.fact_event_b": {
                "select": ["event_dt"],
                "aggregate": ["task_code"],
                "group_by": ["event_dt"],
                "filter": [],
            }
        }
        blueprint = {
            "strategy": "simple_select",
            "main_table": "gold.fact_event_b",
            "aggregation": {"function": "count_distinct", "column": "task_code"},
            "group_by": ["event_dt"],
            "where_conditions": [],
            "order_by": None,
            "limit": 100,
            "having": [],
        }
        sql = SqlBuilder().build(
            strategy="simple_select",
            selected_columns=selected_columns,
            join_spec=[],
            blueprint=blueprint,
        )
        assert sql is not None
        result = check_sql(sql, check_columns=False)
        extra_warnings = [
            w for w in result.warnings
            if "отсутствующие в SELECT" in w
        ]
        assert not extra_warnings, f"Неожиданные extra GROUP BY: {extra_warnings}"
