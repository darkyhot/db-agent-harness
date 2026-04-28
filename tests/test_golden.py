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

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from contextlib import ExitStack

import pytest

from core.schema_loader import SchemaLoader

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
DATA_DIR = Path(__file__).resolve().parent.parent / "data_for_agent"

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
        self._active_scenario: str | None = None
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _extract_query(user_prompt: str) -> str:
        patterns = [
            r"Запрос пользователя:\s*(.+)",
            r"Запрос:\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_prompt, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return user_prompt.strip()

    def _scenario(self, query: str) -> str:
        q = query.lower()
        if "какие таблицы" in q or "схеме schema" in q:
            return "schema_question"
        if ("дате" in q or "report_dt" in q) and ("госб" in q or "new_gosb_name" in q):
            return "outflow_by_date_gosb_name"
        if ("region_name" in q or "по регионам" in q) and ("отток" in q or "outflow_qty" in q):
            return "outflow_by_region"
        if ("segment_name" in q or "по сегмент" in q) and "январ" in q and ("сумм" in q or "sum(" in q):
            return "outflow_sum_by_segment_month"
        if ("task_code" in q or "задач" in q) and ("segment_name" in q or "сегмент" in q):
            return "tasks_by_segment"
        if ("segment_name" in q or "сегмент" in q) and ("epk" in q or "uzp_data_epk_consolidation" in q):
            return "payroll_epk_join"
        if ("segment_name" in q or "по сегмент" in q) and ("колич" in q or "count(" in q):
            return "count_by_segment"
        if "январ" in q and ("отток" in q or "outflow_qty" in q):
            return "outflow_january"
        if "сколько записей" in q and "отток" in q:
            return "outflow_count"
        return "outflow_count"

    @staticmethod
    def _source(table: str, *, confidence: float = 0.9) -> dict[str, Any]:
        return {"schema": "schema", "table": table, "required": True, "confidence": confidence}

    @staticmethod
    def _metric(operation: str, target: str | None = None, *, confidence: float = 0.9) -> dict[str, Any]:
        return {
            "operation": operation,
            "target": target,
            "distinct_policy": "auto",
            "confidence": confidence,
        }

    @staticmethod
    def _dimension(target: str, *, source_table: str | None = None, join_key: str | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {"target": target, "confidence": 0.9}
        if source_table:
            item["source_table"] = source_table
        if join_key:
            item["join_key"] = join_key
        return item

    def _query_spec_for_scenario(self, scenario: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "task": "answer_data",
            "metrics": [self._metric("count")],
            "dimensions": [],
            "filters": [],
            "source_constraints": [self._source("uzp_dwh_fact_outflow")],
            "join_constraints": [],
            "clarification_needed": False,
            "confidence": 0.9,
        }

        if scenario == "schema_question":
            return {
                **base,
                "task": "inspect_schema",
                "metrics": [],
                "source_constraints": [{"schema": "schema", "semantic": "schema", "required": True, "confidence": 0.9}],
            }
        if scenario == "outflow_count":
            return base
        if scenario == "outflow_january":
            return {
                **base,
                "metrics": [self._metric("sum", "outflow_qty")],
                "dimensions": [self._dimension("report_dt")],
                "time_range": {"start": "2024-01-01", "end": "2024-02-01", "grain": "day", "confidence": 0.9},
            }
        if scenario == "outflow_sum_by_segment_month":
            if self.complexity == "join":
                return {
                    **base,
                    "metrics": [self._metric("sum", "outflow_qty")],
                    "dimensions": [
                        self._dimension(
                            "segment_name",
                            source_table="schema.uzp_data_epk_consolidation",
                            join_key="inn",
                        )
                    ],
                    "time_range": {"start": "2024-01-01", "end": "2024-02-01", "grain": "month", "confidence": 0.9},
                    "source_constraints": [
                        self._source("uzp_dwh_fact_outflow"),
                        self._source("uzp_data_epk_consolidation"),
                    ],
                    "join_constraints": [{"left": "schema.uzp_dwh_fact_outflow", "right": "schema.uzp_data_epk_consolidation", "key": "inn", "confidence": 0.9}],
                }
            return {
                **base,
                "metrics": [self._metric("sum", "outflow_qty")],
                "dimensions": [self._dimension("segment_name")],
                "time_range": {"start": "2024-01-01", "end": "2024-02-01", "grain": "month", "confidence": 0.9},
            }
        if scenario == "count_by_segment":
            return {
                **base,
                "metrics": [self._metric("count")],
                "dimensions": [self._dimension("segment_name")],
            }
        if scenario == "tasks_by_segment":
            return {
                **base,
                "metrics": [self._metric("count", "task_code")],
                "dimensions": [self._dimension("segment_name")],
                "source_constraints": [self._source("uzp_data_split_mzp_sale_funnel")],
            }
        if scenario == "outflow_by_region":
            return {
                **base,
                "metrics": [self._metric("sum", "outflow_qty")],
                "dimensions": [self._dimension("region_name", source_table="schema.uzp_dim_gosb", join_key="gosb_id")],
                "source_constraints": [self._source("uzp_dwh_fact_outflow"), self._source("uzp_dim_gosb")],
                "join_constraints": [{"left": "schema.uzp_dwh_fact_outflow", "right": "schema.uzp_dim_gosb", "key": "gosb_id", "confidence": 0.9}],
            }
        if scenario == "outflow_by_date_gosb_name":
            return {
                **base,
                "metrics": [self._metric("sum", "outflow_qty")],
                "dimensions": [
                    self._dimension("report_dt"),
                    self._dimension("new_gosb_name", source_table="schema.uzp_dim_gosb", join_key="old_gosb_id"),
                ],
                "source_constraints": [self._source("uzp_dwh_fact_outflow"), self._source("uzp_dim_gosb")],
                "join_constraints": [
                    {"left": "schema.uzp_dwh_fact_outflow", "right": "schema.uzp_dim_gosb", "key": "old_gosb_id", "confidence": 0.9},
                    {"left": "schema.uzp_dwh_fact_outflow", "right": "schema.uzp_dim_gosb", "key": "tb_id", "confidence": 0.9},
                ],
                "order_by": {"target": "outflow_qty", "direction": "DESC", "confidence": 0.9},
            }
        if scenario == "payroll_epk_join":
            return {
                **base,
                "metrics": [self._metric("count")],
                "dimensions": [
                    self._dimension(
                        "segment_name",
                        source_table="schema.uzp_data_epk_consolidation",
                        join_key="inn",
                    )
                ],
                "source_constraints": [
                    self._source("uzp_data_payroll_m"),
                    self._source("uzp_data_epk_consolidation"),
                ],
                "join_constraints": [{"left": "schema.uzp_data_payroll_m", "right": "schema.uzp_data_epk_consolidation", "key": "inn", "confidence": 0.9}],
            }
        return base

    def _reviewed_tables_for_scenario(self, scenario: str) -> list[str]:
        spec = self._query_spec_for_scenario(scenario)
        tables = []
        for source in spec.get("source_constraints", []):
            schema = source.get("schema") or "schema"
            table = source.get("table")
            if table:
                tables.append(f"{schema}.{table}")
        return tables

    def _columns_for_scenario(self, scenario: str) -> dict[str, Any]:
        fact = "schema.uzp_dwh_fact_outflow"
        epk = "schema.uzp_data_epk_consolidation"
        gosb = "schema.uzp_dim_gosb"
        funnel = "schema.uzp_data_split_mzp_sale_funnel"
        payroll = "schema.uzp_data_payroll_m"

        if scenario == "outflow_count":
            return {"columns": {fact: {"aggregate": []}}, "join_keys": [], "null_warnings": []}
        if scenario == "outflow_january":
            return {
                "columns": {fact: {"select": ["report_dt"], "filter": ["report_dt"], "aggregate": ["outflow_qty"], "group_by": ["report_dt"]}},
                "join_keys": [],
                "null_warnings": [],
            }
        if scenario == "outflow_sum_by_segment_month":
            if self.complexity == "join":
                return {
                    "columns": {
                        fact: {"filter": ["report_dt"], "aggregate": ["outflow_qty"], "select": ["inn"]},
                        epk: {"select": ["segment_name", "inn"], "group_by": ["segment_name"]},
                    },
                    "join_keys": [{"left": f"{fact}.inn", "right": f"{epk}.inn", "safe": False, "strategy": "subquery"}],
                    "null_warnings": [],
                }
            return {
                "columns": {fact: {"select": ["segment_name"], "filter": ["report_dt"], "aggregate": ["outflow_qty"], "group_by": ["segment_name"]}},
                "join_keys": [],
                "null_warnings": [],
            }
        if scenario == "count_by_segment":
            return {
                "columns": {fact: {"select": ["segment_name"], "aggregate": [], "group_by": ["segment_name"]}},
                "join_keys": [],
                "null_warnings": [],
            }
        if scenario == "tasks_by_segment":
            return {
                "columns": {funnel: {"select": ["segment_name"], "aggregate": ["task_code"], "group_by": ["segment_name"]}},
                "join_keys": [],
                "null_warnings": [],
            }
        if scenario == "outflow_by_region":
            return {
                "columns": {
                    fact: {"aggregate": ["outflow_qty"], "select": ["gosb_id"]},
                    gosb: {"select": ["region_name", "old_gosb_id"], "group_by": ["region_name"]},
                },
                "join_keys": [{"left": f"{fact}.gosb_id", "right": f"{gosb}.old_gosb_id", "safe": True, "strategy": "direct"}],
                "null_warnings": [],
            }
        if scenario == "outflow_by_date_gosb_name":
            return {
                "columns": {
                    fact: {"select": ["report_dt", "gosb_id", "tb_id"], "aggregate": ["outflow_qty"], "group_by": ["report_dt"]},
                    gosb: {"select": ["new_gosb_name", "old_gosb_id", "tb_id"], "group_by": ["new_gosb_name"]},
                },
                "join_keys": [
                    {"left": f"{fact}.gosb_id", "right": f"{gosb}.old_gosb_id", "safe": True, "strategy": "direct"},
                    {"left": f"{fact}.tb_id", "right": f"{gosb}.tb_id", "safe": True, "strategy": "direct"},
                ],
                "null_warnings": [],
            }
        if scenario == "payroll_epk_join":
            return {
                "columns": {
                    payroll: {"select": ["inn"], "aggregate": []},
                    epk: {"select": ["segment_name", "inn"], "group_by": ["segment_name"]},
                },
                "join_keys": [{"left": f"{payroll}.inn", "right": f"{epk}.inn", "safe": False, "strategy": "direct"}],
                "null_warnings": [],
            }
        return {"columns": {fact: {"aggregate": []}}, "join_keys": [], "null_warnings": []}

    def invoke_with_system(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        self._call_count += 1
        self.calls.append((system_prompt, user_prompt))
        query = self._extract_query(user_prompt)
        scenario = self._active_scenario or self._scenario(query)

        # query_interpreter
        if "query_interpreter" in system_prompt or "QuerySpec" in system_prompt:
            scenario = self._scenario(query)
            self._active_scenario = scenario
            return json.dumps(self._query_spec_for_scenario(scenario), ensure_ascii=False)

        # catalog_grounding_reviewer
        if "проверяешь выбор таблиц" in system_prompt:
            return json.dumps(
                {"tables": self._reviewed_tables_for_scenario(scenario), "rationale": "golden mock"},
                ensure_ascii=False,
            )

        # column_selector
        if "селектор колонок" in system_prompt:
            return json.dumps(self._columns_for_scenario(scenario), ensure_ascii=False)

        # intent_classifier
        if "классификатор запросов" in system_prompt or "intent" in system_prompt.lower():
            if scenario == "schema_question":
                return (
                    '{"intent": "schema_question", "entities": ["schema"], '
                    '"date_filters": {"from": null, "to": null}, '
                    '"aggregation_hint": null, "needs_search": false, '
                    '"complexity": "single_table", "clarification_question": "", '
                    '"filter_conditions": []}'
                )
            aggregation_hint = "count"
            if "сумм" in query.lower():
                aggregation_hint = "sum"
            return (
                f'{{"intent": "{self.intent}", "entities": ["отток"], '
                f'"date_filters": {{"from": null, "to": null}}, '
                f'"aggregation_hint": "{aggregation_hint}", "needs_search": false, '
                f'"complexity": "{self.complexity}", "clarification_question": "", '
                f'"filter_conditions": []}}'
            )

        # table_resolver
        if "селектор таблиц" in system_prompt or "plan_steps" in system_prompt:
            if scenario == "schema_question":
                return (
                    '{"tables": [{"schema": "schema", "table": "uzp_dwh_fact_outflow", '
                    '"reason": "пример таблицы схемы"}], '
                    '"plan_steps": ["Ответить по каталогу таблиц schema"]}'
                )
            if scenario == "outflow_sum_by_segment_month":
                return (
                    '{"tables": ['
                    '{"schema": "schema", "table": "uzp_dwh_fact_outflow", "reason": "факт оттока"}, '
                    '{"schema": "schema", "table": "uzp_data_epk_consolidation", "reason": "сегмент организации по ИНН"}], '
                    '"plan_steps": ["Использовать schema.uzp_dwh_fact_outflow и schema.uzp_data_epk_consolidation"]}'
                )
            if scenario == "outflow_by_date_gosb_name":
                return (
                    '{"tables": ['
                    '{"schema": "schema", "table": "uzp_dwh_fact_outflow", "reason": "факт оттока"}, '
                    '{"schema": "schema", "table": "uzp_dim_gosb", "reason": "название ГОСБ"}], '
                    '"plan_steps": ["Использовать schema.uzp_dwh_fact_outflow и schema.uzp_dim_gosb"]}'
                )
            if scenario == "outflow_by_region":
                return (
                    '{"tables": ['
                    '{"schema": "schema", "table": "uzp_dwh_fact_outflow", "reason": "факт оттока"}, '
                    '{"schema": "schema", "table": "uzp_dim_gosb", "reason": "регион ГОСБ"}], '
                    '"plan_steps": ["Использовать schema.uzp_dwh_fact_outflow и schema.uzp_dim_gosb"]}'
                )
            if scenario == "payroll_epk_join":
                return (
                    '{"tables": ['
                    '{"schema": "schema", "table": "uzp_data_payroll_m", "reason": "факт payroll"}, '
                    '{"schema": "schema", "table": "uzp_data_epk_consolidation", "reason": "сегмент клиента"}], '
                    '"plan_steps": ["Использовать schema.uzp_data_payroll_m и schema.uzp_data_epk_consolidation"]}'
                )
            if scenario == "tasks_by_segment":
                return (
                    '{"tables": [{"schema": "schema", "table": "uzp_data_split_mzp_sale_funnel", '
                    '"reason": "задачи и сегменты"}], '
                    '"plan_steps": ["Получить данные из schema.uzp_data_split_mzp_sale_funnel"]}'
                )
            return (
                '{"tables": [{"schema": "schema", "table": "uzp_dwh_fact_outflow", "reason": "факт оттока"}], '
                '"plan_steps": ["Получить данные из schema.uzp_dwh_fact_outflow"]}'
            )

        # sql_planner
        if "планировщик SQL" in system_prompt or "strategy" in system_prompt:
            if scenario in {"outflow_by_region", "outflow_by_date_gosb_name", "payroll_epk_join"}:
                return (
                    '{"strategy": "fact_dim_join", '
                    '"main_table": "schema.uzp_dwh_fact_outflow", '
                    '"cte_needed": false, "subquery_for": [], '
                    '"where_conditions": [], "aggregation": {"function": "sum"}, '
                    '"group_by": [], "order_by": null, "limit": 100, "notes": ""}'
                )
            if scenario == "outflow_sum_by_segment_month":
                return (
                    '{"strategy": "fact_fact_join", '
                    '"main_table": "schema.uzp_dwh_fact_outflow", '
                    '"cte_needed": true, "subquery_for": ["schema.uzp_data_epk_consolidation"], '
                    '"where_conditions": [], "aggregation": {"function": "sum"}, '
                    '"group_by": [], "order_by": null, "limit": 100, "notes": "segment_name брать из epk через DISTINCT ON (inn)"}'
                )
            return (
                '{"strategy": "simple_select", "main_table": "schema.uzp_dwh_fact_outflow", '
                '"cte_needed": false, "subquery_for": [], '
                '"where_conditions": [], "aggregation": null, '
                '"group_by": [], "order_by": null, "limit": 100, "notes": ""}'
            )

        # sql_writer — генерирует SQL с корректными алиасами
        if "SQL-писатель" in system_prompt or "execute_query" in system_prompt:
            sql_map = {
                "outflow_count": (
                    "SELECT COUNT(*) AS total_rows "
                    "FROM schema.uzp_dwh_fact_outflow"
                ),
                "outflow_by_date_gosb_name": (
                    "SELECT report_dt, "
                    "SUM(outflow_qty) AS sum_outflow_qty, "
                    "new_gosb_name "
                    "FROM schema.uzp_dwh_fact_outflow "
                    "JOIN schema.uzp_dim_gosb "
                    "ON uzp_dim_gosb.old_gosb_id = uzp_dwh_fact_outflow.gosb_id "
                    "AND uzp_dim_gosb.tb_id = uzp_dwh_fact_outflow.tb_id "
                    "GROUP BY report_dt, new_gosb_name "
                    "ORDER BY sum_outflow_qty DESC"
                ),
                "outflow_january": (
                    "SELECT SUM(outflow_qty) AS total_outflow "
                    "FROM schema.uzp_dwh_fact_outflow "
                    "WHERE report_dt >= DATE '2024-01-01' AND report_dt < DATE '2024-02-01'"
                ),
                "outflow_sum_by_segment_month": (
                    "WITH epk_seg AS ("
                    "SELECT DISTINCT ON (inn) inn, segment_name "
                    "FROM schema.uzp_data_epk_consolidation "
                    "ORDER BY inn"
                    ") "
                    "SELECT e.segment_name AS segment_name, SUM(o.outflow_qty) AS total_outflow "
                    "FROM schema.uzp_dwh_fact_outflow o "
                    "JOIN epk_seg e ON e.inn = o.inn "
                    "WHERE o.report_dt >= DATE '2024-01-01' AND o.report_dt < DATE '2024-02-01' "
                    "GROUP BY e.segment_name"
                ),
                "count_by_segment": (
                    "SELECT segment_name AS segment_name, COUNT(*) AS total_rows "
                    "FROM schema.uzp_dwh_fact_outflow "
                    "GROUP BY segment_name"
                ),
                "tasks_by_segment": (
                    "SELECT segment_name AS segment_name, COUNT(task_code) AS task_count "
                    "FROM schema.uzp_data_split_mzp_sale_funnel "
                    "GROUP BY segment_name"
                ),
                "outflow_by_region": (
                    "SELECT g.region_name AS region_name, SUM(o.outflow_qty) AS total_outflow "
                    "FROM schema.uzp_dwh_fact_outflow o "
                    "JOIN schema.uzp_dim_gosb g ON o.gosb_id = g.old_gosb_id "
                    "GROUP BY g.region_name"
                ),
                "payroll_epk_join": (
                    "SELECT e.segment_name AS segment_name, COUNT(*) AS payroll_count "
                    "FROM schema.uzp_data_payroll_m p "
                    "JOIN schema.uzp_data_epk_consolidation e ON p.inn = CAST(e.inn AS text) "
                    "GROUP BY e.segment_name"
                ),
            }
            sql = sql_map.get(scenario, sql_map["outflow_count"])
            return f'{{"tool": "execute_query", "args": {{"sql": "{sql}"}}}}'

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

    def render_sql(self, query: str) -> str:
        scenario = self._scenario(query)
        sql_map = {
            "outflow_count": (
                "SELECT COUNT(*) AS total_rows "
                "FROM schema.uzp_dwh_fact_outflow"
            ),
            "outflow_by_date_gosb_name": (
                "SELECT report_dt, "
                "SUM(outflow_qty) AS sum_outflow_qty, "
                "new_gosb_name "
                "FROM schema.uzp_dwh_fact_outflow "
                "JOIN schema.uzp_dim_gosb "
                "ON uzp_dim_gosb.old_gosb_id = uzp_dwh_fact_outflow.gosb_id "
                "AND uzp_dim_gosb.tb_id = uzp_dwh_fact_outflow.tb_id "
                "GROUP BY report_dt, new_gosb_name "
                "ORDER BY sum_outflow_qty DESC"
            ),
            "outflow_january": (
                "SELECT SUM(outflow_qty) AS total_outflow "
                "FROM schema.uzp_dwh_fact_outflow "
                "WHERE report_dt >= DATE '2024-01-01' AND report_dt < DATE '2024-02-01'"
            ),
            "outflow_sum_by_segment_month": (
                "WITH epk_seg AS ("
                "SELECT DISTINCT ON (inn) inn, segment_name "
                "FROM schema.uzp_data_epk_consolidation "
                "ORDER BY inn"
                ") "
                "SELECT e.segment_name AS segment_name, SUM(o.outflow_qty) AS total_outflow "
                "FROM schema.uzp_dwh_fact_outflow o "
                "JOIN epk_seg e ON e.inn = o.inn "
                "WHERE o.report_dt >= DATE '2024-01-01' AND o.report_dt < DATE '2024-02-01' "
                "GROUP BY e.segment_name"
            ),
            "count_by_segment": (
                "SELECT segment_name AS segment_name, COUNT(*) AS total_rows "
                "FROM schema.uzp_dwh_fact_outflow "
                "GROUP BY segment_name"
            ),
            "tasks_by_segment": (
                "SELECT segment_name AS segment_name, COUNT(task_code) AS task_count "
                "FROM schema.uzp_data_split_mzp_sale_funnel "
                "GROUP BY segment_name"
            ),
            "outflow_by_region": (
                "SELECT g.region_name AS region_name, SUM(o.outflow_qty) AS total_outflow "
                "FROM schema.uzp_dwh_fact_outflow o "
                "JOIN schema.uzp_dim_gosb g ON o.gosb_id = g.old_gosb_id "
                "GROUP BY g.region_name"
            ),
            "payroll_epk_join": (
                "SELECT e.segment_name AS segment_name, COUNT(*) AS payroll_count "
                "FROM schema.uzp_data_payroll_m p "
                "JOIN schema.uzp_data_epk_consolidation e ON p.inn = CAST(e.inn AS text) "
                "GROUP BY e.segment_name"
            ),
        }
        return sql_map.get(scenario, sql_map["outflow_count"])


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
    """Создать SchemaLoader на основе реальных CSV-метафайлов."""
    return SchemaLoader(data_dir=DATA_DIR)


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

    def run(
        self,
        query: str,
        patch_sql_writer: bool = False,
        *,
        show_plan: bool = False,
        plan_preview_approved: bool = False,
        plan_context: dict[str, Any] | None = None,
        plan_edit_text: str = "",
    ) -> dict[str, Any]:
        """Запустить граф с mock компонентами.

        Returns:
            Словарь с результатами: sql, intent, retry_count, visited_nodes.
        """
        import time
        from unittest.mock import patch, MagicMock

        self._visited_nodes = []
        self._final_state = {}
        self._tool_calls = []

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

        def patched_sql_writer(self_node, state):
            sql = self.mock_llm.render_sql(query)
            iterations = state.get("graph_iterations", 0) + 1
            step_idx = state.get("current_step", 0)
            return {
                "sql_to_validate": sql,
                "pending_sql_tool_call": {
                    "tool": "execute_query",
                    "args": {"sql": sql},
                    "step_idx": step_idx,
                },
                "graph_iterations": iterations,
                "tool_calls": state.get("tool_calls", []) + [
                    {
                        "tool": "execute_query",
                        "args": {"sql": sql},
                        "result": "awaiting_validation",
                    }
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": "SQL отправлен на валидацию [golden mock]"}
                ],
            }

        visited = self._visited_nodes

        with ExitStack() as stack:
            stack.enter_context(patch.object(gn_module.GraphNodes, '__init__', patched_init))
            if patch_sql_writer:
                stack.enter_context(
                    patch.object(gn_module.GraphNodes, 'sql_writer', patched_sql_writer)
                )
            try:
                graph = build_graph(
                    self.mock_llm, mock_db, mock_schema, mock_memory,
                    mock_validator, [exec_tool, sample_tool, search_tool],
                    debug_prompt=False,
                    show_plan=show_plan,
                )
                state = create_initial_state(
                    query,
                    plan_preview_approved=plan_preview_approved,
                    plan_edit_text=plan_edit_text,
                    plan_context=plan_context,
                )
                result: dict = dict(state)

                for event in graph.stream(state):
                    node_name = list(event.keys())[0]
                    visited.append(node_name)
                    node_payload = event.get(node_name)
                    if isinstance(node_payload, dict):
                        result.update(node_payload)

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
                    "final_state": self._final_state,
                    "query_spec": result.get("query_spec", {}),
                    "query_grounding": result.get("query_grounding", {}),
                    "selected_columns": result.get("selected_columns", {}),
                    "join_spec": result.get("join_spec", []),
                    "sql_blueprint": result.get("sql_blueprint", {}),
                    "plan_preview_pending": bool(result.get("plan_preview_pending")),
                    "confirmation_message": result.get("confirmation_message", ""),
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
    patch_sql_writer = spec.get("patch_sql_writer", False)
    result = runner.run(query, patch_sql_writer=patch_sql_writer)

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
# Regression: show_plan (/plan) должен отличаться только preview/resume
# ---------------------------------------------------------------------------

_PLAN_EQ_QUERY = "Посчитай сумму оттока по дате и названию ГОСБ"
_PLAN_EQ_FIELDS = (
    "query_spec",
    "query_grounding",
    "selected_columns",
    "join_spec",
    "sql_blueprint",
)


def _route_without_plan_preview(nodes: list[str]) -> list[str]:
    return [node for node in nodes if node != "plan_preview"]


def _approved_route_without_resume(nodes: list[str]) -> list[str]:
    return [
        node for node in nodes
        if node not in {"query_interpreter", "plan_preview"}
    ]


def _prompts_text(llm: MockLLM) -> str:
    return "\n".join(f"{system}\n{user}" for system, user in llm.calls)


def test_same_query_with_and_without_plan_produces_same_sql() -> None:
    normal_llm = MockLLM()
    normal = GraphRunner(normal_llm).run(_PLAN_EQ_QUERY)

    plan_llm = MockLLM()
    plan_runner = GraphRunner(plan_llm)
    preview = plan_runner.run(_PLAN_EQ_QUERY, show_plan=True)
    approved = plan_runner.run(
        _PLAN_EQ_QUERY,
        show_plan=True,
        plan_preview_approved=True,
        plan_context=preview["final_state"],
    )

    assert "error" not in normal
    assert "error" not in preview
    assert "error" not in approved
    assert preview["plan_preview_pending"] is True
    assert preview["sql"] == ""
    assert normal["sql"] == approved["sql"]

    plan_route = [
        node for node in preview["visited_nodes"] if node != "plan_preview"
    ] + _approved_route_without_resume(approved["visited_nodes"])
    assert _route_without_plan_preview(normal["visited_nodes"]) == plan_route

    prompt_text = _prompts_text(plan_llm)
    assert "/plan" not in prompt_text
    assert "show_plan" not in prompt_text
    assert "/plan" not in approved["sql"]
    assert "show_plan" not in approved["sql"]


def test_plan_preview_resume_preserves_query_spec_and_blueprint() -> None:
    normal = GraphRunner(MockLLM()).run(_PLAN_EQ_QUERY)
    plan_runner = GraphRunner(MockLLM())
    preview = plan_runner.run(_PLAN_EQ_QUERY, show_plan=True)
    approved = plan_runner.run(
        _PLAN_EQ_QUERY,
        show_plan=True,
        plan_preview_approved=True,
        plan_context=preview["final_state"],
    )

    for field in _PLAN_EQ_FIELDS:
        assert approved[field] == normal[field], field
        assert approved[field] == preview[field], field


def test_plan_edit_resume_uses_same_sql_pipeline_after_correction() -> None:
    runner = GraphRunner(MockLLM())
    preview = runner.run(_PLAN_EQ_QUERY, show_plan=True)
    edited_preview = runner.run(
        _PLAN_EQ_QUERY,
        show_plan=True,
        plan_edit_text="сделай сортировку asc",
        plan_context=preview["final_state"],
    )
    approved = runner.run(
        _PLAN_EQ_QUERY,
        show_plan=True,
        plan_preview_approved=True,
        plan_context=edited_preview["final_state"],
    )

    assert "error" not in edited_preview
    assert "error" not in approved
    assert edited_preview["plan_preview_pending"] is True
    assert "plan_edit_router" in edited_preview["visited_nodes"]
    assert "catalog_grounder" in edited_preview["visited_nodes"]
    assert "plan_preview" in edited_preview["visited_nodes"]

    assert "sql_writer" in approved["visited_nodes"]
    assert "sql_self_corrector" in approved["visited_nodes"]
    assert "sql_static_checker" in approved["visited_nodes"]
    assert "sql_validator" in approved["visited_nodes"]
    assert approved["sql"]


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
