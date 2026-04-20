"""Сборка графа LangGraph с логикой переходов.

Новая архитектура: 12 узлов.
intent_classifier → table_resolver → table_explorer → column_selector
  → sql_planner → sql_writer → sql_static_checker → sql_validator
      → [ошибка] → error_diagnoser → sql_fixer → sql_static_checker → sql_validator
      → [успех] → summarizer → END
"""

import logging
import time
from typing import Any

from langgraph.graph import END, StateGraph

from core.database import DatabaseManager
from core.llm import RateLimitedLLM
from core.memory import MemoryManager
from core.schema_loader import SchemaLoader
import core.column_selector_deterministic as column_selector_module
import core.sql_planner_deterministic as sql_planner_module
from core.sql_validator import SQLValidator
from graph.nodes import GraphNodes
import graph.nodes.intent as intent_module
from graph.state import AgentState

logger = logging.getLogger(__name__)


MAX_GRAPH_ITERATIONS = 15
MAX_WALL_CLOCK_SECONDS = 300  # 5 минут на весь граф


def _is_timed_out(state: AgentState) -> bool:
    """Проверить, превышен ли wall-clock timeout."""
    start = state.get("start_time", 0)
    if start and (time.monotonic() - start) >= MAX_WALL_CLOCK_SECONDS:
        logger.warning(
            "Превышен wall-clock timeout (%d секунд)", MAX_WALL_CLOCK_SECONDS,
        )
        return True
    return False


def _check_limits(state: AgentState) -> str | None:
    """Общая проверка лимитов. Возвращает 'summarizer' если лимит достигнут."""
    if state.get("graph_iterations", 0) >= MAX_GRAPH_ITERATIONS:
        logger.warning("Достигнут лимит итераций графа (%d)", MAX_GRAPH_ITERATIONS)
        return "summarizer"
    if _is_timed_out(state):
        return "summarizer"
    return None


# ======================================================================
# Функции маршрутизации для новых узлов
# ======================================================================

def _route_after_intent_classifier(state: AgentState) -> str:
    """Маршрутизация после intent_classifier."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("needs_clarification"):
        return END

    intent = state.get("intent", {})

    # Вопрос по схеме — сразу к summarizer (ответ из каталога)
    if intent.get("intent") == "schema_question":
        return "summarizer"

    # Нужен поиск таблиц — к tool_dispatcher
    if intent.get("needs_search"):
        return "tool_dispatcher"

    # Обычный путь — через hint_extractor к table_resolver
    return "hint_extractor"


def _route_after_tool_dispatcher(state: AgentState) -> str:
    """Маршрутизация после tool_dispatcher."""
    limit = _check_limits(state)
    if limit:
        return limit

    # Если dispatcher обработал search запрос — возврат к intent_classifier
    intent = state.get("intent", {})
    if intent.get("search_results"):
        return "table_resolver"

    # Если dispatcher загрузил sample — возврат к error_diagnoser
    diagnosis = state.get("error_diagnosis", {})
    if diagnosis.get("sample_data"):
        return "error_diagnoser"

    # Fallback
    return "table_resolver"


def _route_after_table_resolver(state: AgentState) -> str:
    """Маршрутизация после table_resolver."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("needs_disambiguation") or state.get("needs_clarification"):
        return END

    return "table_explorer"


def _route_after_sql_writer(state: AgentState) -> str:
    """Маршрутизация после sql_writer."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("needs_disambiguation"):
        return END

    if state.get("needs_clarification"):
        return END

    if state.get("sql_to_validate"):
        return "sql_static_checker"

    if state.get("last_error"):
        return "error_diagnoser"

    if state.get("final_answer"):
        return "summarizer"

    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

    # Ещё есть шаги — возврат к column_selector для следующего шага
    return "column_selector"


def _route_after_static_checker(state: AgentState) -> str:
    """Маршрутизация после sql_static_checker."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("last_error"):
        return "error_diagnoser"

    if state.get("sql_to_validate"):
        return "sql_validator"

    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

    return "column_selector"


def _route_after_validator(state: AgentState) -> str:
    """Маршрутизация после sql_validator."""
    if state.get("needs_confirmation"):
        return END

    if state.get("needs_clarification"):
        return END

    if _is_timed_out(state):
        return "summarizer"

    if state.get("last_error"):
        return "error_diagnoser"

    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

    # Ещё есть шаги — к column_selector
    return "column_selector"


def _route_after_sql_planner(state: AgentState) -> str:
    """Маршрутизация после sql_planner.

    Если column_selector пропустил dim-таблицу, возвращаемся в column_selector
    с корректирующей подсказкой. Иначе — стандартный путь к sql_writer.
    """
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("column_selector_hint", ""):
        return "column_selector"

    if state.get("needs_clarification"):
        return END

    return "sql_writer"


def _route_after_error_diagnoser(state: AgentState) -> str:
    """Маршрутизация после error_diagnoser."""
    limit = _check_limits(state)
    if limit:
        return limit

    # Нужен replanning — к table_resolver с контекстом ошибки
    if state.get("needs_replan"):
        return "table_resolver"

    # Тривиальный фикс кодом — SQL уже исправлен, к валидатору
    if state.get("sql_to_validate"):
        return "sql_validator"

    # Нужен sample — к tool_dispatcher
    diagnosis = state.get("error_diagnosis", {})
    if diagnosis.get("needs_sample"):
        return "tool_dispatcher"

    # Исчерпаны попытки — финальный ответ
    if state.get("final_answer"):
        return "summarizer"

    # Обычный путь — к sql_fixer для переписывания SQL
    return "sql_fixer"


def _route_after_sql_fixer(state: AgentState) -> str:
    """Маршрутизация после sql_fixer."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("sql_to_validate"):
        return "sql_validator"

    if state.get("final_answer"):
        return "summarizer"

    if state.get("last_error"):
        return "error_diagnoser"

    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

    return "column_selector"


# ======================================================================
# Сборка графа
# ======================================================================

def build_graph(
    llm: RateLimitedLLM,
    db_manager: DatabaseManager,
    schema_loader: SchemaLoader,
    memory: MemoryManager,
    sql_validator: SQLValidator,
    tools: list,
    debug_prompt: bool = False,
) -> StateGraph:
    """Собрать граф агента.

    Новая архитектура с 11 узлами:
    intent_classifier → table_resolver → table_explorer → column_selector
      → sql_planner → sql_writer → sql_validator
          → [ошибка] → error_diagnoser → sql_fixer → sql_validator
          → [успех] → summarizer → END
    """
    logger.info(
        "Runtime modules: column_selector=%s, intent=%s, sql_planner=%s",
        getattr(column_selector_module, "__file__", "<unknown>"),
        getattr(intent_module, "__file__", "<unknown>"),
        getattr(sql_planner_module, "__file__", "<unknown>"),
    )
    nodes = GraphNodes(
        llm, db_manager, schema_loader, memory, sql_validator, tools,
        debug_prompt=debug_prompt,
    )

    graph = StateGraph(AgentState)

    # Добавляем все 13 узлов
    graph.add_node("intent_classifier", nodes.intent_classifier)
    graph.add_node("hint_extractor", nodes.hint_extractor)
    graph.add_node("table_resolver", nodes.table_resolver)
    graph.add_node("table_explorer", nodes.table_explorer)
    graph.add_node("column_selector", nodes.column_selector)
    graph.add_node("sql_planner", nodes.sql_planner)
    graph.add_node("sql_writer", nodes.sql_writer)
    graph.add_node("sql_static_checker", nodes.sql_static_checker)
    graph.add_node("sql_validator", nodes.sql_validator_node)
    graph.add_node("error_diagnoser", nodes.error_diagnoser)
    graph.add_node("sql_fixer", nodes.sql_fixer)
    graph.add_node("summarizer", nodes.summarizer)
    graph.add_node("tool_dispatcher", nodes.tool_dispatcher)

    # Точка входа
    graph.set_entry_point("intent_classifier")

    # === Линейная цепочка: intent → tables → explore → columns → plan → write ===
    # intent_classifier → conditional routing
    graph.add_conditional_edges("intent_classifier", _route_after_intent_classifier, {
        END: END,
        "summarizer": "summarizer",
        "tool_dispatcher": "tool_dispatcher",
        "hint_extractor": "hint_extractor",
    })

    # hint_extractor (детерминированный) → table_resolver
    graph.add_edge("hint_extractor", "table_resolver")

    # tool_dispatcher → conditional routing (back to resolver or diagnoser)
    graph.add_conditional_edges("tool_dispatcher", _route_after_tool_dispatcher, {
        "table_resolver": "table_resolver",
        "error_diagnoser": "error_diagnoser",
        "summarizer": "summarizer",
    })

    # table_resolver → table_explorer или END, если нужно уточнение источника
    graph.add_conditional_edges("table_resolver", _route_after_table_resolver, {
        END: END,
        "table_explorer": "table_explorer",
        "summarizer": "summarizer",
    })
    graph.add_edge("table_explorer", "column_selector")
    graph.add_edge("column_selector", "sql_planner")

    # sql_planner → conditional routing:
    # если dim-таблица пропущена → column_selector (повтор с hint), иначе → sql_writer
    graph.add_conditional_edges("sql_planner", _route_after_sql_planner, {
        END: END,
        "column_selector": "column_selector",
        "sql_writer": "sql_writer",
        "summarizer": "summarizer",
    })

    # sql_writer → sql_static_checker → sql_validator (или error_diagnoser)
    graph.add_conditional_edges("sql_writer", _route_after_sql_writer, {
        END: END,
        "sql_static_checker": "sql_static_checker",
        "error_diagnoser": "error_diagnoser",
        "summarizer": "summarizer",
        "column_selector": "column_selector",
    })

    graph.add_conditional_edges("sql_static_checker", _route_after_static_checker, {
        "sql_validator": "sql_validator",
        "error_diagnoser": "error_diagnoser",
        "summarizer": "summarizer",
        "column_selector": "column_selector",
    })

    # sql_validator → conditional routing
    graph.add_conditional_edges("sql_validator", _route_after_validator, {
        END: END,
        "error_diagnoser": "error_diagnoser",
        "summarizer": "summarizer",
        "column_selector": "column_selector",
    })

    # === Цикл коррекции ===
    # error_diagnoser → conditional routing
    graph.add_conditional_edges("error_diagnoser", _route_after_error_diagnoser, {
        "table_resolver": "table_resolver",
        "sql_validator": "sql_validator",
        "tool_dispatcher": "tool_dispatcher",
        "sql_fixer": "sql_fixer",
        "summarizer": "summarizer",
    })

    # sql_fixer → sql_static_checker → sql_validator (через тот же static check)
    graph.add_conditional_edges("sql_fixer", _route_after_sql_fixer, {
        "sql_validator": "sql_static_checker",
        "error_diagnoser": "error_diagnoser",
        "summarizer": "summarizer",
        "column_selector": "column_selector",
    })

    # summarizer → END
    graph.add_edge("summarizer", END)

    logger.info("Граф агента собран (13 узлов)")
    return graph.compile()


def create_initial_state(
    user_input: str,
    prev_sql: str = "",
    prev_result_summary: str = "",
    user_filter_choices: dict[str, str] | None = None,
    rejected_filter_choices: dict[str, list[str]] | None = None,
) -> AgentState:
    """Создать начальное состояние для запуска графа."""
    return AgentState(
        messages=[],
        plan=[],
        current_step=0,
        tool_calls=[],
        last_error=None,
        retry_count=0,
        total_retry_count=0,
        sql_to_validate=None,
        final_answer=None,
        user_input=user_input,
        needs_confirmation=False,
        confirmation_message="",
        needs_clarification=False,
        clarification_message="",
        needs_disambiguation=False,
        disambiguation_options=[],
        user_filter_choices=dict(user_filter_choices or {}),
        rejected_filter_choices={k: list(v) for k, v in dict(rejected_filter_choices or {}).items()},
        tables_context="",
        graph_iterations=0,
        correction_examples=[],
        join_risk_info={},
        start_time=time.monotonic(),
        replan_count=0,
        needs_replan=False,
        replan_context="",
        # Новые структурированные поля
        intent={},
        selected_tables=[],
        table_structures={},
        table_samples={},
        table_types={},
        join_analysis_data={},
        selected_columns={},
        join_spec=[],
        sql_blueprint={},
        error_diagnosis={},
        pending_sql_tool_call=None,
        column_selector_hint="",
        # Multi-turn context
        prev_sql=prev_sql,
        prev_result_summary=prev_result_summary,
        # Подсказки пользователя (детерминированный экстрактор)
        user_hints={
            "must_keep_tables": [],
            "join_fields": [],
            "dim_sources": {},
            "having_hints": [],
        },
        semantic_frame={},
        where_resolution={},
        join_decision={},
        planning_confidence={},
        evidence_trace={},
        fallback_policy={},
        # Белый список таблиц (заполняется в table_resolver)
        allowed_tables=[],
    )
