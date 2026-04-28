"""Сборка графа LangGraph с логикой переходов.

Новая архитектура: 13 узлов.
intent_classifier → table_resolver → table_explorer → column_selector
  → sql_planner → sql_writer → sql_self_corrector → sql_static_checker → sql_validator
      → [ошибка] → error_diagnoser → sql_fixer → sql_self_corrector → sql_static_checker → sql_validator
      → [успех] → summarizer → END
"""

import logging
import time
import copy
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


def _full_table_name(item: tuple[str, str] | list[str] | str) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return f"{item[0]}.{item[1]}"
    return ""

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

    if state.get("plan_preview_approved") and state.get("sql_blueprint"):
        return "plan_preview"

    if state.get("plan_edit_text") and state.get("sql_blueprint"):
        return "plan_edit_router"

    if state.get("needs_clarification"):
        return END

    intent = state.get("intent", {})

    # Вопрос по схеме — сразу к summarizer (ответ из каталога)
    if intent.get("intent") == "schema_question":
        return "summarizer"

    # Нужен поиск таблиц — к tool_dispatcher
    if intent.get("needs_search"):
        return "tool_dispatcher"

    # Обычный путь — LLM-экстрактор подсказок → regex+merge hint_extractor → table_resolver
    return "hint_extractor_llm"


def _route_after_query_interpreter(state: AgentState) -> str:
    """Маршрутизация после нового QuerySpec-интерпретатора."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("plan_preview_approved") and state.get("sql_blueprint"):
        return "plan_preview"

    if state.get("plan_edit_text") and state.get("sql_blueprint"):
        return "plan_edit_router"

    if state.get("needs_clarification"):
        return END

    return "catalog_grounder"


def _route_after_catalog_grounder(state: AgentState) -> str:
    """Маршрутизация после связывания QuerySpec с каталогом."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("needs_clarification"):
        return END

    intent = state.get("intent", {}) or {}
    if intent.get("intent") == "schema_question":
        return "summarizer"

    return "table_explorer"


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
        return "sql_self_corrector"

    if state.get("last_error"):
        return "error_diagnoser"

    if state.get("final_answer"):
        return "summarizer"

    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

    # Ещё есть шаги — возврат к column_selector для следующего шага
    return "column_selector"


def _route_after_sql_self_corrector(state: AgentState) -> str:
    """Маршрутизация после LLM self-correction SQL."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("needs_clarification"):
        return END

    if state.get("last_error"):
        return "error_diagnoser"

    if state.get("sql_to_validate"):
        return "sql_static_checker"

    if state.get("final_answer"):
        return "summarizer"

    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

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
    с корректирующей подсказкой. Иначе — в plan_verifier (если включён) или
    сразу в plan_preview.
    """
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("column_selector_hint", ""):
        return "column_selector"

    if state.get("needs_clarification"):
        return END

    if state.get("plan_verifier_done"):
        return "plan_preview"

    return "plan_verifier"


def _route_after_plan_verifier(state: AgentState) -> str:
    """После plan_verifier: если применились правки — обратно в sql_planner
    для пересборки blueprint, иначе — в plan_preview.
    """
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("needs_clarification"):
        return END

    if state.get("plan_verifier_applied"):
        return "sql_planner"

    return "plan_preview"


def _route_after_plan_preview(state: AgentState) -> str:
    """Маршрутизация после plan_preview.

    Если plan_preview_pending=True — ждём подтверждения пользователя (END).
    Иначе — транзит к sql_writer.
    """
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("plan_preview_pending"):
        return END

    return "sql_writer"


def _route_after_plan_edit_router(state: AgentState) -> str:
    """Маршрутизация после plan_edit_router."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("plan_edit_needs_clarification") or state.get("needs_clarification"):
        return END

    kind = str(state.get("plan_edit_kind") or "")
    if kind == "query_spec":
        return "catalog_grounder"
    if kind == "patch":
        return "plan_patcher"
    if kind == "rebind":
        return "source_rebinder"
    if kind == "rewrite":
        return "intent_rewriter"
    return END


def _route_after_plan_edit_validator(state: AgentState) -> str:
    """После validator либо просим уточнение, либо рендерим diff и новый preview."""
    limit = _check_limits(state)
    if limit:
        return limit

    if state.get("plan_edit_needs_clarification") or state.get("needs_clarification"):
        return END

    return "plan_diff_renderer"


def _route_after_error_diagnoser(state: AgentState) -> str:
    """Маршрутизация после error_diagnoser."""
    limit = _check_limits(state)
    if limit:
        return limit

    # Нужен replanning — к table_resolver с контекстом ошибки
    if state.get("needs_replan"):
        return "table_resolver"

    # Тривиальный фикс кодом — SQL уже исправлен, к self-correction перед валидатором
    if state.get("sql_to_validate"):
        return "sql_self_corrector"

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
        return "sql_self_corrector"

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
    show_plan: bool = False,
    llm_verifier_enabled: bool = False,
) -> StateGraph:
    """Собрать граф агента.

    Новая архитектура с 13 узлами:
    intent_classifier → table_resolver → table_explorer → column_selector
      → sql_planner → sql_writer → sql_self_corrector → sql_validator
          → [ошибка] → error_diagnoser → sql_fixer → sql_self_corrector → sql_validator
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
        show_plan=show_plan,
        llm_verifier_enabled=llm_verifier_enabled,
    )

    graph = StateGraph(AgentState)

    # Добавляем все узлы
    graph.add_node("query_interpreter", nodes.query_interpreter)
    graph.add_node("catalog_grounder", nodes.catalog_grounder)
    graph.add_node("intent_classifier", nodes.intent_classifier)
    graph.add_node("hint_extractor_llm", nodes.hint_extractor_llm)
    graph.add_node("hint_extractor", nodes.hint_extractor)
    graph.add_node("explicit_mode_dispatcher", nodes.explicit_mode_dispatcher)
    graph.add_node("table_resolver", nodes.table_resolver)
    graph.add_node("table_explorer", nodes.table_explorer)
    graph.add_node("column_selector", nodes.column_selector)
    graph.add_node("sql_planner", nodes.sql_planner)
    graph.add_node("plan_verifier", nodes.plan_verifier)
    graph.add_node("plan_preview", nodes.plan_preview)
    graph.add_node("plan_edit_router", nodes.plan_edit_router)
    graph.add_node("plan_patcher", nodes.plan_patcher)
    graph.add_node("source_rebinder", nodes.source_rebinder)
    graph.add_node("intent_rewriter", nodes.intent_rewriter)
    graph.add_node("plan_edit_validator", nodes.plan_edit_validator)
    graph.add_node("plan_diff_renderer", nodes.plan_diff_renderer)
    graph.add_node("sql_writer", nodes.sql_writer)
    graph.add_node("sql_self_corrector", nodes.sql_self_corrector)
    graph.add_node("sql_static_checker", nodes.sql_static_checker)
    graph.add_node("sql_validator", nodes.sql_validator_node)
    graph.add_node("error_diagnoser", nodes.error_diagnoser)
    graph.add_node("sql_fixer", nodes.sql_fixer)
    graph.add_node("summarizer", nodes.summarizer)
    graph.add_node("tool_dispatcher", nodes.tool_dispatcher)

    # Точка входа
    graph.set_entry_point("query_interpreter")

    # === Новый primary path: QuerySpec → catalog grounding → explore → columns → plan → write ===
    graph.add_conditional_edges("query_interpreter", _route_after_query_interpreter, {
        END: END,
        "catalog_grounder": "catalog_grounder",
        "plan_edit_router": "plan_edit_router",
        "plan_preview": "plan_preview",
        "summarizer": "summarizer",
    })
    graph.add_conditional_edges("catalog_grounder", _route_after_catalog_grounder, {
        END: END,
        "table_explorer": "table_explorer",
        "summarizer": "summarizer",
    })

    # Legacy nodes are still registered for direct tests and emergency tooling,
    # but the main graph no longer routes through a second interpretation path.

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
    # column_selector (если пропущена dim-таблица) или plan_verifier (LLM-валидация плана)
    # или plan_preview (если verifier уже отработал на этой итерации).
    graph.add_conditional_edges("sql_planner", _route_after_sql_planner, {
        END: END,
        "column_selector": "column_selector",
        "plan_verifier": "plan_verifier",
        "plan_preview": "plan_preview",
        "summarizer": "summarizer",
    })

    # plan_verifier → sql_planner (если применились правки) или plan_preview.
    graph.add_conditional_edges("plan_verifier", _route_after_plan_verifier, {
        END: END,
        "sql_planner": "sql_planner",
        "plan_preview": "plan_preview",
        "summarizer": "summarizer",
    })

    # plan_preview → sql_writer (транзит) или END (ожидание подтверждения)
    graph.add_conditional_edges("plan_preview", _route_after_plan_preview, {
        END: END,
        "sql_writer": "sql_writer",
    })

    graph.add_conditional_edges("plan_edit_router", _route_after_plan_edit_router, {
        END: END,
        "catalog_grounder": "catalog_grounder",
        "plan_patcher": "plan_patcher",
        "source_rebinder": "source_rebinder",
        "intent_rewriter": "intent_rewriter",
        "summarizer": "summarizer",
    })
    graph.add_edge("plan_patcher", "plan_edit_validator")
    graph.add_edge("source_rebinder", "plan_edit_validator")
    graph.add_edge("intent_rewriter", "plan_edit_validator")
    graph.add_conditional_edges("plan_edit_validator", _route_after_plan_edit_validator, {
        END: END,
        "plan_diff_renderer": "plan_diff_renderer",
        "summarizer": "summarizer",
    })
    graph.add_edge("plan_diff_renderer", "plan_preview")

    # sql_writer → sql_self_corrector → sql_static_checker → sql_validator
    # (или error_diagnoser)
    graph.add_conditional_edges("sql_writer", _route_after_sql_writer, {
        END: END,
        "sql_self_corrector": "sql_self_corrector",
        "error_diagnoser": "error_diagnoser",
        "summarizer": "summarizer",
        "column_selector": "column_selector",
    })

    graph.add_conditional_edges("sql_self_corrector", _route_after_sql_self_corrector, {
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
        "sql_self_corrector": "sql_self_corrector",
        "tool_dispatcher": "tool_dispatcher",
        "sql_fixer": "sql_fixer",
        "summarizer": "summarizer",
    })

    # sql_fixer → sql_self_corrector → sql_static_checker → sql_validator
    graph.add_conditional_edges("sql_fixer", _route_after_sql_fixer, {
        "sql_self_corrector": "sql_self_corrector",
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
    plan_preview_approved: bool = False,
    plan_preview_iteration: int = 0,
    plan_edit_text: str = "",
    plan_context: dict[str, Any] | None = None,
    rejected_filter_choices: dict[str, list[str]] | None = None,
) -> AgentState:
    """Создать начальное состояние для запуска графа."""
    ctx = dict(plan_context or {})
    selected_tables = list(ctx.get("selected_tables") or [])
    seeded_user_hints = copy.deepcopy(ctx.get("user_hints") or {})
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
        query_spec=dict(ctx.get("query_spec") or {}),
        query_spec_validation_errors=list(ctx.get("query_spec_validation_errors") or []),
        use_legacy_interpreter=bool(ctx.get("use_legacy_interpreter", False)),
        query_grounding=dict(ctx.get("query_grounding") or {}),
        plan_ir=dict(ctx.get("plan_ir") or {}),
        clarification_spec=dict(ctx.get("clarification_spec") or {}),
        # Новые структурированные поля
        intent=dict(ctx.get("intent") or {}),
        selected_tables=selected_tables,
        table_structures=dict(ctx.get("table_structures") or {}),
        table_samples=dict(ctx.get("table_samples") or {}),
        table_types=dict(ctx.get("table_types") or {}),
        join_analysis_data=dict(ctx.get("join_analysis_data") or {}),
        selected_columns=dict(ctx.get("selected_columns") or {}),
        join_spec=list(ctx.get("join_spec") or []),
        sql_blueprint=dict(ctx.get("sql_blueprint") or {}),
        error_diagnosis={},
        pending_sql_tool_call=None,
        column_selector_hint="",
        column_selector_retry_count=int(ctx.get("column_selector_retry_count", 0) or 0),
        # Multi-turn context
        prev_sql=prev_sql,
        prev_result_summary=prev_result_summary,
        # Подсказки пользователя (детерминированный экстрактор)
        user_hints=seeded_user_hints or {
            "must_keep_tables": [],
            "join_fields": [],
            "dim_sources": {},
            "having_hints": [],
            "group_by_hints": [],
            "aggregate_hints": [],
            "aggregation_preferences": {},
            "aggregation_preferences_list": [],
            "time_granularity": None,
            "negative_filters": [],
        },
        semantic_frame=dict(ctx.get("semantic_frame") or {}),
        where_resolution=dict(ctx.get("where_resolution") or {}),
        join_decision=dict(ctx.get("join_decision") or {}),
        planning_confidence=dict(ctx.get("planning_confidence") or {}),
        evidence_trace=dict(ctx.get("evidence_trace") or {}),
        fallback_policy=dict(ctx.get("fallback_policy") or {}),
        sql_self_correction=dict(ctx.get("sql_self_correction") or {}),
        # Белый список таблиц (заполняется в table_resolver)
        allowed_tables=list(ctx.get("allowed_tables") or [_full_table_name(t) for t in selected_tables if _full_table_name(t)]),
        excluded_tables=list(ctx.get("excluded_tables") or []),
        # Explicit mode (задача 2.2)
        explicit_mode=bool(ctx.get("explicit_mode", False)),
        # Plan-preview (задача 2.1)
        plan_preview_pending=False,
        plan_preview_approved=plan_preview_approved,
        plan_preview_iteration=plan_preview_iteration,
        # Plan-verifier (LLM-валидация плана перед preview)
        plan_verifier_done=False,
        plan_verifier_applied=False,
        plan_verdict={},
        # Plan-edit cycle
        plan_edit_text=plan_edit_text,
        plan_edit_kind="",
        plan_edit_confidence=0.0,
        plan_edit_payload={},
        plan_edit_resolution={},
        plan_edit_explanation="",
        plan_edit_needs_clarification=False,
        plan_edit_applied=False,
        plan_edit_history=list(ctx.get("plan_edit_history") or []),
        previous_sql_blueprint=dict(ctx.get("previous_sql_blueprint") or {}),
        plan_diff=dict(ctx.get("plan_diff") or {}),
        plan_diff_summary=str(ctx.get("plan_diff_summary") or ""),
    )
