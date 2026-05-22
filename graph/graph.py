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
import core.column_binding as column_binding_module
import core.sql_planner_deterministic as sql_planner_module
from core.sql_validator import SQLValidator
from graph.nodes import GraphNodes
from graph.state import AgentState


def _full_table_name(item: tuple[str, str] | list[str] | str) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return f"{item[0]}.{item[1]}"
    return ""

logger = logging.getLogger(__name__)


MAX_GRAPH_ITERATIONS = 15
MAX_WALL_CLOCK_SECONDS = 600  # 10 минут на весь граф
MAX_ORCH_STEPS = 12  # guard от зацикливания LLM-планировщика


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

    # Если dispatcher обработал search запрос — повторяем catalog grounding.
    intent = state.get("intent", {})
    if intent.get("search_results"):
        return "catalog_grounder"

    # Если dispatcher загрузил sample — возврат к error_diagnoser
    diagnosis = state.get("error_diagnosis", {})
    if diagnosis.get("sample_data"):
        return "error_diagnoser"

    # Fallback
    return "catalog_grounder"


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

    # Нужен replanning — к catalog_grounder с контекстом ошибки
    if state.get("needs_replan"):
        return "catalog_grounder"

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
# Маршрутизация LLM-оркестратора
# ======================================================================

# Node-id'ы намеренно НЕ совпадают с ключами AgentState (LangGraph запрещает
# имена узлов, конфликтующие со state-ключами, напр. orch_explain_plan).
ORCH_STEP_NODES = {
    "extract_sources": "step_extract_sources",
    "pull_metadata": "step_pull_metadata",
    "explain_plan": "step_explain_plan",
    "explain_sql": "step_explain_sql",
    "execute_sql": "step_execute_sql",
    "create_directory": "step_create_directory",
    "file_operation": "step_file_operation",
    "run_analytics": "step_run_analytics",
}


def _route_after_orchestrator(state: AgentState) -> str:
    """Маршрутизация после LLM-оркестратора.

    Терминалы (пауза на пользователя / финальный ответ) уходят в END или
    summarizer, иначе — в выбранный orchestrator-узлом шаг. Любой неизвестный
    шаг безопасно деградирует в run_analytics (= прежнее поведение агента).
    """
    if state.get("orch_step_count", 0) >= MAX_ORCH_STEPS:
        logger.warning("Достигнут лимит шагов оркестратора (%d)", MAX_ORCH_STEPS)
        return "summarizer"

    limit = _check_limits(state)
    if limit:
        return limit

    # Пауза на пользователя — отдаём управление CLI (он возобновит граф).
    if state.get("needs_confirmation"):
        return END
    if state.get("needs_clarification"):
        return END
    if state.get("plan_preview_pending"):
        return END

    step = str(state.get("orch_next_step") or "")
    terminating = step in ("summarize", "finish") or bool(state.get("final_answer"))

    if terminating:
        # Inner subgraph / orchestrator уже сформировали final_answer →
        # outer summarizer ничего не добавит. Только при полностью пустом
        # final_answer уводим в summarizer.
        if state.get("final_answer"):
            return END
        return "summarizer"

    return ORCH_STEP_NODES.get(step, "step_run_analytics")


# ======================================================================
# Сборка графа
# ======================================================================

def build_analytics_subgraph(
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
    """Собрать детерминированный аналитический подграф.

    Тело и маршрутизация — те же, что и раньше (13 узлов, plan_verifier
    управляется ``llm_verifier_enabled``). Остаётся публично доступным как
    ``build_graph`` (для тестов и инструментов). В оркестрованном графе
    вызывается как один составной шаг ``run_analytics`` через
    ``build_orchestrated_graph``.

    Архитектура (13 узлов):
    intent_classifier → table_resolver → table_explorer → column_selector
      → sql_planner → sql_writer → sql_self_corrector → sql_validator
          → [ошибка] → error_diagnoser → sql_fixer → sql_self_corrector → sql_validator
          → [успех] → summarizer → END
    """
    logger.info(
        "Runtime modules: column_selector=%s, intent=%s, column_binding=%s, sql_planner=%s",
        getattr(column_binding_module, "__file__", "<unknown>"),
        "QuerySpec",
        getattr(column_binding_module, "__file__", "<unknown>"),
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

    # tool_dispatcher → conditional routing (back to grounding or diagnoser)
    graph.add_conditional_edges("tool_dispatcher", _route_after_tool_dispatcher, {
        "catalog_grounder": "catalog_grounder",
        "error_diagnoser": "error_diagnoser",
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
        "catalog_grounder": "catalog_grounder",
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

    logger.info("Аналитический подграф собран (13 узлов)")
    return graph.compile()


# Публичное имя аналитического графа сохраняется: тесты и инструменты,
# импортирующие build_graph, продолжают получать прежний детерминированный
# pipeline. Продуктовый путь (CLI) использует build_orchestrated_graph.
build_graph = build_analytics_subgraph


def build_orchestrated_graph(
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
    """Собрать LLM-центричный граф (plan-and-execute).

    В центре — orchestrator-узел: детерминированный guard сырого SQL, иначе
    LLM выбирает следующий шаг из реестра. Аналитический pipeline вызывается
    как один составной шаг run_analytics через скомпилированный
    build_analytics_subgraph (text2sql-пайплайн не модифицируется и
    продолжает уважать ``llm_verifier_enabled``).

    orchestrator ⇄ {extract_sources, pull_metadata, explain_plan, explain_sql,
    execute_sql, create_directory, file_operation, run_analytics} →
    summarizer → END (или END при паузе на пользователя: confirm/clarify/
    plan_preview).
    """
    nodes = GraphNodes(
        llm, db_manager, schema_loader, memory, sql_validator, tools,
        debug_prompt=debug_prompt,
        show_plan=show_plan,
        llm_verifier_enabled=llm_verifier_enabled,
    )
    # Аналитический pipeline как один вызываемый шаг. Отдельный экземпляр
    # подграфа допустим: llm/db/schema/memory/validator/tools — те же объекты.
    nodes._analytics_subgraph = build_analytics_subgraph(
        llm, db_manager, schema_loader, memory, sql_validator, tools,
        debug_prompt=debug_prompt,
        show_plan=show_plan,
        llm_verifier_enabled=llm_verifier_enabled,
    )

    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", nodes.orchestrator)
    graph.add_node("step_extract_sources", nodes.orch_extract_sources)
    graph.add_node("step_pull_metadata", nodes.orch_pull_metadata)
    graph.add_node("step_explain_plan", nodes.orch_explain_plan)
    graph.add_node("step_explain_sql", nodes.orch_explain_sql)
    graph.add_node("step_execute_sql", nodes.orch_execute_sql)
    graph.add_node("step_create_directory", nodes.orch_create_directory)
    graph.add_node("step_file_operation", nodes.orch_file_operation)
    graph.add_node("step_run_analytics", nodes.orch_run_analytics)
    graph.add_node("summarizer", nodes.summarizer)

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", _route_after_orchestrator, {
        END: END,
        "summarizer": "summarizer",
        "step_extract_sources": "step_extract_sources",
        "step_pull_metadata": "step_pull_metadata",
        "step_explain_plan": "step_explain_plan",
        "step_explain_sql": "step_explain_sql",
        "step_execute_sql": "step_execute_sql",
        "step_create_directory": "step_create_directory",
        "step_file_operation": "step_file_operation",
        "step_run_analytics": "step_run_analytics",
    })
    # Каждый шаг возвращает управление оркестратору (единый хаб маршрутизации).
    for step_node in (
        "step_extract_sources",
        "step_pull_metadata",
        "step_explain_plan",
        "step_explain_sql",
        "step_execute_sql",
        "step_create_directory",
        "step_file_operation",
        "step_run_analytics",
    ):
        graph.add_edge(step_node, "orchestrator")
    graph.add_edge("summarizer", END)

    logger.info("Оркестрованный граф собран (orchestrator + 8 шагов)")
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
        sql_preview=str(ctx.get("sql_preview") or ""),
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
        explorer_error=dict(ctx.get("explorer_error") or {}),
        # LLM-центричный оркестратор: все поля переживают рекурсивный
        # перезапуск графа из CLI (resume через plan_context=result).
        orch_history=list(ctx.get("orch_history") or []),
        orch_plan=list(ctx.get("orch_plan") or []),
        orch_plan_active=bool(ctx.get("orch_plan_active") or False),
        orch_plan_step_answers=list(ctx.get("orch_plan_step_answers") or []),
        orch_next_step=str(ctx.get("orch_next_step") or ""),
        orch_step_count=int(ctx.get("orch_step_count", 0) or 0),
        orch_sql=str(ctx.get("orch_sql") or ""),
        orch_fs_path=str(ctx.get("orch_fs_path") or ""),
        orch_fs_tool=str(ctx.get("orch_fs_tool") or ""),
        orch_fs_content=str(ctx.get("orch_fs_content") or ""),
        orch_sources=list(ctx.get("orch_sources") or []),
        orch_metadata=str(ctx.get("orch_metadata") or ""),
        orch_explain_plan=str(ctx.get("orch_explain_plan") or ""),
        orch_resume_step=str(ctx.get("orch_resume_step") or ""),
    )
