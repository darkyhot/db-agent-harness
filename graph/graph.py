"""Сборка графа LangGraph с логикой переходов."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from core.database import DatabaseManager
from core.llm import RateLimitedLLM
from core.memory import MemoryManager
from core.schema_loader import SchemaLoader
from core.sql_validator import SQLValidator
from graph.nodes import GraphNodes
from graph.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_executor(state: AgentState) -> str:
    """Маршрутизация после узла executor.

    Returns:
        Имя следующего узла.
    """
    # Нужна disambiguation — выходим для уточнения у пользователя
    if state.get("needs_disambiguation"):
        return END

    # Есть SQL для валидации
    if state.get("sql_to_validate"):
        return "sql_validator"

    # Есть ошибка — идём в корректор
    if state.get("last_error"):
        return "corrector"

    # Есть финальный ответ (установлен ранее)
    if state.get("final_answer"):
        return "summarizer"

    # Все шаги выполнены
    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

    # Ещё есть шаги — продолжаем выполнение
    return "executor"


def _route_after_validator(state: AgentState) -> str:
    """Маршрутизация после узла sql_validator.

    Returns:
        Имя следующего узла.
    """
    # Нужно подтверждение пользователя — выходим
    if state.get("needs_confirmation"):
        return END

    # Есть ошибка — идём в корректор
    if state.get("last_error"):
        return "corrector"

    # Все шаги выполнены
    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

    # Продолжаем выполнение
    return "executor"


def _route_after_corrector(state: AgentState) -> str:
    """Маршрутизация после узла corrector.

    Returns:
        Имя следующего узла.
    """
    # Есть SQL для валидации после коррекции
    if state.get("sql_to_validate"):
        return "sql_validator"

    # Есть финальный ответ (исчерпаны попытки)
    if state.get("final_answer"):
        return "summarizer"

    # Всё ещё ошибка — повторяем коррекцию
    if state.get("last_error"):
        return "corrector"

    # Все шаги выполнены
    if state["current_step"] >= len(state["plan"]):
        return "summarizer"

    # Продолжаем выполнение
    return "executor"


def build_graph(
    llm: RateLimitedLLM,
    db_manager: DatabaseManager,
    schema_loader: SchemaLoader,
    memory: MemoryManager,
    sql_validator: SQLValidator,
    tools: list,
) -> StateGraph:
    """Собрать граф агента.

    Args:
        llm: LLM клиент.
        db_manager: Менеджер БД.
        schema_loader: Загрузчик схемы.
        memory: Менеджер памяти.
        sql_validator: Валидатор SQL.
        tools: Список LangChain tools.

    Returns:
        Скомпилированный граф LangGraph.
    """
    nodes = GraphNodes(llm, db_manager, schema_loader, memory, sql_validator, tools)

    graph = StateGraph(AgentState)

    # Добавляем узлы
    graph.add_node("planner", nodes.planner)
    graph.add_node("executor", nodes.executor)
    graph.add_node("sql_validator", nodes.sql_validator_node)
    graph.add_node("corrector", nodes.corrector)
    graph.add_node("summarizer", nodes.summarizer)

    # Точка входа
    graph.set_entry_point("planner")

    # Переходы
    graph.add_edge("planner", "executor")

    graph.add_conditional_edges("executor", _route_after_executor, {
        END: END,
        "sql_validator": "sql_validator",
        "corrector": "corrector",
        "summarizer": "summarizer",
        "executor": "executor",
    })

    graph.add_conditional_edges("sql_validator", _route_after_validator, {
        END: END,
        "corrector": "corrector",
        "summarizer": "summarizer",
        "executor": "executor",
    })

    graph.add_conditional_edges("corrector", _route_after_corrector, {
        "sql_validator": "sql_validator",
        "corrector": "corrector",
        "summarizer": "summarizer",
        "executor": "executor",
    })

    graph.add_edge("summarizer", END)

    logger.info("Граф агента собран")
    return graph.compile()


def create_initial_state(user_input: str) -> AgentState:
    """Создать начальное состояние для запуска графа.

    Args:
        user_input: Запрос пользователя.

    Returns:
        Начальное состояние AgentState.
    """
    return AgentState(
        messages=[],
        plan=[],
        current_step=0,
        tool_calls=[],
        last_error=None,
        retry_count=0,
        sql_to_validate=None,
        final_answer=None,
        user_input=user_input,
        needs_confirmation=False,
        confirmation_message="",
        needs_disambiguation=False,
        disambiguation_options=[],
    )
