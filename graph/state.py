"""TypedDict состояние графа LangGraph."""

from typing import TypedDict


class AgentState(TypedDict):
    """Состояние агента, передаваемое между узлами графа.

    Attributes:
        messages: История диалога текущей сессии.
        plan: Шаги плана текущей задачи.
        current_step: Индекс текущего шага выполнения.
        tool_calls: История вызовов инструментов.
        last_error: Последняя ошибка для корректора.
        retry_count: Счётчик попыток текущего шага.
        sql_to_validate: SQL, ожидающий валидации.
        final_answer: Финальный ответ пользователю.
        user_input: Исходный запрос пользователя.
        needs_confirmation: Флаг ожидания подтверждения от пользователя.
        confirmation_message: Сообщение для подтверждения.
    """

    messages: list
    plan: list[str]
    current_step: int
    tool_calls: list
    last_error: str | None
    retry_count: int
    sql_to_validate: str | None
    final_answer: str | None
    user_input: str
    needs_confirmation: bool
    confirmation_message: str
    needs_disambiguation: bool
    disambiguation_options: list
    tables_context: str
    graph_iterations: int
    correction_examples: list
    join_risk_info: dict
    start_time: float
