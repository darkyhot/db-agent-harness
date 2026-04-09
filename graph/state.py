"""TypedDict состояние графа LangGraph."""

from typing import Any, TypedDict


class AgentState(TypedDict):
    """Состояние агента, передаваемое между узлами графа.

    Поля разделены на:
    - Общие (управление потоком, история, ошибки)
    - Структурированные данные (результаты экстракции каждого узла)

    Каждый LLM-узел получает только НУЖНЫЕ ему поля, а не весь state.
    """

    # === Общие поля ===
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
    graph_iterations: int
    correction_examples: list
    join_risk_info: dict
    start_time: float
    replan_count: int
    needs_replan: bool
    replan_context: str

    # === Структурированные данные (результаты экстракции каждого узла) ===

    # intent_classifier → структурированный интент
    intent: dict[str, Any]
    # Пример: {"intent": "analytics", "entities": [...], "date_filters": {...},
    #          "aggregation_hint": "count", "needs_search": False,
    #          "complexity": "single_table"}

    # table_resolver → выбранные таблицы
    selected_tables: list[tuple[str, str]]
    # Пример: [("dm", "sales"), ("dm", "managers")]

    # table_explorer → структурированные данные по таблицам (вместо монолитного tables_context)
    tables_context: str  # DEPRECATED: сохраняется для обратной совместимости
    table_structures: dict[str, str]
    # {"dm.sales": "column_name | dtype | is_pk | ...", ...}
    table_samples: dict[str, str]
    # {"dm.sales": "| col1 | col2 | ...\n| --- | --- | ...", ...}
    table_types: dict[str, str]
    # {"dm.sales": "fact", "dm.managers": "dim"}
    join_analysis_data: dict[str, Any]
    # {"dm.sales|dm.managers": {"candidates": [...], "safe_pattern": "..."}}

    # column_selector → выбранные колонки и JOIN-спецификация
    selected_columns: dict[str, Any]
    # {"dm.sales": {"select": [...], "filter": [...], "aggregate": [...]}, ...}
    join_spec: list[dict[str, Any]]
    # [{"left": "dm.sales.manager_id", "right": "dm.managers.manager_id",
    #   "safe": True, "strategy": "direct"}, ...]

    # sql_planner → blueprint для SQL
    sql_blueprint: dict[str, Any]
    # {"strategy": "fact_dim_join", "main_table": "dm.sales",
    #  "cte_needed": False, "subquery_for": ["dm.managers"],
    #  "where_conditions": [...], "aggregation": {...}, ...}

    # error_diagnoser → диагноз ошибки
    error_diagnosis: dict[str, Any]
    # {"error_type": "column_not_found", "root_cause": "...",
    #  "fix_strategy": "replace_column", "replacements": [...]}

    # sql_validator → pending tool call info
    pending_sql_tool_call: dict[str, Any] | None

    # sql_planner → корректирующая подсказка для повторного запуска column_selector
    # Устанавливается когда dim-таблица пропущена в selected_columns.
    # Сбрасывается в "" после того как column_selector её использует.
    column_selector_hint: str

    # Multi-turn context: предыдущий успешный SQL и краткое резюме результата
    # Передаётся из CLIInterface при follow-up запросах ("а теперь по регионам")
    prev_sql: str
    prev_result_summary: str
