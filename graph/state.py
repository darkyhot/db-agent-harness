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
    # Direction 6.5: суммарный счётчик retry через все replan-итерации.
    # retry_count сбрасывается при replanning (локальный счётчик шага),
    # total_retry_count продолжает копить, чтобы глобальный бюджет retry был виден.
    total_retry_count: int
    sql_to_validate: str | None
    final_answer: str | None
    user_input: str
    needs_confirmation: bool
    confirmation_message: str
    needs_clarification: bool
    clarification_message: str
    needs_disambiguation: bool
    disambiguation_options: list
    # Явные выборы колонок пользователя на уточнения по фильтрам.
    # {"request_id": "column_name"} — переживают рекурсивный повторный запуск
    # графа из CLI, чтобы where_resolver закрыл соответствующий request_id без
    # повторного вопроса.
    user_filter_choices: dict[str, str]
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

    # table_explorer → структурированные данные по таблицам (источник истины ниже).
    # tables_context оставлен только как compatibility shim для summarizer/старых call sites
    # и должен быть удалён после полного ухода от строкового контекста.
    tables_context: str
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

    # === Контракт таблиц (Блок A) ===
    # Белый список таблиц в формате "schema.table", разрешённых в SQL.
    # Заполняется в table_resolver и остаётся неизменным до конца пайплайна.
    # При followup-запросах — мёрджится (не перезаписывается).
    # sql_writer и sql_static_checker используют его для проверки.
    allowed_tables: list[str]

    # === Подсказки пользователя (детерминированный экстрактор) ===
    # Заполняется в hint_extractor (между intent_classifier и table_resolver).
    # Без LLM, без хардкода таблиц/колонок — только regex + валидация по каталогу.
    # Структура:
    #   {
    #     "must_keep_tables": [(schema, table), ...],
    #     "join_fields": ["inn", "customer_id", ...],
    #     "dim_sources": {"segment": {"table": "schema.t", "join_col": "inn"}},
    #     "having_hints": [{"op": ">=", "value": 3, "unit_hint": "человек"}],
    #   }
    # Используется в table_resolver (hard-lock must_keep_tables),
    # column_selector (dim_sources/join_fields), sql_planner (HAVING).
    user_hints: dict[str, Any]

    # semantic frame запроса и результат where_resolver
    semantic_frame: dict[str, Any]
    where_resolution: dict[str, Any]
    join_decision: dict[str, Any]
    planning_confidence: dict[str, Any]
    evidence_trace: dict[str, Any]
    fallback_policy: dict[str, Any]
