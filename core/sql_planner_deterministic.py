"""Детерминированный планировщик SQL-стратегии.

Заменяет LLM-ноду sql_planner полностью на основе структурированных данных
из AgentState. Не делает ни одного LLM-вызова.

Вход: intent, selected_columns, join_spec, table_types, join_analysis_data
Выход: sql_blueprint (dict) — такой же формат, что ожидает sql_writer.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Сопоставление aggregation_hint → SQL-функция
_HINT_TO_FUNC: dict[str, str] = {
    "sum": "SUM",
    "count": "COUNT",
    "avg": "AVG",
    "average": "AVG",
    "min": "MIN",
    "max": "MAX",
    "list": None,   # нет агрегации — просто перечисление
}

# Типы, которые считаются «dimension»
_DIM_TYPES = {"dim", "ref"}

# Типы колонок, которые НЕ идут в GROUP BY (только агрегируются)
_AGGREGATE_ONLY_DTYPES = re.compile(
    r"^(numeric|decimal|float|double|real|money|int|bigint|smallint|serial)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. Определение стратегии
# ---------------------------------------------------------------------------

def _determine_strategy(
    table_types: dict[str, str],
    join_spec: list[dict],
) -> tuple[str, str]:
    """Вернуть (strategy, main_table) на основе типов таблиц и join_spec.

    Правила:
    - 0-1 таблиц → simple_select
    - fact + dim/ref → fact_dim_join (main=fact) или dim_fact_join (main=dim)
    - fact + fact → fact_fact_join (main=первая fact)
    - dim + dim → dim_dim_join (main=первая dim)
    - unknown → fallback к simple_select / fact_dim_join по количеству таблиц
    """
    if len(table_types) <= 1:
        main = next(iter(table_types), "")
        return "simple_select", main

    facts = [t for t, v in table_types.items() if v == "fact"]
    dims  = [t for t, v in table_types.items() if v in _DIM_TYPES]
    unknowns = [t for t, v in table_types.items() if v == "unknown"]

    # Факт + справочник (самый частый сценарий)
    if facts and dims:
        # Если в join_spec первая пара: left=fact, right=dim → fact_dim_join
        # Иначе смотрим по порядку таблиц
        main = facts[0]
        return "fact_dim_join", main

    # Два и более фактов
    if len(facts) >= 2:
        return "fact_fact_join", facts[0]

    # Только справочники
    if len(dims) >= 2 and not facts:
        return "dim_dim_join", dims[0]

    # Один факт без справочника (unknown на второй стороне)
    if facts and unknowns:
        main = facts[0]
        return "fact_dim_join", main  # unknown обрабатываем как dim для безопасности

    # Два unknown — относимся как к двум фактам
    if len(unknowns) >= 2:
        return "fact_fact_join", unknowns[0]

    # Один unknown — простой запрос
    return "simple_select", next(iter(table_types), "")


# ---------------------------------------------------------------------------
# 2. GROUP BY
# ---------------------------------------------------------------------------

def _compute_group_by(
    selected_columns: dict[str, dict],
    aggregation: dict | None,
) -> list[str]:
    """Вычислить список колонок для GROUP BY.

    Логика: все колонки из select-роли, которые НЕ являются агрегируемыми,
    плюс явно указанные в group_by-роли. Дедупликация сохраняет порядок.
    """
    agg_col = aggregation.get("column") if aggregation else None

    group_by: list[str] = []
    seen: set[str] = set()

    for _table, roles in selected_columns.items():
        # Явный group_by
        for col in roles.get("group_by", []):
            if col not in seen:
                group_by.append(col)
                seen.add(col)

        # select-роль: добавляем всё кроме агрегируемой колонки
        for col in roles.get("select", []):
            if col == agg_col:
                continue
            if col in roles.get("aggregate", []):
                continue
            if col not in seen:
                group_by.append(col)
                seen.add(col)

    return group_by


# ---------------------------------------------------------------------------
# 3. Агрегация
# ---------------------------------------------------------------------------

def _compute_aggregation(
    intent: dict,
    selected_columns: dict[str, dict],
) -> dict | None:
    """Вычислить агрегацию из intent.aggregation_hint + aggregate-роли колонок."""
    hint = (intent.get("aggregation_hint") or "").lower().strip()
    func = _HINT_TO_FUNC.get(hint)

    if not func:
        # Нет агрегации или «list» — просто выборка без агрегата
        return None

    # Находим первую колонку с ролью aggregate
    for _table, roles in selected_columns.items():
        agg_cols = roles.get("aggregate", [])
        if agg_cols:
            col = agg_cols[0]
            alias = f"{func.lower()}_{col}"
            return {"function": func, "column": col, "alias": alias}

    # Нет явной aggregate-роли — COUNT(*) как fallback для count
    if hint == "count":
        return {"function": "COUNT", "column": "*", "alias": "cnt"}

    return None


# ---------------------------------------------------------------------------
# 4. WHERE-условия из date_filters и filter_conditions
# ---------------------------------------------------------------------------

# Операторы, допустимые в filter_conditions
_SAFE_OPERATORS = frozenset({"=", "!=", "<>", "<", ">", "<=", ">=", "LIKE", "IN", "NOT IN"})
_OPERATOR_RE = re.compile(r"^(=|!=|<>|<=|>=|<|>|LIKE|NOT\s+IN|IN)$", re.IGNORECASE)


def _quote_value(value: str, operator: str) -> str:
    """Процитировать значение для SQL WHERE-условия.

    Числа оставляем без кавычек, строки — в одинарных.
    IN/NOT IN ожидают список вида (v1, v2).
    """
    op = operator.strip().upper()
    val = str(value).strip()

    if op in ("IN", "NOT IN"):
        # Значение должно быть уже в формате (v1, v2) или просто v1, v2
        if not val.startswith("("):
            items = [f"'{v.strip()}'" for v in val.split(",")]
            return f"({', '.join(items)})"
        return val

    # Число? Только если нет ведущих нулей (иначе это код/строка)
    if not (val.startswith("0") and len(val) > 1):
        try:
            float(val)
            return val
        except ValueError:
            pass

    # Строка
    safe_val = val.replace("'", "''")  # escape single quotes
    return f"'{safe_val}'"


def _compute_where_from_intent(
    intent: dict,
    selected_columns: dict[str, dict],
) -> list[str]:
    """Сформировать WHERE-условия из структурированных данных в intent.

    Обрабатывает:
    1. date_filters: {"from": "2024-01-01", "to": null} → дата-диапазон
    2. filter_conditions: [{"column_hint": "region", "operator": "=", "value": "Москва"}]
       → WHERE region = 'Москва' (если column_hint совпадает с filter-колонкой)
    """
    conditions: list[str] = []

    # 1. Дата-диапазон
    date_filters = intent.get("date_filters") or {}
    date_from = date_filters.get("from")
    date_to   = date_filters.get("to")

    if date_from or date_to:
        date_col = _find_date_column(selected_columns)
        if date_col:
            if date_from:
                conditions.append(f"{date_col} >= '{date_from}'::date")
            if date_to:
                conditions.append(f"{date_col} < '{date_to}'::date")

    # 2. Literal filter conditions из intent_classifier
    filter_conditions = intent.get("filter_conditions") or []
    if filter_conditions:
        # Строим индекс filter-колонок: нижний регистр → реальное имя
        filter_col_index: dict[str, str] = {}
        for _table, roles in selected_columns.items():
            for col in roles.get("filter", []) + roles.get("select", []):
                filter_col_index[col.lower()] = col

        for fc in filter_conditions:
            if not isinstance(fc, dict):
                continue
            hint = str(fc.get("column_hint", "")).lower().strip()
            operator = str(fc.get("operator", "=")).strip()
            value = str(fc.get("value", "")).strip()

            if not hint or not value:
                continue

            # Проверяем оператор на допустимость
            if not _OPERATOR_RE.match(operator):
                logger.warning(
                    "DeterministicPlanner: недопустимый оператор %r в filter_conditions — пропускаем",
                    operator,
                )
                continue

            # Ищем совпадение column_hint с реальными filter-колонками
            matched_col = None
            # Точное совпадение
            if hint in filter_col_index:
                matched_col = filter_col_index[hint]
            else:
                # Частичное совпадение: hint содержится в имени колонки или наоборот
                for col_lower, col_real in filter_col_index.items():
                    if hint in col_lower or col_lower in hint:
                        matched_col = col_real
                        break

            if matched_col:
                quoted = _quote_value(value, operator)
                op_upper = operator.strip().upper()
                conditions.append(f"{matched_col} {op_upper} {quoted}")
                logger.debug(
                    "DeterministicPlanner: filter_condition %s %s %s → %s",
                    matched_col, op_upper, quoted, f"{matched_col} {op_upper} {quoted}",
                )
            else:
                logger.debug(
                    "DeterministicPlanner: column_hint %r не нашёл совпадения в filter-колонках — "
                    "условие пропущено (LLM построит сам)",
                    hint,
                )

    return conditions


def _find_date_column(selected_columns: dict[str, dict]) -> str | None:
    """Найти первую колонку с date/timestamp-семантикой в filter-ролях."""
    _date_suffixes = ("_dt", "_date", "_dttm", "_timestamp", "_ts", "date", "dttm")

    for _table, roles in selected_columns.items():
        for col in roles.get("filter", []):
            col_lower = col.lower()
            if any(col_lower.endswith(suf) for suf in _date_suffixes):
                return col
            if "date" in col_lower or "dttm" in col_lower or "timestamp" in col_lower:
                return col

    # Fallback: ищем среди select-колонок
    for _table, roles in selected_columns.items():
        for col in roles.get("select", []):
            col_lower = col.lower()
            if any(col_lower.endswith(suf) for suf in _date_suffixes):
                return col

    return None


# ---------------------------------------------------------------------------
# 5. Главная функция
# ---------------------------------------------------------------------------

def build_blueprint(
    intent: dict,
    selected_columns: dict[str, dict],
    join_spec: list[dict],
    table_types: dict[str, str],
    join_analysis_data: dict,
) -> dict:
    """Построить SQL Blueprint детерминированно, без LLM.

    Args:
        intent: Распознанный интент (из intent_classifier).
        selected_columns: Роли колонок по таблицам (из column_selector).
        join_spec: JOIN-спецификация (из column_selector).
        table_types: Тип каждой таблицы — fact/dim/ref/unknown.
        join_analysis_data: Результаты join_analysis (для совместимости).

    Returns:
        dict совместимый с форматом sql_blueprint из AgentState.
    """
    strategy, main_table = _determine_strategy(table_types, join_spec)

    aggregation = _compute_aggregation(intent, selected_columns)
    group_by    = _compute_group_by(selected_columns, aggregation)
    where_conditions = _compute_where_from_intent(intent, selected_columns)

    # CTE нужен при: fact+fact, dim+dim, или при небезопасном JOIN
    cte_needed = (
        strategy in {"fact_fact_join", "dim_dim_join"}
        or any(not j.get("safe", True) for j in join_spec)
    )

    # subquery_for: таблицы, которые идут в подзапрос (dim в fact_dim_join)
    subquery_for: list[str] = []
    if strategy == "fact_dim_join":
        dims = [t for t, v in table_types.items() if v in _DIM_TYPES or v == "unknown"]
        subquery_for = [d for d in dims if d != main_table]
    elif strategy == "dim_fact_join":
        facts = [t for t, v in table_types.items() if v == "fact"]
        subquery_for = facts

    # ORDER BY: по агрегатному алиасу DESC, или нет
    order_by: str | None = None
    if aggregation and aggregation.get("alias"):
        order_by = f"{aggregation['alias']} DESC"

    # LIMIT: для analytics-запросов ставим 100 по умолчанию
    limit: int | None = intent.get("limit") or 100

    blueprint = {
        "strategy": strategy,
        "main_table": main_table,
        "cte_needed": cte_needed,
        "subquery_for": subquery_for,
        "where_conditions": where_conditions,
        "aggregation": aggregation,
        "group_by": group_by,
        "order_by": order_by,
        "limit": limit,
        "notes": f"[deterministic] strategy={strategy}, tables={list(table_types.keys())}",
    }

    logger.info(
        "DeterministicPlanner: strategy=%s, main=%s, cte=%s, group_by=%s, agg=%s",
        strategy, main_table, cte_needed, group_by,
        aggregation.get("function") if aggregation else None,
    )

    return blueprint
