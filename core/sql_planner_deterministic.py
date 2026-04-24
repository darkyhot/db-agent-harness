"""Детерминированный планировщик SQL-стратегии.

Заменяет LLM-ноду sql_planner полностью на основе структурированных данных
из AgentState. Не делает ни одного LLM-вызова.

Вход: intent, selected_columns, join_spec, table_types, join_analysis_data
Выход: sql_blueprint (dict) — такой же формат, что ожидает sql_writer.
"""

import logging
import re

from core.where_resolver import resolve_where

logger = logging.getLogger(__name__)

# Суффиксы date-колонок для определения axis-join по дате
_DATE_AXIS_SUFFIXES = ("_dt", "_date", "_dttm", "_timestamp", "_ts", "date", "dttm", "report_dt")


_RU_MONTHS: dict[str, int] = {
    'январ': 1,
    'феврал': 2,
    'март': 3,
    'апрел': 4,
    'май': 5,
    'мая': 5,
    'июн': 6,
    'июл': 7,
    'август': 8,
    'сентябр': 9,
    'октябр': 10,
    'ноябр': 11,
    'декабр': 12,
}

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

    Колонки с ролью filter (например, дата для WHERE) НЕ попадают в GROUP BY:
    иначе они дают «GROUP BY содержит колонки, отсутствующие в SELECT».
    """
    agg_col = aggregation.get("column") if aggregation else None

    group_by: list[str] = []
    seen: set[str] = set()

    for _table, roles in selected_columns.items():
        filter_set = set(roles.get("filter", []))
        agg_set = set(roles.get("aggregate", []))
        select_set = set(roles.get("select", []))

        # Явный group_by — но только если колонка действительно в SELECT
        # или будет добавлена как dimension. Filter-only колонки не включаем.
        for col in roles.get("group_by", []):
            if col in seen:
                continue
            # filter-only (нет в select и не в group_by вручную из dim) → пропуск
            if col in filter_set and col not in select_set:
                continue
            group_by.append(col)
            seen.add(col)

        # select-роль: добавляем всё кроме агрегируемой колонки
        for col in roles.get("select", []):
            if col == agg_col:
                continue
            if col in agg_set:
                continue
            if col not in seen:
                group_by.append(col)
                seen.add(col)

    return group_by


# ---------------------------------------------------------------------------
# 2a-ext. TIME GRANULARITY — DATE_TRUNC wrap для group_by
# ---------------------------------------------------------------------------

_DATE_LIKE_DTYPES: frozenset[str] = frozenset({
    "date", "timestamp", "timestamptz",
    "timestamp with time zone", "timestamp without time zone",
})


def _apply_time_granularity(
    group_by: list[str],
    selected_columns: dict[str, dict],
    time_granularity: str | None,
    schema_loader,
) -> list[str]:
    """Обернуть date/timestamp-колонки в DATE_TRUNC если задана гранулярность.

    Если колонка уже в group_by — заменяет её на DATE_TRUNC(gran, col).
    Если date-колонка есть в select/filter, но не в group_by — добавляет DATE_TRUNC в группировку.
    Если date-колонок нет — возвращает group_by без изменений.
    """
    if not time_granularity or schema_loader is None:
        return group_by

    date_cols: set[str] = set()
    for table_key, roles in selected_columns.items():
        parts = table_key.split(".", 1)
        if len(parts) != 2:
            continue
        try:
            cols_df = schema_loader.get_table_columns(parts[0], parts[1])
        except Exception:
            continue
        for _, row in cols_df.iterrows():
            dtype_raw = str(row.get("dType") or row.get("dtype") or "").lower().strip()
            if any(d in dtype_raw for d in _DATE_LIKE_DTYPES):
                col_name = str(row.get("column_name", "")).strip()
                if col_name:
                    date_cols.add(col_name.lower())

    if not date_cols:
        return group_by

    result: list[str] = []
    wrapped: set[str] = set()
    for col in group_by:
        if col.lower() in date_cols:
            expr = f"DATE_TRUNC('{time_granularity}', {col})"
            result.append(expr)
            wrapped.add(col.lower())
        else:
            result.append(col)

    # Добавляем date-колонки из select/filter, которых нет в group_by
    all_in_gb = {c.lower() for c in group_by}
    for table_key, roles in selected_columns.items():
        for role in ("select", "filter"):
            for col in roles.get(role, []):
                if col.lower() in date_cols and col.lower() not in all_in_gb:
                    result.append(f"DATE_TRUNC('{time_granularity}', {col})")
                    all_in_gb.add(col.lower())

    return result


# ---------------------------------------------------------------------------
# 2b. HAVING (постагрегатные фильтры из user_hints)
# ---------------------------------------------------------------------------

def _compute_having(
    user_hints: dict | None,
    selected_columns: dict[str, dict],
    schema_loader,
    main_table: str,
) -> list[dict]:
    """Сформировать список HAVING-условий из user_hints.having_hints.

    Преобразует подсказки вида «от 3 человек» в:
        [{"expr": "COUNT(DISTINCT employee_id)", "op": ">=", "value": 3}]

    Колонка для unit подбирается из main_table через
    user_hint_extractor.match_unit_column. Если подобрать не удалось — берём
    первую PK-колонку main_table; если и её нет — пропускаем подсказку.
    """
    if not user_hints:
        return []
    hints = user_hints.get("having_hints", []) or []
    if not hints:
        return []
    if not main_table or not schema_loader:
        return []

    # Импорт здесь, чтобы избежать кругов и зависеть только при наличии хинтов.
    try:
        from core.user_hint_extractor import match_unit_column
    except Exception:  # noqa: BLE001
        return []

    parts = main_table.split(".", 1)
    if len(parts) != 2:
        return []
    main_cols = schema_loader.get_table_columns(parts[0], parts[1])

    result: list[dict] = []
    for hint in hints:
        if not isinstance(hint, dict):
            continue
        op = hint.get("op", ">=")
        value = hint.get("value")
        if value is None:
            continue
        unit = hint.get("unit_hint", "") or ""
        col = match_unit_column(unit, main_table, schema_loader) if unit else None
        if not col and not main_cols.empty:
            try:
                pk_mask = main_cols.get(
                    "is_primary_key",
                ).astype(bool)
                pk_cols = main_cols.loc[pk_mask, "column_name"].tolist()
                if pk_cols:
                    col = pk_cols[0]
            except Exception:  # noqa: BLE001
                col = None
        if not col:
            logger.info(
                "Planner: HAVING-подсказка %s проигнорирована — колонка не подобрана",
                hint,
            )
            continue
        result.append({
            "expr": f"COUNT(DISTINCT {col})",
            "op": op,
            "value": value,
        })
        logger.info(
            "Planner: HAVING %s %s %s (из user_hint unit=%r)",
            f"COUNT(DISTINCT {col})", op, value, unit,
        )
    return result


# ---------------------------------------------------------------------------
# 3. Агрегация
# ---------------------------------------------------------------------------

def _count_column_should_be_distinct(
    selected_columns: dict[str, dict],
    column: str,
    schema_loader=None,
    semantic_frame: dict | None = None,
    strategy: str = "",
) -> bool:
    """Определить, нужно ли считать COUNT по колонке как DISTINCT."""
    if not column or column == "*":
        return False
    if strategy == "simple_select":
        return False
    if (semantic_frame or {}).get("requires_single_entity_count"):
        return True
    if schema_loader is None:
        return False

    for table_key, roles in selected_columns.items():
        if column not in roles.get("aggregate", []):
            continue
        parts = table_key.split(".", 1)
        if len(parts) != 2:
            continue
        schema_name, table_name = parts
        cols_df = schema_loader.get_table_columns(schema_name, table_name)
        if cols_df.empty:
            continue
        matched = cols_df[cols_df["column_name"].astype(str).str.lower() == column.lower()]
        if matched.empty:
            continue
        row = matched.iloc[0]
        if bool(row.get("is_primary_key", False)):
            return True
        semantics = schema_loader.get_column_semantics(schema_name, table_name, column)
        if str(semantics.get("semantic_class", "") or "").lower() == "identifier":
            return True
    return False


def _list_pk_candidates(main_table: str, schema_loader) -> list[str]:
    """Список PK-колонок таблицы (для LLM-резолвера при составном PK)."""
    if not main_table or schema_loader is None or "." not in main_table:
        return []
    schema_name, table_name = main_table.split(".", 1)
    cols_df = schema_loader.get_table_columns(schema_name, table_name)
    if cols_df.empty:
        return []
    pks: list[str] = []
    for _, row in cols_df.iterrows():
        if not bool(row.get("is_primary_key", False)):
            continue
        name = str(row.get("column_name", "") or "").strip()
        if name:
            pks.append(name)
    return pks


def _choose_count_identifier_column(
    main_table: str,
    schema_loader,
    semantic_frame: dict | None = None,
    user_input: str = "",
) -> str | None:
    """Подобрать identifier/PK-колонку для COUNT(DISTINCT ...) safety net."""
    if not main_table or schema_loader is None or "." not in main_table:
        return None
    schema_name, table_name = main_table.split(".", 1)
    cols_df = schema_loader.get_table_columns(schema_name, table_name)
    if cols_df.empty:
        return None
    try:
        from core.filter_ranking import _stem_set, _subject_alias_stems, _text_score
    except Exception:  # noqa: BLE001
        _stem_set = None
        _subject_alias_stems = None
        _text_score = None

    subject = str((semantic_frame or {}).get("subject") or "").strip().lower()
    subject_stems = _subject_alias_stems(subject, schema_loader) if _subject_alias_stems and subject else set()
    query_stems = _stem_set(user_input) if _stem_set else set()
    grain = str(schema_loader.get_table_semantics(schema_name, table_name).get("grain") or "").strip().lower()

    best: tuple[float, str] | None = None
    for _, row in cols_df.iterrows():
        col_name = str(row.get("column_name", "") or "").strip()
        if not col_name:
            continue
        score = 0.0
        if bool(row.get("is_primary_key", False)):
            score += 100.0
        unique_perc = float(row.get("unique_perc", 0) or 0)
        if unique_perc >= 95.0:
            score += 40.0
        semantics = schema_loader.get_column_semantics(schema_name, table_name, col_name)
        sem_class = str(semantics.get("semantic_class", "") or "").lower()
        if sem_class == "identifier":
            score += 60.0
        lower_name = col_name.lower()
        col_stems = _stem_set(col_name.replace("_", " ")) if _stem_set else set()
        if col_stems & (subject_stems | query_stems):
            score += 90.0
        if _text_score is not None:
            score += min(_text_score(subject or user_input, col_name, str(row.get("description", "") or "")), 14.0)
        if lower_name.endswith("_code"):
            score += 20.0
        if lower_name.endswith("_id"):
            score += 10.0
        is_date_axis = lower_name.endswith(_DATE_AXIS_SUFFIXES) or sem_class in {"date", "datetime", "timestamp"}
        if is_date_axis:
            score -= 80.0
            if grain == "snapshot" and unique_perc >= 95.0:
                score -= 40.0
        candidate = (score, col_name)
        if best is None or candidate > best:
            best = candidate
    if best is None or best[0] < 40.0:
        return None
    return best[1]


def _is_time_axis_count_column(
    main_table: str,
    column: str,
    schema_loader=None,
) -> bool:
    """True если count-колонка похожа на дату/ось времени и опасна для COUNT."""
    lower_col = str(column or "").strip().lower()
    if not lower_col or lower_col == "*":
        return False
    if lower_col.endswith(_DATE_AXIS_SUFFIXES):
        return True
    if schema_loader is None or "." not in str(main_table or ""):
        return "date" in lower_col or "dttm" in lower_col or "timestamp" in lower_col

    schema_name, table_name = str(main_table).split(".", 1)
    cols_df = schema_loader.get_table_columns(schema_name, table_name)
    if cols_df.empty:
        return "date" in lower_col or "dttm" in lower_col or "timestamp" in lower_col
    matched = cols_df[cols_df["column_name"].astype(str).str.lower() == lower_col]
    if matched.empty:
        return "date" in lower_col or "dttm" in lower_col or "timestamp" in lower_col
    row = matched.iloc[0]
    dtype = str(row.get("dType", "") or "").lower().strip()
    semantics = schema_loader.get_column_semantics(schema_name, table_name, lower_col)
    sem_class = str(semantics.get("semantic_class", "") or "").lower().strip()
    return (
        dtype.startswith("date")
        or dtype.startswith("timestamp")
        or sem_class in {"date", "datetime", "timestamp", "system_timestamp"}
    )


def _compute_aggregation(
    intent: dict,
    selected_columns: dict[str, dict],
    *,
    user_hints: dict | None = None,
    schema_loader=None,
    semantic_frame: dict | None = None,
    strategy: str = "",
    main_table: str = "",
) -> dict | None:
    """Backward-compatible shim: вернуть первую агрегацию из canonical aggregations."""
    aggregations = _compute_aggregations(
        intent,
        selected_columns,
        user_hints=user_hints,
        schema_loader=schema_loader,
        semantic_frame=semantic_frame,
        strategy=strategy,
        main_table=main_table,
    )
    if not aggregations:
        return None
    first = dict(aggregations[0])
    first.pop("source_table", None)
    return first


def _column_available_for_aggregation(
    column: str,
    selected_columns: dict[str, dict],
    schema_loader=None,
    main_table: str = "",
) -> bool:
    if column == "*":
        return True
    selected_cols = {
        c
        for roles in selected_columns.values()
        for role_name in ("aggregate", "select", "group_by", "filter")
        for c in (roles.get(role_name, []) or [])
    }
    if column in selected_cols:
        return True
    if schema_loader is None or "." not in main_table:
        return False
    schema_name, table_name = main_table.split(".", 1)
    cols_df = schema_loader.get_table_columns(schema_name, table_name)
    if cols_df.empty:
        return False
    return bool((cols_df["column_name"].astype(str).str.lower() == column.lower()).any())


def _build_aggregation_result(
    func: str,
    col: str,
    *,
    distinct: bool = False,
    source_table: str = "",
) -> dict[str, str | bool]:
    alias = "count_all" if func == "COUNT" and col == "*" else f"{func.lower()}_{col}"
    result: dict[str, str | bool] = {"function": func, "column": col, "alias": alias}
    if distinct and col != "*":
        result["distinct"] = True
    if source_table:
        result["source_table"] = source_table
    return result


def _compute_aggregations(
    intent: dict,
    selected_columns: dict[str, dict],
    *,
    user_hints: dict | None = None,
    schema_loader=None,
    semantic_frame: dict | None = None,
    strategy: str = "",
    main_table: str = "",
) -> list[dict]:
    hint = (intent.get("aggregation_hint") or "").lower().strip()
    func = _HINT_TO_FUNC.get(hint)
    if not func:
        return []

    user_hints = user_hints or {}
    explicit_overrides = list(user_hints.get("aggregation_preferences_list") or [])
    legacy_override = (user_hints.get("aggregation_preferences") or {}) if isinstance(user_hints, dict) else {}
    if legacy_override and legacy_override not in explicit_overrides:
        explicit_overrides.insert(0, legacy_override)

    results: list[dict] = []
    seen: set[tuple[str, str, bool]] = set()
    preserve_distinct = False

    if hint == "count" and explicit_overrides:
        for override in explicit_overrides:
            override_func = str(override.get("function") or "").strip().lower()
            col = str(override.get("column") or "").strip()
            if override_func not in {"", "count"} or not col:
                continue
            if not _column_available_for_aggregation(col, selected_columns, schema_loader=schema_loader, main_table=main_table):
                continue
            distinct = bool(override.get("distinct")) and col != "*"
            preserve_distinct = preserve_distinct or distinct
            key = ("COUNT", col, distinct)
            if key in seen:
                continue
            seen.add(key)
            results.append(_build_aggregation_result("COUNT", col, distinct=distinct, source_table=main_table))
        if results:
            if strategy == "simple_select" and not preserve_distinct:
                for item in results:
                    item.pop("distinct", None)
            return results

    for table_key, roles in selected_columns.items():
        agg_cols = list(roles.get("aggregate", []) or [])
        if not agg_cols:
            continue
        col = agg_cols[0]
        distinct = False
        if hint == "count" and (
            col.lower().endswith("_count_distinct")
            or _count_column_should_be_distinct(
                selected_columns,
                col,
                schema_loader=schema_loader,
                semantic_frame=semantic_frame,
                strategy=strategy,
            )
        ):
            distinct = True
        key = (func, col, distinct)
        if key in seen:
            continue
        seen.add(key)
        results.append(_build_aggregation_result(func, col, distinct=distinct, source_table=table_key))
        break

    if not results and hint == "count":
        results.append(_build_aggregation_result("COUNT", "*", distinct=False, source_table=main_table))

    if strategy == "simple_select" and func == "COUNT" and not preserve_distinct:
        for item in results:
            item.pop("distinct", None)

    return results


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
    user_input: str = "",
) -> list[str]:
    """Сформировать WHERE-условия из структурированных данных в intent.

    Обрабатывает:
    1. date_filters: {"from": "2024-01-01", "to": null} → дата-диапазон
    2. filter_conditions: [{"column_hint": "region", "operator": "=", "value": "Москва"}]
       → WHERE region = 'Москва' (если column_hint совпадает с filter-колонкой)
    """
    conditions: list[str] = []

    # 1. Дата-диапазон
    date_filters = dict(intent.get("date_filters") or {})
    if not date_filters.get("from") and not date_filters.get("to"):
        date_filters.update(_derive_date_filters_from_text(user_input))
    date_from = date_filters.get("from")
    date_to   = date_filters.get("to")

    if date_from or date_to:
        date_col = _find_date_column(selected_columns)
        if date_col:
            if date_from and date_from != "NEEDS_YEAR":
                conditions.append(f"{date_col} >= '{date_from}'::date")
            elif date_from == "NEEDS_YEAR":
                # Маркер: год не указан — добавляем placeholder вместо реального условия.
                # sql_planner перехватит этот маркер и прервёт pipeline с clarification.
                conditions.append("NEEDS_YEAR")
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


def _derive_date_filters_from_text(user_input: str) -> dict[str, str | None]:
    """Вытащить простой месячный диапазон из текста пользователя.

    Если найден месяц БЕЗ года — возвращает маркер NEEDS_YEAR, чтобы
    sql_planner мог прервать pipeline и запросить уточнение у пользователя.
    """
    q = (user_input or "").lower()
    # Проверяем наличие месяца
    month = None
    for stem, num in _RU_MONTHS.items():
        if stem in q:
            month = num
            break

    if month is None:
        return {"from": None, "to": None}

    year_match = re.search(r"\b(20\d{2})\b", q)
    if not year_match:
        short_year_match = re.search(
            r"(?:уточнение пользователя:\s*|\b)(\d{2})\b",
            q,
        )
        if short_year_match:
            year = 2000 + int(short_year_match.group(1))
            next_year = year + 1 if month == 12 else year
            next_month = 1 if month == 12 else month + 1
            return {
                "from": f"{year:04d}-{month:02d}-01",
                "to": f"{next_year:04d}-{next_month:02d}-01",
            }

    if not year_match:
        # Месяц есть, года нет → маркер для запроса уточнения
        return {"from": "NEEDS_YEAR", "to": None}

    year = int(year_match.group(1))
    next_year = year + 1 if month == 12 else year
    next_month = 1 if month == 12 else month + 1
    return {
        "from": f"{year:04d}-{month:02d}-01",
        "to": f"{next_year:04d}-{next_month:02d}-01",
    }


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

def _is_date_axis_join(join_spec: list[dict]) -> tuple[bool, str]:
    """Определить, является ли JOIN осевым по дате (а не по PK).

    Returns:
        (is_date_axis, axis_column_name)
    """
    for jk in join_spec:
        strategy = str(jk.get("strategy", "")).lower()
        left_col = jk.get("left", "").rsplit(".", 1)[-1].lower()
        right_col = jk.get("right", "").rsplit(".", 1)[-1].lower()
        # Явный указатель от пользователя с date-hint
        if strategy == "explicit_user":
            for col in (left_col, right_col):
                if any(col == suf.lstrip("_") or col.endswith(suf) for suf in _DATE_AXIS_SUFFIXES):
                    return True, col
        # Авто-обнаружение: оба ключа — date-колонки
        left_is_date = any(
            left_col == suf.lstrip("_") or left_col.endswith(suf)
            for suf in _DATE_AXIS_SUFFIXES
        )
        right_is_date = any(
            right_col == suf.lstrip("_") or right_col.endswith(suf)
            for suf in _DATE_AXIS_SUFFIXES
        )
        if left_is_date and right_is_date:
            return True, left_col
    return False, ""


def build_blueprint(
    intent: dict,
    selected_columns: dict[str, dict],
    join_spec: list[dict],
    table_types: dict[str, str],
    join_analysis_data: dict,
    user_input: str = "",
    user_hints: dict | None = None,
    schema_loader=None,
    semantic_frame: dict | None = None,
    user_filter_choices: dict[str, str] | None = None,
    rejected_filter_choices: dict[str, list[str]] | None = None,
    count_identifier_resolver=None,
    filter_tiebreaker=None,
    filter_specs: list[dict] | None = None,
) -> dict:
    """Построить SQL Blueprint детерминированно, без LLM.

    Args:
        intent: Распознанный интент (из intent_classifier).
        selected_columns: Роли колонок по таблицам (из column_selector).
        join_spec: JOIN-спецификация (из column_selector).
        table_types: Тип каждой таблицы — fact/dim/ref/unknown.
        join_analysis_data: Результаты join_analysis (для совместимости).
        user_input: Текст запроса (для derive date filters).
        user_hints: Подсказки из hint_extractor (having_hints используются здесь).
        schema_loader: SchemaLoader для подбора unit_col в HAVING.
        count_identifier_resolver: Опциональный callable(main_table, pk_candidates, user_input) -> str | None,
            который спрашивает LLM, какая колонка — identifier сущности для
            COUNT(DISTINCT). Срабатывает ТОЛЬКО при составном PK (≥2 кандидатов)
            в safety-net пути. При None/невалидном ответе — fallback на
            детерминированный `_choose_count_identifier_column`.

    Returns:
        dict совместимый с форматом sql_blueprint из AgentState.
    """
    strategy, main_table = _determine_strategy(table_types, join_spec)
    aggregations = _compute_aggregations(
        intent,
        selected_columns,
        user_hints=user_hints,
        schema_loader=schema_loader,
        semantic_frame=semantic_frame,
        strategy=strategy,
        main_table=main_table,
    )
    aggregation = aggregations[0] if aggregations else None

    # --- Блок C: date-aligned JOIN (axis-join по дате) ---
    # Если join_spec указывает на date-колонки — это "аналитический JOIN по оси времени":
    # обе таблицы агрегируются независимо по дате, затем соединяются.
    # Форсируем fact_fact_join и добавляем флаг join_by_axis.
    _is_axis, _axis_col = _is_date_axis_join(join_spec)
    join_by_axis = False
    axis_column = ""
    if _is_axis and len(selected_columns) >= 2:
        strategy = "fact_fact_join"
        join_by_axis = True
        axis_column = _axis_col
        # Убираем "main_table" в пользу первой таблицы со select-агрегацией
        _agg_tables = [t for t, r in selected_columns.items() if r.get("aggregate")]
        if _agg_tables:
            main_table = _agg_tables[0]
        logger.info(
            "DeterministicPlanner: date-aligned JOIN по оси '%s' → fact_fact_join",
            axis_column,
        )

    if (
        str((intent or {}).get("aggregation_hint") or "").lower().strip() == "count"
        and (semantic_frame or {}).get("requires_single_entity_count")
        and schema_loader is not None
        and main_table
        and not ((user_hints or {}).get("aggregation_preferences") or {}).get("force_count_star")
    ):
        main_roles = selected_columns.setdefault(main_table, {})
        suspicious_group_by = list(main_roles.get("group_by", []) or [])
        current_agg_cols = list(main_roles.get("aggregate", []) or [])
        suspicious_agg_col = next(
            (
                col for col in current_agg_cols
                if _is_time_axis_count_column(main_table, col, schema_loader=schema_loader)
            ),
            None,
        )
        fallback_col = None
        force_count_star = strategy == "simple_select" and (
            main_roles.get("aggregate") == ["*"] or suspicious_agg_col is not None
        )
        if (main_roles.get("aggregate") == ["*"] or suspicious_group_by or suspicious_agg_col) and not force_count_star:
            # 1. LLM-резолвер при составном PK — срабатывает только если
            #    предоставлен callback И в таблице ≥2 PK-кандидата.
            if count_identifier_resolver is not None:
                pk_candidates = _list_pk_candidates(main_table, schema_loader)
                if len(pk_candidates) >= 2:
                    try:
                        llm_choice = count_identifier_resolver(
                            main_table=main_table,
                            pk_candidates=pk_candidates,
                            user_input=user_input,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "count_identifier_resolver упал: %s — fallback на detreministic",
                            exc,
                        )
                        llm_choice = None
                    if llm_choice and llm_choice in pk_candidates:
                        fallback_col = llm_choice
                        logger.info(
                            "DeterministicPlanner: COUNT identifier выбран LLM → %s.%s",
                            main_table, fallback_col,
                        )
            # 2. Fallback на детерминистику.
            if fallback_col is None:
                fallback_col = _choose_count_identifier_column(
                    main_table,
                    schema_loader,
                    semantic_frame=semantic_frame,
                    user_input=user_input,
                )
        if force_count_star:
            main_roles["aggregate"] = ["*"]
            main_roles.pop("group_by", None)
            aggregations = _compute_aggregations(
                intent,
                selected_columns,
                user_hints=user_hints,
                schema_loader=schema_loader,
                semantic_frame=semantic_frame,
                strategy=strategy,
                main_table=main_table,
            )
            aggregation = aggregations[0] if aggregations else None
            logger.info(
                "DeterministicPlanner: single-entity safety net → %s.COUNT(*)",
                main_table,
            )
        elif fallback_col:
            main_roles["aggregate"] = [fallback_col]
            main_roles["select"] = [fallback_col]
            main_roles.pop("group_by", None)
            aggregations = _compute_aggregations(
                intent,
                selected_columns,
                user_hints=user_hints,
                schema_loader=schema_loader,
                semantic_frame=semantic_frame,
                strategy=strategy,
                main_table=main_table,
            )
            aggregation = aggregations[0] if aggregations else None
            logger.info(
                "DeterministicPlanner: single-entity safety net → %s.%s",
                main_table, fallback_col,
            )

    # Если одна таблица даёт метрику, а вторая только атрибуты для группировки/селекта,
    # рассматриваем вторую как dimension-like источник атрибута, даже если её тип unknown/fact.
    if not join_by_axis and len(selected_columns) == 2:
        aggregate_tables = [t for t, roles in selected_columns.items() if roles.get("aggregate")]
        attribute_tables = [
            t for t, roles in selected_columns.items()
            if roles.get("group_by") or [c for c in roles.get("select", []) if c not in roles.get("aggregate", [])]
        ]
        if len(aggregate_tables) == 1 and strategy in {"fact_fact_join", "fact_dim_join"}:
            main_table = aggregate_tables[0]
            other_tables = [t for t in selected_columns if t != main_table]
            if other_tables and other_tables[0] in attribute_tables:
                strategy = "fact_dim_join"

    group_by    = _compute_group_by(selected_columns, aggregation)

    # Применяем DATE_TRUNC из time_granularity (если задана)
    _time_gran = (user_hints or {}).get("time_granularity")
    if _time_gran:
        group_by = _apply_time_granularity(group_by, selected_columns, _time_gran, schema_loader)
        logger.info("DeterministicPlanner: time_granularity='%s' → group_by=%s", _time_gran, group_by)

    base_where_conditions = _compute_where_from_intent(intent, selected_columns, user_input=user_input)
    where_resolution = resolve_where(
        user_input=user_input,
        intent=intent,
        selected_columns=selected_columns,
        selected_tables=list(selected_columns.keys()),
        schema_loader=schema_loader,
        semantic_frame=semantic_frame,
        base_conditions=base_where_conditions,
        user_filter_choices=user_filter_choices,
        rejected_filter_choices=rejected_filter_choices,
        filter_tiebreaker=filter_tiebreaker,
        filter_specs=filter_specs,
    )
    where_conditions = where_resolution.get("conditions", base_where_conditions)

    # --- Блок D: required_output enforcement ---
    # Колонки из required_output (обязательные в SELECT) добавляем в group_by если их нет.
    required_output: list[str] = list(intent.get("required_output") or [])
    if required_output and aggregation:
        _gb_lower = {c.lower() for c in group_by}
        for _req in required_output:
            _req_norm = _req.lower().strip()
            # Ищем в selected_columns колонку, семантически совпадающую с required_output
            for _tbl, _roles in selected_columns.items():
                _all_cols = (
                    _roles.get("select", [])
                    + _roles.get("group_by", [])
                    + _roles.get("filter", [])
                )
                for _col in _all_cols:
                    _col_lower = _col.lower()
                    if _req_norm in _col_lower or _col_lower in _req_norm:
                        if _col_lower not in _gb_lower:
                            group_by.append(_col)
                            _gb_lower.add(_col_lower)
                            logger.info(
                                "DeterministicPlanner: required_output %r → добавляем %s в group_by",
                                _req, _col,
                            )
                        break

    # Filter-only колонки не должны автоматически утекать в GROUP BY.
    # Добавляем их только если пользователь действительно просил измерение
    # в required_output/output_dimensions.
    if aggregation:
        where_str = " ".join(where_conditions).lower()
        seen_gb: set[str] = set(group_by)
        requested_dimensions = {
            str(v).strip().lower()
            for v in list((semantic_frame or {}).get("output_dimensions") or [])
            + list(required_output)
            if str(v).strip()
        }
        for _table, roles in selected_columns.items():
            for col in roles.get("filter", []):
                col_lower = col.lower()
                should_group = any(
                    dim in col_lower or col_lower in dim
                    for dim in requested_dimensions
                )
                if not should_group and "date" in requested_dimensions:
                    should_group = any(
                        col_lower.endswith(suf)
                        for suf in ("_dt", "_date", "_dttm", "_timestamp", "_ts")
                    ) or "date" in col_lower or "dttm" in col_lower or "timestamp" in col_lower
                if should_group and col not in seen_gb and col_lower not in where_str:
                    group_by.append(col)
                    seen_gb.add(col)

    if (
        aggregation
        and (semantic_frame or {}).get("requires_single_entity_count")
        and not list((semantic_frame or {}).get("output_dimensions") or [])
    ):
        if group_by:
            logger.info(
                "DeterministicPlanner: single-entity count → очищаю group_by=%s",
                group_by,
            )
        group_by = []

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

    # LIMIT: только если явно задан в intent; без умолчания
    limit: int | None = intent.get("limit") or None

    # HAVING-условия из user_hints (пост-агрегатный фильтр).
    having_clauses = _compute_having(
        user_hints=user_hints,
        selected_columns=selected_columns,
        schema_loader=schema_loader,
        main_table=main_table,
    )

    legacy_aggregation = dict(aggregation) if aggregation else None
    if legacy_aggregation:
        legacy_aggregation.pop("source_table", None)

    blueprint = {
        "strategy": strategy,
        "main_table": main_table,
        "cte_needed": cte_needed,
        "subquery_for": subquery_for,
        "where_conditions": where_conditions,
        "aggregation": legacy_aggregation,
        "aggregations": aggregations,
        "group_by": group_by,
        "having": having_clauses,
        "order_by": order_by,
        "limit": limit,
        "join_by_axis": join_by_axis,
        "axis_column": axis_column,
        "required_output": required_output,
        "where_resolution": where_resolution,
        "notes": f"[deterministic] strategy={strategy}, tables={list(table_types.keys())}",
    }

    logger.info(
        "DeterministicPlanner: strategy=%s, main=%s, cte=%s, group_by=%s, agg=%s, where_rules=%s",
        strategy, main_table, cte_needed, group_by,
        aggregation.get("function") if aggregation else None,
        where_resolution.get("applied_rules", []),
    )

    return blueprint
