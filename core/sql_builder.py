"""Шаблонный SQL-генератор (без LLM).

Строит SQL из структурированных данных AgentState по стратегии из blueprint.
Возвращает готовый SQL-строку или None если шаблон не может покрыть случай
(→ fallback на LLM sql_writer).

Покрываемые стратегии:
- simple_select
- fact_dim_join   (fact → JOIN dim через DISTINCT ON если safe=False)
- dim_fact_join   (dim → JOIN aggregated fact subquery)
- fact_fact_join  (два CTE + JOIN)
- dim_dim_join    (два CTE с DISTINCT ON + JOIN)
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Сопоставление hint → SQL-функция (для проверки)
_AGG_FUNCS = frozenset({"SUM", "COUNT", "AVG", "MIN", "MAX", "STDDEV"})

# Шаблон для генерации короткого алиаса таблицы из schema.table
_ALIAS_RE = re.compile(r"(?:.*\.)?([a-zA-Z][a-zA-Z0-9_]*)")
_GENERIC_TABLE_TOKENS = {
    "schema", "data", "dwh", "fact", "dim", "ref", "table", "tbl", "mart", "dm", "uzp"
}
_GENERIC_COL_TOKENS = {"id", "name", "dt", "date", "dttm", "amt", "qty", "num", "code"}
_TOKEN_ABBREVIATIONS = {"segment": "seg", "region": "reg"}


def _short_alias(full_name: str, used: set[str]) -> str:
    """Сгенерировать короткий, уникальный алиас для таблицы.

    dm.fact_outflow → fo; dm.clients → c; если занят — c2, c3, ...
    """
    m = _ALIAS_RE.match(full_name)
    table = m.group(1) if m else full_name
    # Берём первые буквы каждого слова в snake_case
    words = [w for w in table.split("_") if w]
    candidate = "".join(w[0] for w in words[:3]) if words else table[:2]
    candidate = candidate.lower()
    if candidate not in used:
        used.add(candidate)
        return candidate
    for i in range(2, 10):
        c2 = f"{candidate}{i}"
        if c2 not in used:
            used.add(c2)
            return c2
    used.add(candidate + "_x")
    return candidate + "_x"


def _significant_tokens(name: str, stopwords: set[str]) -> list[str]:
    parts = [p.lower() for p in re.split(r"[_\W]+", name) if p]
    return [p for p in parts if p not in stopwords]


def _semantic_alias(
    full_name: str,
    roles: dict[str, list[str]],
    aggregation: dict | None,
    used: set[str],
) -> str:
    tokens: list[str] = []
    agg_col = aggregation.get("column") if aggregation else None
    if agg_col and agg_col != "*":
        tokens.extend(_significant_tokens(agg_col, _GENERIC_COL_TOKENS))
    for col in roles.get("select", []) + roles.get("group_by", []):
        tokens.extend(_significant_tokens(col, _GENERIC_COL_TOKENS))
    tokens.extend(_significant_tokens(full_name.split(".")[-1], _GENERIC_TABLE_TOKENS))

    for token in tokens:
        candidate = token[0]
        if candidate.isalpha() and candidate not in used:
            used.add(candidate)
            return candidate
    return _short_alias(full_name, used)


def _derive_cte_alias(table_full: str, roles: dict[str, list[str]]) -> str:
    table_tokens = _significant_tokens(table_full.split(".")[-1], _GENERIC_TABLE_TOKENS)
    attr_tokens: list[str] = []
    for col in roles.get("select", []):
        attr_tokens.extend(_significant_tokens(col, _GENERIC_COL_TOKENS))
    left = table_tokens[0] if table_tokens else "src"
    right = attr_tokens[0] if attr_tokens else "attr"
    right = _TOKEN_ABBREVIATIONS.get(right, right)
    return f"{left}_{right}"


def _build_select_items(
    selected_columns: dict[str, dict],
    table_alias_map: dict[str, str],
    aggregation: dict | None,
) -> list[str]:
    """Сформировать список выражений для SELECT-клаузы.

    Возвращает список строк вида: "t.region", "SUM(t.amount) AS sum_amount".
    """
    items: list[str] = []
    seen_cols: set[str] = set()

    agg_col = aggregation.get("column") if aggregation else None
    agg_func = aggregation.get("function") if aggregation else None
    agg_alias = aggregation.get("alias") if aggregation else None

    agg_distinct = aggregation.get("distinct", False) if aggregation else False

    for table, roles in selected_columns.items():
        alias = table_alias_map.get(table, "t")
        agg_set = set(roles.get("aggregate", []))
        for col in roles.get("select", []):
            if col in seen_cols:
                continue
            seen_cols.add(col)
            if col == agg_col and agg_func:
                # Основная агрегируемая колонка
                if col == "*":
                    items.append(f"{agg_func}(*) AS {agg_alias}")
                else:
                    d = "DISTINCT " if agg_distinct else ""
                    items.append(f"{agg_func}({d}{alias}.{col}) AS {agg_alias}")
            elif col in agg_set and agg_func:
                # Дополнительная агрегируемая колонка (напр. второй PK при COUNT DISTINCT).
                # Агрегируем её вместо bare-select, чтобы не потребовался GROUP BY.
                d = "DISTINCT " if agg_distinct else ""
                items.append(f"{agg_func}({d}{alias}.{col}) AS {agg_func.lower()}_{col}")
            else:
                items.append(f"{alias}.{col}")

        # aggregate-роль (если не в select)
        for col in roles.get("aggregate", []):
            if col in seen_cols:
                continue
            seen_cols.add(col)
            if agg_func:
                if col == "*":
                    items.append(f"{agg_func}(*) AS {agg_alias or 'agg_val'}")
                else:
                    d = "DISTINCT " if agg_distinct else ""
                    items.append(
                        f"{agg_func}({d}{alias}.{col}) AS "
                        f"{agg_alias if col == agg_col else f'{agg_func.lower()}_{col}'}"
                    )

    if not items:
        # Совсем пусто — возвращаем count(*)
        items = ["COUNT(*) AS cnt"]

    return items


def _build_where_clause(
    where_conditions: list[str],
    alias: str = "",
) -> str:
    """Сформировать WHERE-клаузу из списка условий."""
    if not where_conditions:
        return ""
    return "WHERE " + "\n  AND ".join(where_conditions)


def _build_group_by(group_by: list[str], alias: str = "") -> str:
    """Сформировать GROUP BY клаузу."""
    if not group_by:
        return ""
    cols = [f"{alias}.{c}" if alias and "." not in c else c for c in group_by]
    return "GROUP BY " + ", ".join(cols)


def _build_order_by(order_by: str | None) -> str:
    if not order_by:
        return ""
    return f"ORDER BY {order_by}"


def _build_limit(limit: int | None) -> str:
    if not limit:
        return ""
    return f"LIMIT {limit}"


# ---------------------------------------------------------------------------
# Стратегии
# ---------------------------------------------------------------------------

def _build_simple_select(
    selected_columns: dict[str, dict],
    blueprint: dict,
) -> str:
    """simple_select: одна таблица, агрегация + фильтры."""
    main_table = blueprint.get("main_table", "")
    if not main_table:
        tables = list(selected_columns.keys())
        main_table = tables[0] if tables else ""

    used: set[str] = set()
    alias = _short_alias(main_table, used)
    alias_map = {main_table: alias}

    aggregation = blueprint.get("aggregation")
    select_items = _build_select_items(selected_columns, alias_map, aggregation)

    group_by_cols = blueprint.get("group_by", [])
    where_clause = _build_where_clause(blueprint.get("where_conditions", []))
    group_by_clause = _build_group_by(group_by_cols, alias)
    order_by_clause = _build_order_by(blueprint.get("order_by"))
    limit_clause = _build_limit(blueprint.get("limit"))

    parts = [
        f"SELECT {', '.join(select_items)}",
        f"FROM {main_table} {alias}",
    ]
    if where_clause:
        parts.append(where_clause)
    if group_by_clause:
        parts.append(group_by_clause)
    if order_by_clause:
        parts.append(order_by_clause)
    if limit_clause:
        parts.append(limit_clause)

    return "\n".join(parts)


def _resolve_join_key(join_spec: list[dict], left_table: str, right_table: str) -> tuple[str, str]:
    """Найти первую пару колонок для JOIN между двумя таблицами из join_spec.

    Returns:
        (left_col, right_col) или ("", "") если не найдено.
    """
    pairs = _resolve_join_keys_composite(join_spec, left_table, right_table)
    return pairs[0] if pairs else ("", "")


def _resolve_join_keys_composite(
    join_spec: list[dict], left_table: str, right_table: str
) -> list[tuple[str, str]]:
    """Вернуть ВСЕ join-пары для данной пары таблиц (поддержка составного PK).

    Returns:
        Список (left_col, right_col). Пустой список если пар не найдено.
    """
    pairs: list[tuple[str, str]] = []
    for jk in join_spec:
        left_full = jk.get("left", "")
        right_full = jk.get("right", "")
        left_tbl = ".".join(left_full.split(".")[:2]) if left_full.count(".") >= 2 else ""
        right_tbl = ".".join(right_full.split(".")[:2]) if right_full.count(".") >= 2 else ""
        left_col = left_full.rsplit(".", 1)[-1] if left_full else ""
        right_col = right_full.rsplit(".", 1)[-1] if right_full else ""

        if left_tbl == left_table and right_tbl == right_table:
            pairs.append((left_col, right_col))
        elif left_tbl == right_table and right_tbl == left_table:
            pairs.append((right_col, left_col))  # swap

    return pairs


def _get_select_cols(roles: dict, exclude_col: str = "") -> list[str]:
    """Собрать список select-колонок из ролей, исключая join-ключ если нужно."""
    cols = list(roles.get("select", []))
    # Добавляем из других ролей если не перечислены в select
    for role in ("filter", "group_by"):
        for c in roles.get(role, []):
            if c not in cols:
                cols.append(c)
    return cols


def _build_fact_dim_join(
    selected_columns: dict[str, dict],
    join_spec: list[dict],
    blueprint: dict,
    table_types: dict[str, str],
) -> str | None:
    """fact_dim_join: факт-таблица + справочник.

    Паттерн:
    - safe=True → прямой JOIN
    - safe=False → DISTINCT ON в подзапросе для справочника
    """
    main_table = blueprint.get("main_table", "")
    if not main_table:
        return None

    # Определяем dim-таблицу
    dim_tables = [t for t in selected_columns if t != main_table]
    if not dim_tables:
        return None
    dim_table = dim_tables[0]

    # JOIN-ключи (все пары для составного PK)
    join_pairs = _resolve_join_keys_composite(join_spec, main_table, dim_table)
    if not join_pairs:
        # Нет join_spec — не можем построить JOIN без LLM
        return None
    # Первичная пара (для обратной совместимости)
    join_key_fact, join_key_dim = join_pairs[0]

    # Безопасность: берём из join_spec
    is_safe = any(
        j.get("safe", False)
        for j in join_spec
        if main_table in j.get("left", "") or main_table in j.get("right", "")
    )
    use_full_refs = is_safe

    aggregation = blueprint.get("aggregation")

    # SELECT из факт-таблицы
    fact_roles = selected_columns.get(main_table, {})
    dim_roles = selected_columns.get(dim_table, {})

    used: set[str] = set()
    f_alias = _semantic_alias(main_table, fact_roles, aggregation, used)
    d_alias = _semantic_alias(dim_table, dim_roles, None, used)
    initial_d_alias = d_alias

    # Строим SELECT-items
    select_items: list[str] = []
    seen: set[str] = set()

    # Колонки факт-таблицы
    agg_col = aggregation.get("column") if aggregation else None
    agg_func = aggregation.get("function") if aggregation else None
    agg_alias = aggregation.get("alias") if aggregation else None
    agg_distinct = aggregation.get("distinct", False) if aggregation else False
    fact_agg_set = set(fact_roles.get("aggregate", []))

    for col in (fact_roles.get("select", []) + fact_roles.get("group_by", [])):
        if col in seen:
            continue
        seen.add(col)
        if col == agg_col and agg_func:
            d = "DISTINCT " if agg_distinct else ""
            source = col if use_full_refs else f"{f_alias}.{col}"
            select_items.append(f"{agg_func}({d}{source}) AS {agg_alias}")
        elif col in fact_agg_set and agg_func:
            d = "DISTINCT " if agg_distinct else ""
            source = col if use_full_refs else f"{f_alias}.{col}"
            select_items.append(f"{agg_func}({d}{source}) AS {agg_func.lower()}_{col}")
        else:
            select_items.append(col if use_full_refs else f"{f_alias}.{col}")

    for col in fact_roles.get("aggregate", []):
        if col in seen:
            continue
        seen.add(col)
        if agg_func:
            d = "DISTINCT " if agg_distinct else ""
            source = col if use_full_refs else f"{f_alias}.{col}"
            select_items.append(f"{agg_func}({d}{source}) AS {agg_alias or f'{agg_func.lower()}_{col}'}")

    # Колонки из справочника (исключаем все dim join-ключи)
    dim_join_cols = {p[1] for p in join_pairs}
    for col in dim_roles.get("select", []):
        if col in dim_join_cols or col in seen:
            continue
        seen.add(col)
        select_items.append(col if use_full_refs else f"{d_alias}.{col}")

    if not select_items:
        select_items = [f"COUNT(*) AS cnt"]

    # WHERE
    where_clause = _build_where_clause(blueprint.get("where_conditions", []))

    group_by_cols = blueprint.get("group_by", [])
    order_by_clause = _build_order_by(blueprint.get("order_by"))
    limit_clause = _build_limit(blueprint.get("limit"))

    # Формируем dim-подзапрос или прямой JOIN
    # dim_select_cols: все PK-ключи + остальные нужные колонки справочника
    all_dim_join_cols = list(dict.fromkeys(p[1] for p in join_pairs))  # порядок сохранён
    dim_extra_cols = [c for c in dim_roles.get("select", []) if c not in dim_join_cols]
    dim_select_cols = all_dim_join_cols + dim_extra_cols

    cte_sql = ""
    from_sql = f"FROM {main_table}" if use_full_refs else f"FROM {main_table} {f_alias}"

    if is_safe:
        main_ref = main_table.split(".")[-1]
        dim_ref = dim_table.split(".")[-1]
        if len(join_pairs) > 1:
            on_conditions = " AND ".join(
                f"{dim_ref}.{p[1]} = {main_ref}.{p[0]}" for p in join_pairs
            )
        else:
            on_conditions = f"{dim_ref}.{join_key_dim} = {main_ref}.{join_key_fact}"
        join_sql = f"JOIN {dim_table} ON {on_conditions}"
    elif len(join_pairs) == 1 and dim_extra_cols:
        cte_alias = _derive_cte_alias(dim_table, dim_roles)
        cte_alias_short = cte_alias[0].lower()
        if cte_alias_short in used:
            d_alias = _short_alias(cte_alias, used)
        else:
            used.add(cte_alias_short)
            d_alias = cte_alias_short
        cte_sql = (
            f"WITH {cte_alias} AS (\n"
            f"    SELECT DISTINCT ON ({join_key_dim}) {', '.join(dim_select_cols)}\n"
            f"    FROM {dim_table}\n"
            f"    ORDER BY {join_key_dim}\n"
            f")\n"
        )
        join_sql = f"JOIN {cte_alias} {d_alias} ON {d_alias}.{join_key_dim} = {f_alias}.{join_key_fact}"
        if d_alias != initial_d_alias:
            select_items = [item.replace(f"{initial_d_alias}.", f"{d_alias}.") for item in select_items]
    elif len(join_pairs) > 1:
        # Составной DISTINCT ON по всем ключам PK
        distinct_on_str = ", ".join(p[1] for p in join_pairs)
        order_by_str = ", ".join(p[1] for p in join_pairs)
        dim_cols_str = ", ".join(dim_select_cols) if dim_select_cols else distinct_on_str
        on_conditions = " AND ".join(
            f"{d_alias}.{p[1]} = {f_alias}.{p[0]}" for p in join_pairs
        )
        join_sql = (
            f"JOIN (\n"
            f"    SELECT DISTINCT ON ({distinct_on_str}) {dim_cols_str}\n"
            f"    FROM {dim_table}\n"
            f"    ORDER BY {order_by_str}\n"
            f") {d_alias} ON {on_conditions}"
        )
    else:
        # Одиночный ключ — DISTINCT ON по нему
        dim_cols_str = ", ".join(dim_select_cols) if dim_select_cols else join_key_dim
        join_sql = (
            f"JOIN (\n"
            f"    SELECT DISTINCT ON ({join_key_dim}) {dim_cols_str}\n"
            f"    FROM {dim_table}\n"
            f"    ORDER BY {join_key_dim}\n"
            f") {d_alias} ON {d_alias}.{join_key_dim} = {f_alias}.{join_key_fact}"
        )

    gb_items: list[str] = []
    fact_gb_items: list[str] = []
    dim_gb_items: list[str] = []
    other_gb_items: list[str] = []
    for col in group_by_cols:
        if col in (fact_roles.get("select", []) + fact_roles.get("group_by", [])):
            fact_gb_items.append(col if use_full_refs else f"{f_alias}.{col}")
        elif col in dim_roles.get("select", []):
            dim_gb_items.append(col if use_full_refs else f"{d_alias}.{col}")
        else:
            other_gb_items.append(col)
    gb_items = fact_gb_items + dim_gb_items + other_gb_items
    group_by_clause = ("GROUP BY " + ", ".join(gb_items)) if gb_items else ""

    parts = [
        cte_sql.rstrip() if cte_sql else "",
        f"SELECT {', '.join(select_items)}",
        from_sql,
        join_sql,
    ]
    parts = [p for p in parts if p]
    if where_clause:
        parts.append(where_clause)
    if group_by_clause:
        parts.append(group_by_clause)
    if order_by_clause:
        parts.append(order_by_clause)
    if limit_clause:
        parts.append(limit_clause)

    return "\n".join(parts)


def _build_dim_fact_join(
    selected_columns: dict[str, dict],
    join_spec: list[dict],
    blueprint: dict,
    table_types: dict[str, str],
) -> str | None:
    """dim_fact_join: справочник + агрегированная факт-таблица в подзапросе."""
    main_table = blueprint.get("main_table", "")
    fact_tables = [t for t, v in table_types.items() if v == "fact"]
    dim_tables = [t for t, v in table_types.items() if v in ("dim", "ref")]

    if not fact_tables or not dim_tables:
        return None

    dim_table = dim_tables[0]
    fact_table = fact_tables[0]

    join_pairs = _resolve_join_keys_composite(join_spec, dim_table, fact_table)
    if not join_pairs:
        return None
    join_key_dim, join_key_fact = join_pairs[0]

    used: set[str] = set()
    d_alias = _short_alias(dim_table, used)

    aggregation = blueprint.get("aggregation")
    agg_func = aggregation.get("function") if aggregation else "COUNT"
    agg_col = aggregation.get("column") if aggregation else "*"
    agg_alias = aggregation.get("alias") if aggregation else "agg_val"

    fact_roles = selected_columns.get(fact_table, {})
    dim_roles = selected_columns.get(dim_table, {})

    # Dim SELECT — исключаем все join-ключи справочника
    dim_join_cols = {p[0] for p in join_pairs}
    dim_select_cols = list(dict.fromkeys(p[0] for p in join_pairs)) + [
        c for c in dim_roles.get("select", []) if c not in dim_join_cols
    ]
    dim_select_str = ", ".join(f"{d_alias}.{c}" for c in dim_select_cols)

    # Fact aggregation subquery — GROUP BY по всем join-ключам факта (составной)
    fact_join_cols = list(dict.fromkeys(p[1] for p in join_pairs))
    fact_group_by_str = ", ".join(fact_join_cols)
    fact_select_keys_str = ", ".join(fact_join_cols)
    fact_agg_col = "*" if agg_col == "*" else agg_col
    fact_subquery = (
        f"SELECT {fact_select_keys_str}, {agg_func}({fact_agg_col}) AS {agg_alias}\n"
        f"    FROM {fact_table}\n"
        f"    GROUP BY {fact_group_by_str}"
    )

    # ON-условие: составной JOIN по всем парам
    on_conditions = " AND ".join(
        f"agg.{p[1]} = {d_alias}.{p[0]}" for p in join_pairs
    )

    where_clause = _build_where_clause(blueprint.get("where_conditions", []))
    order_by_clause = _build_order_by(blueprint.get("order_by"))
    limit_clause = _build_limit(blueprint.get("limit"))

    parts = [
        f"SELECT {dim_select_str}, agg.{agg_alias}",
        f"FROM {dim_table} {d_alias}",
        f"JOIN (\n    {fact_subquery}\n) agg ON {on_conditions}",
    ]
    if where_clause:
        parts.append(where_clause)
    if order_by_clause:
        parts.append(order_by_clause)
    if limit_clause:
        parts.append(limit_clause)

    return "\n".join(parts)


def _build_fact_fact_join(
    selected_columns: dict[str, dict],
    join_spec: list[dict],
    blueprint: dict,
) -> str | None:
    """fact_fact_join: два CTE с агрегацией + финальный JOIN.

    Поддерживает два режима:
    - join_by_axis=True: JOIN по временной оси (report_dt и т.д.) — дата не PK,
      обе таблицы агрегируются независимо по axis_column, затем JOIN по нему.
    - Обычный режим: JOIN по join_spec-ключу.
    """
    tables = list(selected_columns.keys())
    if len(tables) < 2:
        return None

    t1, t2 = tables[0], tables[1]

    join_by_axis: bool = bool(blueprint.get("join_by_axis"))
    axis_column: str = str(blueprint.get("axis_column") or "")

    if join_by_axis and axis_column:
        # --- Axis-join режим: GROUP BY axis_column, JOIN по нему ---
        join_key_1 = axis_column
        join_key_2 = axis_column
    else:
        join_key_1, join_key_2 = _resolve_join_key(join_spec, t1, t2)
        if not join_key_1 or not join_key_2:
            return None

    used: set[str] = set()
    a1 = _short_alias(t1, used)
    a2 = _short_alias(t2, used)
    cte1 = f"{a1}_agg"
    cte2 = f"{a2}_agg"

    aggregation = blueprint.get("aggregation")
    agg_func = aggregation.get("function") if aggregation else "COUNT"

    roles1 = selected_columns[t1]
    roles2 = selected_columns[t2]

    # Агрегации для каждой таблицы: ищем aggregate-роль, fallback COUNT(*)
    def _agg_for(roles: dict, suffix: str, key_col: str) -> str:
        agg_cols = [c for c in roles.get("aggregate", []) if c != key_col]
        if agg_cols:
            col = agg_cols[0]
            return f"{agg_func}({col}) AS {agg_func.lower()}_{col}"
        # Попробуем найти числовую метрику в select
        for col in roles.get("select", []):
            if col == key_col:
                continue
            return f"{agg_func}({col}) AS {agg_func.lower()}_{col}"
        return f"COUNT(*) AS cnt_{suffix}"

    agg1_expr = _agg_for(roles1, a1, join_key_1)
    agg2_expr = _agg_for(roles2, a2, join_key_2)

    # В axis-join режиме group_by включает axis_column + дополнительные измерения
    group_extra_1: list[str] = []
    group_extra_2: list[str] = []
    if join_by_axis:
        # Добавляем колонки group_by каждой таблицы (кроме axis и агрегируемых)
        agg1_col_name = agg1_expr.split(" AS ")[0].replace(f"{agg_func}(", "").rstrip(")")
        agg2_col_name = agg2_expr.split(" AS ")[0].replace(f"{agg_func}(", "").rstrip(")")
        for col in roles1.get("group_by", []) + roles1.get("select", []):
            if col not in (join_key_1, agg1_col_name) and col not in group_extra_1:
                group_extra_1.append(col)
        for col in roles2.get("group_by", []) + roles2.get("select", []):
            if col not in (join_key_2, agg2_col_name) and col not in group_extra_2:
                group_extra_2.append(col)

    gb1_cols = [join_key_1] + group_extra_1
    gb2_cols = [join_key_2] + group_extra_2
    gb1_str = ", ".join(gb1_cols)
    gb2_str = ", ".join(gb2_cols)

    sel1_extra = "".join(f", {c}" for c in group_extra_1)
    sel2_extra = "".join(f", {c}" for c in group_extra_2)

    where_clause = _build_where_clause(blueprint.get("where_conditions", []))
    cte1_where = f"\n    {where_clause}" if where_clause else ""
    cte2_where = f"\n    {where_clause}" if where_clause else ""

    cte1_sql = (
        f"{cte1} AS (\n"
        f"    SELECT {join_key_1}{sel1_extra}, {agg1_expr}\n"
        f"    FROM {t1}"
        f"{cte1_where}\n"
        f"    GROUP BY {gb1_str}\n"
        f")"
    )
    cte2_sql = (
        f"{cte2} AS (\n"
        f"    SELECT {join_key_2}{sel2_extra}, {agg2_expr}\n"
        f"    FROM {t2}"
        f"{cte2_where}\n"
        f"    GROUP BY {gb2_str}\n"
        f")"
    )

    # Финальный SELECT
    agg1_col = agg1_expr.split(" AS ")[-1]
    agg2_col = agg2_expr.split(" AS ")[-1]

    final_extra_1 = "".join(f", c1.{c}" for c in group_extra_1)
    final_extra_2 = "".join(f", c2.{c}" for c in group_extra_2)
    select_str = f"c1.{join_key_1}{final_extra_1}, c1.{agg1_col}{final_extra_2}, c2.{agg2_col}"

    limit_clause = _build_limit(blueprint.get("limit"))
    order_by_clause = _build_order_by(blueprint.get("order_by"))

    parts = [
        f"WITH {cte1_sql},\n{cte2_sql}",
        f"SELECT {select_str}",
        f"FROM {cte1} c1",
        f"JOIN {cte2} c2 ON c2.{join_key_2} = c1.{join_key_1}",
    ]
    if order_by_clause:
        parts.append(order_by_clause)
    if limit_clause:
        parts.append(limit_clause)

    return "\n".join(parts)


def _build_dim_dim_join(
    selected_columns: dict[str, dict],
    join_spec: list[dict],
    blueprint: dict,
) -> str | None:
    """dim_dim_join: два CTE с DISTINCT ON + JOIN."""
    tables = list(selected_columns.keys())
    if len(tables) < 2:
        return None

    t1, t2 = tables[0], tables[1]
    join_key_1, join_key_2 = _resolve_join_key(join_spec, t1, t2)
    if not join_key_1 or not join_key_2:
        return None

    used: set[str] = set()
    a1 = _short_alias(t1, used)
    a2 = _short_alias(t2, used)

    roles1 = selected_columns[t1]
    roles2 = selected_columns[t2]

    def _dim_cte(alias, table, join_key, roles):
        cols = [join_key] + [c for c in roles.get("select", []) if c != join_key]
        cols_str = ", ".join(cols)
        return (
            f"{alias}_cte AS (\n"
            f"    SELECT DISTINCT ON ({join_key}) {cols_str}\n"
            f"    FROM {table}\n"
            f"    ORDER BY {join_key}\n"
            f")"
        )

    cte1 = _dim_cte(a1, t1, join_key_1, roles1)
    cte2 = _dim_cte(a2, t2, join_key_2, roles2)
    cte1_name = f"{a1}_cte"
    cte2_name = f"{a2}_cte"

    # SELECT
    sel1 = [f"c1.{c}" for c in ([join_key_1] + [c for c in roles1.get("select", []) if c != join_key_1])]
    sel2 = [f"c2.{c}" for c in roles2.get("select", []) if c != join_key_2]
    select_str = ", ".join(sel1 + sel2) or "c1.*, c2.*"

    limit_clause = _build_limit(blueprint.get("limit"))
    order_by_clause = _build_order_by(blueprint.get("order_by"))

    parts = [
        f"WITH {cte1},\n{cte2}",
        f"SELECT {select_str}",
        f"FROM {cte1_name} c1",
        f"JOIN {cte2_name} c2 ON c2.{join_key_2} = c1.{join_key_1}",
    ]
    if order_by_clause:
        parts.append(order_by_clause)
    if limit_clause:
        parts.append(limit_clause)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Главная точка входа
# ---------------------------------------------------------------------------

class SqlBuilder:
    """Шаблонный SQL-генератор без LLM.

    Использование:
        builder = SqlBuilder()
        sql = builder.build(strategy, selected_columns, join_spec, blueprint, table_types)
        if sql:
            # Используем готовый SQL
        else:
            # Fallback на LLM sql_writer
    """

    def build(
        self,
        strategy: str,
        selected_columns: dict[str, dict],
        join_spec: list[dict],
        blueprint: dict,
        table_types: dict[str, str] | None = None,
    ) -> str | None:
        """Построить SQL по стратегии.

        Returns:
            Строка SQL или None если шаблон не покрывает данный случай.
        """
        if not selected_columns:
            logger.warning("SqlBuilder: selected_columns пуст — пропускаем шаблон")
            return None

        table_types = table_types or {}

        # Fallback-условия: когда шаблон не подходит
        # - Нетривиальные WHERE-литералы (фильтры по значениям, кроме дат)
        has_filter_cols = any(
            bool(roles.get("filter")) for roles in selected_columns.values()
        )
        has_date_where = any(
            "::date" in w or "::timestamp" in w
            for w in blueprint.get("where_conditions", [])
        )
        has_non_date_filters = has_filter_cols and not has_date_where and not blueprint.get("where_conditions")

        if has_non_date_filters:
            logger.info(
                "SqlBuilder: обнаружены filter-колонки без WHERE-условий — "
                "нужны литеральные значения, передаём LLM"
            )
            return None

        try:
            if strategy == "simple_select":
                return _build_simple_select(selected_columns, blueprint)

            elif strategy == "fact_dim_join":
                return _build_fact_dim_join(selected_columns, join_spec, blueprint, table_types)

            elif strategy == "dim_fact_join":
                return _build_dim_fact_join(selected_columns, join_spec, blueprint, table_types)

            elif strategy == "fact_fact_join":
                return _build_fact_fact_join(selected_columns, join_spec, blueprint)

            elif strategy == "dim_dim_join":
                return _build_dim_dim_join(selected_columns, join_spec, blueprint)

            else:
                logger.info("SqlBuilder: стратегия %r не поддерживается шаблоном", strategy)
                return None

        except Exception as e:
            logger.warning("SqlBuilder: ошибка генерации шаблона (%s): %s", strategy, e)
            return None
