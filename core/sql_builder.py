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

    for table, roles in selected_columns.items():
        alias = table_alias_map.get(table, "t")
        for col in roles.get("select", []):
            if col in seen_cols:
                continue
            seen_cols.add(col)
            if col == agg_col and agg_func:
                # Агрегируем
                if col == "*":
                    items.append(f"{agg_func}(*) AS {agg_alias}")
                else:
                    items.append(f"{agg_func}({alias}.{col}) AS {agg_alias}")
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
                    items.append(f"{agg_func}({alias}.{col}) AS {agg_alias or f'{agg_func.lower()}_{col}'}")

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
    """Найти колонки для JOIN между двумя таблицами из join_spec.

    Returns:
        (left_col, right_col) или ("", "") если не найдено.
    """
    for jk in join_spec:
        left_full = jk.get("left", "")
        right_full = jk.get("right", "")
        # Разбиваем schema.table.column
        left_parts = left_full.rsplit(".", 1)
        right_parts = right_full.rsplit(".", 1)
        left_tbl = ".".join(left_full.split(".")[:2]) if left_full.count(".") >= 2 else ""
        right_tbl = ".".join(right_full.split(".")[:2]) if right_full.count(".") >= 2 else ""
        left_col = left_parts[-1] if left_parts else ""
        right_col = right_parts[-1] if right_parts else ""

        if left_tbl == left_table and right_tbl == right_table:
            return left_col, right_col
        if left_tbl == right_table and right_tbl == left_table:
            return right_col, left_col  # swap

    return "", ""


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

    # JOIN-ключи
    join_key_fact, join_key_dim = _resolve_join_key(join_spec, main_table, dim_table)
    if not join_key_fact or not join_key_dim:
        # Нет join_spec — не можем построить JOIN без LLM
        return None

    # Безопасность: берём из join_spec
    is_safe = any(
        j.get("safe", False)
        for j in join_spec
        if main_table in j.get("left", "") or main_table in j.get("right", "")
    )

    used: set[str] = set()
    f_alias = _short_alias(main_table, used)
    d_alias = _short_alias(dim_table, used)

    aggregation = blueprint.get("aggregation")
    alias_map_fact = {main_table: f_alias}
    alias_map_dim = {dim_table: d_alias}

    # SELECT из факт-таблицы
    fact_roles = selected_columns.get(main_table, {})
    dim_roles = selected_columns.get(dim_table, {})

    # Строим SELECT-items
    select_items: list[str] = []
    seen: set[str] = set()

    # Колонки факт-таблицы
    agg_col = aggregation.get("column") if aggregation else None
    agg_func = aggregation.get("function") if aggregation else None
    agg_alias = aggregation.get("alias") if aggregation else None

    for col in (fact_roles.get("select", []) + fact_roles.get("group_by", [])):
        if col in seen:
            continue
        seen.add(col)
        if col == agg_col and agg_func:
            select_items.append(f"{agg_func}({f_alias}.{col}) AS {agg_alias}")
        else:
            select_items.append(f"{f_alias}.{col}")

    for col in fact_roles.get("aggregate", []):
        if col in seen:
            continue
        seen.add(col)
        if agg_func:
            select_items.append(f"{agg_func}({f_alias}.{col}) AS {agg_alias or f'{agg_func.lower()}_{col}'}")

    # Колонки из справочника
    for col in dim_roles.get("select", []):
        if col == join_key_dim or col in seen:
            continue
        seen.add(col)
        select_items.append(f"{d_alias}.{col}")

    if not select_items:
        select_items = [f"COUNT(*) AS cnt"]

    # WHERE
    where_clause = _build_where_clause(blueprint.get("where_conditions", []))

    # GROUP BY
    group_by_cols = blueprint.get("group_by", [])
    gb_items: list[str] = []
    for col in group_by_cols:
        # Определяем принадлежность колонки
        if col in (fact_roles.get("select", []) + fact_roles.get("group_by", [])):
            gb_items.append(f"{f_alias}.{col}")
        elif col in dim_roles.get("select", []):
            gb_items.append(f"{d_alias}.{col}")
        else:
            gb_items.append(col)

    group_by_clause = ("GROUP BY " + ", ".join(gb_items)) if gb_items else ""
    order_by_clause = _build_order_by(blueprint.get("order_by"))
    limit_clause = _build_limit(blueprint.get("limit"))

    # Формируем dim-подзапрос или прямой JOIN
    dim_select_cols = [join_key_dim] + [c for c in dim_roles.get("select", []) if c != join_key_dim]

    if is_safe:
        # Прямой JOIN
        join_sql = (
            f"JOIN {dim_table} {d_alias} ON {d_alias}.{join_key_dim} = {f_alias}.{join_key_fact}"
        )
    else:
        # DISTINCT ON для безопасности
        dim_cols_str = ", ".join(dim_select_cols) if dim_select_cols else join_key_dim
        join_sql = (
            f"JOIN (\n"
            f"    SELECT DISTINCT ON ({join_key_dim}) {dim_cols_str}\n"
            f"    FROM {dim_table}\n"
            f"    ORDER BY {join_key_dim}\n"
            f") {d_alias} ON {d_alias}.{join_key_dim} = {f_alias}.{join_key_fact}"
        )

    parts = [
        f"SELECT {', '.join(select_items)}",
        f"FROM {main_table} {f_alias}",
        join_sql,
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

    join_key_dim, join_key_fact = _resolve_join_key(join_spec, dim_table, fact_table)
    if not join_key_dim or not join_key_fact:
        return None

    used: set[str] = set()
    d_alias = _short_alias(dim_table, used)

    aggregation = blueprint.get("aggregation")
    agg_func = aggregation.get("function") if aggregation else "COUNT"
    agg_col = aggregation.get("column") if aggregation else "*"
    agg_alias = aggregation.get("alias") if aggregation else "agg_val"

    fact_roles = selected_columns.get(fact_table, {})
    dim_roles = selected_columns.get(dim_table, {})

    # Dim SELECT
    dim_select_cols = [join_key_dim] + [
        c for c in dim_roles.get("select", []) if c != join_key_dim
    ]
    dim_select_str = ", ".join(f"{d_alias}.{c}" for c in dim_select_cols)

    # Fact aggregation subquery
    fact_agg_col = "*" if agg_col == "*" else agg_col
    fact_subquery = (
        f"SELECT {join_key_fact}, {agg_func}({fact_agg_col}) AS {agg_alias}\n"
        f"    FROM {fact_table}\n"
        f"    GROUP BY {join_key_fact}"
    )

    where_clause = _build_where_clause(blueprint.get("where_conditions", []))
    order_by_clause = _build_order_by(blueprint.get("order_by"))
    limit_clause = _build_limit(blueprint.get("limit"))

    parts = [
        f"SELECT {dim_select_str}, agg.{agg_alias}",
        f"FROM {dim_table} {d_alias}",
        f"JOIN (\n    {fact_subquery}\n) agg ON agg.{join_key_fact} = {d_alias}.{join_key_dim}",
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
    """fact_fact_join: два CTE с агрегацией + финальный JOIN."""
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
    cte1 = f"{a1}_agg"
    cte2 = f"{a2}_agg"

    aggregation = blueprint.get("aggregation")
    agg_func = aggregation.get("function") if aggregation else "COUNT"
    agg_col = aggregation.get("column") if aggregation else "*"
    agg_alias_base = aggregation.get("alias") if aggregation else "agg_val"

    roles1 = selected_columns[t1]
    roles2 = selected_columns[t2]

    # Агрегации для каждой таблицы
    def _agg_for(roles, agg_suffix):
        agg_cols = roles.get("aggregate", [])
        if agg_cols:
            col = agg_cols[0]
            return f"{agg_func}({col}) AS {agg_func.lower()}_{col}"
        return f"COUNT(*) AS cnt_{agg_suffix}"

    agg1_expr = _agg_for(roles1, a1)
    agg2_expr = _agg_for(roles2, a2)

    cte1_sql = f"{cte1} AS (\n    SELECT {join_key_1}, {agg1_expr}\n    FROM {t1}\n    GROUP BY {join_key_1}\n)"
    cte2_sql = f"{cte2} AS (\n    SELECT {join_key_2}, {agg2_expr}\n    FROM {t2}\n    GROUP BY {join_key_2}\n)"

    # Финальный SELECT
    agg1_col = agg1_expr.split(" AS ")[-1]
    agg2_col = agg2_expr.split(" AS ")[-1]

    select_str = f"c1.{join_key_1}, c1.{agg1_col}, c2.{agg2_col}"

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
