"""Узел plan_preview: детерминированный рендер плана SQL перед исполнением (opt-in).

Запускается между sql_planner и sql_writer. Без LLM — читает sql_blueprint,
selected_columns, join_spec, where_resolution, user_hints и строит
Markdown-описание плана. Активируется при:
  - config.show_plan == True  (устанавливается через BaseNodeMixin.show_plan)
  - state["explicit_mode"] == True  (выставляется explicit_mode_dispatcher)
  - state["plan_preview_approved"] == True  → транзит (план уже подтверждён)

При активации: needs_confirmation=False (не используем SQL-confirmation),
plan_preview_pending=True, confirmation_message=<Markdown план>.
"""

import logging
from typing import Any

from core.sql_builder import SqlBuilder
from core.sql_formatter import format_sql_safe
from core.sql_static_checker import check_sql
from graph.state import AgentState

logger = logging.getLogger(__name__)


def _iter_aggregations(sql_blueprint: dict[str, Any]) -> list[dict[str, Any]]:
    aggregations = sql_blueprint.get("aggregations")
    if isinstance(aggregations, list) and aggregations:
        return [dict(item) for item in aggregations if isinstance(item, dict)]
    aggregation = sql_blueprint.get("aggregation") or {}
    return [dict(aggregation)] if aggregation else []


def _render_plan(
    sql_blueprint: dict[str, Any],
    selected_columns: dict[str, Any],
    join_spec: list[dict[str, Any]],
    where_resolution: dict[str, Any],
    user_hints: dict[str, Any],
    plan_diff_summary: str = "",
    sql_preview: str = "",
    sql_preview_note: str = "",
) -> str:
    """Собрать человекочитаемый Markdown-план запроса."""
    lines: list[str] = ["**План запроса:**", ""]

    if plan_diff_summary:
        lines.append("**Изменения после правки:**")
        lines.extend(plan_diff_summary.splitlines())
        lines.append("")

    # Главная таблица
    main_table = sql_blueprint.get("main_table") or ""
    if main_table:
        lines.append(f"- **Главная таблица:** `{main_table}`")

    # Стратегия
    strategy = sql_blueprint.get("strategy") or ""
    if strategy:
        lines.append(f"- **Стратегия:** {strategy}")

    # JOIN — показываем только когда стратегия реально использует join.
    # Иначе пользователь видит ложный JOIN (column_selector строит join_spec
    # для любых ≥2 таблиц, даже если стратегия simple_select).
    join_strategies = {"fact_dim_join", "dim_fact_join", "fact_fact_join", "dim_dim_join"}
    if join_spec and strategy in join_strategies:
        join_parts = []
        for j in join_spec:
            left = j.get("left") or ""
            right = j.get("right") or ""
            if left and right:
                join_parts.append(f"`{left}` = `{right}`")
        if join_parts:
            lines.append(f"- **JOIN:** {', '.join(join_parts)}")

    # Агрегация
    aggregations = _iter_aggregations(sql_blueprint)
    if aggregations:
        if len(aggregations) == 1:
            aggregation = aggregations[0]
            func = aggregation.get("function") or ""
            col = aggregation.get("column") or ""
            alias = aggregation.get("alias") or ""
            if func and col:
                distinct_sql = "DISTINCT " if aggregation.get("distinct") else ""
                agg_str = f"{func.upper()}({distinct_sql}{col})"
                if alias:
                    agg_str += f" AS {alias}"
                lines.append(f"- **Агрегация:** `{agg_str}`")
        else:
            lines.append("- **Агрегации:**")
            for aggregation in aggregations:
                func = aggregation.get("function") or ""
                col = aggregation.get("column") or ""
                alias = aggregation.get("alias") or ""
                if not func or not col:
                    continue
                distinct_sql = "DISTINCT " if aggregation.get("distinct") else ""
                agg_str = f"{func.upper()}({distinct_sql}{col})"
                if alias:
                    agg_str += f" AS {alias}"
                lines.append(f"  - `{agg_str}`")

    # Фильтры
    where_conditions = sql_blueprint.get("where_conditions") or []
    neg_filters = (user_hints or {}).get("negative_filters") or []
    filter_parts = list(where_conditions)
    if neg_filters:
        filter_parts.append(f"исключая: {', '.join(neg_filters)}")
    if filter_parts:
        lines.append(f"- **Фильтры:** {', '.join(filter_parts)}")

    # Группировка
    group_by = sql_blueprint.get("group_by") or []
    if group_by:
        lines.append(f"- **Группировка:** {', '.join(f'`{c}`' for c in group_by)}")

    # HAVING
    having = sql_blueprint.get("having") or []
    if having:
        lines.append(f"- **HAVING:** {', '.join(having)}")

    # Сортировка / лимит
    order_by = sql_blueprint.get("order_by") or ""
    if order_by:
        lines.append(f"- **Сортировка:** {order_by}")
    limit = sql_blueprint.get("limit")
    if limit:
        lines.append(f"- **Лимит:** {limit}")

    # Временная гранулярность из хинтов
    time_gran = (user_hints or {}).get("time_granularity")
    if time_gran:
        lines.append(f"- **Гранулярность:** {time_gran}")

    if sql_preview:
        lines.append("")
        lines.append("**SQL preview:**")
        lines.append("```sql")
        lines.append(sql_preview)
        lines.append("```")
    elif sql_preview_note:
        lines.append("")
        lines.append(f"**SQL preview:** {sql_preview_note}")

    lines.append("")
    lines.append("_Введите «ок» для выполнения или уточните запрос (до 3 итераций)._")

    return "\n".join(lines)


def _build_deterministic_sql_preview(
    *,
    sql_blueprint: dict[str, Any],
    selected_columns: dict[str, Any],
    join_spec: list[dict[str, Any]],
    table_types: dict[str, str],
    schema_loader: Any,
    allowed_tables: list[str] | None = None,
) -> tuple[str, str]:
    """Build SQL text for preview without executing it or calling the LLM."""
    try:
        sql = SqlBuilder().build(
            strategy=str(sql_blueprint.get("strategy") or "simple_select"),
            selected_columns=selected_columns,
            join_spec=join_spec,
            blueprint=sql_blueprint,
            table_types=table_types,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan_preview: deterministic SQL preview failed: %s", exc)
        return "", "детерминированный SQL-preview недоступен для этого плана"
    if not sql:
        return "", "детерминированный SQL-preview недоступен для этого плана"
    sql = format_sql_safe(sql)
    try:
        check = check_sql(
            sql,
            schema_loader=schema_loader,
            allowed_tables=allowed_tables if allowed_tables else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan_preview: SQL preview static check failed: %s", exc)
        return "", "SQL-preview не прошёл статическую проверку"
    if not check.is_valid:
        logger.info("plan_preview: SQL preview suppressed by static checker: %s", check.summary())
        return "", "SQL-preview не прошёл статическую проверку"
    return sql, ""


class PlanPreviewNodes:
    """Mixin с узлом plan_preview (детерминированный, без LLM)."""

    def plan_preview(self, state: AgentState) -> dict[str, Any]:
        """Показать человекочитаемый план запроса перед выполнением SQL.

        Транзит (пропуск) если:
        - show_plan=False И explicit_mode=False
        - план уже подтверждён (plan_preview_approved=True)

        Ожидание подтверждения если:
        - show_plan=True ИЛИ explicit_mode=True — выставляет plan_preview_pending=True.
        """
        # Если план уже подтверждён — транзит
        if state.get("plan_preview_approved"):
            logger.debug("plan_preview: approved — транзит")
            return {
                "plan_preview_pending": False,
                "plan_preview_approved": False,
            }

        show_plan: bool = getattr(self, "show_plan", False)
        explicit_mode: bool = bool(state.get("explicit_mode"))

        # plan_preview активируется только если show_plan=True в конфиге.
        # explicit_mode=True усиливает эффект (конфидентность и strict-хинты),
        # но не форсирует показ плана при show_plan=False — это обеспечивает
        # стабильность golden-тестов (acceptance 2.1: show_plan=false → тесты не ломаются).
        if not show_plan:
            logger.debug("plan_preview: show_plan=False — транзит")
            return {}

        # Собираем данные для рендера
        sql_blueprint = state.get("sql_blueprint") or {}
        selected_columns = state.get("selected_columns") or {}
        join_spec = state.get("join_spec") or []
        where_resolution = state.get("where_resolution") or {}
        user_hints = state.get("user_hints") or {}
        table_types = state.get("table_types") or {}
        allowed_tables = state.get("allowed_tables") or []

        # Если blueprint пуст — транзит (нечего показывать)
        if not sql_blueprint:
            logger.warning("plan_preview: sql_blueprint пуст — транзит")
            return {}

        sql_preview, sql_preview_note = _build_deterministic_sql_preview(
            sql_blueprint=sql_blueprint,
            selected_columns=selected_columns,
            join_spec=join_spec,
            table_types=table_types,
            schema_loader=getattr(self, "schema", None),
            allowed_tables=allowed_tables,
        )
        plan_md = _render_plan(
            sql_blueprint=sql_blueprint,
            selected_columns=selected_columns,
            join_spec=join_spec,
            where_resolution=where_resolution,
            user_hints=user_hints,
            plan_diff_summary=str(state.get("plan_diff_summary") or ""),
            sql_preview=sql_preview,
            sql_preview_note=sql_preview_note,
        )

        iteration = state.get("plan_preview_iteration", 0)
        logger.info(
            "plan_preview: показываем план (explicit_mode=%s, show_plan=%s, iteration=%d)",
            explicit_mode, show_plan, iteration,
        )

        return {
            "plan_preview_pending": True,
            "confirmation_message": plan_md,
            "sql_preview": sql_preview,
            "plan_preview_iteration": iteration,
        }
