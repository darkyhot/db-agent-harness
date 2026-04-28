"""Валидация и применение PlanEdit-правок к плану и selected_columns.

Принципы:
- Никогда не верить LLM на слово: каждая правка валидируется по каталогу.
- Невалидные правки логируются и отбрасываются, не падая весь pipeline.
- Только whitelisted операции; никакого free-form rewrite.
- После применения plan-узлы (sql_planner) пересобирают blueprint —
  здесь мы трогаем только selected_columns и сами условия.
"""

from __future__ import annotations

import logging
from typing import Any

from core.plan_verifier_models import PlanEdit

logger = logging.getLogger(__name__)


def _split_ref(ref: str) -> tuple[str | None, str | None, str | None]:
    """Разбить `schema.table.column` или `schema.table` на части."""
    parts = [p.strip() for p in str(ref or "").split(".") if p.strip()]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], None
    return None, None, None


def _column_exists(schema_loader, schema: str, table: str, column: str) -> bool:
    try:
        cols = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        return False
    if cols is None or cols.empty or "column_name" not in cols.columns:
        return False
    return column.lower() in {str(c).strip().lower() for c in cols["column_name"]}


def _table_exists(schema_loader, schema: str, table: str) -> bool:
    df = getattr(schema_loader, "tables_df", None)
    if df is None or df.empty:
        return False
    mask = (
        df["schema_name"].astype(str).str.lower() == schema.lower()
    ) & (
        df["table_name"].astype(str).str.lower() == table.lower()
    )
    return not df[mask].empty


def apply_plan_edits(
    *,
    selected_columns: dict[str, dict[str, list[str]]],
    where_conditions: list[str],
    edits: list[PlanEdit],
    schema_loader,
) -> dict[str, Any]:
    """Применить серию edits к selected_columns и where_conditions.

    Возвращает dict с обновлёнными selected_columns/where_conditions и
    списками `applied`/`rejected` (PlanEdit + причина).
    """
    new_columns: dict[str, dict[str, list[str]]] = {
        table: {role: list(cols) for role, cols in roles.items()}
        for table, roles in (selected_columns or {}).items()
    }
    new_conditions: list[str] = list(where_conditions or [])
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for edit in edits or []:
        try:
            verdict = _apply_single_edit(edit, new_columns, new_conditions, schema_loader)
        except Exception as exc:  # noqa: BLE001
            verdict = {"ok": False, "reason": f"exception: {exc}"}
        item = {"edit": edit.model_dump(), "reason": verdict.get("reason", "")}
        if verdict.get("ok"):
            applied.append(item)
            logger.info("plan_edit applied: %s (%s)", edit.op, item["reason"])
        else:
            rejected.append(item)
            logger.warning(
                "plan_edit rejected: op=%s from=%s to=%s — %s",
                edit.op, edit.from_ref, edit.to_ref, verdict.get("reason"),
            )

    return {
        "selected_columns": new_columns,
        "where_conditions": new_conditions,
        "applied": applied,
        "rejected": rejected,
    }


def _apply_single_edit(
    edit: PlanEdit,
    columns: dict[str, dict[str, list[str]]],
    conditions: list[str],
    schema_loader,
) -> dict[str, Any]:
    op = edit.op

    if op == "replace_column":
        from_schema, from_table, from_col = _split_ref(edit.from_ref or "")
        to_schema, to_table, to_col = _split_ref(edit.to_ref or "")
        if not (from_schema and from_table and from_col and to_schema and to_table and to_col):
            return {"ok": False, "reason": "ref must be schema.table.column"}
        if not _column_exists(schema_loader, to_schema, to_table, to_col):
            return {"ok": False, "reason": f"target column {edit.to_ref} not in catalog"}
        target_table_key = f"{to_schema}.{to_table}"
        from_table_key = f"{from_schema}.{from_table}"
        target_role = edit.target_role
        # Если to_table отсутствует в selected_columns — добавляем пустой dict.
        columns.setdefault(target_table_key, {})
        replaced_anywhere = False
        # Заменяем from в указанной роли (или во всех ролях, если target_role не задан).
        roles_to_touch = [target_role] if target_role else list(
            columns.get(from_table_key, {}).keys()
        )
        for role in roles_to_touch:
            if not role:
                continue
            cols = columns.get(from_table_key, {}).get(role) or []
            if from_col in cols:
                cols.remove(from_col)
                replaced_anywhere = True
                if not cols:
                    columns[from_table_key].pop(role, None)
            target_cols = columns[target_table_key].setdefault(role, [])
            if to_col not in target_cols:
                target_cols.append(to_col)
        if not columns.get(from_table_key):
            columns.pop(from_table_key, None)
        if not replaced_anywhere:
            return {"ok": False, "reason": "from_ref not found in selected_columns"}
        return {"ok": True, "reason": f"{edit.from_ref} → {edit.to_ref}"}

    if op == "add_filter":
        condition = (edit.to_ref or "").strip()
        if not condition:
            return {"ok": False, "reason": "add_filter requires to_ref as full SQL condition"}
        if condition in conditions:
            return {"ok": False, "reason": "condition already present"}
        conditions.append(condition)
        return {"ok": True, "reason": f"+ {condition}"}

    if op == "drop_filter":
        target = (edit.from_ref or "").strip()
        if not target:
            return {"ok": False, "reason": "drop_filter requires from_ref"}
        before = len(conditions)
        target_lower = target.lower()
        kept: list[str] = []
        dropped = 0
        for cond in conditions:
            cond_lower = cond.lower()
            if cond.strip() == target or target_lower in cond_lower:
                dropped += 1
                continue
            kept.append(cond)
        if dropped == 0:
            return {"ok": False, "reason": "no matching condition"}
        conditions[:] = kept
        return {"ok": True, "reason": f"- {target} ({dropped} cond removed)"}

    if op == "swap_aggregation":
        from_schema, from_table, from_col = _split_ref(edit.from_ref or "")
        if not (from_schema and from_table and from_col):
            return {"ok": False, "reason": "from_ref must be schema.table.column"}
        new_func = (edit.to_ref or "").strip().upper()
        if new_func not in {"SUM", "COUNT", "AVG", "MIN", "MAX"}:
            return {"ok": False, "reason": f"unknown aggregation '{new_func}'"}
        table_key = f"{from_schema}.{from_table}"
        roles = columns.get(table_key)
        if not roles or from_col not in (roles.get("aggregate") or []):
            return {"ok": False, "reason": f"{edit.from_ref} not in aggregate role"}
        # На уровне selected_columns aggregation function не хранится отдельно —
        # она восстанавливается sql_planner_deterministic из user_hints
        # `aggregate_hints`. Этот edit применяется как маркер для downstream.
        roles.setdefault("_aggregation_overrides", {})
        roles["_aggregation_overrides"][from_col] = new_func
        return {"ok": True, "reason": f"{from_col} → {new_func}"}

    if op == "add_table":
        to_schema, to_table, _to_col = _split_ref(edit.to_ref or "")
        if not (to_schema and to_table):
            return {"ok": False, "reason": "add_table requires to_ref schema.table"}
        if not _table_exists(schema_loader, to_schema, to_table):
            return {"ok": False, "reason": f"table {edit.to_ref} not in catalog"}
        table_key = f"{to_schema}.{to_table}"
        columns.setdefault(table_key, {})
        return {"ok": True, "reason": f"+ table {table_key}"}

    return {"ok": False, "reason": f"unknown op '{op}'"}
