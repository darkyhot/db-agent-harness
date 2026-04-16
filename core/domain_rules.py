"""Data-driven table support and bonus logic based on semantic registry."""

from __future__ import annotations

from typing import Any


def _table_columns(schema_loader, schema: str, table: str) -> list[dict[str, Any]]:
    cols_df = schema_loader.get_table_columns(schema, table)
    if cols_df.empty:
        return []
    return cols_df.to_dict(orient="records")


def _table_column_keys(schema_loader, schema: str, table: str) -> set[str]:
    return {
        f"{schema}.{table}.{str(row.get('column_name', '') or '')}".lower()
        for row in _table_columns(schema_loader, schema, table)
        if str(row.get("column_name", "") or "").strip()
    }


def _table_subject_match(schema_loader, schema: str, table: str, semantic_frame: dict[str, Any] | None) -> bool:
    if not semantic_frame:
        return True
    requested_subject = str(semantic_frame.get("subject") or "").strip().lower()
    if not requested_subject:
        return True
    table_sem = schema_loader.get_table_semantics(schema, table)
    subjects = {str(v).strip().lower() for v in (table_sem.get("primary_subjects") or []) if str(v).strip()}
    grain = str(table_sem.get("grain") or "").strip().lower()
    return requested_subject in subjects or requested_subject == grain


def _supported_filter_intents(schema_loader, schema: str, table: str, semantic_frame: dict[str, Any] | None) -> list[dict[str, Any]]:
    filter_intents = list((semantic_frame or {}).get("filter_intents") or [])
    if not filter_intents:
        return []
    column_keys = _table_column_keys(schema_loader, schema, table)
    supported: list[dict[str, Any]] = []
    for item in filter_intents:
        column_key = str(item.get("column_key") or "").lower()
        if column_key and column_key in column_keys:
            supported.append(item)
        elif not column_key:
            supported.append(item)
    return supported


def table_bonus_for_frame(
    schema_loader,
    schema: str,
    table: str,
    semantic_frame: dict[str, Any] | None,
) -> float:
    """Additional weight from subject and filter-support compatibility."""
    if not semantic_frame:
        return 0.0

    bonus = 0.0
    if _table_subject_match(schema_loader, schema, table, semantic_frame):
        bonus += 120.0
    else:
        bonus -= 60.0

    supported = _supported_filter_intents(schema_loader, schema, table, semantic_frame)
    all_filters = list((semantic_frame or {}).get("filter_intents") or [])
    if all_filters:
        support_ratio = len(supported) / max(1, len(all_filters))
        bonus += support_ratio * 140.0
        if support_ratio == 0.0:
            bonus -= 40.0

    table_sem = schema_loader.get_table_semantics(schema, table)
    filter_friendliness = float(table_sem.get("filter_friendliness", 0.0) or 0.0)
    bonus += filter_friendliness * 25.0
    return bonus


def table_can_satisfy_frame(
    schema_loader,
    schema: str,
    table: str,
    semantic_frame: dict[str, Any] | None,
) -> bool:
    """Can one table satisfy the current semantic frame without extra JOIN."""
    if not semantic_frame:
        return True
    if not _table_subject_match(schema_loader, schema, table, semantic_frame):
        return False

    filter_intents = list((semantic_frame or {}).get("filter_intents") or [])
    if not filter_intents:
        return True
    supported = _supported_filter_intents(schema_loader, schema, table, semantic_frame)
    return len(supported) == len(filter_intents)
