"""Классификация колонок по универсальным semantic classes."""

from __future__ import annotations

from typing import Any


def classify_column(row: dict[str, Any]) -> dict[str, Any]:
    """Классифицировать колонку по её имени, описанию, dtype и stats."""
    column = str(row.get("column_name", "") or "")
    dtype = str(row.get("dType", "") or "").lower()
    description = str(row.get("description", "") or "").lower()
    is_pk = bool(row.get("is_primary_key", False))
    unique_pct = float(row.get("unique_perc", 0.0) or 0.0)
    not_null_pct = float(row.get("not_null_perc", 0.0) or 0.0)

    lower = column.lower()
    text = f"{lower} {description}"
    tags: set[str] = set()

    if any(token in dtype for token in ("timestamp", "datetime")) or lower.endswith("_dttm"):
        semantic_class = "system_timestamp" if any(token in lower for token in ("modified", "created", "updated", "inserted")) else "date"
    elif any(token in dtype for token in ("date",)):
        semantic_class = "date"
    elif is_pk or lower.endswith(("_id", "_code", "_key")) and unique_pct >= 70.0:
        semantic_class = "join_key" if unique_pct < 98.0 else "identifier"
    elif lower.startswith("is_") or "признак" in description or "flag" in lower:
        semantic_class = "flag"
    elif any(token in text for token in ("subtype", "subtipe", "status", "state", "category", "segment", "reason", "тип", "подтип", "статус", "сегмент", "категор")):
        semantic_class = "enum_like"
    elif any(token in lower for token in ("_name", "_fio", "label", "title")) or any(token in description for token in ("наименование", "фио", "название")):
        semantic_class = "label"
    elif any(token in dtype for token in ("int", "numeric", "decimal", "float", "double", "real")):
        semantic_class = "metric"
    elif any(token in dtype for token in ("text", "char", "varchar")):
        semantic_class = "free_text"
    else:
        semantic_class = "free_text"

    if semantic_class in {"flag", "enum_like", "label", "date"}:
        tags.add("filter_candidate")
    if semantic_class in {"label", "enum_like"} and unique_pct <= 35.0:
        tags.add("categorical")
    if semantic_class in {"identifier", "join_key"}:
        tags.add("join_candidate")
    if semantic_class == "date" and not any(token in lower for token in ("created", "updated", "inserted", "modified")):
        tags.add("time_axis")
    if semantic_class == "metric" and not is_pk:
        tags.add("aggregate_candidate")
    if not_null_pct >= 95.0:
        tags.add("dense")

    return {
        "semantic_class": semantic_class,
        "semantic_tags": sorted(tags),
    }


def build_column_semantics(attrs_df) -> dict[str, dict[str, Any]]:
    """Построить semantics для всех колонок каталога."""
    result: dict[str, dict[str, Any]] = {}
    if attrs_df is None or attrs_df.empty:
        return result
    for _, row in attrs_df.iterrows():
        schema = str(row.get("schema_name", "") or "")
        table = str(row.get("table_name", "") or "")
        column = str(row.get("column_name", "") or "")
        if not (schema and table and column):
            continue
        key = f"{schema}.{table}.{column}".lower()
        result[key] = {
            "schema": schema,
            "table": table,
            "column": column,
            **classify_column(row.to_dict()),
        }
    return result
