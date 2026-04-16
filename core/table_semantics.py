"""Универсальная семантика таблиц на основе описания и семантики колонок."""

from __future__ import annotations

from collections import Counter
from typing import Any


def infer_table_role(table_name: str, description: str, grain: str, column_classes: list[str]) -> str:
    text = f"{table_name} {description}".lower()
    if any(token in text for token in ("dim", "dict", "lookup", "reference", "справочник")):
        return "reference"
    if grain == "snapshot":
        return "snapshot"
    if grain == "event":
        return "event"
    if grain == "dictionary":
        return "reference"
    if sum(1 for cls in column_classes if cls == "metric") >= 2:
        return "fact"
    if sum(1 for cls in column_classes if cls in {"label", "enum_like"}) >= 3:
        return "dimension"
    return "fact" if grain and grain not in {"other", "dictionary"} else "other"


def infer_primary_subjects(table_name: str, description: str, grain: str) -> list[str]:
    subjects: list[str] = []
    if grain:
        subjects.append(grain)
    return list(dict.fromkeys(subjects))


def build_table_semantics(tables_df, attrs_df, column_semantics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Построить table semantics для всего каталога."""
    result: dict[str, dict[str, Any]] = {}
    if tables_df is None or tables_df.empty:
        return result

    for _, row in tables_df.iterrows():
        schema = str(row.get("schema_name", "") or "")
        table = str(row.get("table_name", "") or "")
        description = str(row.get("description", "") or "")
        grain = str(row.get("grain", "") or "").lower().strip()
        if not (schema and table):
            continue

        table_attrs = attrs_df[
            (attrs_df["schema_name"] == schema) & (attrs_df["table_name"] == table)
        ].copy()
        class_counter: Counter[str] = Counter()
        time_axis_columns: list[str] = []
        join_key_count = 0
        filter_candidate_count = 0
        for _, a_row in table_attrs.iterrows():
            col = str(a_row.get("column_name", "") or "")
            key = f"{schema}.{table}.{col}".lower()
            semantics = column_semantics.get(key, {})
            semantic_class = str(semantics.get("semantic_class", "") or "")
            semantic_tags = set(semantics.get("semantic_tags", []) or [])
            if semantic_class:
                class_counter[semantic_class] += 1
            if "time_axis" in semantic_tags:
                time_axis_columns.append(col)
            if "join_candidate" in semantic_tags:
                join_key_count += 1
            if "filter_candidate" in semantic_tags:
                filter_candidate_count += 1

        table_role = infer_table_role(table, description, grain, list(class_counter.elements()))
        primary_subjects = infer_primary_subjects(table, description, grain)
        filter_friendliness = round(
            min(1.0, filter_candidate_count / max(1, len(table_attrs))),
            3,
        )
        join_richness = round(
            min(1.0, join_key_count / max(1, len(table_attrs))),
            3,
        )

        result[f"{schema}.{table}".lower()] = {
            "schema": schema,
            "table": table,
            "grain": grain,
            "table_role": table_role,
            "primary_subjects": primary_subjects,
            "time_axis_columns": sorted(set(time_axis_columns)),
            "filter_friendliness": filter_friendliness,
            "join_richness": join_richness,
            "column_class_counts": dict(class_counter),
        }

    return result
