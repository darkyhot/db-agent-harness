"""Bind QuerySpec semantic slots to physical catalog columns.

This module is intentionally not an intent parser: it never inspects raw user
text. It consumes only the structured QuerySpec produced by the LLM and catalog
metadata selected by grounding.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.join_analysis import detect_table_type
from core.query_ir import QuerySpec

logger = logging.getLogger(__name__)


_ENTITY_COLUMN_ALIASES: dict[str, list[str]] = {
    "tb": ["tb_id"],
    "тб": ["tb_id"],
    "tb_id": ["tb_id"],
    "gosb": ["gosb_id", "old_gosb_id"],
    "госб": ["gosb_id", "old_gosb_id"],
    "gosb_id": ["gosb_id", "old_gosb_id"],
    "задача": ["task_code", "task_id"],
    "задачи": ["task_code", "task_id"],
    "task": ["task_code", "task_id"],
}


def bind_columns(
    *,
    query_spec: QuerySpec | dict[str, Any],
    table_structures: dict[str, str],
    table_types: dict[str, str],
    schema_loader: Any,
) -> dict[str, Any] | None:
    """Bind physical columns for a QuerySpec.

    Returns None when the QuerySpec strategy is not handled by this binder.
    """
    spec = _coerce_spec(query_spec)
    if spec is None:
        return None
    if spec.strategy != "count_attributes":
        return _bind_metric_dimension_columns(
            spec=spec,
            table_structures=table_structures,
            table_types=table_types,
            schema_loader=schema_loader,
        )

    targets = _count_attribute_targets(spec)
    if len(targets) < 2:
        return {
            "selected_columns": {},
            "join_spec": [],
            "confidence": 0.0,
            "reason": "count_attributes has fewer than two targets",
        }

    best_table: tuple[float, str, dict[str, str]] | None = None
    table_count = len(table_structures)
    for idx, table_key in enumerate(table_structures):
        parts = table_key.split(".", 1)
        if len(parts) != 2:
            continue
        cols_df = schema_loader.get_table_columns(parts[0], parts[1])
        if cols_df.empty:
            continue
        bindings: dict[str, str] = {}
        score = 0.0
        for target in targets:
            col_score = _choose_count_attribute_column(cols_df, target)
            if col_score is None:
                break
            col, value = col_score
            bindings[target] = col
            score += value
        if len(bindings) != len(targets):
            continue
        t_type = table_types.get(table_key) or detect_table_type(parts[1], cols_df)
        if t_type in {"dim", "ref"}:
            score += 500.0
        elif t_type == "fact":
            score -= 250.0
        table_name_score = sum(_text_score(table_key, "", target) for target in targets)
        score += table_name_score * 80.0
        candidate = (score, table_key, bindings)
        if best_table is None or candidate > best_table:
            best_table = candidate

    if best_table is None:
        logger.warning("ColumnBinding: count_attributes targets unresolved: %s", targets)
        return {
            "selected_columns": {},
            "join_spec": [],
            "confidence": 0.0,
            "reason": "count_attributes columns unresolved",
        }

    _score, table_key, bindings = best_table
    aggregate_cols = list(dict.fromkeys(bindings.values()))
    selected_columns = {
        table_key: {
            "select": aggregate_cols,
            "aggregate": aggregate_cols,
        }
    }
    logger.info(
        "ColumnBinding: count_attributes → %s.%s",
        table_key,
        ",".join(aggregate_cols),
    )
    return {
        "selected_columns": selected_columns,
        "join_spec": [],
        "confidence": 0.95,
        "reason": f"count_attributes bound to {table_key}: {bindings}",
    }


def _bind_metric_dimension_columns(
    *,
    spec: QuerySpec,
    table_structures: dict[str, str],
    table_types: dict[str, str],
    schema_loader: Any,
) -> dict[str, Any] | None:
    if not spec.metrics and not spec.dimensions and not spec.filters:
        return None

    selected: dict[str, dict[str, list[str]]] = {}

    for metric in spec.metrics:
        target = str(metric.target or "").strip()
        if not target:
            if metric.operation == "count" and table_structures:
                table_key = next(iter(table_structures))
                selected.setdefault(table_key, {}).setdefault("aggregate", []).append("*")
            continue
        choice = _choose_column_across_tables(
            table_structures=table_structures,
            table_types=table_types,
            schema_loader=schema_loader,
            target=target,
            prefer_fact=metric.operation != "count",
        )
        if not choice:
            continue
        table_key, col = choice
        roles = selected.setdefault(table_key, {})
        roles.setdefault("select", [])
        roles.setdefault("aggregate", [])
        if col not in roles["select"]:
            roles["select"].append(col)
        if col not in roles["aggregate"]:
            roles["aggregate"].append(col)

    for dim in spec.dimensions:
        choice = _choose_column_across_tables(
            table_structures=table_structures,
            table_types=table_types,
            schema_loader=schema_loader,
            target=dim.target,
            prefer_fact=False,
        )
        if not choice:
            continue
        table_key, col = choice
        roles = selected.setdefault(table_key, {})
        roles.setdefault("select", [])
        roles.setdefault("group_by", [])
        if col not in roles["select"]:
            roles["select"].append(col)
        if col not in roles["group_by"]:
            roles["group_by"].append(col)

    for flt in spec.filters:
        choice = _choose_column_across_tables(
            table_structures=table_structures,
            table_types=table_types,
            schema_loader=schema_loader,
            target=flt.target,
            prefer_fact=True,
        )
        if not choice:
            continue
        table_key, col = choice
        roles = selected.setdefault(table_key, {})
        roles.setdefault("filter", [])
        if col not in roles["filter"]:
            roles["filter"].append(col)

    if not selected:
        return None
    logger.info("ColumnBinding: QuerySpec metrics/dimensions → %s", selected)
    return {
        "selected_columns": selected,
        "join_spec": [],
        "confidence": 0.82,
        "reason": "QuerySpec metric/dimension binding",
    }


def _choose_column_across_tables(
    *,
    table_structures: dict[str, str],
    table_types: dict[str, str],
    schema_loader: Any,
    target: str,
    prefer_fact: bool,
) -> tuple[str, str] | None:
    best: tuple[float, str, str] | None = None
    table_count = len(table_structures)
    for idx, table_key in enumerate(table_structures):
        parts = table_key.split(".", 1)
        if len(parts) != 2:
            continue
        cols_df = schema_loader.get_table_columns(parts[0], parts[1])
        if cols_df.empty:
            continue
        col_score = _choose_count_attribute_column(cols_df, target)
        if col_score is None:
            continue
        col, score = col_score
        t_type = table_types.get(table_key) or detect_table_type(parts[1], cols_df)
        if prefer_fact and t_type == "fact":
            score += 120.0
        if not prefer_fact and t_type in {"dim", "ref"}:
            score += 120.0
        score += (table_count - idx) * 0.01
        candidate = (score, table_key, col)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    return best[1], best[2]


def _coerce_spec(query_spec: QuerySpec | dict[str, Any]) -> QuerySpec | None:
    if isinstance(query_spec, QuerySpec):
        return query_spec
    if isinstance(query_spec, dict):
        spec, errors = QuerySpec.from_dict(query_spec)
        if spec is None:
            logger.warning("ColumnBinding: invalid QuerySpec: %s", "; ".join(errors))
        return spec
    return None


def _count_attribute_targets(spec: QuerySpec) -> list[str]:
    targets: list[str] = []
    for entity in spec.entities:
        target = entity.target_column_hint or entity.canonical or entity.name
        if target:
            targets.append(str(target))
    if targets:
        return list(dict.fromkeys(targets))
    for metric in spec.metrics:
        if metric.operation == "count" and metric.target:
            targets.append(metric.target)
    return list(dict.fromkeys(targets))


def _choose_count_attribute_column(cols_df, target: str) -> tuple[str, float] | None:
    aliases = _ENTITY_COLUMN_ALIASES.get(_normalize(target), [])
    by_lower = {
        str(row.get("column_name") or "").strip().lower(): row
        for _, row in cols_df.iterrows()
        if str(row.get("column_name") or "").strip()
    }
    for idx, alias in enumerate(aliases):
        row = by_lower.get(alias)
        if row is not None:
            return str(row.get("column_name")), 1000.0 - idx

    best: tuple[float, str] | None = None
    for _, row in cols_df.iterrows():
        col_name = str(row.get("column_name") or "").strip()
        if not col_name:
            continue
        desc = str(row.get("description") or "")
        score = _text_score(col_name, desc, target) * 100.0
        if score <= 0:
            continue
        if bool(row.get("is_primary_key", False)):
            score += 160.0
        unique = _float(row.get("unique_perc"))
        not_null = _float(row.get("not_null_perc"))
        score += min(unique, 100.0) * 1.2 + min(not_null, 100.0) * 0.8
        lower = col_name.lower()
        if lower.endswith("_id"):
            score += 60.0
        if lower.startswith(("old_", "legacy_", "prev_")):
            score -= 20.0
        candidate = (score, col_name)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    return best[1], best[0]


def _text_score(name: str, description: str, target: str) -> float:
    haystack = _normalize(f"{name} {description}")
    term = _normalize(target)
    if not term:
        return 0.0
    if term == _normalize(name):
        return 5.0
    if term in haystack:
        return 3.0
    term_tokens = set(term.split())
    hay_tokens = set(haystack.split())
    if term_tokens and term_tokens <= hay_tokens:
        return 2.0
    return 0.0


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zа-яё_]+", " ", str(value).lower())).strip()


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
