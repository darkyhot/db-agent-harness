"""Bind QuerySpec semantic slots to physical catalog columns.

This module is intentionally not an intent parser: it never inspects raw user
text. It consumes only the structured QuerySpec produced by the LLM and catalog
metadata selected by grounding.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.entity_resolver import resolve_entity_to_columns
from core.join_analysis import detect_table_type
from core.query_ir import QuerySpec

logger = logging.getLogger(__name__)


def bind_columns(
    *,
    query_spec: QuerySpec | dict[str, Any],
    table_structures: dict[str, str],
    table_types: dict[str, str],
    schema_loader: Any,
    llm_invoker: Any = None,
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
            llm_invoker=llm_invoker,
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
            col_score = _resolve_column_for_target(
                target=target,
                table_key=table_key,
                schema_loader=schema_loader,
                llm_invoker=llm_invoker,
                role_hint="id",
            )
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
    llm_invoker: Any = None,
) -> dict[str, Any] | None:
    if not spec.metrics and not spec.dimensions and not spec.filters:
        return None

    selected: dict[str, dict[str, list[str]]] = {}

    for metric in spec.metrics:
        target = str(metric.target or "").strip()
        if not target:
            if metric.operation == "count" and table_structures:
                table_key = next(iter(table_structures))
                count_col = _choose_count_column_for_table(
                    table_key=table_key,
                    spec=spec,
                    schema_loader=schema_loader,
                )
                selected.setdefault(table_key, {}).setdefault("aggregate", []).append(count_col or "*")
            continue
        choice = _choose_column_across_tables(
            table_structures=table_structures,
            table_types=table_types,
            schema_loader=schema_loader,
            target=target,
            prefer_fact=metric.operation != "count",
            role_hint="metric" if metric.operation != "count" else "id",
            llm_invoker=llm_invoker,
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
            role_hint="any",
            llm_invoker=llm_invoker,
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
            role_hint="filter",
            llm_invoker=llm_invoker,
        )
        if not choice:
            continue
        table_key, col = choice
        roles = selected.setdefault(table_key, {})
        roles.setdefault("filter", [])
        if col not in roles["filter"]:
            roles["filter"].append(col)

    if spec.time_range is not None:
        _ensure_time_axis_filters(selected, table_structures, schema_loader)

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
    role_hint: str = "any",
    llm_invoker: Any = None,
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
        col_score = _resolve_column_for_target(
            target=target,
            table_key=table_key,
            schema_loader=schema_loader,
            llm_invoker=llm_invoker,
            role_hint=role_hint,
        )
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


def _resolve_column_for_target(
    *,
    target: str,
    table_key: str,
    schema_loader: Any,
    llm_invoker: Any,
    role_hint: str,
) -> tuple[str, float] | None:
    """Универсальный матчинг колонки через entity_resolver (без алиасов)."""
    resolution = resolve_entity_to_columns(
        entity_term=target,
        user_input="",
        candidate_table_keys=[table_key],
        schema_loader=schema_loader,
        llm_invoker=llm_invoker,
        role_hint=role_hint,
    )
    if not resolution.matched or resolution.column is None:
        return None
    # Масштаб 0..1000 для совместимости с table-level бонусами выше
    # (т-тип +500/+120, table_name_score *80).
    return resolution.column, resolution.confidence * 1000.0


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


def _choose_count_column_for_table(
    *,
    table_key: str,
    spec: QuerySpec,
    schema_loader: Any,
) -> str | None:
    if "." not in table_key or not spec.entities:
        return None
    schema, table = table_key.split(".", 1)
    table_sem = schema_loader.get_table_semantics(schema, table)
    grain = str(table_sem.get("grain") or "").strip().lower()
    subjects = {str(v).strip().lower() for v in (table_sem.get("primary_subjects") or []) if str(v).strip()}
    if grain in {"event", "snapshot"} and "task" not in subjects:
        return None
    try:
        cols_df = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        return None
    if cols_df.empty:
        return None
    best: tuple[float, str] | None = None
    for _, row in cols_df.iterrows():
        col = str(row.get("column_name") or "").strip()
        if not col:
            continue
        sem = schema_loader.get_column_semantics(schema, table, col)
        sem_class = str(sem.get("semantic_class") or "").lower()
        score = 0.0
        if bool(row.get("is_primary_key", False)):
            score += 10.0
        if sem_class in {"identifier", "join_key"}:
            score += 5.0
        if col.lower().endswith(("_id", "_code")):
            score += 2.0
        if score <= 0:
            continue
        candidate = (score, col)
        if best is None or candidate > best:
            best = candidate
    return best[1] if best else None


def _ensure_time_axis_filters(
    selected: dict[str, dict[str, list[str]]],
    table_structures: dict[str, str],
    schema_loader: Any,
) -> None:
    candidate_tables = list(selected) or list(table_structures)
    for table_key in candidate_tables:
        if "." not in table_key:
            continue
        date_col = _choose_time_axis_column(table_key, schema_loader)
        if not date_col:
            continue
        roles = selected.setdefault(table_key, {})
        filters = roles.setdefault("filter", [])
        if date_col not in filters:
            filters.append(date_col)


def _choose_time_axis_column(table_key: str, schema_loader: Any) -> str | None:
    schema, table = table_key.split(".", 1)
    table_sem = schema_loader.get_table_semantics(schema, table)
    time_axis = [str(v).strip() for v in (table_sem.get("time_axis_columns") or []) if str(v).strip()]
    try:
        cols_df = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        cols_df = None
    if cols_df is None or cols_df.empty:
        return time_axis[0] if time_axis else None
    known = {str(row.get("column_name") or "").lower(): row for _, row in cols_df.iterrows()}
    ranked: list[tuple[int, str]] = []
    for col in time_axis:
        if col.lower() in known:
            ranked.append((_date_priority(schema_loader, schema, table, col, known[col.lower()]), col))
    for col_lower, row in known.items():
        col = str(row.get("column_name") or "").strip()
        if not col or col in time_axis:
            continue
        dtype = str(row.get("dType") or row.get("dtype") or "").lower()
        sem = schema_loader.get_column_semantics(schema, table, col)
        sem_class = str(sem.get("semantic_class") or "").lower()
        if sem_class == "date" or dtype.startswith(("date", "timestamp")):
            ranked.append((_date_priority(schema_loader, schema, table, col, row), col))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def _date_priority(schema_loader: Any, schema: str, table: str, col: str, row: Any) -> int:
    name = col.lower()
    sem = schema_loader.get_column_semantics(schema, table, col)
    tags = {str(v).lower() for v in (sem.get("semantic_tags") or [])}
    dtype = str(row.get("dType") or row.get("dtype") or "").lower()
    if "time_axis" in tags and name in {"report_dt", "report_date"}:
        return 0
    if "time_axis" in tags:
        return 1
    if name in {"report_dt", "report_date"}:
        return 2
    if name.startswith(("inserted_", "updated_", "modified_", "created_", "load_", "etl_")):
        return 8
    if dtype.startswith("date"):
        return 3
    if dtype.startswith("timestamp"):
        return 5
    return 9


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
