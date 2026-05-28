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
from core.join_selection import _norm_col_name
from core.query_ir import QuerySpec, _parse_calendar_period, _target_looks_calendar

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
        result = _bind_metric_dimension_columns(
            spec=spec,
            table_structures=table_structures,
            table_types=table_types,
            schema_loader=schema_loader,
            llm_invoker=llm_invoker,
        )
        # bind_columns сам по себе не строит join_spec. Для fact↔dim с составным
        # PK достраиваем его детерминированно из PK dim-таблицы — иначе ON-клаузу
        # пишет LLM sql_writer и теряет вторую пару составного ключа (tb_id).
        if (
            result
            and result.get("selected_columns")
            and len(result["selected_columns"]) >= 2
            and not result.get("join_spec")
        ):
            result["join_spec"] = _derive_join_spec_from_pk(
                result["selected_columns"], table_types, schema_loader
            )
        return result

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


def _derive_join_spec_from_pk(
    selected_columns: dict[str, dict[str, list[str]]],
    table_types: dict[str, str],
    schema_loader: Any,
) -> list[dict[str, Any]]:
    """Построить составной join_spec для пар fact↔dim из PK dim-таблицы.

    Якоримся на PRIMARY KEY справочника, а не на ранжированном кандидате: это
    однозначно выбирает правильный FK-таргет. Пример: dim PK (tb_id, old_gosb_id)
    маппится на fact (tb_id, gosb_id), и НИКОГДА на non-PK new_gosb_id (который
    нормализуется в тот же стем gosb_id и путает ранжирующий выбор).

    safe=True ставим, только когда пары покрывают ВЕСЬ PK справочника — тогда
    join к dim по полному уникальному ключу не размножает строки. Это вывод из
    метаданных (is_primary_key), без обращения к БД.
    """
    tables = list(selected_columns.keys())
    join_spec: list[dict[str, Any]] = []
    for i, t1 in enumerate(tables):
        for t2 in tables[i + 1:]:
            ty1 = (table_types or {}).get(t1, "unknown")
            ty2 = (table_types or {}).get(t2, "unknown")
            if ty1 == "fact" and ty2 in {"dim", "ref"}:
                fact, dim = t1, t2
            elif ty2 == "fact" and ty1 in {"dim", "ref"}:
                fact, dim = t2, t1
            else:
                continue
            ds = dim.split(".", 1)
            fs = fact.split(".", 1)
            if len(ds) != 2 or len(fs) != 2:
                continue
            try:
                dim_cols = schema_loader.get_table_columns(ds[0], ds[1])
                fact_cols = schema_loader.get_table_columns(fs[0], fs[1])
            except Exception:  # noqa: BLE001
                continue
            if dim_cols is None or dim_cols.empty or "is_primary_key" not in dim_cols.columns:
                continue
            pk_cols = dim_cols.loc[
                dim_cols["is_primary_key"].astype(bool), "column_name"
            ].tolist()
            if not pk_cols:
                continue
            fact_names = (
                fact_cols["column_name"].tolist()
                if fact_cols is not None and not fact_cols.empty
                else []
            )
            pairs: list[tuple[str, str]] = []
            for pk in pk_cols:
                if pk in fact_names:
                    fact_col: str | None = pk
                else:
                    norm_pk = _norm_col_name(pk)
                    fact_col = next(
                        (f for f in fact_names if _norm_col_name(f) == norm_pk), None
                    )
                if fact_col:
                    pairs.append((fact_col, pk))
            if not pairs:
                continue
            safe = len(pairs) == len(pk_cols)
            for fact_col, pk in pairs:
                join_spec.append({
                    "left": f"{fact}.{fact_col}",
                    "right": f"{dim}.{pk}",
                    "safe": safe,
                    "strategy": "fact_dim_join",
                })
            logger.info(
                "ColumnBinding: составной join_spec из PK %s → %s (%d пар, safe=%s)",
                dim, fact, len(pairs), safe,
            )
    return join_spec


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
        # Структурный hint резолверу: календарные dimension-таргеты идут как
        # "date" (отсекает не-date dtypes), всё остальное — как "label"
        # (отсекает system_timestamp / metric / id и принуждает резолвер
        # выбирать именно label-колонки в dim/ref).
        dim_target_str = str(dim.target or "")
        dim_role = "date" if _target_looks_calendar(dim_target_str) else "label"
        choice = _choose_column_across_tables(
            table_structures=table_structures,
            table_types=table_types,
            schema_loader=schema_loader,
            target=dim.target,
            prefer_fact=False,
            role_hint=dim_role,
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
        if _filter_is_calendar_literal(flt):
            continue
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


def _filter_is_calendar_literal(flt: Any) -> bool:
    target = str(getattr(flt, "target", "") or "")
    value = getattr(flt, "value", None)
    return bool(_parse_calendar_period(value) and _target_looks_calendar(f"{target} {value}"))


_ENTITY_FLAG_PREFIXES = ("is_", "has_", "flag_")


def derive_entity_flag_filters(
    *,
    query_spec: QuerySpec | dict[str, Any],
    selected_columns: dict[str, dict[str, list[str]]],
    schema_loader: Any,
    llm_invoker: Any = None,
    user_input: str = "",
) -> list[dict[str, Any]]:
    """When the user's entity (e.g. «Задача») resolves to a boolean flag on
    the main table (e.g. `is_task`), emit a synthetic `target=column,
    value=True` filter spec so the query semantics ("count tasks") are
    preserved.

    Uses the existing entity_resolver for semantic matching (cross-lingual,
    embedding-based) and then checks whether the resolved column is a
    boolean/flag type. Empty when no entity-flag match is found.

    Why this is needed: without it, the boolean column gets pulled into the
    WHERE pipeline only via filter_intents — and if the intent's value is a
    text literal like "фактический отток", a dtype-check kills the
    condition, silently dropping `is_task = TRUE`. Surfacing the entity →
    flag match here keeps the structural filter independent of any text
    filter on the same row.
    """
    spec = _coerce_spec(query_spec)
    if spec is None or not spec.entities or not selected_columns:
        return []
    existing_filter_targets = {
        str(f.target or "").strip().lower()
        for f in (spec.filters or [])
    }
    table_keys = list(selected_columns.keys())
    synthetic: list[dict[str, Any]] = []
    seen_cols: set[tuple[str, str]] = set()
    # Защита от галлюцинаций LLM: если entity ВООБЩЕ не упомянут в
    # исходном запросе (ни через стем, ни буквально), не используем его
    # для synthetic flag-filter. DeepSeek иногда добавляет «задача» в
    # entities для запроса про отток — без проверки это вешает is_task=TRUE.
    user_input_stems = {
        _stem_token(tok) for tok in re.findall(r"\w+", (user_input or "").lower())
        if len(tok) >= 3
    }
    for entity in spec.entities:
        entity_name = (entity.canonical or entity.name or "").strip()
        if not entity_name:
            continue
        if user_input_stems:
            entity_stems = {
                _stem_token(tok) for tok in re.findall(r"\w+", entity_name.lower())
                if len(tok) >= 3
            }
            if entity_stems and not (entity_stems & user_input_stems):
                logger.debug(
                    "ColumnBinding: entity «%s» отсутствует в user_input — "
                    "пропускаем synthetic flag-filter",
                    entity_name,
                )
                continue
        resolved_table: str | None = None
        resolved_column: str | None = None
        resolved_confidence: float = 0.9
        resolution_source: str = ""

        # Pass 1: semantic match via entity_resolver (cross-lingual via LLM
        # tiebreak when available, embeddings otherwise).
        try:
            resolution = resolve_entity_to_columns(
                entity_term=entity_name,
                user_input="",
                candidate_table_keys=table_keys,
                schema_loader=schema_loader,
                llm_invoker=llm_invoker,
                role_hint="filter",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "ColumnBinding: entity_flag resolver failed for %r: %s",
                entity_name, exc,
            )
            resolution = None  # type: ignore[assignment]
        if (
            resolution is not None
            and resolution.matched
            and resolution.column
            and resolution.table_key
            and _column_is_boolean_flag(
                schema_loader=schema_loader,
                table_key=resolution.table_key,
                column=resolution.column,
            )
        ):
            resolved_table = resolution.table_key
            resolved_column = resolution.column
            resolved_confidence = float(resolution.confidence or 0.9)
            resolution_source = "entity_resolver"

        # Pass 2: description-based fallback. The entity_resolver can miss
        # cross-lingual cases (e.g. «Задача» ↔ `is_task` with English column
        # name + Russian description). We walk boolean columns directly and
        # match the entity stem against the column's description text —
        # robust without requiring synonym tables or LLM calls.
        if resolved_column is None:
            fallback = _find_flag_column_by_description(
                entity_name=entity_name,
                table_keys=table_keys,
                schema_loader=schema_loader,
            )
            if fallback is not None:
                resolved_table, resolved_column = fallback
                resolved_confidence = 0.8
                resolution_source = "description_match"

        if resolved_table is None or resolved_column is None:
            continue
        col_key = (resolved_table, resolved_column.lower())
        if col_key in seen_cols:
            continue
        if resolved_column.lower() in existing_filter_targets:
            continue
        seen_cols.add(col_key)
        roles = selected_columns.setdefault(resolved_table, {})
        roles.setdefault("filter", [])
        if resolved_column not in roles["filter"]:
            roles["filter"].append(resolved_column)
        synthetic.append({
            "target": resolved_column,
            "operator": "=",
            "value": True,
            "value_kind": "literal",
            "evidence": [{
                "source": "column_binding",
                "text": (
                    f"entity «{entity_name}» → flag column {resolved_column} "
                    f"(via {resolution_source})"
                ),
                "confidence": resolved_confidence,
            }],
            "confidence": resolved_confidence,
        })
        existing_filter_targets.add(resolved_column.lower())
        logger.info(
            "ColumnBinding: entity «%s» → synthetic filter %s.%s = TRUE (via %s)",
            entity_name, resolved_table, resolved_column, resolution_source,
        )
    return synthetic


_RU_STEM_SUFFIXES = (
    "ами", "ями", "ого", "ему", "ыми", "его", "ой", "ом",
    "ам", "ям", "ах", "ях", "ев", "ов", "ью", "ия", "ие",
    "ый", "ий", "ая", "яя", "ое", "ее",
    "ы", "и", "а", "я", "е", "у", "ю", "о",
)


def _stem_token(token: str) -> str:
    """Tiny Russian-friendly stemmer: strip one common suffix, keep ≥3 chars."""
    t = token.lower()
    for suffix in _RU_STEM_SUFFIXES:
        if len(t) > len(suffix) + 2 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def _find_flag_column_by_description(
    *,
    entity_name: str,
    table_keys: list[str],
    schema_loader: Any,
) -> tuple[str, str] | None:
    """Walk boolean columns across the given tables; return (table_key,
    column) when the column's description shares a ≥4-char stem with the
    entity name. Used as a cross-lingual fallback when the embedding-based
    entity_resolver misses (e.g. «Задача» vs English `is_task`).
    """
    entity_stems = {
        _stem_token(tok) for tok in re.findall(r"\w+", entity_name.lower())
        if len(tok) >= 3
    }
    entity_stems = {s for s in entity_stems if s and len(s) >= 4}
    if not entity_stems:
        return None
    for table_key in table_keys:
        if "." not in table_key:
            continue
        schema, table = table_key.split(".", 1)
        try:
            cols_df = schema_loader.get_table_columns(schema, table)
        except Exception:  # noqa: BLE001
            continue
        if cols_df is None or getattr(cols_df, "empty", True):
            continue
        for _, row in cols_df.iterrows():
            col_name = str(row.get("column_name") or "").strip()
            if not col_name:
                continue
            col_lower = col_name.lower()
            dtype = str(row.get("data_type") or "").lower()
            is_flag = ("bool" in dtype) or col_lower.startswith(_ENTITY_FLAG_PREFIXES)
            if not is_flag:
                continue
            description = str(row.get("description") or "").lower()
            if not description:
                continue
            # Содержимое скобок — контекстная сноска, не основная семантика.
            # «Признак ОКТМО (принадлежит ГОСБ эмиссии карт)» не должен
            # матчиться по entity «ГОСБ» — это даст ложный фильтр. Срезаем
            # parenthetical content универсально.
            description = re.sub(r"\([^)]*\)", " ", description)
            desc_tokens = re.findall(r"\w+", description)
            desc_stems = {_stem_token(tok) for tok in desc_tokens if len(tok) >= 3}
            for left in entity_stems:
                for right in desc_stems:
                    if not right or len(right) < 4:
                        continue
                    if left == right:
                        return table_key, col_name
                    if min(len(left), len(right)) >= 4 and (
                        left.startswith(right) or right.startswith(left)
                    ):
                        return table_key, col_name
    return None


def _column_is_boolean_flag(
    *,
    schema_loader: Any,
    table_key: str,
    column: str,
) -> bool:
    if not column or "." not in table_key:
        return False
    schema, table = table_key.split(".", 1)
    try:
        cols_df = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        return False
    if cols_df is None or getattr(cols_df, "empty", True):
        return False
    col_lower = column.lower()
    for _, row in cols_df.iterrows():
        if str(row.get("column_name") or "").strip().lower() != col_lower:
            continue
        dtype = str(row.get("data_type") or "").lower()
        if "bool" in dtype:
            return True
        if col_lower.startswith(_ENTITY_FLAG_PREFIXES):
            return True
        return False
    return False


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
            # Для label/dimension-целей канонический справочник должен побеждать
            # денормализованную копию колонки внутри факта (часто fact-таблицы
            # дублируют label-колонки ради удобства аналитики, но source of truth —
            # это dim/ref). Бонус выбран больше, чем максимальная разница
            # text_overlap × 1000 между exact-match (1.0) и substring (0.85) ≈ 150,
            # с запасом на структурные бонусы (PK, not_null).
            score += 250.0
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
