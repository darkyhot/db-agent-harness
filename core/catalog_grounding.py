"""Ground QuerySpec against the physical catalog.

The grounding layer is deliberately contract-oriented: it binds semantic
requests to catalog objects, records confidence/evidence, and asks for a typed
clarification when the binding is too weak.  Language understanding belongs to
QuerySpec creation, not to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any

from core.join_analysis import detect_table_type

logger = logging.getLogger(__name__)
from core.filter_ranking import rank_filter_candidates
from core.query_ir import (
    ClarificationSpec,
    Evidence,
    PlanIR,
    QuerySpec,
    SourceBinding,
)
from core.semantic_registry import find_tables_for_term


@dataclass
class GroundingResult:
    """Physical bindings and compatibility projections for the graph."""

    query_spec: QuerySpec
    sources: list[SourceBinding] = field(default_factory=list)
    plan_ir: PlanIR | None = None
    clarification: ClarificationSpec | None = None
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": [
                {
                    "schema": s.schema,
                    "table": s.table,
                    "reason": s.reason,
                    "confidence": s.confidence,
                    "evidence": [e.to_dict() for e in s.evidence],
                }
                for s in self.sources
            ],
            "warnings": list(self.warnings),
            "confidence": self.confidence,
            "needs_clarification": self.needs_clarification,
            "clarification": self.clarification.to_dict() if self.clarification else None,
        }


def ground_query_spec(
    *,
    query_spec: QuerySpec,
    schema_loader,
    user_input: str,
    max_sources: int = 3,
    min_confidence: float = 0.35,
) -> GroundingResult:
    """Bind QuerySpec sources to real catalog tables.

    Args:
        query_spec: Semantic request produced by query_interpreter.
        schema_loader: Existing SchemaLoader instance.
        user_input: Original text, used only as a search term fallback.
        max_sources: Maximum number of candidate source tables to keep.
        min_confidence: Below this confidence we ask for clarification instead
            of silently selecting a table.
    """
    semantic_frame: dict[str, Any] = {}

    if query_spec.clarification_needed and query_spec.clarification:
        return GroundingResult(
            query_spec=query_spec,
            clarification=query_spec.clarification,
            confidence=query_spec.confidence,
        )

    if query_spec.task == "inspect_schema":
        # Schema questions may be answered from the catalog without forcing a
        # physical table binding.
        return GroundingResult(
            query_spec=query_spec,
            sources=[],
            plan_ir=PlanIR(
                metrics=query_spec.metrics,
                dimensions=query_spec.dimensions,
                filters=query_spec.filters,
                joins=query_spec.join_constraints,
                time_range=query_spec.time_range,
                having=query_spec.having,
                order_by=query_spec.order_by,
                limit=query_spec.limit,
                confidence=max(query_spec.confidence, 0.5),
            ),
            confidence=max(query_spec.confidence, 0.5),
        )

    sources: list[SourceBinding] = []
    warnings: list[str] = []
    excluded = _excluded_source_names(query_spec, schema_loader)

    for constraint in query_spec.source_constraints:
        bound = _bind_explicit_source(constraint, schema_loader)
        if bound is not None:
            if bound.full_name.lower() not in excluded:
                sources.append(bound)
            continue
        if constraint.required:
            warnings.append(
                "required source was not found in catalog: "
                + ".".join(part for part in (constraint.schema, constraint.table or constraint.semantic) if part)
            )

    search_terms = _search_terms(query_spec, user_input)
    has_required_explicit_sources = any(
        constraint.required and constraint.schema and constraint.table
        for constraint in query_spec.source_constraints
    )
    if not has_required_explicit_sources:
        for binding in _lexicon_anchor_bindings(query_spec, schema_loader):
            if binding.full_name.lower() in excluded:
                continue
            if not _has_source(sources, binding):
                sources.append(binding)
            if len(sources) >= max_sources:
                break

        for binding in _score_catalog_bindings(
            query_spec,
            schema_loader,
            user_input,
            top_n=max_sources,
            semantic_frame=semantic_frame,
        ):
            if binding.full_name.lower() in excluded:
                continue
            if not _has_source(sources, binding):
                sources.append(binding)
            if len(sources) >= max_sources:
                break

        for term in search_terms:
            for binding in _search_table_bindings(term, schema_loader, top_n=max_sources):
                if binding.full_name.lower() in excluded:
                    continue
                if not _has_source(sources, binding):
                    sources.append(binding)
                if len(sources) >= max_sources:
                    break
            if len(sources) >= max_sources:
                break

    sources = _enrich_low_quality_dimension_sources(
        sources,
        query_spec=query_spec,
        schema_loader=schema_loader,
        user_input=user_input,
        semantic_frame=semantic_frame,
    )
    if excluded:
        sources = [source for source in sources if source.full_name.lower() not in excluded]

    sources = _prune_count_sources_covered_by_single_dictionary(
        sources,
        query_spec=query_spec,
        schema_loader=schema_loader,
    )

    sources = _prune_sources_to_minimal_covering_table(
        sources,
        query_spec=query_spec,
        schema_loader=schema_loader,
        user_input=user_input,
    )

    sources = _prune_unrequested_helper_sources(
        sources,
        query_spec=query_spec,
        schema_loader=schema_loader,
        user_input=user_input,
        semantic_frame=semantic_frame,
    )

    if not sources and query_spec.task == "answer_data":
        clarification = ClarificationSpec(
            question="Не удалось уверенно выбрать таблицу по запросу. Уточните, пожалуйста, источник данных или добавьте описание нужной витрины.",
            reason="catalog_grounding_no_source",
            field="source_constraints",
            evidence=[Evidence(source="catalog_grounder", text=user_input, confidence=0.0)],
        )
        return GroundingResult(
            query_spec=query_spec,
            clarification=clarification,
            warnings=warnings,
            confidence=0.0,
        )

    best_conf = max((source.confidence for source in sources), default=0.0)
    if sources and best_conf < min_confidence:
        options = [source.full_name for source in sources[:max_sources]]
        clarification = ClarificationSpec(
            question="Я нашёл несколько слабых кандидатов на источник данных. Уточните, какую таблицу использовать.",
            reason="catalog_grounding_low_confidence",
            field="source_constraints",
            options=options,
            evidence=[Evidence(source="catalog_grounder", text=user_input, confidence=best_conf)],
        )
        return GroundingResult(
            query_spec=query_spec,
            sources=sources,
            clarification=clarification,
            warnings=warnings,
            confidence=best_conf,
        )

    plan_ir = PlanIR(
        main_source=sources[0] if sources else None,
        sources=sources,
        metrics=query_spec.metrics,
        dimensions=query_spec.dimensions,
        filters=query_spec.filters,
        joins=query_spec.join_constraints,
        time_range=query_spec.time_range,
        having=query_spec.having,
        order_by=query_spec.order_by,
        limit=query_spec.limit,
        confidence=min(1.0, (query_spec.confidence + best_conf) / 2.0) if sources else query_spec.confidence,
        warnings=warnings,
    )
    return GroundingResult(
        query_spec=query_spec,
        sources=sources,
        plan_ir=plan_ir,
        warnings=warnings,
        confidence=plan_ir.confidence,
    )


def _bind_explicit_source(constraint, schema_loader) -> SourceBinding | None:
    schema = constraint.schema
    table = constraint.table
    if table and "." in table and not schema:
        schema, table = table.split(".", 1)
    if not schema or not table:
        return None
    df = schema_loader.tables_df
    if df.empty:
        return None
    mask = (
        df["schema_name"].astype(str).str.lower() == str(schema).lower()
    ) & (
        df["table_name"].astype(str).str.lower() == str(table).lower()
    )
    if df[mask].empty:
        return None
    row = df[mask].iloc[0]
    return SourceBinding(
        schema=str(row["schema_name"]),
        table=str(row["table_name"]),
        reason="explicit_source_constraint",
        confidence=max(0.95, constraint.confidence),
        evidence=constraint.evidence or [Evidence(source="query_spec", text=f"{schema}.{table}", confidence=1.0)],
    )


def _excluded_source_names(query_spec: QuerySpec, schema_loader) -> set[str]:
    excluded: set[str] = set()
    for constraint in getattr(query_spec, "excluded_source_constraints", []) or []:
        bound = _bind_explicit_source(constraint, schema_loader)
        if bound is not None:
            excluded.add(bound.full_name.lower())
            continue
        schema = constraint.schema
        table = constraint.table
        if table and "." in table and not schema:
            schema, table = table.split(".", 1)
        if schema and table:
            excluded.add(f"{schema}.{table}".lower())
    return excluded


def _lexicon_anchor_bindings(
    query_spec: QuerySpec,
    schema_loader,
) -> list[SourceBinding]:
    """Подобрать таблицы детерминированно через semantic_lexicon.

    Для каждого «термина» из QuerySpec (entities/metrics/dimensions) дёргаем
    `find_tables_for_term` и считаем, сколько раз каждая таблица встретилась.
    Чем выше покрытие термов, тем выше confidence — это якоря для дальнейшего
    scoring'а, не финальный список.
    """
    try:
        lexicon = schema_loader.get_semantic_lexicon()
    except Exception as exc:  # noqa: BLE001
        logger.debug("lexicon anchors: get_semantic_lexicon failed: %s", exc)
        return []
    if not lexicon:
        return []

    terms: list[str] = []
    for entity in query_spec.entities or []:
        for value in (entity.canonical, entity.name, entity.target_column_hint):
            text = str(value or "").strip()
            if text and text not in terms:
                terms.append(text)
    for metric in query_spec.metrics or []:
        text = str(metric.target or "").strip()
        if text and text not in terms:
            terms.append(text)
    for dim in query_spec.dimensions or []:
        for value in (dim.target, getattr(dim, "label", "")):
            text = str(value or "").strip()
            if text and text not in terms:
                terms.append(text)
    for fil in query_spec.filters or []:
        text = str(fil.target or "").strip()
        if text and text not in terms:
            terms.append(text)
    if not terms:
        return []

    table_hits: dict[str, dict[str, Any]] = {}
    for term in terms:
        for table_key in find_tables_for_term(term, lexicon):
            entry = table_hits.setdefault(
                table_key.lower(),
                {"key": table_key, "terms": [], "count": 0},
            )
            entry["count"] += 1
            if term not in entry["terms"]:
                entry["terms"].append(term)

    if not table_hits:
        return []

    bindings: list[SourceBinding] = []
    for entry in table_hits.values():
        parts = entry["key"].split(".", 1)
        if len(parts) != 2:
            continue
        schema_name, table_name = parts
        coverage = entry["count"] / max(len(terms), 1)
        confidence = min(0.95, 0.55 + 0.4 * coverage)
        matched_terms = ", ".join(entry["terms"])
        bindings.append(
            SourceBinding(
                schema=schema_name,
                table=table_name,
                reason=f"semantic_lexicon:{matched_terms}",
                confidence=confidence,
                evidence=[
                    Evidence(
                        source="semantic_lexicon",
                        text=matched_terms,
                        confidence=confidence,
                    )
                ],
            )
        )
    bindings.sort(key=lambda b: b.confidence, reverse=True)
    return bindings


def _search_table_bindings(term: str, schema_loader, top_n: int) -> list[SourceBinding]:
    result: list[SourceBinding] = []
    try:
        df = schema_loader.search_tables(term, top_n=top_n)
    except Exception:
        df = None
    if df is not None and not df.empty:
        for idx, row in df.iterrows():
            score = 0.8 if idx == 0 else max(0.35, 0.7 - idx * 0.1)
            result.append(
                SourceBinding(
                    schema=str(row.get("schema_name") or ""),
                    table=str(row.get("table_name") or ""),
                    reason=f"catalog_search:{term}",
                    confidence=score,
                    evidence=[Evidence(source="catalog_search", text=term, confidence=score)],
                )
            )
    try:
        semantic_hits = schema_loader.semantic_search_tables(term, top_k=top_n)
    except Exception:
        semantic_hits = []
    for schema, table, score in semantic_hits:
        binding = SourceBinding(
            schema=schema,
            table=table,
            reason=f"semantic_catalog_search:{term}",
            confidence=max(0.0, min(1.0, float(score))),
            evidence=[Evidence(source="semantic_catalog_search", text=term, confidence=float(score))],
        )
        if not _has_source(result, binding):
            result.append(binding)
    return [item for item in result if item.schema and item.table]


def _score_catalog_bindings(
    query_spec: QuerySpec,
    schema_loader,
    user_input: str,
    top_n: int,
    semantic_frame: dict[str, Any] | None = None,
) -> list[SourceBinding]:
    """Score real catalog tables against semantic slots from QuerySpec."""
    df = getattr(schema_loader, "tables_df", None)
    if df is None or df.empty:
        return []

    metric_terms = [m.target for m in query_spec.metrics if m.target]
    entity_terms = _entity_terms(query_spec)
    dimension_terms = [d.target for d in query_spec.dimensions if d.target]
    filter_terms = [f.target for f in query_spec.filters if f.target]
    join_terms = [j.key for j in query_spec.join_constraints if j.key]
    source_terms = [
        item
        for source in query_spec.source_constraints
        for item in (source.table, source.semantic)
        if item
    ]
    all_terms = list(dict.fromkeys(entity_terms + metric_terms + dimension_terms + filter_terms + join_terms + source_terms))
    if not all_terms:
        all_terms = [user_input]

    scored: list[tuple[float, SourceBinding]] = []
    for _, row in df.iterrows():
        schema = str(row.get("schema_name") or "")
        table = str(row.get("table_name") or "")
        if not schema or not table:
            continue
        description = str(row.get("description") or "")
        try:
            cols = schema_loader.get_table_columns(schema, table)
        except Exception:  # noqa: BLE001
            cols = None
        if cols is None or cols.empty:
            continue

        table_type = detect_table_type(table, cols)
        score = _score_text(table, description, source_terms, 1.4)
        if _source_mentioned_in_text(schema, table, user_input):
            score += 6.0
        metric_score = 0.0
        entity_score = 0.0
        entity_coverage = 0
        dimension_score = 0.0
        for _, col in cols.iterrows():
            col_name = str(col.get("column_name") or "")
            col_desc = _row_text(col)
            metric_score += _score_text(col_name, col_desc, metric_terms, 3.0)
            for term in entity_terms:
                if _best_attribute_column_score(col, term) > 0:
                    entity_coverage += 1
                    entity_score += _best_attribute_column_score(col, term)
                    break
            dimension_score += _score_text(col_name, col_desc, dimension_terms, 2.0)
            score += _score_text(col_name, col_desc, filter_terms, 1.3)
            score += _score_text(col_name, col_desc, join_terms, 1.0)

        if query_spec.strategy == "count_attributes":
            full_coverage = _table_count_attribute_coverage(cols, entity_terms)
            if full_coverage < len(entity_terms):
                continue
            score += full_coverage * 6.0 + entity_score
            score += _score_text(table, description, entity_terms, 2.0)
            if table_type in {"dim", "ref"}:
                score += 8.0
            elif table_type == "fact":
                score -= 4.0
        else:
            score += metric_score + dimension_score
        frame_score = _frame_table_support_score(
            schema_loader=schema_loader,
            schema=schema,
            table=table,
            query_spec=query_spec,
            user_input=user_input,
            semantic_frame=semantic_frame,
        )
        score += frame_score
        if metric_score and table_type == "fact":
            score += 0.7
        if metric_score and _count_id_metrics(query_spec) and table_type in {"dim", "ref"}:
            score += 1.6
        if dimension_score and table_type in {"dim", "ref"}:
            score += 0.8
        if not metric_score and dimension_score and table_type == "fact":
            score -= 0.5

        if score <= 0:
            continue
        confidence = min(0.95, max(0.35, score / 12.0))
        scored.append((
            score,
            SourceBinding(
                schema=schema,
                table=table,
                reason="query_spec_slot_score",
                confidence=confidence,
                evidence=[Evidence(source="catalog_slot_score", text=", ".join(all_terms), confidence=confidence)],
            ),
        ))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [binding for _, binding in scored[:top_n]]


def _count_id_metrics(query_spec: QuerySpec) -> bool:
    metrics = query_spec.metrics or []
    return bool(metrics) and all(
        metric.operation == "count"
        and str(metric.target or "").lower().endswith(("_id", "id"))
        for metric in metrics
    )


def _entity_terms(query_spec: QuerySpec) -> list[str]:
    terms: list[str] = []
    for entity in query_spec.entities or []:
        value = entity.target_column_hint or entity.canonical or entity.name
        if value:
            terms.append(value)
    if query_spec.strategy == "count_attributes" and not terms:
        terms.extend(metric.target for metric in query_spec.metrics if metric.operation == "count" and metric.target)
    return list(dict.fromkeys(str(term).strip() for term in terms if str(term).strip()))


def _table_count_attribute_coverage(cols, entity_terms: list[str]) -> int:
    covered = 0
    for term in entity_terms:
        if any(_best_attribute_column_score(row, term) > 0 for _, row in cols.iterrows()):
            covered += 1
    return covered


def _best_attribute_column_score(row: Any, term: str) -> float:
    """Универсальный score «колонка ↔ entity term» через description/name overlap.

    Без захардкоженных доменных алиасов — все «тб»/«госб»-маппинги уходят на
    уровень entity_resolver. Здесь только структурные структурные бонусы:
    PK и `_id`-суффикс (не язык, а SQL-конвенция).
    """
    col_name = str(row.get("column_name") or "").strip()
    if not col_name:
        return 0.0
    desc = _row_text(row)
    score = _score_text(col_name, desc, [term], 1.0)
    if score <= 0:
        return 0.0
    if bool(row.get("is_primary_key", False)):
        score += 1.5
    if col_name.lower().endswith("_id"):
        score += 0.8
    return score


def _score_text(name: str, description: str, terms: list[str], weight: float) -> float:
    score = 0.0
    haystack = _normalize(f"{name} {description}")
    name_norm = _normalize(name)
    for term in terms:
        term_norm = _normalize(str(term or ""))
        if not term_norm:
            continue
        if term_norm == name_norm:
            score += weight * 2.0
        elif term_norm in name_norm or name_norm in term_norm:
            score += weight * 1.5
        elif term_norm in haystack:
            score += weight
        else:
            term_tokens = set(term_norm.split())
            hay_tokens = set(haystack.split())
            if term_tokens and term_tokens <= hay_tokens:
                score += weight * 0.8
    return score


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zа-яё_]+", " ", str(text).lower())).strip()


def _row_text(row: Any) -> str:
    return str(row.get("description") or "").strip()


def _column_match_score(column_name: str, description: str, term: str) -> float:
    """Score a physical column as a candidate for a semantic slot."""
    return _score_text(column_name, description, [term], 1.0)


def _float_meta(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _table_key(schema: str, table: str) -> str:
    return f"{schema}.{table}"


def _source_key(source: SourceBinding) -> str:
    return source.full_name.lower()


def _dimension_terms(query_spec: QuerySpec) -> list[str]:
    terms: list[str] = []
    for dim in query_spec.dimensions:
        target = str(dim.target or "").strip()
        if not target:
            continue
        if target.lower() in {"date", "дата", "дате", "day", "month", "year"}:
            continue
        terms.append(target)
    return list(dict.fromkeys(terms))


def _best_dimension_column(
    schema_loader,
    schema: str,
    table: str,
    term: str,
) -> dict[str, Any] | None:
    try:
        cols = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        return None
    if cols is None or cols.empty:
        return None

    best: tuple[float, dict[str, Any]] | None = None
    for _, row in cols.iterrows():
        col_name = str(row.get("column_name") or "").strip()
        if not col_name:
            continue
        desc = _row_text(row)
        score = _column_match_score(col_name, desc, term)
        if score <= 0:
            continue
        dtype = str(row.get("dType") or "").lower()
        if dtype.startswith(("int", "bigint", "smallint", "numeric", "decimal", "float", "double", "real")):
            score *= 0.55
        not_null = _float_meta(row.get("not_null_perc"))
        unique = _float_meta(row.get("unique_perc"))
        item = {
            "schema": schema,
            "table": table,
            "column": col_name,
            "description": desc,
            "score": score,
            "not_null_perc": not_null,
            "unique_perc": unique,
        }
        rank = score * 1000.0 + not_null * 2.0 + min(unique, 100.0) * 0.1
        if best is None or rank > best[0]:
            best = (rank, item)
    return best[1] if best else None


def _all_dimension_candidates(
    schema_loader,
    term: str,
) -> list[dict[str, Any]]:
    df = getattr(schema_loader, "tables_df", None)
    if df is None or df.empty:
        return []

    candidates: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        schema = str(row.get("schema_name") or "").strip()
        table = str(row.get("table_name") or "").strip()
        if not schema or not table:
            continue
        candidate = _best_dimension_column(schema_loader, schema, table, term)
        if candidate:
            candidates.append(candidate)
    return candidates


def _joinable_key_between(
    schema_loader,
    left_schema: str,
    left_table: str,
    right_schema: str,
    right_table: str,
) -> dict[str, Any] | None:
    """Find a metadata-supported key that can join two tables."""
    try:
        left_cols = schema_loader.get_table_columns(left_schema, left_table)
        right_cols = schema_loader.get_table_columns(right_schema, right_table)
    except Exception:  # noqa: BLE001
        return None
    if left_cols is None or right_cols is None or left_cols.empty or right_cols.empty:
        return None

    right_by_name = {
        str(row.get("column_name") or "").lower(): row
        for _, row in right_cols.iterrows()
        if str(row.get("column_name") or "").strip()
    }
    best: tuple[float, dict[str, Any]] | None = None

    def _key_like(row: Any) -> bool:
        name = str(row.get("column_name") or "").lower()
        if bool(row.get("is_primary_key", False)):
            return True
        if name.endswith(("_id", "_code", "_num", "_no")) or name in {"inn", "kpp", "ogrn"}:
            return True
        return _float_meta(row.get("unique_perc")) >= 50.0 and _float_meta(row.get("not_null_perc")) >= 50.0

    for _, lrow in left_cols.iterrows():
        left_col = str(lrow.get("column_name") or "").strip()
        if not left_col:
            continue
        l_fk = str(lrow.get("foreign_key_target") or "").strip().lower()
        for rname, rrow in right_by_name.items():
            right_col = str(rrow.get("column_name") or "").strip()
            if not right_col:
                continue
            exact_name = left_col.lower() == rname
            r_ref = f"{right_schema}.{right_table}.{right_col}".lower()
            l_ref = f"{left_schema}.{left_table}.{left_col}".lower()
            r_fk = str(rrow.get("foreign_key_target") or "").strip().lower()
            fk_match = bool(l_fk and l_fk == r_ref) or bool(r_fk and r_fk == l_ref)
            if not exact_name and not fk_match:
                continue
            if exact_name and not fk_match and not (_key_like(lrow) or _key_like(rrow)):
                continue

            left_nn = _float_meta(lrow.get("not_null_perc"))
            right_nn = _float_meta(rrow.get("not_null_perc"))
            left_unique = _float_meta(lrow.get("unique_perc"))
            right_unique = _float_meta(rrow.get("unique_perc"))
            score = left_nn + right_nn + min(left_unique, 100.0) * 0.5 + min(right_unique, 100.0) * 0.5
            if fk_match:
                score += 80.0
            lower = left_col.lower()
            if lower.endswith(("_id", "_code")) or lower in {"inn", "kpp", "ogrn"}:
                score += 20.0
            item = {
                "left": f"{left_schema}.{left_table}.{left_col}",
                "right": f"{right_schema}.{right_table}.{right_col}",
                "score": score,
            }
            if best is None or score > best[0]:
                best = (score, item)
    return best[1] if best else None


def _enrich_low_quality_dimension_sources(
    sources: list[SourceBinding],
    *,
    query_spec: QuerySpec,
    schema_loader,
    user_input: str,
    semantic_frame: dict[str, Any] | None,
    low_fill_threshold: float = 50.0,
    good_fill_threshold: float = 80.0,
) -> list[SourceBinding]:
    """Add better joinable sources for requested dimensions with poor fill rate."""
    if not sources or not query_spec.dimensions:
        return sources

    result = list(sources)
    existing = {_source_key(source) for source in result}
    table_types: dict[str, str] = {}
    for source in result:
        try:
            cols = schema_loader.get_table_columns(source.schema, source.table)
            table_types[source.full_name] = detect_table_type(source.table, cols)
        except Exception:  # noqa: BLE001
            table_types[source.full_name] = "unknown"

    anchors = [
        source for source in result
        if table_types.get(source.full_name) == "fact"
    ] or result[:1]

    for term in _dimension_terms(query_spec):
        anchor_quality: list[float] = []
        for anchor in anchors:
            current = _best_dimension_column(schema_loader, anchor.schema, anchor.table, term)
            anchor_quality.append(current["not_null_perc"] if current else 0.0)
        if anchor_quality and max(anchor_quality) >= low_fill_threshold:
            continue

        candidates = [
            item for item in _all_dimension_candidates(schema_loader, term)
            if item["not_null_perc"] >= good_fill_threshold
            and f"{item['schema']}.{item['table']}".lower() not in existing
        ]
        ranked: list[tuple[float, dict[str, Any], dict[str, Any], SourceBinding]] = []
        for candidate in candidates:
            candidate_key = _table_key(candidate["schema"], candidate["table"])
            try:
                cand_cols = schema_loader.get_table_columns(candidate["schema"], candidate["table"])
                cand_type = detect_table_type(candidate["table"], cand_cols)
            except Exception:  # noqa: BLE001
                cand_type = "unknown"
            type_bonus = 40.0 if cand_type in {"dim", "ref", "unknown"} else -35.0
            for anchor in anchors:
                join_key = _joinable_key_between(
                    schema_loader,
                    anchor.schema,
                    anchor.table,
                    candidate["schema"],
                    candidate["table"],
                )
                if not join_key:
                    continue
                rank = (
                    candidate["score"] * 1000.0
                    + candidate["not_null_perc"] * 3.0
                    + join_key["score"]
                    + type_bonus
                )
                binding = SourceBinding(
                    schema=candidate["schema"],
                    table=candidate["table"],
                    reason="dimension_quality_enrichment",
                    confidence=min(0.95, max(0.55, rank / 1500.0)),
                    evidence=[
                        Evidence(
                            source="dimension_quality",
                            text=(
                                f"{term}: {candidate_key}.{candidate['column']} "
                                f"not_null={candidate['not_null_perc']:.2f}; "
                                f"join={join_key['left']}={join_key['right']}"
                            ),
                            confidence=0.9,
                        )
                    ],
                )
                ranked.append((rank, candidate, join_key, binding))

        if not ranked:
            continue
        ranked.sort(key=lambda item: item[0], reverse=True)
        binding = ranked[0][3]
        if binding.full_name.lower() not in existing:
            result.append(binding)
            existing.add(binding.full_name.lower())

    return result


def _search_terms(query_spec: QuerySpec, user_input: str) -> list[str]:
    terms: list[str] = []
    for source in query_spec.source_constraints:
        if source.semantic:
            terms.append(source.semantic)
        if source.table:
            terms.append(source.table)
    terms.extend(metric.target for metric in query_spec.metrics if metric.target)
    terms.extend(_entity_terms(query_spec))
    terms.extend(dim.target for dim in query_spec.dimensions if dim.target)
    terms.extend(item.target for item in query_spec.filters if item.target)
    terms.append(user_input)
    clean_terms: list[str] = []
    for term in terms:
        text = str(term or "").strip()
        if text and text not in clean_terms:
            clean_terms.append(text)
    return clean_terms


def _has_source(sources: list[SourceBinding], binding: SourceBinding) -> bool:
    key = binding.full_name.lower()
    return any(item.full_name.lower() == key for item in sources)


def _frame_table_support_score(
    *,
    schema_loader,
    schema: str,
    table: str,
    query_spec: QuerySpec,
    user_input: str,
    semantic_frame: dict[str, Any] | None,
) -> float:
    table_key = f"{schema}.{table}"
    score = 0.0
    frame = semantic_frame or {}
    subject = str(frame.get("subject") or "").strip().lower()
    table_sem = schema_loader.get_table_semantics(schema, table)
    subjects = {str(v).strip().lower() for v in (table_sem.get("primary_subjects") or []) if str(v).strip()}
    grain = str(table_sem.get("grain") or "").strip().lower()
    if subject and (subject in subjects or subject == grain):
        score += 2.2
    elif subject and table_sem:
        score -= 0.4

    try:
        intent = query_spec.to_legacy_intent()
        if (semantic_frame or {}).get("filter_intents"):
            intent = {**intent, "filter_conditions": []}
        ranked = rank_filter_candidates(
            user_input=user_input,
            intent=intent,
            selected_tables=[table_key],
            schema_loader=schema_loader,
            semantic_frame=frame,
        )
    except Exception:  # noqa: BLE001
        ranked = {}
    for candidates in ranked.values():
        if not candidates:
            continue
        best = candidates[0]
        confidence = str(best.get("confidence") or "")
        if confidence == "high":
            score += 3.0
        elif confidence == "medium":
            score += 1.7
        else:
            score += 0.6

    event = str(frame.get("business_event") or "").strip()
    if event:
        score += min(2.0, _score_text(table, str(schema_loader.get_table_info(schema, table)), [event], 1.2))
    return score


def _prune_unrequested_helper_sources(
    sources: list[SourceBinding],
    *,
    query_spec: QuerySpec,
    schema_loader,
    user_input: str,
    semantic_frame: dict[str, Any] | None,
) -> list[SourceBinding]:
    if len(sources) <= 1 or query_spec.join_constraints or query_spec.dimensions:
        return sources
    frame = semantic_frame or {}
    if not (
        frame.get("filter_intents")
        or frame.get("subject")
        or frame.get("business_event")
        or query_spec.filters
    ):
        return sources

    kept: list[SourceBinding] = []
    min_support = 1.0 if query_spec.filters else 0.5
    for source in sources:
        score = _frame_table_support_score(
            schema_loader=schema_loader,
            schema=source.schema,
            table=source.table,
            query_spec=query_spec,
            user_input=user_input,
            semantic_frame=frame,
        )
        if score >= min_support:
            kept.append(source)
            logger.debug(
                "PruneHelper: keep %s (score=%.2f >= %.2f)",
                source.full_name, score, min_support,
            )
        else:
            logger.info(
                "PruneHelper: drop %s (frame_support_score=%.2f < %.2f)",
                source.full_name, score, min_support,
            )

    if not kept:
        logger.info(
            "PruneHelper: all candidates dropped, falling back to first source %s",
            sources[0].full_name if sources else "<none>",
        )
        return sources[:1]
    return kept


def _required_explicit_source_count(query_spec: QuerySpec) -> int:
    return sum(
        1 for source in (query_spec.source_constraints or [])
        if source.required and (source.table or source.schema or source.semantic)
    )


def _prune_sources_to_minimal_covering_table(
    sources: list[SourceBinding],
    *,
    query_spec: QuerySpec,
    schema_loader,
    user_input: str = "",
) -> list[SourceBinding]:
    """Keep one source when it covers all LLM-requested slots.

    This is a guardrail, not language interpretation: the LLM provides metrics,
    dimensions, filters, and joins; the deterministic layer only checks whether
    a single physical table already satisfies that contract.
    """
    if len(sources) <= 1:
        return sources
    if query_spec.join_constraints or _required_explicit_source_count(query_spec) > 1:
        return sources

    required_terms = [
        str(item.target or "").strip()
        for item in (query_spec.metrics or [])
        if str(item.target or "").strip() and item.target != "*"
    ]
    required_terms.extend(
        str(item.target or "").strip()
        for item in (query_spec.dimensions or [])
        if str(item.target or "").strip()
    )
    required_terms.extend(
        str(item.target or "").strip()
        for item in (query_spec.filters or [])
        if str(item.target or "").strip()
    )
    required_terms = list(dict.fromkeys(required_terms))
    if not required_terms:
        return sources

    candidates: list[tuple[float, SourceBinding]] = []
    for source in sources:
        if not _source_covers_query_slots(
            source,
            query_spec=query_spec,
            schema_loader=schema_loader,
            user_input=user_input,
        ):
            continue
        try:
            cols = schema_loader.get_table_columns(source.schema, source.table)
        except Exception:  # noqa: BLE001
            cols = None
        table_type = detect_table_type(source.table, cols) if cols is not None and not cols.empty else "unknown"
        type_bonus = 0.0
        if _count_id_metrics(query_spec) and table_type in {"dim", "ref", "unknown"}:
            type_bonus += 1.5
        mention_bonus = 2.0 if _source_mentioned_in_text(source.schema, source.table, user_input) else 0.0
        rank = source.confidence + type_bonus + mention_bonus
        candidates.append((rank, source))

    if not candidates:
        return sources
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = candidates[0][1]
    if len(sources) > 1:
        dropped = [src.full_name for src in sources if src.full_name != chosen.full_name]
        logger.info(
            "PruneMinimalCovering: keep %s (rank=%.2f), drop %s — single table covers all slots",
            chosen.full_name, candidates[0][0], dropped,
        )
    return [chosen]


_LABEL_TARGET_TOKENS = (
    "name", "label", "title",
    "название", "наименование", "имя",
)


def _target_is_label_slot(target: str) -> bool:
    """True если dimension target явно просит человеко-читаемое имя сущности."""
    t = str(target or "").lower()
    if not t:
        return False
    if t.endswith(("_name", "_label", "_title")):
        return True
    return any(token in t for token in _LABEL_TARGET_TOKENS)


def _dimension_requests_label(dim: Any) -> bool:
    return (
        _target_is_label_slot(getattr(dim, "target", ""))
        or _target_is_label_slot(getattr(dim, "label", ""))
    )


def _source_has_label_column_for_term(
    schema_loader,
    schema: str,
    table: str,
    term: str,
) -> bool:
    """Проверить, есть ли в таблице колонка с semantic_class='label', связанная с термином.

    Используется как guard для label-слотов: fact-таблица с одним лишь `gosb_id`
    не должна считаться покрывающей слот «название ГОСБ» — для этого нужен
    реальный label-атрибут (gosb_name/old_gosb_name/etc.).

    Связь термина с колонкой проверяется через `_column_match_score` — он
    учитывает и имя, и описание, что важно для случая «ГОСБ» (кириллица) →
    `gosb_name` (латиница): термин совпадает с описанием «Название ГОСБ».
    """
    try:
        cols = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        return False
    if cols is None or cols.empty:
        return False
    for _, row in cols.iterrows():
        col_name = str(row.get("column_name") or "").strip()
        if not col_name:
            continue
        try:
            sem = schema_loader.get_column_semantics(schema, table, col_name) or {}
        except Exception:  # noqa: BLE001
            sem = {}
        sem_class = str(sem.get("semantic_class") or "").lower()
        col_lower = col_name.lower()
        is_label_col = (
            sem_class == "label"
            or any(col_lower.endswith(suffix) for suffix in ("_name", "_label", "_title"))
        )
        if not is_label_col:
            continue
        desc = _row_text(row)
        if _column_match_score(col_name, desc, term) > 0:
            return True
    return False


def _source_covers_query_slots(
    source: SourceBinding,
    *,
    query_spec: QuerySpec,
    schema_loader,
    user_input: str = "",
) -> bool:
    try:
        cols = schema_loader.get_table_columns(source.schema, source.table)
    except Exception:  # noqa: BLE001
        return False
    if cols is None or cols.empty:
        return False

    source_name = source.full_name.lower()
    source_is_explicit = _source_mentioned_in_text(source.schema, source.table, user_input)
    for metric in query_spec.metrics or []:
        target = str(metric.target or "").strip()
        if not target or target == "*":
            continue
        if _best_metric_column_in_table(cols, target) is None:
            return False

    for dim in query_spec.dimensions or []:
        target = str(dim.target or "").strip()
        if not target:
            continue
        if dim.source_table and str(dim.source_table).strip().lower() != source_name:
            return False
        if _dimension_requests_label(dim):
            if not _source_has_label_column_for_term(
                schema_loader, source.schema, source.table, target
            ):
                return False
        dim_match = _best_dimension_column(schema_loader, source.schema, source.table, target)
        if dim_match is None:
            return False
        if not source_is_explicit and _float_meta(dim_match.get("not_null_perc")) < 50.0:
            return False

    for filt in query_spec.filters or []:
        target = str(filt.target or "").strip()
        if not target:
            continue
        if _best_metric_column_in_table(cols, target) is None:
            return False

    return True


def _source_mentioned_in_text(schema: str, table: str, text: str) -> bool:
    haystack = str(text or "").lower()
    table_l = str(table or "").lower()
    schema_l = str(schema or "").lower()
    return bool(
        table_l
        and (
            table_l in haystack
            or (schema_l and f"{schema_l}.{table_l}" in haystack)
        )
    )


def _prune_count_sources_covered_by_single_dictionary(
    sources: list[SourceBinding],
    *,
    query_spec: QuerySpec,
    schema_loader,
) -> list[SourceBinding]:
    """For dictionary cardinality counts, keep one source that covers all metrics.

    Queries like "сколько всего есть ТБ и ГОСБ" should be answered from the
    dictionary table containing both entity keys. Joinable fact/helper tables do
    not add requested information and only create row-count ambiguity.
    """
    if len(sources) <= 1 or query_spec.dimensions or query_spec.filters or query_spec.join_constraints:
        return sources
    targets = _entity_terms(query_spec)
    if not targets:
        targets = [
            str(metric.target or "").strip()
            for metric in (query_spec.metrics or [])
            if metric.operation == "count" and str(metric.target or "").strip()
        ]
    if len(targets) < 2:
        return sources

    best: tuple[int, float, SourceBinding] | None = None
    for source in sources:
        try:
            cols = schema_loader.get_table_columns(source.schema, source.table)
        except Exception:  # noqa: BLE001
            continue
        if cols is None or cols.empty:
            continue
        table_type = detect_table_type(source.table, cols)
        if table_type not in {"dim", "ref", "unknown"}:
            continue
        covered = 0
        pk_bonus = 0.0
        for target in targets:
            match = _best_metric_column_in_table(cols, target)
            if match is None:
                break
            covered += 1
            if bool(match.get("is_primary_key", False)):
                pk_bonus += 1.0
        if covered != len(targets):
            continue
        rank = (covered, pk_bonus + source.confidence)
        candidate = (rank[0], rank[1], source)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return sources
    chosen = best[2]
    if len(sources) > 1:
        dropped = [src.full_name for src in sources if src.full_name != chosen.full_name]
        logger.info(
            "PruneCountSingleDictionary: keep %s (covered=%d, rank=%.2f), drop %s",
            chosen.full_name, best[0], best[1], dropped,
        )
    return [chosen]


def _best_metric_column_in_table(cols, target: str):
    """Подобрать колонку под target по описанию/имени без доменных алиасов.

    Базовый сигнал — `_score_text` (токен-overlap по имени и описанию). PK-бонус
    повышает score, чтобы при равенстве выбрать основной идентификатор.
    """
    target_norm = _normalize(target)
    best: tuple[float, Any] | None = None
    for _, row in cols.iterrows():
        col_name = str(row.get("column_name") or "")
        desc = _row_text(row)
        score = _score_text(col_name, desc, [target_norm], 1.0)
        if score <= 0:
            continue
        if bool(row.get("is_primary_key", False)):
            score += 0.5
        candidate = (score, row)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best else None
