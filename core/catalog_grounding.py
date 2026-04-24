"""Ground QuerySpec against the physical catalog.

The grounding layer is deliberately contract-oriented: it binds semantic
requests to catalog objects, records confidence/evidence, and asks for a typed
clarification when the binding is too weak.  Language understanding belongs to
QuerySpec creation, not to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.query_ir import (
    ClarificationSpec,
    Evidence,
    PlanIR,
    QuerySpec,
    SourceBinding,
)


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
                confidence=max(query_spec.confidence, 0.5),
            ),
            confidence=max(query_spec.confidence, 0.5),
        )

    sources: list[SourceBinding] = []
    warnings: list[str] = []

    for constraint in query_spec.source_constraints:
        bound = _bind_explicit_source(constraint, schema_loader)
        if bound is not None:
            sources.append(bound)
            continue
        if constraint.required:
            warnings.append(
                "required source was not found in catalog: "
                + ".".join(part for part in (constraint.schema, constraint.table or constraint.semantic) if part)
            )

    search_terms = _search_terms(query_spec, user_input)
    for term in search_terms:
        for binding in _search_table_bindings(term, schema_loader, top_n=max_sources):
            if not _has_source(sources, binding):
                sources.append(binding)
            if len(sources) >= max_sources:
                break
        if len(sources) >= max_sources:
            break

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


def _search_terms(query_spec: QuerySpec, user_input: str) -> list[str]:
    terms: list[str] = []
    for source in query_spec.source_constraints:
        if source.semantic:
            terms.append(source.semantic)
        if source.table:
            terms.append(source.table)
    terms.extend(metric.target for metric in query_spec.metrics if metric.target)
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
