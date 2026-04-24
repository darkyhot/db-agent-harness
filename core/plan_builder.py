"""Build PlanIR from grounded QuerySpec data."""

from __future__ import annotations

from core.query_ir import PlanIR, QuerySpec, SourceBinding


def build_plan_ir(
    *,
    query_spec: QuerySpec,
    sources: list[SourceBinding],
    confidence: float,
    warnings: list[str] | None = None,
) -> PlanIR:
    """Create the structured plan boundary before SQL compilation."""
    best_source = sources[0] if sources else None
    return PlanIR(
        main_source=best_source,
        sources=sources,
        metrics=query_spec.metrics,
        dimensions=query_spec.dimensions,
        filters=query_spec.filters,
        joins=query_spec.join_constraints,
        time_range=query_spec.time_range,
        confidence=confidence,
        warnings=list(warnings or []),
    )
