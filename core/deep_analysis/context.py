"""Build business context for deep table analysis."""

from __future__ import annotations

from typing import Any

from core.deep_analysis.types import ColumnRole, TableAnalysisContext, TableProfile
from core.schema_loader import SchemaLoader

_ENTITY_NAME_PRIORITIES: tuple[tuple[str, int], ...] = (
    ("inn", 120),
    ("epk", 110),
    ("saphr", 105),
    ("employee", 100),
    ("client", 95),
    ("person", 90),
    ("account", 85),
    ("acc_", 85),
    ("agrmnt", 82),
    ("deal", 80),
    ("gosb", 70),
    ("tb_", 65),
)

_MEASURE_HINTS = (
    "amount", "amt", "sum", "salary", "payroll", "qty", "cnt", "count",
    "revenue", "outflow", "score", "rate", "perc", "доля", "сумм",
)


def build_table_analysis_context(
    profile: TableProfile,
    schema_loader: SchemaLoader | None,
) -> TableAnalysisContext:
    """Combine catalog metadata and live profile into business context."""
    table_meta: dict[str, Any] = {}
    if schema_loader is not None:
        try:
            table_meta = schema_loader.get_table_semantics(profile.schema, profile.table)
        except Exception:
            table_meta = {}

    descriptions = _column_descriptions(profile, schema_loader)
    semantic_classes = _column_semantic_classes(profile, schema_loader)
    synonyms = _column_synonyms(profile, schema_loader)

    time_axes = _rank_time_axes(profile, table_meta)
    entity_keys = _rank_entity_keys(profile, semantic_classes)
    measures = _rank_measures(profile, semantic_classes)
    dimensions = _rank_dimensions(profile, semantic_classes)
    flags = profile.representatives(profile.flag_columns())
    peer_groups = _rank_peer_groups(profile, dimensions)
    primary_subjects = [
        str(x) for x in (table_meta.get("primary_subjects") or [])
        if str(x).strip()
    ]
    grain = str(table_meta.get("grain") or "").strip()
    business_subject = _infer_business_subject(
        profile, grain, primary_subjects, entity_keys
    )

    return TableAnalysisContext(
        schema=profile.schema,
        table=profile.table,
        description=profile.table_description,
        grain=grain,
        table_role=str(table_meta.get("table_role") or "").strip(),
        business_subject=business_subject,
        primary_subjects=primary_subjects,
        entity_keys=entity_keys,
        time_axes=time_axes,
        measures=measures,
        dimensions=dimensions,
        flags=flags,
        peer_groups=peer_groups,
        column_descriptions=descriptions,
        column_semantic_classes=semantic_classes,
        business_synonyms=synonyms,
    )


def _column_descriptions(
    profile: TableProfile,
    schema_loader: SchemaLoader | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if schema_loader is None:
        return out
    try:
        attrs = schema_loader.attrs_df
        rows = attrs[
            (attrs["schema_name"] == profile.schema)
            & (attrs["table_name"] == profile.table)
        ]
        for _, row in rows.iterrows():
            col = str(row.get("column_name") or "")
            desc = str(row.get("description") or "").strip()
            if col in profile.columns and desc:
                out[col] = desc
    except Exception:
        return out
    return out


def _column_semantic_classes(
    profile: TableProfile,
    schema_loader: SchemaLoader | None,
) -> dict[str, str]:
    out: dict[str, str] = {}
    if schema_loader is None:
        return out
    for col in profile.columns:
        try:
            meta = schema_loader.get_column_semantics(profile.schema, profile.table, col)
        except Exception:
            meta = {}
        sem_class = str(meta.get("semantic_class") or "").strip()
        if sem_class:
            out[col] = sem_class
    return out


def _column_synonyms(
    profile: TableProfile,
    schema_loader: SchemaLoader | None,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if schema_loader is None:
        return out
    try:
        attrs = schema_loader.attrs_df
        rows = attrs[
            (attrs["schema_name"] == profile.schema)
            & (attrs["table_name"] == profile.table)
        ]
        for _, row in rows.iterrows():
            col = str(row.get("column_name") or "")
            if col not in profile.columns:
                continue
            raw = str(row.get("synonyms") or "").strip()
            values = [
                x.strip() for x in raw.replace("|", ",").split(",")
                if x.strip()
            ]
            if values:
                out[col] = values
    except Exception:
        return out
    return out


def _rank_time_axes(profile: TableProfile, table_meta: dict[str, Any]) -> list[str]:
    preferred = [
        str(c) for c in (table_meta.get("time_axis_columns") or [])
        if str(c) in profile.columns
    ]
    candidates = profile.date_columns()
    ranked = sorted(
        candidates,
        key=lambda c: (
            0 if c in preferred else 1,
            -profile.columns[c].n_unique,
            c,
        ),
    )
    return _dedupe(profile.representatives(ranked))


def _rank_entity_keys(
    profile: TableProfile,
    semantic_classes: dict[str, str],
) -> list[str]:
    candidates = profile.entity_candidates(min_card=5, max_card=100_000)
    candidates = profile.representatives(candidates)
    return sorted(
        candidates,
        key=lambda c: (
            -_entity_score(c, profile.columns[c].n_unique, semantic_classes.get(c, "")),
            -profile.columns[c].n_unique,
            c,
        ),
    )


def _entity_score(name: str, cardinality: int, semantic_class: str) -> int:
    low = name.lower()
    score = 0
    if semantic_class in {"identifier", "join_key"}:
        score += 40
    if cardinality >= 50:
        score += 20
    if cardinality >= 500:
        score += 20
    for token, bonus in _ENTITY_NAME_PRIORITIES:
        if token in low:
            score += bonus
            break
    return score


def _rank_measures(
    profile: TableProfile,
    semantic_classes: dict[str, str],
) -> list[str]:
    nums = profile.representatives(profile.numeric_columns())
    return sorted(
        nums,
        key=lambda c: (
            -_measure_score(c, profile.columns[c].role, semantic_classes.get(c, "")),
            c,
        ),
    )


def _measure_score(name: str, role: ColumnRole, semantic_class: str) -> int:
    low = name.lower()
    score = 0
    if role == ColumnRole.MONEY:
        score += 80
    elif role == ColumnRole.PERCENT:
        score += 70
    elif role == ColumnRole.NUMERIC:
        score += 40
    if semantic_class == "metric":
        score += 20
    if any(h in low for h in _MEASURE_HINTS):
        score += 20
    return score


def _rank_dimensions(
    profile: TableProfile,
    semantic_classes: dict[str, str],
) -> list[str]:
    cats = profile.representatives(profile.category_columns(max_cardinality=500))
    return sorted(
        cats,
        key=lambda c: (
            0 if semantic_classes.get(c) in {"enum_like", "label"} else 1,
            profile.columns[c].n_unique,
            c,
        ),
    )


def _rank_peer_groups(profile: TableProfile, dimensions: list[str]) -> list[str]:
    candidates = [
        c for c in dimensions
        if 2 <= profile.columns[c].n_unique <= 100
    ]
    # Prefer organizational / segment dimensions, then compact low-card groups.
    return sorted(
        candidates,
        key=lambda c: (
            0 if any(t in c.lower() for t in ("segment", "status", "type", "tb", "gosb", "region")) else 1,
            profile.columns[c].n_unique,
            c,
        ),
    )


def _infer_business_subject(
    profile: TableProfile,
    grain: str,
    primary_subjects: list[str],
    entity_keys: list[str],
) -> str:
    if primary_subjects:
        return primary_subjects[0]
    if grain and grain != "other":
        return grain
    if entity_keys:
        name = entity_keys[0].lower()
        if "inn" in name or "client" in name or "epk" in name:
            return "client"
        if "employee" in name or "saphr" in name:
            return "employee"
        if "gosb" in name or name.startswith("tb"):
            return "organization"
    return profile.table_description or profile.table


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
