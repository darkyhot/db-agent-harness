"""Построение value profiles для фильтровых колонок."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from core.exceptions import KerberosAuthError

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_MAX_SAMPLE_VALUES = 12
_TABLE_SAMPLE_LIMIT = 100_000


def _is_identifier(value: str) -> bool:
    return bool(_IDENTIFIER_RE.match(value or ""))


def discover_profile_candidates(
    attrs_df: pd.DataFrame,
    column_semantics: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Найти колонки, для которых имеет смысл собирать value profiles."""
    if attrs_df.empty:
        return []

    candidates: list[dict[str, Any]] = []
    for _, row in attrs_df.iterrows():
        schema = str(row.get("schema_name", "") or "")
        table = str(row.get("table_name", "") or "")
        column = str(row.get("column_name", "") or "")
        if not (schema and table and column):
            continue

        key = f"{schema}.{table}.{column}".lower()
        semantics = (column_semantics or {}).get(key, {})
        semantic_class = str(semantics.get("semantic_class", "") or "")
        semantic_tags = {str(t) for t in (semantics.get("semantic_tags") or [])}

        if semantic_class in {"flag", "enum_like", "date"}:
            candidates.append(row.to_dict())
            continue
        if semantic_class in {"free_text", "system_timestamp", "identifier", "join_key"}:
            continue
        if {"filter_candidate", "categorical"} & semantic_tags:
            candidates.append(row.to_dict())

    return candidates


def build_metadata_profile(
    row: dict[str, Any],
    semantics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Построить value profile только по CSV-метаданным."""
    semantics = semantics or {}
    schema = str(row.get("schema_name", "") or "")
    table = str(row.get("table_name", "") or "")
    column = str(row.get("column_name", "") or "")
    dtype = str(row.get("dType", "") or "").lower()
    description = str(row.get("description", "") or "")
    distinct_pct = float(row.get("unique_perc", 0.0) or 0.0)
    null_pct = max(0.0, 100.0 - float(row.get("not_null_perc", 0.0) or 0.0))

    semantic_class = str(semantics.get("semantic_class", "") or "")
    semantic_tags = list(semantics.get("semantic_tags", []) or [])
    operators = ["=", "ILIKE"]
    value_mode = "dictionary_like"
    supports_exact = True
    supports_pattern = True

    if semantic_class == "flag":
        value_mode = "boolean_like"
        operators = ["="]
        supports_pattern = False
    elif semantic_class == "date":
        value_mode = "date_range"
        operators = ["=", ">=", "<=", "<"]
        supports_pattern = False
    elif semantic_class == "metric":
        value_mode = "numeric_range"
        operators = ["=", ">", ">=", "<", "<="]
        supports_pattern = False
    elif semantic_class == "enum_like":
        value_mode = "enum_like"
    elif semantic_class == "label":
        value_mode = "dictionary_like"
    elif semantic_class == "free_text":
        value_mode = "text_pattern"
        supports_exact = False

    return {
        "schema": schema,
        "table": table,
        "column": column,
        "semantic_class": semantic_class,
        "semantic_tags": semantic_tags,
        "value_mode": value_mode,
        "allowed_operators": operators,
        "supports_exact_match": supports_exact,
        "supports_pattern_match": supports_pattern,
        "null_pct": round(null_pct, 2),
        "distinct_pct": round(distinct_pct, 2),
        "known_terms": [],
        "source": "metadata",
    }


def fetch_table_profile_sample(
    db_manager: Any,
    *,
    schema: str,
    table: str,
    columns: list[str],
    sample_limit: int = _TABLE_SAMPLE_LIMIT,
) -> pd.DataFrame:
    """Загрузить один sample DataFrame по таблице для набора колонок."""
    if db_manager is None:
        return pd.DataFrame()
    if not (_is_identifier(schema) and _is_identifier(table)):
        return pd.DataFrame()
    safe_columns = [col for col in columns if _is_identifier(col)]
    if not safe_columns:
        return pd.DataFrame()

    projection = ", ".join(f'"{column}"' for column in safe_columns)
    sql = (
        f"SELECT {projection} "
        f'FROM "{schema}"."{table}" '
        f"ORDER BY random() "
        f"LIMIT {int(sample_limit)}"
    )
    try:
        return db_manager.execute_query(sql, limit=sample_limit)
    except KerberosAuthError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.info("ValueProfiler: table sample skipped for %s.%s: %s", schema, table, exc)
        return pd.DataFrame()


def build_db_profile(
    sample_df: pd.DataFrame,
    *,
    column: str,
    metadata_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Построить value profile по sample DataFrame таблицы."""
    metadata_profile = metadata_profile or {}
    if sample_df.empty or column not in sample_df.columns:
        return {"top_values": [], "sample_size": 0, "source": "db"}

    series = sample_df[column].dropna()
    if series.empty:
        return {"top_values": [], "sample_size": 0, "source": "db"}

    sample_size = int(series.shape[0])
    vc = series.astype(str).value_counts(dropna=False).head(_MAX_SAMPLE_VALUES)
    top_values = vc.index.tolist()
    top_value_freq = [int(v) for v in vc.tolist()]

    profile: dict[str, Any] = {
        "top_values": top_values,
        "top_value_freq": top_value_freq,
        "sample_size": sample_size,
        "source": "db",
    }

    if pd.api.types.is_numeric_dtype(series):
        try:
            profile["min_value"] = float(pd.to_numeric(series, errors="coerce").min())
            profile["max_value"] = float(pd.to_numeric(series, errors="coerce").max())
        except Exception:  # noqa: BLE001
            pass

    return profile


def merge_profiles(
    metadata_profile: dict[str, Any],
    db_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Слить metadata profile и optional db profile."""
    result = dict(metadata_profile)
    db_profile = db_profile or {}
    if db_profile:
        result.update(db_profile)

        top_values = [str(v).lower() for v in db_profile.get("top_values", []) if str(v).strip()]
        known_terms = list(result.get("known_terms", []) or [])
        for value in top_values:
            if len(value) >= 3 and value not in known_terms:
                known_terms.append(value)
        result["known_terms"] = known_terms[:_MAX_SAMPLE_VALUES]

    return result
