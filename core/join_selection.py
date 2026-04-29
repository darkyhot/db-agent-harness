"""Shared join-spec cleanup utilities.

The LLM-only intent path still needs deterministic validation of join specs,
but this module does not parse user language.
"""

from __future__ import annotations

from typing import Any


def normalize_join_spec(
    join_spec: list[dict[str, Any]],
    schema_loader: Any,
    table_types: dict[str, str],
) -> list[dict[str, Any]]:
    """Validate shape, orient fact-to-dim joins, deduplicate, and refresh safety."""
    if not join_spec:
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in join_spec:
        if not isinstance(raw, dict):
            continue
        left = str(raw.get("left") or "").strip()
        right = str(raw.get("right") or "").strip()
        if not left or not right:
            continue
        left_table = ".".join(left.split(".")[:2])
        right_table = ".".join(right.split(".")[:2])
        if table_types.get(left_table, "unknown") != "fact" and table_types.get(right_table, "unknown") == "fact":
            left, right = right, left
            left_table, right_table = right_table, left_table
        key = tuple(sorted((left.lower(), right.lower())))
        if key in seen:
            continue
        seen.add(key)
        entry = dict(raw)
        entry["left"] = left
        entry["right"] = right
        entry["safe"] = _is_unique_side(right, schema_loader)
        if entry["safe"]:
            entry.pop("risk", None)
        elif "risk" not in entry:
            entry["risk"] = f"{right.rsplit('.', 1)[-1]} не подтверждён как уникальный ключ"
        entry["strategy"] = str(entry.get("strategy") or "direct")
        normalized.append(entry)
    return normalized


def _pick_join_candidate(
    text: str,
    t1: str,
    t2: str,
    schema_loader: Any,
    user_input: str = "",
    hint_join_fields: list[str] | None = None,
) -> dict[str, str] | None:
    """Pick the strongest common key-like column between two catalog tables."""
    del text, user_input, hint_join_fields
    parts1 = t1.split(".", 1)
    parts2 = t2.split(".", 1)
    if len(parts1) != 2 or len(parts2) != 2:
        return None
    cols1 = schema_loader.get_table_columns(parts1[0], parts1[1])
    cols2 = schema_loader.get_table_columns(parts2[0], parts2[1])
    if cols1.empty or cols2.empty:
        return None

    rows1 = {
        str(row.get("column_name") or "").strip(): row
        for _, row in cols1.iterrows()
        if str(row.get("column_name") or "").strip()
    }
    rows2_lower = {
        str(row.get("column_name") or "").strip().lower(): row
        for _, row in cols2.iterrows()
        if str(row.get("column_name") or "").strip()
    }
    best: tuple[float, str, str] | None = None
    for left_name, left_row in rows1.items():
        right_row = rows2_lower.get(left_name.lower())
        if right_row is None:
            continue
        right_name = str(right_row.get("column_name") or "").strip()
        if not (_key_like(left_row) or _key_like(right_row)):
            continue
        score = _float(left_row.get("not_null_perc")) + _float(right_row.get("not_null_perc"))
        score += min(_float(left_row.get("unique_perc")), 100.0)
        score += min(_float(right_row.get("unique_perc")), 100.0)
        if left_name.lower().endswith(("_id", "_code")):
            score += 50.0
        candidate = (score, left_name, right_name)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        return None
    return {"col1": best[1], "col2": best[2]}


def _key_like(row: Any) -> bool:
    name = str(row.get("column_name") or "").lower()
    if bool(row.get("is_primary_key", False)):
        return True
    if name.endswith(("_id", "_code", "_num", "_no")):
        return True
    return _float(row.get("unique_perc")) >= 50.0 and _float(row.get("not_null_perc")) >= 50.0


def _is_unique_side(column_ref: str, schema_loader: Any) -> bool:
    parts = column_ref.rsplit(".", 2)
    if len(parts) != 3:
        return False
    schema, table, column = parts
    try:
        result = schema_loader.check_key_uniqueness(schema, table, [column])
    except Exception:
        return False
    return result.get("is_unique") is True


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
