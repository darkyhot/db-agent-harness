"""Shared join-spec cleanup utilities.

Both the LLM-driven explorer path and the deterministic column selector go
through `normalize_join_spec`. The function:
  1. Validates each entry, drops malformed ones, deduplicates table-pair keys.
  2. Reorients so the fact side is on the left for fact↔dim pairs.
  3. Auto-completes composite PK keys (e.g. dim PK = (tb_id, old_gosb_id))
     so a single LLM-supplied pair does not silently cause row multiplication.
  4. Recomputes `safe`/`risk` on the full key set for each table-pair.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_PK_NORM_RE = re.compile(r"^(old|new|prev|cur|current|actual|base|src|tgt)_", re.I)
_DIM_TYPES = {"dim", "ref"}


def _norm_col_name(name: str) -> str:
    """Нормализовать имя ключевой колонки: убрать old_/new_/prev_ префиксы."""
    return _PK_NORM_RE.sub("", name.lower())


def _check_safe(table_full: str, col_name: str, schema_loader: Any) -> bool:
    """Детерминированная проверка уникальности JOIN-ключа из CSV-метаданных."""
    parts = table_full.split(".", 1)
    if len(parts) != 2:
        return False
    try:
        result = schema_loader.check_key_uniqueness(parts[0], parts[1], [col_name])
        return bool(result.get("is_unique", False))
    except Exception:
        return False


def _infer_strategy(t1_type: str, t2_type: str, safe: bool) -> str:
    """Определить стратегию JOIN по типам таблиц."""
    if t1_type == "fact" and t2_type in _DIM_TYPES:
        return "direct" if safe else "through_dim"
    if t1_type in _DIM_TYPES and t2_type == "fact":
        return "subquery"
    if t1_type == "fact" and t2_type == "fact":
        return "subquery"
    return "direct"


def _complete_composite_join(
    initial_entry: dict[str, Any],
    t1: str,
    t2: str,
    table_types: dict[str, str],
    schema_loader: Any,
) -> list[dict[str, Any]]:
    """Найти дополнительные join-пары для составного PK dim-таблицы.

    Если dim-таблица имеет составной PK, но initial_entry покрывает лишь одну его
    колонку — ищем пары для оставшихся PK-колонок в fact-таблице.
    Пример: dim PK = (tb_id, old_gosb_id), initial = fact.gosb_id↔dim.old_gosb_id →
    добавляем fact.tb_id↔dim.tb_id (exact same-name match).
    """
    t1_type = table_types.get(t1, "unknown")
    t2_type = table_types.get(t2, "unknown")

    if t2_type in _DIM_TYPES:
        dim_table, fact_table = t2, t1
    elif t1_type in _DIM_TYPES:
        dim_table, fact_table = t1, t2
    else:
        return []

    dim_parts = dim_table.split(".", 1)
    if len(dim_parts) != 2:
        return []
    try:
        dim_cols_df = schema_loader.get_table_columns(dim_parts[0], dim_parts[1])
        if dim_cols_df.empty or "column_name" not in dim_cols_df.columns:
            return []
        pk_mask = dim_cols_df.get("is_primary_key", pd.Series(dtype=bool)).astype(bool)
        pk_cols: list[str] = dim_cols_df.loc[pk_mask, "column_name"].tolist()
    except Exception:
        return []

    if len(pk_cols) < 2:
        return []

    initial_left_tbl = ".".join(initial_entry["left"].split(".")[:2])
    if initial_left_tbl == dim_table:
        covered_dim_col = initial_entry["left"].rsplit(".", 1)[-1]
        fact_is_left = False
    else:
        covered_dim_col = initial_entry["right"].rsplit(".", 1)[-1]
        fact_is_left = True

    fact_parts = fact_table.split(".", 1)
    try:
        fact_cols_df = schema_loader.get_table_columns(fact_parts[0], fact_parts[1])
        fact_col_names: list[str] = (
            fact_cols_df["column_name"].tolist() if not fact_cols_df.empty else []
        )
    except Exception:
        fact_col_names = []

    additional: list[dict[str, Any]] = []
    for pk_col in pk_cols:
        if pk_col == covered_dim_col:
            continue

        fact_col: str | None = None
        if pk_col in fact_col_names:
            fact_col = pk_col
        else:
            norm_pk = _norm_col_name(pk_col)
            for fc in fact_col_names:
                if _norm_col_name(fc) == norm_pk:
                    fact_col = fc
                    break

        if not fact_col:
            continue

        if fact_is_left:
            entry: dict[str, Any] = {
                "left": f"{fact_table}.{fact_col}",
                "right": f"{dim_table}.{pk_col}",
                "safe": False,
                "strategy": initial_entry.get("strategy", "fact_dim_join"),
                "risk": f"{fact_col} — composite PK pair, не уникален в {fact_table}",
            }
        else:
            entry = {
                "left": f"{dim_table}.{pk_col}",
                "right": f"{fact_table}.{fact_col}",
                "safe": False,
                "strategy": initial_entry.get("strategy", "dim_fact_join"),
                "risk": f"{fact_col} — composite PK pair, не уникален в {fact_table}",
            }
        additional.append(entry)

    return additional


def _apply_composite_safety(
    pair_entries: list[dict[str, Any]],
    schema_loader: Any,
    table_types: dict[str, str],
) -> None:
    """Если составной join покрывает уникальный ключ одной стороны, считаем его safe."""
    if len(pair_entries) < 2:
        return

    left_table = ".".join(pair_entries[0]["left"].split(".")[:2])
    right_table = ".".join(pair_entries[0]["right"].split(".")[:2])
    left_cols = [e["left"].rsplit(".", 1)[-1] for e in pair_entries]
    right_cols = [e["right"].rsplit(".", 1)[-1] for e in pair_entries]

    candidate_sides = [
        (left_table, left_cols, table_types.get(left_table, "unknown")),
        (right_table, right_cols, table_types.get(right_table, "unknown")),
    ]
    candidate_sides.sort(key=lambda x: 0 if x[2] in _DIM_TYPES else 1)

    for table_full, cols, _ttype in candidate_sides:
        parts = table_full.split(".", 1)
        if len(parts) != 2:
            continue
        uniq = schema_loader.check_key_uniqueness(parts[0], parts[1], cols)
        if uniq.get("is_unique") is True:
            for entry in pair_entries:
                entry["safe"] = True
                entry.pop("risk", None)
                lt = ".".join(entry["left"].split(".")[:2])
                rt = ".".join(entry["right"].split(".")[:2])
                entry["strategy"] = _infer_strategy(
                    table_types.get(lt, "unknown"),
                    table_types.get(rt, "unknown"),
                    True,
                )
            return


def normalize_join_spec(
    join_spec: list[dict[str, Any]],
    schema_loader: Any,
    table_types: dict[str, str],
) -> list[dict[str, Any]]:
    """Validate, orient, dedup, complete composite keys, recompute safety.

    Used by both the LLM explorer path and the deterministic column selector.
    A single key that points to a dim/ref composite PK is auto-completed so the
    full key-set determines safety — preventing row multiplication when the LLM
    returns just one pair of a composite PK.
    """
    if not join_spec:
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(entry: dict[str, Any]) -> None:
        left = str(entry.get("left") or "").strip()
        right = str(entry.get("right") or "").strip()
        if not left or not right:
            return
        left_table = ".".join(left.split(".")[:2])
        right_table = ".".join(right.split(".")[:2])
        if (
            table_types.get(left_table, "unknown") != "fact"
            and table_types.get(right_table, "unknown") == "fact"
        ):
            left, right = right, left
            left_table, right_table = right_table, left_table
        key = tuple(sorted((left.lower(), right.lower())))
        if key in seen:
            return
        seen.add(key)
        clean = dict(entry)
        clean["left"] = left
        clean["right"] = right
        clean["safe"] = bool(clean.get("safe", False))
        clean["strategy"] = str(clean.get("strategy") or "direct")
        normalized.append(clean)

    for entry in join_spec:
        if not isinstance(entry, dict):
            continue
        _add(entry)
        left_table = ".".join(str(entry.get("left") or "").split(".")[:2])
        right_table = ".".join(str(entry.get("right") or "").split(".")[:2])
        if not left_table or not right_table:
            continue
        for extra in _complete_composite_join(
            entry,
            left_table,
            right_table,
            table_types,
            schema_loader,
        ):
            _add(extra)

    groups: dict[frozenset[str], list[dict[str, Any]]] = {}
    for entry in normalized:
        left_table = ".".join(entry["left"].split(".")[:2])
        right_table = ".".join(entry["right"].split(".")[:2])
        if left_table and right_table:
            groups.setdefault(frozenset((left_table, right_table)), []).append(entry)

    for entries in groups.values():
        if len(entries) > 1:
            _apply_composite_safety(entries, schema_loader, table_types)
        else:
            entry = entries[0]
            right_parts = entry["right"].rsplit(".", 2)
            if len(right_parts) != 3:
                continue
            r_schema, r_table, r_col = right_parts
            try:
                uniq = schema_loader.check_key_uniqueness(r_schema, r_table, [r_col])
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "normalize_join_spec: uniqueness check failed for %s: %s",
                    entry["right"],
                    exc,
                )
                continue
            if uniq.get("is_unique") is True:
                entry["safe"] = True
                entry.pop("risk", None)
            elif uniq.get("is_unique") is False:
                entry["safe"] = False
                entry["risk"] = (
                    f"{r_col} не уникален в {r_schema}.{r_table} "
                    f"(~{uniq.get('duplicate_pct', '?')}% дублей)"
                )

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


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
