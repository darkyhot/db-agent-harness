"""Generic ranking of filter candidates based on metadata registry and value profiles."""

from __future__ import annotations

import re
from typing import Any


def _normalize_text(text: str) -> str:
    text = str(text or "").lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я_ ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _normalize_text(text).split() if len(tok) >= 2]


def _stem(token: str) -> str:
    token = _normalize_text(token)
    for suffix in (
        "ыми", "ими", "ого", "его", "ому", "ему", "ая", "яя", "ое", "ее",
        "ые", "ие", "ый", "ий", "ой", "ом", "ем", "ым", "им", "ах", "ях",
        "ов", "ев", "ей", "ам", "ям", "у", "ю", "а", "я", "ы", "и", "е", "о",
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _stem_set(text: str) -> set[str]:
    return {_stem(tok) for tok in _tokenize(text)}


def _text_score(query_text: str, column: str, description: str) -> float:
    q = _stem_set(query_text)
    if not q:
        return 0.0
    hay = _stem_set(f"{column} {description}")
    return float(len(q & hay) * 14.0)


def _profile_value_match(profile: dict[str, Any], query_text: str) -> tuple[str | None, float]:
    query_tokens = _stem_set(query_text)
    best_value = None
    best_score = 0.0
    for value in list(profile.get("top_values", []) or []) + list(profile.get("known_terms", []) or []):
        value_str = str(value).strip()
        if not value_str:
            continue
        score = float(len(query_tokens & _stem_set(value_str)) * 18.0)
        if score > best_score:
            best_score = score
            best_value = value_str
    return best_value, best_score


def _escape_sql_literal(value: str) -> str:
    return str(value).replace("'", "''")


def _build_condition(column: str, operator: str, value: Any, profile: dict[str, Any], semantics: dict[str, Any]) -> str:
    op = str(operator or "=").upper()
    semantic_class = str(semantics.get("semantic_class", "") or "")
    value_mode = str(profile.get("value_mode", "") or "")
    dtype = str(profile.get("dType", "") or "").lower()

    if semantic_class == "flag" or value_mode == "boolean_like":
        if value is True:
            if "bool" in dtype:
                return f"{column} = true"
            if any(token in dtype for token in ("int", "numeric", "decimal", "number")):
                return f"{column} = 1"
        if value is False:
            if "bool" in dtype:
                return f"{column} = false"
            if any(token in dtype for token in ("int", "numeric", "decimal", "number")):
                return f"{column} = 0"
        return f"{column} = '{_escape_sql_literal(str(value))}'"

    if op == "ILIKE":
        tokens = [tok for tok in _tokenize(str(value)) if len(tok) >= 4]
        if len(tokens) >= 2:
            unique_tokens = list(dict.fromkeys(tokens))
            return " AND ".join(
                f"{column} ILIKE '%{_escape_sql_literal(token)}%'" for token in unique_tokens
            )
        return f"{column} ILIKE '%{_escape_sql_literal(str(value))}%'"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{column} {op} {value}"
    if value_mode == "date_range" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        return f"{column} {op} '{value}'::date"
    return f"{column} {op} '{_escape_sql_literal(str(value))}'"


def _collect_requests(intent: dict[str, Any], semantic_frame: dict[str, Any] | None) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for item in (semantic_frame or {}).get("filter_intents", []) or []:
        requests.append(dict(item))
    # explicit filters are already mirrored into filter_intents by semantic_frame,
    # but keep backward compatibility when semantic_frame was built elsewhere.
    for idx, raw in enumerate((intent or {}).get("filter_conditions", []) or []):
        if not isinstance(raw, dict):
            continue
        request_id = f"explicit:{idx}"
        if any(req.get("request_id") == request_id for req in requests):
            continue
        requests.append({
            "request_id": request_id,
            "kind": "explicit_filter",
            "query_text": str(raw.get("value") or ""),
            "column_hint": str(raw.get("column_hint") or ""),
            "operator": str(raw.get("operator") or "=").upper(),
            "value": raw.get("value"),
        })
    return requests


def rank_filter_candidates(
    *,
    user_input: str,
    intent: dict[str, Any],
    selected_tables: list[str],
    schema_loader,
    semantic_frame: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """Rank filter candidates for metadata-driven filter intents."""
    requests = _collect_requests(intent, semantic_frame)
    ranked: dict[str, list[dict[str, Any]]] = {str(req.get("request_id")): [] for req in requests}
    if not requests:
        return ranked

    for table_key in selected_tables:
        if "." not in table_key:
            continue
        schema, table = table_key.split(".", 1)
        table_semantics = schema_loader.get_table_semantics(schema, table)
        filter_friendliness = float(table_semantics.get("filter_friendliness", 0.0) or 0.0)

        cols_df = schema_loader.get_table_columns(schema, table)
        if cols_df.empty:
            continue
        for _, row in cols_df.iterrows():
            column = str(row.get("column_name", "") or "")
            description = str(row.get("description", "") or "")
            if not column:
                continue
            key = f"{schema}.{table}.{column}".lower()
            profile = schema_loader.get_value_profile(schema, table, column)
            semantics = schema_loader.get_column_semantics(schema, table, column)
            semantic_class = str(semantics.get("semantic_class", "") or "")
            semantic_tags = {str(v) for v in (semantics.get("semantic_tags") or [])}

            for request in requests:
                query_text = str(request.get("query_text") or request.get("value") or "")
                if not query_text:
                    continue
                score = _text_score(query_text, column, description)
                evidence: list[str] = []
                candidate_value = request.get("value")
                operator = str(request.get("operator") or request.get("preferred_operator") or "=").upper()
                kind = str(request.get("kind") or "")

                if kind == "explicit_filter":
                    if "filter_candidate" in semantic_tags:
                        score += 16.0
                    if request.get("column_hint"):
                        score += _text_score(str(request.get("column_hint") or ""), column, description)
                    matched_value, value_score = _profile_value_match(profile, query_text)
                    score += value_score
                    if matched_value and _stem(request.get("query_text") or "") in _stem(matched_value):
                        candidate_value = matched_value
                        evidence.append(f"profile_value={matched_value}")
                    elif candidate_value is None:
                        candidate_value = query_text

                elif kind == "boolean_true":
                    if semantic_class == "flag":
                        score += 44.0
                        candidate_value = True
                        operator = "="
                        evidence.append("boolean_flag")
                    else:
                        score -= 20.0

                else:
                    if semantic_class in {"enum_like", "label", "free_text"}:
                        score += 20.0
                        if kind == "phrase_filter":
                            token_count = len([tok for tok in _tokenize(query_text) if len(tok) >= 4])
                            if token_count >= 2:
                                score += 18.0
                    matched_value, value_score = _profile_value_match(profile, query_text)
                    score += value_score
                    if matched_value:
                        candidate_value = matched_value
                        evidence.append(f"value_match={matched_value}")
                    elif candidate_value is None:
                        candidate_value = query_text
                    if semantic_class in {"enum_like", "label", "free_text"}:
                        operator = "ILIKE"

                score += filter_friendliness * 20.0
                if score <= 0 or candidate_value in (None, ""):
                    continue

                ranked[str(request.get("request_id"))].append({
                    "request_id": str(request.get("request_id")),
                    "request_kind": kind,
                    "table_key": table_key,
                    "schema": schema,
                    "table": table,
                    "column": column,
                    "semantic_class": semantic_class,
                    "operator": operator,
                    "value": candidate_value,
                    "condition": _build_condition(
                        column,
                        operator,
                        candidate_value,
                        profile | {"dType": row.get("dType", "")},
                        semantics,
                    ),
                    "score": round(score, 3),
                    "evidence": evidence,
                    "value_source": "profile" if evidence else "query",
                    "confidence": "high" if score >= 75.0 else ("medium" if score >= 45.0 else "low"),
                    "column_key": key,
                })

    for request_id, candidates in ranked.items():
        candidates.sort(key=lambda item: item["score"], reverse=True)
        deduped: list[dict[str, Any]] = []
        seen = set()
        for candidate in candidates:
            key = (candidate["column"], candidate["condition"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        ranked[request_id] = deduped
    return ranked
