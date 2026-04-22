"""Generic ranking of filter candidates based on metadata registry and value profiles."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

# Порог Левенштейна-подобного ratio для «нечёткого» сравнения стемов.
# 0.85 покрывает опечатки вида замены/вставки/пропуска одной буквы на словах
# длины 5+, но не срабатывает на разных словах («факт»/«акт»).
_FUZZY_STEM_RATIO = 0.85
_FUZZY_STEM_MIN_LEN = 5
_FLAG_PREFIXES = {"is", "has", "flag", "flg", "bool", "boolean", "priznak", "признак"}
_FLAG_SUFFIXES = {"flag", "flg", "ind", "indicator", "bool", "boolean", "priznak", "признак"}


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


def _fuzzy_stem_match(stem: str, candidates: set[str]) -> bool:
    """Есть ли в множестве candidates стем, достаточно похожий на данный.

    Exact-попадание — fast path. Иначе — SequenceMatcher.ratio ≥ 0.85 на
    стемах длины ≥ 5 (коротких слов не трогаем, чтобы не ловить ложные
    срабатывания на «акт»/«факт»).
    """
    if stem in candidates:
        return True
    if len(stem) < _FUZZY_STEM_MIN_LEN:
        return False
    for cand in candidates:
        if len(cand) < _FUZZY_STEM_MIN_LEN:
            continue
        if SequenceMatcher(None, stem, cand).ratio() >= _FUZZY_STEM_RATIO:
            return True
    return False


def _fuzzy_overlap_count(value_stems: set[str], query_stems: set[str]) -> int:
    """Сколько value_stems имеют exact- или fuzzy-матч в query_stems."""
    return sum(1 for vs in value_stems if _fuzzy_stem_match(vs, query_stems))


def _subject_alias_stems(subject: str, schema_loader) -> set[str]:
    subject = _normalize_text(subject)
    if not subject:
        return set()
    aliases = [subject]
    try:
        lexicon = schema_loader.get_semantic_lexicon()
    except Exception:  # noqa: BLE001
        lexicon = {}
    meta = ((lexicon or {}).get("subjects") or {}).get(subject) or {}
    aliases.extend(str(v) for v in (meta.get("aliases") or []) if str(v).strip())
    stems: set[str] = set()
    for alias in aliases:
        stems.update(_stem_set(alias))
    return stems


def _column_looks_like_subject_flag(column: str, subject_stems: set[str]) -> bool:
    """Определить, что flag-колонка кодирует принадлежность строки к subject.

    Правило умышленно строгое: после снятия типичных boolean-prefix/suffix имя
    должно схлопываться к одному subject-токену. Это даёт универсальный механизм
    для is_task / client_flag / has_employee и не задевает более узкие признаки
    вроде is_task_leader.
    """
    tokens = _tokenize(str(column or "").replace("_", " "))
    if not tokens:
        return False
    while tokens and tokens[0] in _FLAG_PREFIXES:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _FLAG_SUFFIXES:
        tokens = tokens[:-1]
    if len(tokens) != 1:
        return False
    token_stem = _stem(tokens[0])
    return _fuzzy_stem_match(token_stem, subject_stems)


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


def _known_terms_overlap_score(profile: dict[str, Any], query_text: str) -> float:
    """Сильный бонус, если в known_terms/top_values есть хорошее смысловое совпадение.

    Использует fuzzy-матчинг стемов — устойчив к опечаткам в 1 букву на
    токенах длины ≥ 5 («фоктический» всё ещё попадает в «фактическ»).
    """
    query_stems = _stem_set(query_text)
    if not query_stems:
        return 0.0
    best_overlap = 0
    for value in list(profile.get("top_values", []) or []) + list(profile.get("known_terms", []) or []):
        value_str = str(value).strip()
        if not value_str:
            continue
        overlap = _fuzzy_overlap_count(_stem_set(value_str), query_stems)
        if overlap > best_overlap:
            best_overlap = overlap
    if best_overlap >= 2:
        return 42.0
    if best_overlap == 1:
        return 18.0
    return 0.0


def _phrase_in_known_terms(profile: dict[str, Any], query_text: str) -> tuple[str | None, int]:
    """Найти known_term/top_value, стем-токены которого полностью покрываются запросом.

    Покрытие допускает fuzzy-матч стемов: опечатка в 1 букву на слове длины ≥ 5
    тоже считается совпадением. Возвращает (значение, число стем-токенов) для
    самой «длинной» такой фразы или (None, 0), если ни одна не покрылась.

    Используется как детерминированный тай-брейкер: если конкретное значение
    колонки ("фактический отток") встречается в пользовательском запросе
    целиком, колонка считается точным кандидатом по семантике значений.
    """
    query_stems = _stem_set(query_text)
    if not query_stems:
        return None, 0
    best_value: str | None = None
    best_len = 0
    for value in list(profile.get("top_values", []) or []) + list(profile.get("known_terms", []) or []):
        value_str = str(value).strip()
        if not value_str:
            continue
        value_stems = _stem_set(value_str)
        if not value_stems:
            continue
        covered = all(_fuzzy_stem_match(vs, query_stems) for vs in value_stems)
        if covered and len(value_stems) > best_len:
            best_value = value_str
            best_len = len(value_stems)
    return best_value, best_len


def _column_reference_score(
    user_input: str,
    column: str,
    description: str,
    *,
    semantic_index: Any = None,
    column_key: str = "",
) -> float:
    """Высокий бонус, если пользователь явно назвал колонку или её описание.

    Опциональный semantic_index добавляет бонус (до +20) на основе косинусного
    сходства между запросом и эмбеддингом колонки.
    """
    normalized = _normalize_text(user_input)
    column_norm = _normalize_text(column)
    if column_norm and column_norm in normalized:
        base = 120.0
    else:
        desc_score = _text_score(user_input, "", description)
        base = 35.0 if desc_score >= 14.0 else 0.0

    if semantic_index is not None and column_key:
        try:
            sem_scores = semantic_index.similarity(user_input, [column_key])
            sem_score = sem_scores.get(column_key, 0.0)
        except Exception:
            sem_score = 0.0
        bonus = 20.0 if sem_score >= 0.8 else (10.0 if sem_score >= 0.6 else 0.0)
        return base + bonus

    return base


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


def _collect_implicit_subject_flag_requests(
    *,
    selected_tables: list[str],
    schema_loader,
    semantic_frame: dict[str, Any] | None,
    existing_requests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    subject = str((semantic_frame or {}).get("subject") or "").strip().lower()
    subject_stems = _subject_alias_stems(subject, schema_loader)
    if not subject_stems:
        return []

    existing_keys = {
        str(req.get("column_key") or "").strip().lower()
        for req in existing_requests
        if str(req.get("column_key") or "").strip()
    }
    implicit_requests: list[dict[str, Any]] = []

    for table_key in selected_tables:
        if "." not in table_key:
            continue
        schema, table = table_key.split(".", 1)
        cols_df = schema_loader.get_table_columns(schema, table)
        if cols_df.empty:
            continue

        best_match: tuple[float, str] | None = None
        for _, row in cols_df.iterrows():
            column = str(row.get("column_name", "") or "").strip()
            if not column:
                continue
            column_key = f"{schema}.{table}.{column}".lower()
            if column_key in existing_keys:
                continue

            semantics = schema_loader.get_column_semantics(schema, table, column)
            semantic_class = str(semantics.get("semantic_class", "") or "")
            profile = schema_loader.get_value_profile(schema, table, column)
            value_mode = str(profile.get("value_mode", "") or "")
            if semantic_class != "flag" and value_mode != "boolean_like":
                continue
            if not _column_looks_like_subject_flag(column, subject_stems):
                continue

            score = 100.0 + _text_score(subject, column, str(row.get("description", "") or ""))
            candidate = (score, column_key)
            if best_match is None or candidate > best_match:
                best_match = candidate

        if best_match is None:
            continue
        implicit_requests.append({
            "request_id": f"implicit_subject_flag:{best_match[1]}",
            "kind": "boolean_true",
            "query_text": subject,
            "column_key": best_match[1],
            "match_source": "implicit_subject_flag",
        })

    return implicit_requests


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
    requests.extend(
        _collect_implicit_subject_flag_requests(
            selected_tables=selected_tables,
            schema_loader=schema_loader,
            semantic_frame=semantic_frame,
            existing_requests=requests,
        )
    )
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
            try:
                unique_perc = float(row.get("unique_perc", 0.0) or 0.0)
            except (TypeError, ValueError):
                unique_perc = 0.0

            for request in requests:
                request_column_key = str(request.get("column_key") or "").strip().lower()
                if request_column_key and key != request_column_key:
                    continue
                query_text = str(request.get("query_text") or request.get("value") or "")
                if not query_text:
                    continue
                score = _text_score(query_text, column, description)
                evidence: list[str] = []
                candidate_value = request.get("value")
                operator = str(request.get("operator") or request.get("preferred_operator") or "=").upper()
                kind = str(request.get("kind") or "")
                matched_example: str | None = None

                if kind == "explicit_filter":
                    if "filter_candidate" in semantic_tags:
                        score += 16.0
                    if request.get("column_hint"):
                        score += _text_score(str(request.get("column_hint") or ""), column, description)
                    _sem_idx = getattr(schema_loader, "semantic_index", None)
                    score += _column_reference_score(
                        user_input, column, description,
                        semantic_index=_sem_idx,
                        column_key=key,
                    )
                    matched_value, value_score = _profile_value_match(profile, query_text)
                    score += value_score
                    if matched_value and _stem(request.get("query_text") or "") in _stem(matched_value):
                        candidate_value = matched_value
                        matched_example = matched_value
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
                                # Для узких смысловых фраз предпочитаем более "богатые" categorical-поля.
                                # dense/низкокардинальные атрибуты вроде category/type обычно слишком грубые.
                                score += min(unique_perc * 5.0, 18.0)
                                if "dense" in semantic_tags:
                                    score -= 12.0
                    matched_value, value_score = _profile_value_match(profile, query_text)
                    score += value_score
                    score += _known_terms_overlap_score(profile, query_text)
                    phrase_value, phrase_len = _phrase_in_known_terms(profile, query_text)
                    if phrase_len >= 2:
                        score += 70.0
                        evidence.append(f"known_term_phrase={phrase_value}")
                        if not matched_value:
                            matched_value = phrase_value
                    elif phrase_len == 1 and not matched_value:
                        matched_value = phrase_value
                    if matched_value:
                        candidate_value = matched_value
                        matched_example = matched_value
                        evidence.append(f"value_match={matched_value}")
                    elif candidate_value is None:
                        candidate_value = query_text
                    if semantic_class in {"enum_like", "label", "free_text"}:
                        operator = "ILIKE"

                score += filter_friendliness * 20.0
                if score <= 0 or candidate_value in (None, ""):
                    continue

                explicit_column_choice = _column_reference_score(user_input, column, description) >= 120.0
                final_confidence = "high" if explicit_column_choice else (
                    "high" if score >= 75.0 else ("medium" if score >= 45.0 else "low")
                )
                # Примеры значений колонки для пояснительных подписей в clarification-сообщениях.
                # Отдельно от matched_example — используется как fallback, когда совпадения
                # с пользовательской фразой нет, но хочется показать «характер» колонки.
                example_values: list[str] = []
                for _val in list(profile.get("top_values", []) or []) + list(profile.get("known_terms", []) or []):
                    _val_str = str(_val).strip()
                    if _val_str and _val_str not in example_values:
                        example_values.append(_val_str)
                    if len(example_values) >= 2:
                        break
                ranked[str(request.get("request_id"))].append({
                    "request_id": str(request.get("request_id")),
                    "request_kind": kind,
                    "table_key": table_key,
                    "schema": schema,
                    "table": table,
                    "column": column,
                    "description": description,
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
                    "confidence": final_confidence,
                    "column_key": key,
                    "matched_example": matched_example,
                    "example_values": example_values,
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
