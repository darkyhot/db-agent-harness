"""Semantic frame driven by metadata lexicon and generic phrase extraction."""

from __future__ import annotations

import re
from typing import Any

from core.semantic_registry import (
    builtin_subject_aliases,
    find_best_subject,
    find_matching_dimensions,
    find_matching_rules,
)


_RU_MONTH_STEMS = (
    "январ", "феврал", "март", "апрел", "май", "мая",
    "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр",
)
_PREPOSITIONS = ("с ", "со ", "по ", "без ", "для ")
_PROJECTION_VERBS = ("подтяни", "дотяни", "возьми", "выведи", "покажи")
_DIMENSION_WORD_STEMS = ("сегмент", "регион", "госб", "тб", "филиал")
_SERVICE_SUFFIX_RE = re.compile(
    r"\(\s*использовать\s+таблицу\s+[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\s*\)",
    re.IGNORECASE,
)
_LOW_SIGNAL_DIM_TOKENS = frozenset({
    "тип", "задач", "задача", "дат", "фактическ", "фактическая", "фактическому",
})


def sanitize_user_input_for_semantics(user_input: str) -> str:
    """Убрать служебные CLI-аннотации, не относящиеся к бизнес-смыслу запроса."""
    text = str(user_input or "")
    text = _SERVICE_SUFFIX_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _collect_text(user_input: str, intent: dict[str, Any] | None) -> str:
    entities = [str(e) for e in (intent or {}).get("entities", []) if e]
    required = [str(e) for e in (intent or {}).get("required_output", []) if e]
    return " ".join([user_input or ""] + entities + required).lower()


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


def _metric_intent(intent: dict[str, Any] | None, haystack: str) -> str | None:
    hint = str((intent or {}).get("aggregation_hint") or "").lower().strip()
    if hint in {"count", "sum", "avg", "min", "max", "list"}:
        return hint
    if any(token in haystack for token in ("сколько", "посчитай", "количество", "count")):
        return "count"
    if any(token in haystack for token in ("сумм", "sum")):
        return "sum"
    if any(token in haystack for token in ("средн", "avg", "average")):
        return "avg"
    if any(token in haystack for token in ("список", "перечень", "list")):
        return "list"
    return None


def _extract_freeform_phrases(user_input: str) -> list[str]:
    normalized = _normalize_text(user_input)
    phrases: list[str] = []
    for prep in _PREPOSITIONS:
        start = 0
        while True:
            idx = normalized.find(prep, start)
            if idx == -1:
                break
            tail = normalized[idx + len(prep):]
            words = tail.split()
            if not words:
                break
            phrase_words: list[str] = []
            for word in words[:4]:
                if word in {"дате", "дате", "дата", "региону", "регион", "сегменту", "сегмент"} and not phrase_words:
                    break
                if word in _PROJECTION_VERBS and phrase_words:
                    break
                if word == "из" and phrase_words:
                    break
                if word in {"и", "или", "за"} and phrase_words:
                    break
                if any(word.startswith(stem) for stem in _RU_MONTH_STEMS) and phrase_words:
                    break
                if re.fullmatch(r"\d{2,4}", word) and phrase_words:
                    break
                phrase_words.append(word)
            if phrase_words:
                phrase = " ".join(phrase_words).strip()
                if len(phrase) >= 4:
                    phrases.append(phrase)
            start = idx + len(prep)
    return list(dict.fromkeys(phrases))


def _token_overlap(a: str, b: str) -> int:
    a_tokens = {_stem(tok) for tok in _tokenize(a)}
    b_tokens = {_stem(tok) for tok in _tokenize(b)}
    used: set[str] = set()
    matches = 0
    for left in a_tokens:
        for right in b_tokens:
            if right in used:
                continue
            if left == right:
                used.add(right)
                matches += 1
                break
            if min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left)):
                used.add(right)
                matches += 1
                break
    return matches


def _looks_like_output_dimension_filter(item: dict[str, Any], output_dimensions: list[str]) -> bool:
    """Не путать `group by`-измерение с filter_intent.

    Пример: запрос `по сегментам` должен давать output dimension `segment`,
    а не просьбу фильтровать по `segment_name`.
    """
    if not output_dimensions:
        return False
    # Если матч произошёл по значению enum-like колонки ("фактический отток"),
    # трактуем это как фильтр по значению, а не как измерение.
    if str(item.get("match_source") or "") == "value_candidate":
        return False
    column_key = str(item.get("column_key") or "")
    column_name = column_key.rsplit(".", 1)[-1] if column_key else ""
    matched_phrase = str(item.get("query_text") or item.get("matched_phrase") or "")
    haystacks = [column_name, matched_phrase]
    for dim in output_dimensions:
        dim_norm = str(dim or "").strip().lower()
        if not dim_norm:
            continue
        for hay in haystacks:
            hay_norm = str(hay or "").strip().lower()
            if not hay_norm:
                continue
            if dim_norm in hay_norm or hay_norm in dim_norm:
                return True
            if _token_overlap(dim_norm, hay_norm) >= 1:
                return True
    return False


def _dimension_shadowed_by_value_filter(
    dimension: str,
    filter_intents: list[dict[str, Any]],
) -> bool:
    """Если dimension конфликтует с более сильным value_candidate фильтром — убираем её."""
    dim_norm = str(dimension or "").strip().lower()
    if not dim_norm:
        return False
    for item in filter_intents:
        if str(item.get("match_source") or "") != "value_candidate":
            continue
        phrase = str(item.get("query_text") or item.get("matched_phrase") or "").strip().lower()
        if not phrase:
            continue
        if _token_overlap(dim_norm, phrase) >= 1:
            return True
    return False


def _extract_output_dimension_hints(user_input: str) -> list[str]:
    """Вытащить явные измерения из phrasing `по X`, `по дате`, `по region_name`."""
    normalized = _normalize_text(user_input)
    dimensions: list[str] = []

    for match in re.finditer(r"(?:по|в разбивке по)\s+([a-zа-я_]+)", normalized):
        token = str(match.group(1) or "").strip().lower()
        if not token:
            continue
        stemmed = _stem(token)
        if stemmed in _LOW_SIGNAL_DIM_TOKENS:
            continue
        if token in {"дате", "дата", "date", "report_dt"}:
            dimensions.append("date")
            continue
        if token.endswith(("_name", "_label", "_title")):
            dimensions.append(token)
            continue
        if "_" in token and not token.endswith(("_id", "_code")):
            dimensions.append(token)
            continue
        if any(token.startswith(stem) for stem in _DIMENSION_WORD_STEMS):
            dimensions.append(token)
            continue

    for match in re.finditer(r"(?:названи[еяю]|наименовани[еяю])\s+([a-zа-я_]+)", normalized):
        token = str(match.group(1) or "").strip().lower()
        if token:
            dimensions.append(token)

    return list(dict.fromkeys(dimensions))


def _has_explicit_grouping_request(
    user_input: str,
    intent: dict[str, Any] | None,
) -> bool:
    """Определить, просит ли пользователь явную разбивку в результате."""
    if (intent or {}).get("required_output"):
        return True
    return bool(_extract_output_dimension_hints(user_input))


def _derive_filter_intents(
    *,
    user_input: str,
    intent: dict[str, Any] | None,
    schema_loader=None,
) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []
    registry = schema_loader.get_rule_registry() if schema_loader is not None else {"rules": []}
    matched_rules = find_matching_rules(user_input, registry)
    for idx, rule in enumerate(matched_rules):
        matched_phrase = str(rule.get("matched_phrase") or "")
        normalized_input = _normalize_text(user_input)
        normalized_phrase = _normalize_text(matched_phrase)
        matched_from_column_phrase = normalized_phrase in {
            _normalize_text(v) for v in (rule.get("match_phrases") or [])
        }
        if (
            matched_from_column_phrase
            and normalized_phrase
            and (
                f"по {normalized_phrase}" in normalized_input
                or f"подтяни {normalized_phrase}" in normalized_input
                or f"выведи {normalized_phrase}" in normalized_input
                or f"покажи {normalized_phrase}" in normalized_input
            )
        ):
            continue
        intents.append({
            "request_id": rule["rule_id"],
            "kind": str(rule.get("match_kind") or "text_search"),
            "query_text": matched_phrase,
            "column_key": str(rule.get("column_key") or ""),
            "semantic_class": str(rule.get("semantic_class") or ""),
            "match_score": float(rule.get("match_score", 0.0) or 0.0),
            "match_source": str(rule.get("match_source") or ""),
        })

    for idx, raw in enumerate((intent or {}).get("filter_conditions", []) or []):
        if not isinstance(raw, dict):
            continue
        column_hint = str(raw.get("column_hint") or "").strip()
        value = raw.get("value")
        if not column_hint or value in (None, ""):
            continue
        intents.append({
            "request_id": f"explicit:{idx}",
            "kind": "explicit_filter",
            "query_text": str(value),
            "column_hint": column_hint,
            "operator": str(raw.get("operator") or "=").upper(),
            "value": value,
        })

    freeform_phrases = _extract_freeform_phrases(user_input)
    normalized_input = _normalize_text(user_input)
    known_query_texts = {str(item.get("query_text") or "") for item in intents}
    for idx, phrase in enumerate(freeform_phrases):
        if phrase in known_query_texts:
            continue
        phrase_token_count = len(_tokenize(phrase))
        normalized_phrase = _normalize_text(phrase)
        if (
            phrase_token_count == 1
            and normalized_phrase
            and (
                f"по {normalized_phrase}" in normalized_input
                or f"подтяни {normalized_phrase}" in normalized_input
            )
        ):
            continue
        if any(
            _token_overlap(phrase, existing) >= (
                1
                if phrase_token_count == 1 and len(_tokenize(existing)) == 1
                else 2
            )
            for existing in known_query_texts
            if existing
        ):
            continue
        intents.append({
            "request_id": f"phrase:{idx}",
            "kind": "phrase_filter",
            "query_text": phrase,
        })

    return intents


def derive_semantic_frame(
    user_input: str,
    intent: dict[str, Any] | None = None,
    schema_loader=None,
) -> dict[str, Any]:
    """Build semantic frame using metadata-driven lexicon when available."""
    clean_input = sanitize_user_input_for_semantics(user_input)
    haystack = _collect_text(clean_input, intent)
    lexicon = schema_loader.get_semantic_lexicon() if schema_loader is not None else {}

    subject = find_best_subject(haystack, lexicon) if lexicon else None
    if subject is None:
        for candidate, aliases in builtin_subject_aliases().items():
            if any(token in haystack for token in aliases):
                subject = candidate
                break

    raw_filter_intents = _derive_filter_intents(user_input=clean_input, intent=intent, schema_loader=schema_loader)

    requested_grain = subject
    period_kind = None
    if re.search(r"\b(20\d{2}|\d{2})\b", haystack) or any(stem in haystack for stem in _RU_MONTH_STEMS):
        period_kind = "calendar"

    metric_intent = _metric_intent(intent, haystack)
    output_dimensions = (
        find_matching_dimensions(haystack, lexicon) if lexicon else []
    )
    output_dimensions = list(dict.fromkeys(output_dimensions + _extract_output_dimension_hints(clean_input)))
    required_output = [str(v).strip().lower() for v in ((intent or {}).get("required_output") or []) if str(v).strip()]
    output_dimensions = list(dict.fromkeys(required_output + output_dimensions))
    qualifier = None
    business_event = None
    ambiguities: list[str] = []

    filter_intents = raw_filter_intents
    filter_intents = [
        item for item in filter_intents
        if not _looks_like_output_dimension_filter(item, output_dimensions)
    ]
    output_dimensions = [
        dim for dim in output_dimensions
        if not _dimension_shadowed_by_value_filter(dim, filter_intents)
    ]
    requires_listing = metric_intent == "list" or any(token in haystack for token in ("список", "перечень"))
    has_explicit_grouping = _has_explicit_grouping_request(clean_input, intent)
    requires_single_entity_count = (
        metric_intent == "count"
        and bool(subject)
        and not output_dimensions
        and not has_explicit_grouping
        and not requires_listing
    )

    qualifier = None
    business_event = None
    top_filter = filter_intents[0] if filter_intents else None
    if top_filter:
        request_id = str(top_filter.get("request_id") or "")
        if ":" in request_id:
            qualifier = request_id.split(":", 1)[0]
        matched_phrase = str(top_filter.get("query_text") or top_filter.get("matched_phrase") or "").strip()
        if matched_phrase:
            business_event = matched_phrase
    elif any(token in haystack for token in ("outflow", "отток", "churn", "attrition")):
        business_event = "outflow"

    if not metric_intent and any(token in haystack for token in ("покажи", "выведи")):
        ambiguities.append("metric_intent")
    if not subject and business_event:
        ambiguities.append("subject")

    return {
        "subject": subject,
        "metric_intent": metric_intent,
        "business_event": business_event,
        "qualifier": qualifier,
        "output_dimensions": output_dimensions,
        "requires_listing": requires_listing,
        "requires_single_entity_count": requires_single_entity_count,
        "requested_grain": requested_grain,
        "period_kind": period_kind,
        "ambiguities": ambiguities,
        "filter_intents": filter_intents,
    }
