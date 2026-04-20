"""Semantic frame driven by metadata lexicon and generic phrase extraction."""

from __future__ import annotations

import re
from typing import Any

from core.semantic_registry import find_best_subject, find_matching_dimensions, find_matching_rules


_RU_MONTH_STEMS = (
    "январ", "феврал", "март", "апрел", "май", "мая",
    "июн", "июл", "август", "сентябр", "октябр", "ноябр", "декабр",
)
_PREPOSITIONS = ("с ", "со ", "по ", "без ", "для ")


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
                if word in {"и", "или"} and phrase_words:
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
        intents.append({
            "request_id": rule["rule_id"],
            "kind": str(rule.get("match_kind") or "text_search"),
            "query_text": str(rule.get("matched_phrase") or ""),
            "column_key": str(rule.get("column_key") or ""),
            "semantic_class": str(rule.get("semantic_class") or ""),
            "match_score": float(rule.get("match_score", 0.0) or 0.0),
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
    known_query_texts = {str(item.get("query_text") or "") for item in intents}
    for idx, phrase in enumerate(freeform_phrases):
        if phrase in known_query_texts:
            continue
        phrase_token_count = len(_tokenize(phrase))
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
    haystack = _collect_text(user_input, intent)
    lexicon = schema_loader.get_semantic_lexicon() if schema_loader is not None else {}

    subject = find_best_subject(haystack, lexicon) if lexicon else None
    if subject is None:
        if any(token in haystack for token in ("task", "ticket", "issue", "задач", "воронка")):
            subject = "task"
        elif any(token in haystack for token in ("client", "customer", "клиент")):
            subject = "client"
        elif any(token in haystack for token in ("employee", "staff", "сотрудник")):
            subject = "employee"
        elif any(token in haystack for token in ("organization", "org", "branch", "госб", "тб")):
            subject = "organization"

    matched_rules = find_matching_rules(haystack, schema_loader.get_rule_registry()) if schema_loader is not None else []
    qualifier = None
    business_event = None
    top_rule = matched_rules[0] if matched_rules else None
    if top_rule:
        rule_id = str(top_rule.get("rule_id") or "")
        if str(top_rule.get("match_kind") or "") == "boolean_true":
            qualifier = rule_id.split(":", 1)[0]
        else:
            qualifier = rule_id.split(":", 1)[0]
        business_event = str(top_rule.get("matched_phrase") or "")
    elif any(token in haystack for token in ("outflow", "отток", "churn", "attrition")):
        business_event = "outflow"

    requested_grain = subject
    period_kind = None
    if re.search(r"\b(20\d{2}|\d{2})\b", haystack) or any(stem in haystack for stem in _RU_MONTH_STEMS):
        period_kind = "calendar"

    metric_intent = _metric_intent(intent, haystack)
    output_dimensions = (
        find_matching_dimensions(haystack, lexicon) if lexicon else []
    )
    required_output = [str(v).strip().lower() for v in ((intent or {}).get("required_output") or []) if str(v).strip()]
    output_dimensions = list(dict.fromkeys(required_output + output_dimensions))
    requires_listing = metric_intent == "list" or any(token in haystack for token in ("список", "перечень"))
    requires_single_entity_count = (
        metric_intent == "count"
        and bool(subject)
        and not output_dimensions
        and not requires_listing
    )
    ambiguities: list[str] = []
    if not metric_intent and any(token in haystack for token in ("покажи", "выведи")):
        ambiguities.append("metric_intent")
    if not subject and business_event:
        ambiguities.append("subject")

    filter_intents = _derive_filter_intents(user_input=user_input, intent=intent, schema_loader=schema_loader)

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
