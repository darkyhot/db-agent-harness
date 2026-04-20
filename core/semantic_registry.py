"""Metadata-driven semantic lexicon and rule registry."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


_STOPWORDS = {
    "и", "или", "по", "с", "со", "без", "для", "из", "в", "на", "за", "от", "до",
    "the", "a", "an", "of", "for", "to", "from", "in", "on", "with",
    "table", "таблица", "колонка", "column", "признак", "flag", "field",
    "значение", "type", "тип", "code", "код", "name", "наименование",
}

_FLAG_WORDS = {"признак", "flag", "is", "bool", "boolean"}
_DATE_WORDS = {"date", "дата", "period", "период", "report"}


def _normalize_phrase(text: str) -> str:
    text = str(text or "").lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я_ ]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_phrase(text)
    return [tok for tok in normalized.split() if len(tok) >= 2 and tok not in _STOPWORDS]


def _raw_tokenize(text: str) -> list[str]:
    normalized = _normalize_phrase(text)
    return [tok for tok in normalized.split() if len(tok) >= 2]


def _token_stem(token: str) -> str:
    token = _normalize_phrase(token)
    for suffix in (
        "ыми", "ими", "ого", "его", "ому", "ему", "ыми", "ими", "ыми", "ими",
        "ыми", "ими", "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий", "ой",
        "ом", "ем", "ым", "им", "ах", "ях", "ов", "ев", "ей", "ам", "ям",
        "у", "ю", "а", "я", "ы", "и", "е", "о",
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _stem_tokens(tokens: list[str]) -> list[str]:
    return [_token_stem(tok) for tok in tokens if tok]


def _tokens_overlap(query_tokens: set[str], candidate_tokens: set[str]) -> int:
    used: set[str] = set()
    matches = 0
    for candidate in candidate_tokens:
        for query in query_tokens:
            if query in used:
                continue
            if candidate == query:
                used.add(query)
                matches += 1
                break
            if min(len(candidate), len(query)) >= 5 and (candidate.startswith(query) or query.startswith(candidate)):
                used.add(query)
                matches += 1
                break
    return matches


def _phrase_variants(text: str) -> list[str]:
    raw_tokens = _raw_tokenize(text)
    tokens = _tokenize(text)
    if not tokens:
        return []
    # Если исходная фраза была многословной, но после выкидывания stopwords
    # схлопнулась в одно слишком общее слово ("Тип задачи" -> "задачи"),
    # такой alias только засоряет registry и даёт ложные text-match'и.
    if len(raw_tokens) > 1 and len(tokens) < 2:
        return []
    variants = {
        " ".join(tokens),
        " ".join(_stem_tokens(tokens)),
    }
    return [variant for variant in variants if variant.strip()]


def _slug(text: str) -> str:
    normalized = _normalize_phrase(text).replace(" ", "_")
    return re.sub(r"_+", "_", normalized).strip("_")


def _collect_aliases(*parts: str) -> list[str]:
    aliases: list[str] = []
    for part in parts:
        aliases.extend(_phrase_variants(part))
    result: list[str] = []
    seen = set()
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            result.append(alias)
    return result


def _is_overly_generic_match_phrase(column: str, phrase: str) -> bool:
    """Отсечь слишком общие match_phrases для type/category-полей.

    Пример нежелательного поведения: `task_type` начинает матчиться по словам
    "task"/"задача", из-за чего любой запрос про задачи провоцирует лишний
    фильтр по типу. Для таких полей оставляем только более специфичные фразы.
    """
    column_norm = _normalize_phrase(column)
    phrase_tokens = _tokenize(phrase)
    if not phrase_tokens:
        return True

    if column_norm.endswith("_type") or column_norm.endswith("_category"):
        return len(phrase_tokens) < 2
    return False


def build_semantic_lexicon(
    tables_df,
    attrs_df,
    *,
    table_semantics: dict[str, dict[str, Any]],
    column_semantics: dict[str, dict[str, Any]],
    value_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build metadata-driven lexicon for subjects, dimensions, filters, and time phrases."""
    lexicon: dict[str, Any] = {
        "subjects": {},
        "dimensions": {},
        "filter_phrases": {},
        "time_phrases": {},
    }

    for key, meta in (table_semantics or {}).items():
        grain = str(meta.get("grain") or "").strip().lower()
        if not grain or grain == "other":
            continue
        aliases = _collect_aliases(grain, str(meta.get("table") or ""), str(meta.get("schema") or ""))
        if aliases:
            entry = lexicon["subjects"].setdefault(grain, {"aliases": [], "tables": []})
            for alias in aliases:
                if alias not in entry["aliases"]:
                    entry["aliases"].append(alias)
            if key not in entry["tables"]:
                entry["tables"].append(key)

    if attrs_df is not None and not attrs_df.empty:
        for _, row in attrs_df.iterrows():
            schema = str(row.get("schema_name", "") or "")
            table = str(row.get("table_name", "") or "")
            column = str(row.get("column_name", "") or "")
            if not (schema and table and column):
                continue
            key = f"{schema}.{table}.{column}".lower()
            semantics = column_semantics.get(key, {})
            sem_class = str(semantics.get("semantic_class", "") or "")
            description = str(row.get("description", "") or "")
            aliases = _collect_aliases(column.replace("_", " "), description)
            if sem_class == "date":
                entry = lexicon["time_phrases"].setdefault("date", {"aliases": []})
                for alias in aliases + _collect_aliases(*_DATE_WORDS):
                    if alias not in entry["aliases"]:
                        entry["aliases"].append(alias)
            if sem_class in {"label", "enum_like", "date"}:
                dim_key = _slug(description or column)
                entry = lexicon["dimensions"].setdefault(dim_key, {
                    "column_keys": [],
                    "aliases": [],
                    "semantic_class": sem_class,
                })
                if key not in entry["column_keys"]:
                    entry["column_keys"].append(key)
                for alias in aliases:
                    if alias not in entry["aliases"]:
                        entry["aliases"].append(alias)

            if sem_class in {"flag", "enum_like", "label", "free_text"}:
                phrase_key = _slug(description or column)
                entry = lexicon["filter_phrases"].setdefault(phrase_key, {
                    "column_keys": [],
                    "aliases": [],
                    "semantic_class": sem_class,
                })
                if key not in entry["column_keys"]:
                    entry["column_keys"].append(key)
                for alias in aliases:
                    if alias not in entry["aliases"]:
                        entry["aliases"].append(alias)
                profile = value_profiles.get(key, {})
                for term in list(profile.get("top_values", []) or []) + list(profile.get("known_terms", []) or []):
                    for alias in _collect_aliases(str(term)):
                        if alias not in entry["aliases"]:
                            entry["aliases"].append(alias)

    return lexicon


def build_rule_registry(
    attrs_df,
    *,
    column_semantics: dict[str, dict[str, Any]],
    value_profiles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build generic rule registry from metadata and value profiles."""
    rules: dict[str, Any] = {"rules": []}
    if attrs_df is None or attrs_df.empty:
        return rules

    for _, row in attrs_df.iterrows():
        schema = str(row.get("schema_name", "") or "")
        table = str(row.get("table_name", "") or "")
        column = str(row.get("column_name", "") or "")
        if not (schema and table and column):
            continue
        key = f"{schema}.{table}.{column}".lower()
        semantics = column_semantics.get(key, {})
        profile = value_profiles.get(key, {})
        sem_class = str(semantics.get("semantic_class", "") or "")
        if sem_class not in {"flag", "enum_like", "label", "free_text"}:
            continue

        description = str(row.get("description", "") or "")
        aliases = _collect_aliases(column.replace("_", " "), description)
        match_phrases = [
            alias
            for alias in aliases
            if alias
            and alias not in _FLAG_WORDS
            and not _is_overly_generic_match_phrase(column, alias)
        ]
        values = [str(v) for v in list(profile.get("top_values", []) or []) + list(profile.get("known_terms", []) or []) if str(v).strip()]

        if sem_class == "flag":
            description_tokens = [tok for tok in _tokenize(description) if tok not in _FLAG_WORDS]
            match_phrases.extend(_collect_aliases(" ".join(description_tokens)))
            rule = {
                "rule_id": f"flag:{key}",
                "column_key": key,
                "semantic_class": sem_class,
                "match_kind": "boolean_true",
                "match_phrases": sorted(set(match_phrases)),
                "value_candidates": [],
            }
            rules["rules"].append(rule)
            continue

        rule = {
            "rule_id": f"text:{key}",
            "column_key": key,
            "semantic_class": sem_class,
            "match_kind": "text_search",
            "match_phrases": sorted(set(match_phrases)),
            "value_candidates": values[:20],
        }
        rules["rules"].append(rule)

    return rules


def find_best_subject(query: str, lexicon: dict[str, Any]) -> str | None:
    """Pick best subject from lexicon aliases."""
    query_tokens = set(_stem_tokens(_tokenize(query)))
    best_subject = None
    best_score = 0
    for subject, meta in (lexicon.get("subjects") or {}).items():
        aliases = meta.get("aliases", []) or []
        score = 0
        for alias in aliases:
            alias_tokens = set(_stem_tokens(_tokenize(alias)))
            score = max(score, len(query_tokens & alias_tokens))
        if score > best_score:
            best_score = score
            best_subject = subject
    return best_subject


def find_matching_dimensions(query: str, lexicon: dict[str, Any]) -> list[str]:
    query_tokens = set(_stem_tokens(_tokenize(query)))
    results: list[str] = []
    for dim_key, meta in (lexicon.get("dimensions") or {}).items():
        for alias in meta.get("aliases", []) or []:
            alias_tokens = set(_stem_tokens(_tokenize(alias)))
            if alias_tokens and len(query_tokens & alias_tokens) >= max(1, min(2, len(alias_tokens))):
                results.append(dim_key)
                break
    return list(dict.fromkeys(results))


def find_matching_rules(query: str, registry: dict[str, Any]) -> list[dict[str, Any]]:
    query_tokens = set(_stem_tokens(_tokenize(query)))
    matched: list[dict[str, Any]] = []
    for rule in (registry.get("rules") or []):
        best = 0
        matched_phrase = ""
        match_source = ""
        for phrase in rule.get("match_phrases", []) or []:
            phrase_tokens = set(_stem_tokens(_tokenize(phrase)))
            overlap = _tokens_overlap(query_tokens, phrase_tokens)
            raw_phrase_tokens = _raw_tokenize(phrase)
            threshold = 1 if len(raw_phrase_tokens) <= 1 else min(2, len(raw_phrase_tokens))
            if overlap >= threshold and overlap > best:
                best = overlap
                matched_phrase = phrase
                match_source = "match_phrase"
        for value in rule.get("value_candidates", []) or []:
            value_tokens = set(_stem_tokens(_tokenize(value)))
            overlap = _tokens_overlap(query_tokens, value_tokens)
            raw_value_tokens = _raw_tokenize(value)
            threshold = 1 if len(raw_value_tokens) <= 1 else min(2, len(raw_value_tokens))
            if overlap >= threshold and overlap > best:
                best = overlap
                matched_phrase = value
                match_source = "value_candidate"
        if best > 0:
            enriched = dict(rule)
            enriched["match_score"] = float(best)
            enriched["matched_phrase"] = matched_phrase
            enriched["match_source"] = match_source
            matched.append(enriched)

    # Если запрос уже содержит более специфичную фразу, не оставляем рядом
    # общий «вложенный» матч по её подмножеству. Пример:
    # - "фактический отток" -> task_subtype
    # - "отток" -> task_type
    # Для запроса "по фактическому оттоку" второй матч не должен провоцировать
    # clarification: пользователь уже задал более точный признак.
    pruned: list[dict[str, Any]] = []
    matched_token_sets: list[set[str]] = [
        set(_stem_tokens(_tokenize(str(item.get("matched_phrase") or ""))))
        for item in matched
    ]
    for idx, item in enumerate(matched):
        current_tokens = matched_token_sets[idx]
        dominated = False
        if current_tokens:
            for other_idx, other in enumerate(matched):
                if other_idx == idx:
                    continue
                other_tokens = matched_token_sets[other_idx]
                if not other_tokens or current_tokens == other_tokens:
                    continue
                if current_tokens < other_tokens:
                    dominated = True
                    break
        if not dominated:
            pruned.append(item)
    pruned.sort(
        key=lambda item: (
            item.get("match_score", 0.0),
            len(_stem_tokens(_tokenize(str(item.get("matched_phrase") or "")))),
        ),
        reverse=True,
    )
    return pruned
