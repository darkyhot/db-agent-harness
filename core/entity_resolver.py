"""Универсальный матчинг «сущность → колонка» через embeddings + LLM.

Идея: для произвольного entity_term (например, «ТБ», «отток клиентов», «sku_code»)
ранжируем колонки из разрешённого пула таблиц по семантической близости (cosine
по эмбеддингам имени+описания, либо токен-overlap как fallback) и при нечётком
лидере просим LLM выбрать конкретную колонку из top-N с учётом описаний.

Принципиально: НЕТ захардкоженных доменных алиасов. Всё, что нужно для матчинга,
живёт в метаданных каталога (`column_name`, `description`, `dType`, `is_primary_key`,
`semantic_class` из column_semantics.json) и в эмбеддингах GigaChat.

Используется в `column_binding`, `catalog_grounding`, `column_selector_deterministic`
вместо разрозненных alias-словарей и спец-кейсов вроде `gosb`→`gosb_id`.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Структурные фильтры по dtype/semantic_class — не доменные.
_NUMERIC_DTYPE_RE = re.compile(
    r"^(integer|int[248]?|bigint|smallint|numeric|decimal|real|double|float[48]?|money|number|int)",
    re.I,
)
_DATE_DTYPE_RE = re.compile(r"^(date|timestamp)", re.I)

_LRU_CAPACITY = 256

_SYSTEM_PROMPT = (
    "Ты — матчер сущностей пользователя на колонки SQL-каталога.\n"
    "На вход даётся: исходный запрос пользователя, термин-сущность (то, как пользователь "
    "назвал бизнес-понятие), желаемая роль колонки (id/label/metric/dimension/filter/any) "
    "и список кандидатов с их описаниями и dtype.\n"
    "Твоя задача — выбрать ОДНУ колонку из списка кандидатов, которая лучше всего "
    "соответствует термину и роли.\n"
    "Учитывай:\n"
    "- описание колонки (description) — главный сигнал; если в нём прямо упомянут термин, "
    "это сильное совпадение;\n"
    "- роль: для role=id — предпочитай идентификатор/первичный ключ, не наименование;\n"
    "  для role=label — предпочитай *_name/*_label, не *_id;\n"
    "  для role=metric — предпочитай численные не-ключевые колонки;\n"
    "- если в пуле есть «канонический» вариант (без префиксов old_/new_/legacy_/prev_) "
    "и пользователь не уточнил «старый»/«новый», предпочитай канонический; если канонического "
    "нет — выбирай «новый»/актуальный.\n"
    "Если ни один кандидат не подходит — верни chosen_ref=null.\n"
    "Верни строго JSON: "
    '{"chosen_ref":"<schema.table.column> или null","confidence":0..1,"reason":"..."}'
)

_MAX_DESC_LEN = 160


@dataclass
class Candidate:
    """Один кандидат-колонка для матчинга."""

    ref: str  # "schema.table.column"
    column: str
    table_key: str  # "schema.table"
    description: str
    dtype: str
    is_primary_key: bool
    unique_perc: float
    not_null_perc: float
    semantic_class: str = ""
    embedding_score: float = 0.0
    text_score: float = 0.0
    combined_score: float = 0.0


@dataclass
class EntityResolution:
    """Результат матчинга entity_term → колонка."""

    column_ref: str | None
    table_key: str | None
    column: str | None
    confidence: float
    candidates: list[Candidate] = field(default_factory=list)
    decision_path: str = "no_match"  # embedding_only | llm_tiebreak | text_only | no_match
    reason: str = ""

    @property
    def matched(self) -> bool:
        return self.column_ref is not None


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zа-яё_]+", " ", str(text).lower())).strip()


def _truncate_desc(text: str) -> str:
    s = (text or "").strip().replace("\n", " ")
    return s if len(s) <= _MAX_DESC_LEN else s[: _MAX_DESC_LEN - 1] + "…"


def _metadata_text(row: Any) -> str:
    return str(row.get("description") or "").strip()


def _is_numeric_dtype(dtype: str) -> bool:
    return bool(_NUMERIC_DTYPE_RE.match(str(dtype or "").strip()))


def _is_date_dtype(dtype: str) -> bool:
    return bool(_DATE_DTYPE_RE.match(str(dtype or "").strip()))


def _structural_role_pass(cand: Candidate, role_hint: str) -> bool:
    """Структурный фильтр по dtype/semantic_class. Не доменный."""
    role = (role_hint or "any").lower()
    sem = (cand.semantic_class or "").lower()
    dtype = (cand.dtype or "").lower()
    name_lower = cand.column.lower()

    if role == "metric":
        if cand.is_primary_key:
            return False
        if not _is_numeric_dtype(dtype):
            return False
        if sem in {"date", "system_timestamp", "join_key", "identifier"}:
            return False
        return True
    if role == "label":
        if sem in {"label", "enum_like", "free_text"}:
            return True
        return name_lower.endswith(("_name", "_label", "_title"))
    if role == "id":
        if cand.is_primary_key:
            return True
        if sem in {"join_key", "identifier"}:
            return True
        return name_lower.endswith(("_id", "_code"))
    if role == "date":
        return _is_date_dtype(dtype) or sem in {"date"}
    if role == "filter":
        return sem != "system_timestamp"
    return True


def _text_overlap_score(entity_term: str, name: str, description: str) -> float:
    """Fallback-оценка без эмбеддингов: совпадение токенов entity_term в имени/описании."""
    term = _normalize(entity_term)
    if not term:
        return 0.0
    name_n = _normalize(name)
    desc_n = _normalize(description)
    haystack = f"{name_n} {desc_n}"
    score = 0.0
    if term == name_n:
        score = 1.0
    elif term in name_n.split() or term in name_n:
        score = 0.85
    elif term in haystack:
        score = 0.7
    else:
        term_tokens = set(term.split())
        hay_tokens = set(haystack.split())
        if term_tokens and term_tokens <= hay_tokens:
            score = 0.55
    return score


# ---------------------------------------------------------------------------
# Сбор и ранжирование кандидатов
# ---------------------------------------------------------------------------


def _collect_candidates(
    *,
    entity_term: str,
    candidate_table_keys: Iterable[str],
    schema_loader: Any,
    role_hint: str,
) -> list[Candidate]:
    column_semantics = getattr(schema_loader, "_column_semantics", None) or {}
    out: list[Candidate] = []
    for table_key in candidate_table_keys:
        parts = str(table_key).split(".", 1)
        if len(parts) != 2:
            continue
        schema, table = parts
        try:
            cols_df = schema_loader.get_table_columns(schema, table)
        except Exception:  # noqa: BLE001
            continue
        if cols_df is None or cols_df.empty:
            continue
        for _, row in cols_df.iterrows():
            col = str(row.get("column_name") or "").strip()
            if not col:
                continue
            sem_key = f"{schema}.{table}.{col}".lower()
            sem_info = column_semantics.get(sem_key) or {}
            cand = Candidate(
                ref=f"{schema}.{table}.{col}",
                column=col,
                table_key=f"{schema}.{table}",
                description=_metadata_text(row),
                dtype=str(row.get("dType") or "").strip(),
                is_primary_key=bool(row.get("is_primary_key", False)),
                unique_perc=_safe_float(row.get("unique_perc")),
                not_null_perc=_safe_float(row.get("not_null_perc")),
                semantic_class=str(sem_info.get("semantic_class") or ""),
            )
            if not _structural_role_pass(cand, role_hint):
                continue
            cand.text_score = _text_overlap_score(entity_term, col, cand.description)
            out.append(cand)
    return out


def _apply_embedding_scores(
    candidates: list[Candidate],
    entity_term: str,
    schema_loader: Any,
) -> bool:
    """Заполняет embedding_score у кандидатов. Возвращает True если индекс был задействован."""
    sem_idx = getattr(schema_loader, "semantic_index", None)
    if sem_idx is None or not getattr(sem_idx, "is_ready", False):
        return False
    refs = [c.ref.lower() for c in candidates]
    try:
        scores = sem_idx.similarity(entity_term, refs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("EntityResolver: ошибка similarity: %s", exc)
        return False
    for cand in candidates:
        cand.embedding_score = float(scores.get(cand.ref.lower(), 0.0))
    return True


def _combine_scores(candidates: list[Candidate], have_embeddings: bool) -> None:
    """Заполняет combined_score: embedding (если есть) + text overlap + лёгкие структурные.

    Структурные бонусы (PK, not_null) применяются ТОЛЬКО при наличии семантического
    сигнала — иначе кандидат с нулевой релевантностью получал бы ненулевой score
    и проходил вместо честного no_match.
    """
    for cand in candidates:
        if have_embeddings:
            base = cand.embedding_score * 0.7 + cand.text_score * 0.3
            has_signal = cand.embedding_score > 0.15 or cand.text_score > 0.0
        else:
            base = cand.text_score
            has_signal = cand.text_score > 0.0
        if has_signal:
            if cand.is_primary_key:
                base += 0.05
            if cand.not_null_perc >= 95.0:
                base += 0.02
        cand.combined_score = base if has_signal else 0.0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# LLM-вызов
# ---------------------------------------------------------------------------


def _build_llm_payload(
    *,
    user_input: str,
    entity_term: str,
    role_hint: str,
    top_candidates: list[Candidate],
) -> dict[str, Any]:
    return {
        "user_input": user_input,
        "entity_term": entity_term,
        "role_hint": role_hint,
        "candidates": [
            {
                "ref": c.ref,
                "column": c.column,
                "table": c.table_key,
                "dtype": c.dtype,
                "description": _truncate_desc(c.description),
                "is_pk": c.is_primary_key,
                "unique_perc": round(c.unique_perc, 1),
            }
            for c in top_candidates
        ],
    }


def _llm_pick(
    *,
    llm_invoker: Any,
    payload: dict[str, Any],
    failure_tag: str,
) -> tuple[str | None, float, str] | None:
    """Возвращает (chosen_ref, confidence, reason) или None при сбое."""
    if llm_invoker is None or not hasattr(llm_invoker, "_llm_json_with_retry"):
        return None
    import json as _json

    try:
        verdict = llm_invoker._llm_json_with_retry(
            _SYSTEM_PROMPT,
            _json.dumps(payload, ensure_ascii=False, indent=2),
            temperature=0.0,
            failure_tag=failure_tag,
            expect="object",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("EntityResolver: исключение в LLM: %s", exc)
        return None
    if not isinstance(verdict, dict):
        return None
    chosen = verdict.get("chosen_ref")
    if chosen in (None, "", "null"):
        chosen_ref: str | None = None
    else:
        chosen_ref = str(chosen).strip()
    try:
        conf = float(verdict.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    reason = str(verdict.get("reason") or "").strip()
    return chosen_ref, max(0.0, min(1.0, conf)), reason


# ---------------------------------------------------------------------------
# Кэш
# ---------------------------------------------------------------------------


class _ResolverCache:
    def __init__(self, capacity: int = _LRU_CAPACITY) -> None:
        self._cache: OrderedDict[tuple, EntityResolution] = OrderedDict()
        self._capacity = capacity

    def get(self, key: tuple) -> EntityResolution | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: tuple, value: EntityResolution) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._capacity:
                self._cache.popitem(last=False)
        self._cache[key] = value


_GLOBAL_CACHE = _ResolverCache()


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------


def resolve_entity_to_columns(
    *,
    entity_term: str,
    user_input: str,
    candidate_table_keys: list[str],
    schema_loader: Any,
    llm_invoker: Any = None,
    role_hint: str = "any",
    top_k_embeddings: int = 6,
    high_confidence_gap: float = 0.15,
    high_confidence_floor: float = 0.55,
    failure_tag: str = "entity_resolver",
    use_cache: bool = True,
) -> EntityResolution:
    """Подобрать одну колонку под entity_term из candidate_table_keys.

    Args:
        entity_term: пользовательский термин (например, «ТБ», «outflow_amt», «sku»).
        user_input: исходный текст запроса (для контекста LLM).
        candidate_table_keys: пул таблиц «schema.table», в которых ищем колонку.
        schema_loader: SchemaLoader с доступом к колонкам и (опционально) semantic_index.
        llm_invoker: BaseNodeMixin-подобный объект с `_llm_json_with_retry`. Если None —
            LLM-tiebreak пропускается.
        role_hint: одно из {"any","id","label","metric","dimension","filter","date"}.
        top_k_embeddings: сколько кандидатов передавать LLM.
        high_confidence_gap: если top1.combined - top2.combined ≥ gap, LLM не вызываем.
        high_confidence_floor: дополнительно требуем top1.combined ≥ floor для пропуска LLM.
        failure_tag: тег для логов parse-failures.
        use_cache: использовать LRU-кэш в рамках процесса.

    Returns:
        EntityResolution с выбранной колонкой и обоснованием. Если ничего не подошло —
        column_ref=None, decision_path="no_match".
    """
    cache_key = (
        _normalize(entity_term),
        tuple(sorted(t.lower() for t in candidate_table_keys)),
        role_hint.lower(),
    )
    if use_cache:
        cached = _GLOBAL_CACHE.get(cache_key)
        if cached is not None:
            return cached

    candidates = _collect_candidates(
        entity_term=entity_term,
        candidate_table_keys=candidate_table_keys,
        schema_loader=schema_loader,
        role_hint=role_hint,
    )
    if not candidates:
        result = EntityResolution(
            column_ref=None, table_key=None, column=None, confidence=0.0,
            candidates=[], decision_path="no_match",
            reason=f"нет кандидатов для term='{entity_term}' role={role_hint}",
        )
        if use_cache:
            _GLOBAL_CACHE.put(cache_key, result)
        return result

    have_embeddings = _apply_embedding_scores(candidates, entity_term, schema_loader)
    _combine_scores(candidates, have_embeddings)
    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    top = candidates[: max(top_k_embeddings, 2)]

    if not top or top[0].combined_score <= 0.0:
        result = EntityResolution(
            column_ref=None, table_key=None, column=None, confidence=0.0,
            candidates=top, decision_path="no_match",
            reason=f"нулевой score у всех кандидатов для term='{entity_term}'",
        )
        if use_cache:
            _GLOBAL_CACHE.put(cache_key, result)
        return result

    top1 = top[0]
    top2 = top[1] if len(top) > 1 else None
    gap = top1.combined_score - (top2.combined_score if top2 else 0.0)
    high_confidence = (
        top1.combined_score >= high_confidence_floor
        and (top2 is None or gap >= high_confidence_gap)
    )

    if high_confidence or llm_invoker is None:
        path = "embedding_only" if have_embeddings else "text_only"
        result = EntityResolution(
            column_ref=top1.ref,
            table_key=top1.table_key,
            column=top1.column,
            confidence=min(0.95, top1.combined_score),
            candidates=top,
            decision_path=path,
            reason=(
                f"top={top1.column} score={top1.combined_score:.3f} "
                f"gap={gap:.3f} (no LLM call: {'high_confidence' if high_confidence else 'no_invoker'})"
            ),
        )
        if use_cache:
            _GLOBAL_CACHE.put(cache_key, result)
        return result

    payload = _build_llm_payload(
        user_input=user_input,
        entity_term=entity_term,
        role_hint=role_hint,
        top_candidates=top,
    )
    llm_result = _llm_pick(llm_invoker=llm_invoker, payload=payload, failure_tag=failure_tag)
    if llm_result is None:
        # LLM-сбой → fallback на top1
        result = EntityResolution(
            column_ref=top1.ref,
            table_key=top1.table_key,
            column=top1.column,
            confidence=min(0.7, top1.combined_score),
            candidates=top,
            decision_path="embedding_only" if have_embeddings else "text_only",
            reason=f"LLM-сбой, fallback на top1={top1.column} score={top1.combined_score:.3f}",
        )
        if use_cache:
            _GLOBAL_CACHE.put(cache_key, result)
        return result

    chosen_ref, conf, llm_reason = llm_result
    if chosen_ref is None:
        result = EntityResolution(
            column_ref=None, table_key=None, column=None, confidence=0.0,
            candidates=top, decision_path="llm_tiebreak",
            reason=f"LLM отказался выбирать: {llm_reason}",
        )
        if use_cache:
            _GLOBAL_CACHE.put(cache_key, result)
        return result

    chosen_norm = chosen_ref.strip().lower()
    chosen_cand = next((c for c in top if c.ref.lower() == chosen_norm), None)
    if chosen_cand is None:
        # LLM придумал ref не из списка → fallback на top1
        logger.warning(
            "EntityResolver: LLM вернул ref %s вне списка кандидатов, fallback на top1=%s",
            chosen_ref, top1.ref,
        )
        result = EntityResolution(
            column_ref=top1.ref,
            table_key=top1.table_key,
            column=top1.column,
            confidence=min(0.7, top1.combined_score),
            candidates=top,
            decision_path="embedding_only" if have_embeddings else "text_only",
            reason=f"LLM hallucinated ref={chosen_ref}, fallback на {top1.column}",
        )
        if use_cache:
            _GLOBAL_CACHE.put(cache_key, result)
        return result

    result = EntityResolution(
        column_ref=chosen_cand.ref,
        table_key=chosen_cand.table_key,
        column=chosen_cand.column,
        confidence=max(conf, min(0.85, top1.combined_score)),
        candidates=top,
        decision_path="llm_tiebreak",
        reason=f"LLM выбрал {chosen_cand.column}: {llm_reason}",
    )
    if use_cache:
        _GLOBAL_CACHE.put(cache_key, result)
    return result


def reset_resolver_cache() -> None:
    """Очистить процесс-локальный кэш resolver'а (для тестов и rebuild каталога)."""
    global _GLOBAL_CACHE
    _GLOBAL_CACHE = _ResolverCache()
