"""Детерминированный селектор колонок (без LLM).

Заменяет LLM-узел column_selector для стандартных аналитических запросов.
Работает на основе:
- CSV-метаданных (типы, уникальность, PK-флаги из attr_list.csv)
- intent из intent_classifier (aggregation_hint, entities, date_filters, filter_conditions)
- join_analysis_data из table_explorer (pre-computed JOIN кандидаты с scoring)

Логика выбора:
  aggregate  → числовые колонки, матчинг с aggregation_hint/entities
  group_by   → категориальные колонки с низкой кардинальностью, матчинг с entities
  filter     → date-колонки (если есть date_filters) + колонки из filter_conditions
  select     → union(group_by, aggregate)
  join_keys  → top-1 кандидат из join_analysis_data, safe из check_key_uniqueness

Returns:
    {
        "selected_columns": dict,  # schema.table → {select, filter, aggregate, group_by}
        "join_spec": list,         # [{left, right, safe, strategy}, ...]
        "confidence": float,       # 0..1; >= 0.70 → используем без LLM
        "reason": str,             # для логирования
    }

Confidence < 0.70 → caller должен упасть в LLM column_selector как fallback.
"""

import logging
import re
from typing import Any

import pandas as pd

from core.synonym_map import expand_with_synonyms

logger = logging.getLogger(__name__)

# Префиксы PK-нормализации: old_gosb_id → gosb_id
_PK_NORM_RE = re.compile(r"^(old|new|prev|cur|current|actual|base|src|tgt)_", re.I)


# ---------------------------------------------------------------------------
# Sanitize (Direction 5.2): корректирует галлюцинированные колонки от LLM
# через fuzzy-match по allowed-списку (Левенштейн ≤ 2).
# ---------------------------------------------------------------------------


def _levenshtein_le2(a: str, b: str) -> int:
    """Расстояние Левенштейна с ранним выходом при > 2.

    Возвращает 0/1/2 при точных значениях либо 3 (или больше) — тогда мы
    считаем такие пары несовместимыми.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 3
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        min_row = curr[0]
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,         # удаление
                curr[j - 1] + 1,     # вставка
                prev[j - 1] + cost,  # замена
            )
            if curr[j] < min_row:
                min_row = curr[j]
        if min_row > 2:
            return 3
        prev = curr
    return prev[n] if prev[n] <= 2 else 3


def sanitize_selected_columns(
    columns: list[str],
    allowed: list[str],
) -> dict[str, Any]:
    """Отсеять галлюцинированные имена, скорректировать близкие к допустимым.

    Args:
        columns: Имена колонок, предложенные LLM.
        allowed: Список реально существующих колонок (lowercase не требуется).

    Returns:
        {
          "columns": итоговые валидные колонки (в порядке входа),
          "coerced": [(original, corrected), ...] — мягкие fuzzy-фиксы,
          "rejected": [...] — имена, не имеющие пары в allowed,
          "warnings": [...] — человекочитаемые пояснения.
        }
    """
    allowed_norm: dict[str, str] = {}
    for a in allowed:
        a_str = str(a).strip()
        if a_str:
            allowed_norm.setdefault(a_str.lower(), a_str)

    out: list[str] = []
    coerced: list[tuple[str, str]] = []
    rejected: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for col in columns:
        raw = str(col or "").strip()
        if not raw:
            continue
        low = raw.lower()
        if low in allowed_norm:
            canonical = allowed_norm[low]
            if canonical not in seen:
                out.append(canonical)
                seen.add(canonical)
            continue

        # Fuzzy match по Levenshtein ≤ 2
        best: tuple[int, str] | None = None
        for al_low, al_orig in allowed_norm.items():
            dist = _levenshtein_le2(low, al_low)
            if dist <= 2 and (best is None or dist < best[0]):
                best = (dist, al_orig)
                if dist == 0:
                    break
        if best is not None:
            canonical = best[1]
            coerced.append((raw, canonical))
            warnings.append(
                f"'{raw}' → '{canonical}' (fuzzy-match, dist={best[0]})"
            )
            if canonical not in seen:
                out.append(canonical)
                seen.add(canonical)
        else:
            rejected.append(raw)
            warnings.append(
                f"'{raw}' отклонён: нет в allowed-списке и fuzzy-match не нашёл близких"
            )

    return {
        "columns": out,
        "coerced": coerced,
        "rejected": rejected,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Паттерны типов колонок
# ---------------------------------------------------------------------------

# Числовые типы → кандидаты для SUM/AVG/MIN/MAX
_NUMERIC_RE = re.compile(
    r'^(integer|int[248]?|bigint|smallint|numeric|decimal|real|double\s+precision'
    r'|float[48]?|money|number|int)',
    re.I,
)

# Типы дат/времени → кандидаты для date_filters
_DATE_RE = re.compile(r'^(date|timestamp)', re.I)

# Категориальные типы (низкая кардинальность ожидаема)
_CATEGORICAL_RE = re.compile(
    r'^(varchar|character\s+varying|bpchar|char\b|boolean|text)',
    re.I,
)

# Ключевые слова в именах метрических колонок
_METRIC_PARTS = frozenset({
    'amount', 'amt', 'sum', 'total', 'qty', 'quantity',
    'value', 'val', 'revenue', 'cost', 'price', 'balance', 'volume',
    'outflow', 'inflow', 'sales', 'profit', 'loss', 'income',
    'num', 'rate', 'perc', 'pct', 'percent', 'score', 'cnt',
})

# Ключевые слова в именах статусных/типовых колонок (кандидаты для filter)
_STATUS_PARTS = frozenset({'status', 'state', 'flag', 'type', 'code', 'category', 'kind'})
_METRIC_SUFFIXES = ('_qty', '_amt', '_sum', '_cnt', '_perc', '_pct', '_amount', '_value')
_LABEL_SUFFIXES = ('_name', '_label', '_title')
_DATE_HINTS = frozenset({'date', 'dt', 'dttm', 'timestamp', 'data'})
_COUNT_HINTS = frozenset({'count', 'kolichestvo', 'qty', 'cnt'})
_TRANSLIT_TABLE = str.maketrans({
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
})


def _is_numeric(dtype: str) -> bool:
    return bool(_NUMERIC_RE.match(dtype.strip()))


def _is_date(dtype: str) -> bool:
    return bool(_DATE_RE.match(dtype.strip()))


def _is_categorical(dtype: str) -> bool:
    return bool(_CATEGORICAL_RE.match(dtype.strip()))


def _normalize_query_text(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').lower()).strip()


def _translit(text: str) -> str:
    return text.lower().translate(_TRANSLIT_TABLE)


def _tokenize(text: str) -> list[str]:
    raw = re.findall(r'[a-zA-Zа-яА-ЯёЁ_]+', text or '')
    tokens: list[str] = []
    for chunk in raw:
        chunk = chunk.strip('_')
        if not chunk:
            continue
        parts = [p for p in chunk.split('_') if p]
        for part in parts:
            tokens.append(part.lower())
            tokens.append(_translit(part))
    return [t for t in tokens if t]


def _normalize_concept(text: str) -> str:
    tokens = _tokenize(text)
    return "_".join(dict.fromkeys(tokens[:3]))


def _normalize_metric_concept(terms: list[str]) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for term in terms:
        for token in _tokenize(term):
            if token in seen:
                continue
            seen.add(token)
            tokens.append(token)
            if len(tokens) >= 6:
                return "_".join(tokens)
    return "_".join(tokens)


def _looks_like_explicit_column(token: str) -> bool:
    lower = token.lower()
    return "_" in lower or lower.endswith(_METRIC_SUFFIXES + _LABEL_SUFFIXES + ('_id', '_code'))


def _is_label_slot(slot: str) -> bool:
    return slot.endswith(_LABEL_SUFFIXES) or slot in {'name', 'label'}


def _is_metric_slot(slot: str) -> bool:
    return slot.endswith(_METRIC_SUFFIXES) or slot.endswith(('_code', '_score'))


def _is_dimension_slot(slot: str) -> bool:
    lower = (slot or '').lower().strip()
    return bool(lower) and lower != 'date' and not _is_metric_slot(lower)


def _fuzzy_overlap_score(slot_tokens: set[str], source_tokens: set[str]) -> float:
    matched = 0.0
    for st in slot_tokens:
        if len(st) < 3:
            continue
        if st in source_tokens:
            matched += 1.0
            continue
        for src in source_tokens:
            if len(src) < 3:
                continue
            if st.startswith(src[:4]) or src.startswith(st[:4]):
                matched += 0.6
                break
    return matched


def _derive_requested_slots(user_input: str, intent: dict[str, Any]) -> dict[str, Any]:
    """Вытащить из user_input явные аналитические слоты без LLM."""
    query = _normalize_query_text(user_input)
    agg_hint = str(intent.get('aggregation_hint') or '').lower().strip()
    query_tokens = _tokenize(query)
    dimensions: list[str] = []

    explicit_cols = [
        t for t in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query)
        if _looks_like_explicit_column(t)
    ]
    for col in explicit_cols:
        lower = col.lower()
        if lower.endswith(_LABEL_SUFFIXES) or any(tok in _DATE_HINTS for tok in lower.split('_')):
            dimensions.append(lower)

    for m in re.finditer(r'(?:по|в разбивке по)\s+([a-zA-Zа-яА-ЯёЁ_]+)', query):
        concept = _normalize_concept(m.group(1))
        if not concept:
            continue
        if any(tok in _DATE_HINTS for tok in concept.split('_')):
            dimensions.append('date')
        elif _looks_like_explicit_column(m.group(1)):
            dimensions.append(m.group(1).lower())
        else:
            dimensions.append(f'{concept}_name')

    for m in re.finditer(r'(?:названи[еяю]|наименовани[еяю])\s+([a-zA-Zа-яА-ЯёЁ_]+)', query):
        concept = _normalize_concept(m.group(1))
        if concept:
            dimensions.append(f'{concept}_name')

    metric: str | None = None
    metric_requires_numeric = True
    metric_cols = [c.lower() for c in explicit_cols if _is_metric_slot(c.lower())]
    if metric_cols:
        metric = metric_cols[0]
        metric_requires_numeric = metric.endswith(_METRIC_SUFFIXES)
    elif agg_hint in {'sum', 'avg', 'min', 'max', 'count'}:
        for entity in intent.get('entities') or []:
            seed = str(entity).strip()
            synonym_terms = sorted({str(term).strip() for term in expand_with_synonyms(seed) if str(term).strip()})
            expanded_terms = [seed] + [term for term in synonym_terms if term != seed]
            concept = _normalize_metric_concept(expanded_terms)
            if concept:
                metric = concept
                break

    join_key_hints = [
        token.lower() for token in explicit_cols
        if token.lower().endswith(('_id', '_code', '_num', '_no'))
    ]
    for m in re.finditer(r'(?:по|через|join|using|ключ[ауеом]?)\s+([a-zA-Z_][a-zA-Z0-9_]*)', query):
        token = m.group(1).lower()
        if _looks_like_explicit_column(token) or re.fullmatch(r'[a-z]{3,10}', token):
            join_key_hints.append(token)

    # ---- Дополняем из LLM-поля required_output (уже распознанные измерения) ----
    # Надёжнее regex: LLM понимает "по X и Y", "по X, Y", "X и Y" одинаково.
    # Добавляем без суффикса _name, чтобы не активировать label-slot фильтр
    # в _choose_best_column (иначе пропустятся колонки без 'name' в токенах).
    for ro_item in (intent.get('required_output') or []):
        concept = _normalize_concept(str(ro_item))
        if not concept:
            continue
        if any(tok in _DATE_HINTS for tok in concept.split('_')):
            if 'date' not in dimensions:
                dimensions.append('date')
        elif concept not in dimensions and f'{concept}_name' not in dimensions:
            dimensions.append(concept)

    return {
        'dimensions': list(dict.fromkeys(dimensions)),
        'metric': metric,
        'metric_requires_numeric': metric_requires_numeric,
        'join_key_hints': list(dict.fromkeys(join_key_hints)),
        'has_date_filter': ('январ' in query or 'феврал' in query or 'март' in query or
                            'апрел' in query or 'май' in query or
                            'мая' in query or 'июн' in query or
                            'июл' in query or 'август' in query or 'сентябр' in query or
                            'октябр' in query or 'ноябр' in query or 'декабр' in query),
        'explicit_count_metric': agg_hint == 'count' and bool(metric_cols),
    }


def _semantic_match_score(col_name: str, desc: str, slot: str) -> float:
    col_tokens = set(_tokenize(col_name))
    desc_tokens = set(_tokenize(desc))
    slot_tokens = [t for t in slot.split('_') if t]
    slot_set = set(slot_tokens)
    if _is_label_slot(slot):
        slot_set = {t for t in slot_set if t not in {'name', 'label', 'title'}}

    score = 0.0
    if not slot_set:
        slot_set = set(slot_tokens)
        if not slot_set:
            return 0.0

    name_overlap = _fuzzy_overlap_score(slot_set, col_tokens)
    desc_overlap = _fuzzy_overlap_score(slot_set, desc_tokens)
    if name_overlap:
        score += min(0.8, 0.35 * name_overlap)
    if desc_overlap:
        score += min(0.6, 0.25 * desc_overlap)

    semantic_hit = name_overlap > 0 or desc_overlap > 0

    if _is_label_slot(slot) and semantic_hit:
        if any(t in col_tokens for t in ('name', 'label', 'title')):
            score += 0.25
        if any(t in desc_tokens for t in ('naimenovanie', 'nazvanie', 'name')):
            score += 0.2
        lower = col_name.lower()
        if lower.startswith(('new_', 'current_', 'actual_')):
            score += 0.8
        if lower.startswith(('old_', 'prev_')):
            score -= 0.8

    if slot == 'date':
        if any(t in col_tokens for t in _DATE_HINTS):
            score += 0.6
        if any(t in desc_tokens for t in ('data', 'date', 'otchetnaya')):
            score += 0.2
        lower = col_name.lower()
        if 'report' in lower or 'otchetn' in desc_tokens:
            score += 0.2
        if lower.startswith(('inserted_', 'modified_', 'created_', 'updated_')):
            score -= 0.2

    if semantic_hit and slot.endswith('_name') and col_name.lower() == slot:
        score += 0.2
    if semantic_hit and col_name.lower() == slot:
        score += 0.3

    return min(score, 1.0)


def _choose_best_column(
    table_structures: dict[str, str],
    table_types: dict[str, str],
    schema_loader: Any,
    slot: str,
    allowed_tables: set[str] | None = None,
    require_numeric: bool = False,
    agg_hint: str | None = None,
    dim_source_table: str | None = None,
) -> tuple[str, str] | None:
    """Выбрать лучший источник атрибута/метрики по имени, описанию и метаданным.

    Если задан dim_source_table — поиск ограничивается этой таблицей
    (приходит из user_hints.dim_sources). Это даёт пользователю явный
    контроль: «сегмент возьми в TABLE» — и колонка строго оттуда.
    """
    best: tuple[float, str, str] | None = None

    # Приоритет user_hints.dim_sources: ограничиваем поиск указанной таблицей.
    if dim_source_table:
        bound_allowed = {dim_source_table}
        if allowed_tables:
            bound_allowed = bound_allowed & allowed_tables
        if bound_allowed:
            allowed_tables = bound_allowed
        else:
            allowed_tables = {dim_source_table}

    is_dimension_slot = _is_dimension_slot(slot)

    for table_key in table_structures:
        if allowed_tables and table_key not in allowed_tables:
            continue
        parts = table_key.split('.', 1)
        if len(parts) != 2:
            continue
        cols_df = schema_loader.get_table_columns(parts[0], parts[1])
        if cols_df.empty:
            continue
        t_type = table_types.get(table_key, 'unknown')

        for _, row in cols_df.iterrows():
            col_name = str(row.get('column_name', '')).strip()
            if not col_name:
                continue
            dtype = str(row.get('dType', '') or '').lower().strip()
            is_pk = bool(row.get('is_primary_key', False))
            if require_numeric and not _is_numeric(dtype):
                continue

            desc = str(row.get('description', '') or '')
            semantic = _semantic_match_score(col_name, desc, slot)
            if semantic <= 0 and require_numeric and _is_numeric(dtype) and not is_pk:
                # Fallback для англоязычных metric-name колонок, когда
                # описание не повторяет термин сущности из пользовательского запроса.
                metric_semantic = _metric_score(col_name, [slot])
                if metric_semantic >= 0.4:
                    semantic = metric_semantic * 0.35
            if semantic <= 0:
                continue
            col_tokens = set(_tokenize(col_name))
            desc_tokens = set(_tokenize(desc))
            if _is_label_slot(slot) and not (
                {'name', 'label', 'title'} & col_tokens
                or {'naimenovanie', 'nazvanie', 'name'} & desc_tokens
            ):
                continue

            not_null = float(row.get('not_null_perc', 0) or 0)
            unique = float(row.get('unique_perc', 0) or 0)
            score = semantic * 1000 + not_null * 1.5 + min(unique, 100.0) * 0.1
            lower_name = col_name.lower()
            if lower_name.startswith(('calc_', 'plan_', 'prev_', 'next_')):
                score -= 180
            elif lower_name.startswith(('fact_', 'avg_')):
                score -= 40

            if _is_label_slot(slot):
                if t_type in ('dim', 'ref', 'unknown'):
                    score += 40
                if t_type == 'fact':
                    score -= 30
            elif is_dimension_slot:
                # Для обычных dimension-slot'ов вроде "сегмент" или "регион"
                # также слегка предпочитаем dim/ref-источники, даже если slot
                # не был выражен как явный *_name.
                if t_type in ('dim', 'ref', 'unknown'):
                    score += 25
                if t_type == 'fact':
                    score -= 20
            elif _is_metric_slot(slot) and t_type == 'fact':
                score += 25
            if (agg_hint or '').lower() == 'sum' and any(
                marker in lower_name for marker in ('perc', 'pct', 'rate', 'avg')
            ):
                score -= 320
            if (agg_hint or '').lower() == 'sum' and re.match(
                r'^(is_|has_|was_|are_)', lower_name,
            ):
                score -= 200

            candidate = (score, table_key, col_name)
            if best is None or candidate > best:
                best = candidate

    if best is None:
        return None
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Scoring вспомогательные функции
# ---------------------------------------------------------------------------

def _entity_score(col_name: str, col_desc: str, entities: list[str]) -> float:
    """Оценить насколько колонка совпадает с entity-ключевыми словами запроса.

    Логика:
    - точное вхождение entity в имя колонки / имя колонки в entity → 0.6
    - вхождение по частям snake_case → 0.35
    - вхождение в описание → 0.2
    """
    if not entities:
        return 0.0
    col_lower = col_name.lower()
    desc_lower = (col_desc or '').lower()
    col_parts = set(col_lower.split('_'))
    score = 0.0
    for entity in entities:
        e = entity.lower().strip()
        if not e or len(e) < 2:
            continue
        if e in col_lower or col_lower in e:
            score += 0.6
        elif col_parts & set(e.split('_')):
            score += 0.35
        elif e in desc_lower:
            score += 0.2
    return min(score, 1.0)


def _metric_score(col_name: str, entities: list[str]) -> float:
    """Оценить метрическую «весомость» колонки по имени + entities."""
    col_parts = set(col_name.lower().split('_'))
    base = 0.4 if col_parts & _METRIC_PARTS else 0.0
    entity_hit = _entity_score(col_name, '', entities)
    return min(base + entity_hit * 0.4, 1.0)


def _choose_single_entity_count_column(
    table_key: str,
    schema_loader: Any,
    entities: list[str],
    subject: str = "",
) -> str | None:
    """Подобрать PK/identifier-колонку для single-entity COUNT на факте."""
    parts = table_key.split(".", 1)
    if len(parts) != 2:
        return None
    schema_name, table_name = parts
    cols_df = schema_loader.get_table_columns(schema_name, table_name)
    if cols_df.empty:
        return None

    best: tuple[float, str] | None = None
    subject = str(subject or "").strip().lower()
    subject_tokens = [subject] if subject else []
    query_entities = [str(e).strip() for e in entities if str(e).strip()]

    for _, row in cols_df.iterrows():
        col_name = str(row.get("column_name", "") or "").strip()
        if not col_name:
            continue
        desc = str(row.get("description", "") or "")
        is_pk = bool(row.get("is_primary_key", False))
        dtype = str(row.get("dType", "") or "").lower().strip()
        unique_perc = float(row.get("unique_perc", 0) or 0)
        semantics = schema_loader.get_column_semantics(schema_name, table_name, col_name)
        sem_class = str(semantics.get("semantic_class", "") or "")

        score = 0.0
        if is_pk:
            score += 140.0
        if sem_class == "identifier":
            score += 80.0
        if sem_class == "join_key":
            score += 20.0
        if unique_perc >= 95.0:
            score += 45.0
        elif unique_perc >= 80.0:
            score += 20.0
        if _is_numeric(dtype):
            score -= 40.0

        entity_match = _entity_score(col_name, desc, query_entities + subject_tokens)
        score += entity_match * 90.0

        lower_name = col_name.lower()
        if lower_name.endswith("_code"):
            score += 35.0
        if lower_name.endswith("_id"):
            score += 15.0
        if subject == "task" and "task" in lower_name:
            score += 25.0
        if subject == "client" and any(tok in lower_name for tok in ("client", "cust", "inn")):
            score += 25.0
        if subject == "employee" and any(tok in lower_name for tok in ("employee", "staff", "emp")):
            score += 25.0

        candidate = (score, col_name)
        if best is None or candidate > best:
            best = candidate

    if best is None or best[0] < 60.0:
        return None
    return best[1]


def _is_scalar_count_request(
    agg_hint: str,
    semantic_output_dimensions: list[str],
    explicit_count_metric: bool,
) -> bool:
    """True для запросов вида «сколько задач ...» без разбивки."""
    if agg_hint != "count":
        return False
    if explicit_count_metric:
        return False
    return not semantic_output_dimensions


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def select_columns(
    intent: dict[str, Any],
    table_structures: dict[str, str],
    table_types: dict[str, str],
    join_analysis_data: dict[str, Any],
    schema_loader: Any,
    user_input: str = "",
    user_hints: dict[str, Any] | None = None,
    semantic_frame: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Детерминированно выбрать колонки и JOIN-спецификацию.

    Args:
        intent:            dict из intent_classifier (aggregation_hint, entities, …)
        table_structures:  dict schema.table → строка-описание (из table_explorer)
        table_types:       dict schema.table → "fact"/"dim"/"ref"/"unknown"
        join_analysis_data: dict из table_explorer (pre-computed join candidates)
        schema_loader:     SchemaLoader для доступа к колонкам и check_key_uniqueness
    user_hints:        dict из hint_extractor (dim_sources, join_fields, having_hints)
    semantic_frame:    Семантический фрейм запроса; нужен для requires_single_entity_count

    Returns:
        dict с ключами: selected_columns, join_spec, confidence, reason
    """
    entities: list[str] = [str(e) for e in (intent.get('entities') or [])]
    agg_hint: str = str(intent.get('aggregation_hint') or '').lower().strip()
    date_filters: dict = intent.get('date_filters') or {}
    filter_conditions: list[dict] = intent.get('filter_conditions') or []
    requested = _derive_requested_slots(user_input, intent)
    frame = dict(semantic_frame or {})
    requires_single_entity_count = bool(frame.get("requires_single_entity_count"))
    semantic_output_dimensions = list(frame.get("output_dimensions") or [])
    if semantic_output_dimensions:
        requires_single_entity_count = False
    scalar_count_request = _is_scalar_count_request(
        agg_hint=agg_hint,
        semantic_output_dimensions=semantic_output_dimensions,
        explicit_count_metric=bool(requested.get("explicit_count_metric")),
    )

    # Подсказки пользователя: связки «слот → таблица», JOIN-поля, HAVING-хинты.
    user_hints = user_hints or {}
    hint_dim_sources: dict[str, dict[str, str]] = user_hints.get('dim_sources', {}) or {}
    hint_join_fields: list[str] = list(user_hints.get('join_fields', []) or [])
    hint_having: list[dict[str, Any]] = list(user_hints.get('having_hints', []) or [])

    def _slot_dim_source(slot_name: str) -> str | None:
        """Найти dim-источник для слота через user_hints (с учётом синонимов)."""
        if not hint_dim_sources or not slot_name:
            return None
        slot_lower = slot_name.lower()
        for key, binding in hint_dim_sources.items():
            if not isinstance(binding, dict):
                continue
            key_lower = key.lower()
            if (
                key_lower == slot_lower
                or key_lower in slot_lower
                or slot_lower in key_lower
            ):
                tbl = binding.get('table')
                if tbl:
                    return tbl
        return None

    has_date_filter = bool(
        date_filters.get('from') or date_filters.get('to') or requested['has_date_filter']
    )
    wants_aggregation = agg_hint not in ('', 'null', 'list', 'None')

    selected_columns: dict[str, dict] = {}
    per_table_confidence: list[float] = []
    chosen_metric_ref: tuple[str, str] | None = None

    for table_key in table_structures:
        parts = table_key.split('.', 1)
        if len(parts) != 2:
            continue
        schema_name, table_name = parts

        cols_df = schema_loader.get_table_columns(schema_name, table_name)
        if cols_df.empty or 'column_name' not in cols_df.columns:
            continue

        t_type = table_types.get(table_key, 'unknown')

        agg_candidates: list[tuple[str, float]] = []   # (col, score)
        gb_candidates: list[tuple[str, float]] = []
        filter_candidates: list[tuple[str, float]] = []

        for _, row in cols_df.iterrows():
            col_name = str(row.get('column_name', '')).strip()
            if not col_name:
                continue

            dtype = str(row.get('dType', '') or '').lower().strip()
            is_pk = bool(row.get('is_primary_key', False))
            unique_perc = float(row.get('unique_perc', 0) or 0)
            desc = str(row.get('description', '') or '')

            # ---- aggregate score ----
            if wants_aggregation and t_type in ('fact', 'unknown') and not is_pk:
                if _is_numeric(dtype):
                    s = 0.4 + _metric_score(col_name, entities) * 0.6
                    if requested['metric']:
                        s = max(s, _semantic_match_score(col_name, desc, requested['metric']))
                    agg_candidates.append((col_name, round(s, 3)))

            # ---- group_by score ----
            # Не кладём PK и числовые метрики в group_by если они не упомянуты в entities
            gb_s = 0.0
            if not is_pk:
                if scalar_count_request:
                    gb_s = 0.0
                else:
                    entity_hit = _entity_score(col_name, desc, entities)
                    slot_hit = max(
                        [_semantic_match_score(col_name, desc, slot) for slot in requested['dimensions']],
                        default=0.0,
                    )
                    if _is_categorical(dtype):
                        gb_s += 0.25
                    if (entity_hit > 0 or slot_hit > 0) and unique_perc > 0 and unique_perc < 30:
                        gb_s += 0.20
                    gb_s += max(entity_hit, slot_hit) * 0.75
                    # Числовые — только если явно упомянуты в entities
                    if _is_numeric(dtype) and max(entity_hit, slot_hit) < 0.3:
                        gb_s = 0.0
                    if _is_date(dtype):
                        if slot_hit >= 0.8:
                            gb_s = max(gb_s, 0.95)
                        else:
                            gb_s = min(gb_s, 0.2)
                        # Подавляем дата-колонки в таблицах, которые выступают dim-источником
                        # для нечасовых слотов. Дата берётся из фактовой таблицы; справочник
                        # нужен только ради своего атрибута (сегмент, регион и т.п.).
                        if gb_s > 0.05 and hint_dim_sources:
                            _is_nondated_dim_src = any(
                                b.get('table') == table_key and sk != 'date'
                                for sk, b in hint_dim_sources.items()
                                if isinstance(b, dict)
                            )
                            if _is_nondated_dim_src:
                                gb_s = 0.05
            if gb_s > 0:
                gb_candidates.append((col_name, round(gb_s, 3)))

            # ---- filter score ----
            flt_s = 0.0
            if _is_date(dtype) and has_date_filter:
                flt_s = 0.90
                if _semantic_match_score(col_name, desc, 'date') >= 0.8:
                    flt_s = 0.98
            for fc in filter_conditions:
                hint = str(fc.get('column_hint', '') or '').lower()
                col_lower = col_name.lower()
                if hint and (hint in col_lower or col_lower in hint):
                    flt_s = max(flt_s, 0.85)
            col_parts = set(col_name.lower().split('_'))
            if col_parts & _STATUS_PARTS and filter_conditions:
                flt_s = max(flt_s, 0.35)
            if flt_s > 0:
                filter_candidates.append((col_name, round(flt_s, 3)))

        # ---- Выбор top-N для каждой роли ----

        # aggregate: top-1 по score (только если score > 0.45)
        agg_cols: list[str] = []
        if wants_aggregation and agg_hint != 'count':
            agg_sorted = sorted(agg_candidates, key=lambda x: x[1], reverse=True)
            agg_cols = [c for c, s in agg_sorted if s > 0.45][:1]

        # COUNT на dim/ref-таблице: используем PK для COUNT(DISTINCT pk_col)
        # Это отвечает на «сколько всего X?» вместо генерации GROUP BY + COUNT(*).
        # Признак: agg_hint=count, таблица-справочник, нет явных числовых агрегатов.
        _use_count_distinct = False
        if agg_hint == 'count' and t_type in ('dim', 'ref') and not agg_cols:
            pk_mask = cols_df.get('is_primary_key', pd.Series(dtype=bool)).astype(bool)
            pk_for_count = cols_df.loc[pk_mask, 'column_name'].tolist()
            if pk_for_count:
                agg_cols = pk_for_count[:2]  # максимум 2 PK для COUNT DISTINCT
                _use_count_distinct = True

        # filter: все с score > 0.5, max 4
        flt_sorted = sorted(filter_candidates, key=lambda x: x[1], reverse=True)
        filter_cols = [c for c, s in flt_sorted if s > 0.50][:4]

        # group_by: top-3 с score > 0.30, не пересекаем с aggregate
        agg_set = set(agg_cols)
        gb_sorted = sorted(gb_candidates, key=lambda x: x[1], reverse=True)
        group_by_cols = [c for c, s in gb_sorted if s > 0.30 and c not in agg_set][:3]

        # COUNT DISTINCT на dim-таблице: group_by не нужен (считаем сами PK-сущности)
        if _use_count_distinct or scalar_count_request:
            group_by_cols = []

        # select = group_by ∪ aggregate (без дублей, порядок: group_by первым)
        seen: dict[str, None] = {}
        for c in group_by_cols:
            seen[c] = None
        for c in agg_cols:
            if c not in seen:
                seen[c] = None
        select_cols = list(seen.keys())

        # ---- confidence для этой таблицы ----
        t_conf = 0.50
        if select_cols:
            t_conf += 0.15
        if wants_aggregation and agg_hint != 'count' and agg_cols:
            t_conf += 0.20
        elif agg_hint == 'count':
            t_conf += 0.15
        elif wants_aggregation and not agg_cols and t_type not in ('dim', 'ref'):
            t_conf -= 0.20   # хотим агрегацию, но не нашли числовых колонок (не применяется к справочникам)
        if entities and not group_by_cols and agg_hint not in ('count', ''):
            t_conf -= 0.10   # есть entities, но GROUP BY пустой — подозрительно

        per_table_confidence.append(max(0.0, min(1.0, t_conf)))

        # ---- Собираем роли ----
        roles: dict[str, list[str]] = {}
        if select_cols:
            roles['select'] = select_cols
        if filter_cols:
            roles['filter'] = filter_cols
        if agg_cols:
            roles['aggregate'] = agg_cols
        if group_by_cols:
            roles['group_by'] = group_by_cols

        if roles:
            selected_columns[table_key] = roles

    # ---- JOIN-спецификация из join_analysis_data ----
    # Таблицы, уже занятые другими dim_source-слотами, — предварительно вычислим
    # для фильтрации при поиске «свободных» слотов (таких как 'date').
    _dim_src_tables: set[str] = {
        b.get('table')
        for b in hint_dim_sources.values()
        if isinstance(b, dict) and b.get('table')
    }

    if requested['dimensions']:
        for slot in requested['dimensions']:
            bound_table = _slot_dim_source(slot)

            # Для слотов без явного dim_source-биндинга (например 'date') запрещаем
            # выбирать колонку из таблиц, которые выступают dim_source для других
            # слотов. Иначе epk_create_dttm из справочника бьёт report_dt из факта
            # за счёт более высокого unique_perc (timestamp vs агрегированная дата).
            _allowed: set[str] | None = None
            if not bound_table and _dim_src_tables:
                _allowed = set(table_structures.keys()) - _dim_src_tables

            choice = _choose_best_column(
                table_structures, table_types, schema_loader, slot,
                dim_source_table=bound_table,
                allowed_tables=_allowed,
            )
            # Fallback: если dim_source указан, но колонки в нём нет —
            # предупреждаем и берём из любой таблицы.
            if not choice and bound_table:
                logger.warning(
                    "ColumnSelectorDet: dim_source '%s' для слота '%s' "
                    "не содержит подходящей колонки — fallback на общий поиск",
                    bound_table, slot,
                )
                choice = _choose_best_column(
                    table_structures, table_types, schema_loader, slot,
                )
            if not choice:
                continue
            table_key, col_name = choice
            roles = selected_columns.setdefault(table_key, {})
            roles.setdefault('select', [])
            roles.setdefault('group_by', [])
            if col_name not in roles['select']:
                roles['select'].append(col_name)
            if col_name not in roles['group_by']:
                roles['group_by'].append(col_name)
            if slot == 'date':
                roles.setdefault('filter', [])
                if col_name not in roles['filter'] and has_date_filter:
                    roles['filter'].append(col_name)

    # ---- Safety-net: dim_source-слоты, не попавшие в requested['dimensions'] ----
    # Срабатывает, если required_output от LLM пустой или не содержит нужного измерения.
    # Используем сырой slot_key (напр. "segment"), а не производный с суффиксом _name —
    # это критично: без суффикса _name не активируется label-slot фильтр в _choose_best_column.
    if hint_dim_sources:
        _matched_dim_keys: set[str] = set()
        for slot in requested['dimensions']:
            slot_lower = slot.lower()
            for key in hint_dim_sources:
                key_lower = key.lower()
                if key_lower == slot_lower or key_lower in slot_lower or slot_lower in key_lower:
                    _matched_dim_keys.add(key)

        for slot_key, binding in hint_dim_sources.items():
            if slot_key in _matched_dim_keys:
                continue
            if not isinstance(binding, dict):
                continue
            bound_table = binding.get('table')
            choice = _choose_best_column(
                table_structures, table_types, schema_loader, slot_key,
                dim_source_table=bound_table,
            )
            if not choice and bound_table:
                logger.warning(
                    "ColumnSelectorDet: dim_source '%s' для слота '%s' "
                    "не содержит подходящей колонки — fallback на общий поиск",
                    bound_table, slot_key,
                )
                choice = _choose_best_column(
                    table_structures, table_types, schema_loader, slot_key,
                )
            if not choice:
                logger.warning(
                    "ColumnSelectorDet: слот '%s' из dim_sources не удалось разрешить",
                    slot_key,
                )
                continue
            t_key, col_name = choice
            roles = selected_columns.setdefault(t_key, {})
            roles.setdefault('select', [])
            roles.setdefault('group_by', [])
            if col_name not in roles['select']:
                roles['select'].append(col_name)
            if col_name not in roles['group_by']:
                roles['group_by'].append(col_name)
            logger.info(
                "ColumnSelectorDet: dim_source-слот '%s' → %s.%s",
                slot_key, t_key, col_name,
            )

    if wants_aggregation and agg_hint != 'count' and requested['metric']:
        metric_choice = _choose_best_column(
            table_structures,
            table_types,
            schema_loader,
            requested['metric'],
            require_numeric=requested.get('metric_requires_numeric', True),
            agg_hint=agg_hint,
        )
        if metric_choice:
            chosen_metric_ref = metric_choice
            table_key, col_name = metric_choice
            roles = selected_columns.setdefault(table_key, {})
            roles.setdefault('select', [])
            roles.setdefault('aggregate', [])
            if col_name not in roles['select']:
                roles['select'].append(col_name)
            if col_name not in roles['aggregate']:
                roles['aggregate'].append(col_name)

    if requested['explicit_count_metric']:
        metric_choice = _choose_best_column(
            table_structures,
            table_types,
            schema_loader,
            requested['metric'] or '',
            require_numeric=requested.get('metric_requires_numeric', False),
            agg_hint=agg_hint,
        )
        if metric_choice:
            chosen_metric_ref = metric_choice
            table_key, col_name = metric_choice
            roles = selected_columns.setdefault(table_key, {})
            roles.setdefault('select', [])
            roles.setdefault('aggregate', [])
            if col_name not in roles['select']:
                roles['select'].append(col_name)
            if col_name not in roles['aggregate']:
                roles['aggregate'].append(col_name)

    if has_date_filter:
        date_choice = _choose_best_column(
            table_structures, table_types, schema_loader, 'date'
        )
        if date_choice:
            table_key, col_name = date_choice
            roles = selected_columns.setdefault(table_key, {})
            roles.setdefault('filter', [])
            if col_name not in roles['filter']:
                roles['filter'].append(col_name)

    if agg_hint == 'count' and not requested['explicit_count_metric']:
        query_norm = _normalize_query_text(user_input)
        explicit_positions: list[tuple[int, str]] = []
        for t in table_structures:
            table_name = t.split('.', 1)[-1].lower()
            pos = query_norm.find(table_name)
            if pos >= 0:
                explicit_positions.append((pos, t))
        explicit_positions.sort()
        main_table = explicit_positions[0][1] if explicit_positions else None
        if main_table is None:
            main_table = next((t for t, tp in table_types.items() if tp == 'fact'), None)
        if main_table is None:
            main_table = next(iter(table_structures), None)
        if main_table:
            roles = selected_columns.setdefault(main_table, {})
            if requires_single_entity_count or scalar_count_request:
                count_col = _choose_single_entity_count_column(
                    main_table,
                    schema_loader,
                    entities=entities,
                    subject=str(frame.get("subject") or requested.get("metric") or ""),
                )
                if count_col:
                    roles['aggregate'] = [count_col]
                    roles['select'] = [count_col]
                    roles.pop('group_by', None)
                    logger.info(
                        "ColumnSelectorDet: single-entity count → %s.%s",
                        main_table, count_col,
                    )
                else:
                    roles.setdefault('aggregate', [])
                    if '*' not in roles['aggregate']:
                        roles['aggregate'].append('*')
            else:
                roles.setdefault('aggregate', [])
                if '*' not in roles['aggregate']:
                    roles['aggregate'].append('*')

    if requires_single_entity_count or scalar_count_request:
        main_fact = next((t for t, tp in table_types.items() if tp == 'fact'), None)
        for table_key, roles in list(selected_columns.items()):
            if table_key != main_fact:
                roles.pop('group_by', None)
                non_agg_select = [
                    c for c in roles.get('select', [])
                    if c in roles.get('aggregate', [])
                ]
                if non_agg_select:
                    roles['select'] = non_agg_select
                elif not roles.get('filter'):
                    selected_columns.pop(table_key, None)
            else:
                roles.pop('group_by', None)

    if requested['dimensions']:
        allowed_refs: set[tuple[str, str]] = set()
        for slot in requested['dimensions']:
            bound_table = _slot_dim_source(slot)
            choice = _choose_best_column(
                table_structures, table_types, schema_loader, slot,
                dim_source_table=bound_table,
            )
            if not choice and bound_table:
                choice = _choose_best_column(
                    table_structures, table_types, schema_loader, slot,
                )
            if choice:
                allowed_refs.add(choice)
        if has_date_filter:
            choice = _choose_best_column(
                table_structures, table_types, schema_loader, 'date'
            )
            if choice:
                allowed_refs.add(choice)
        if requested['metric']:
            choice = _choose_best_column(
                table_structures,
                table_types,
                schema_loader,
                requested['metric'],
                require_numeric=requested.get('metric_requires_numeric', True),
                agg_hint=agg_hint,
            )
            if choice:
                allowed_refs.add(choice)
                chosen_metric_ref = choice

        for table_key, roles in list(selected_columns.items()):
            if chosen_metric_ref and roles.get('aggregate'):
                keep_agg = [
                    c for c in roles['aggregate']
                    if (table_key, c) == chosen_metric_ref or c == '*'
                ]
                if keep_agg:
                    roles['aggregate'] = keep_agg[:1]
                else:
                    roles.pop('aggregate', None)
            for role in ('select', 'group_by', 'filter'):
                cols = roles.get(role, [])
                if not cols:
                    continue
                protected = set(roles.get('aggregate', [])) if role != 'filter' else set()
                pruned = [
                    c for c in cols
                    if (table_key, c) in allowed_refs or c in protected
                ]
                if pruned:
                    roles[role] = pruned
                else:
                    roles.pop(role, None)
            if not roles:
                selected_columns.pop(table_key, None)

    join_spec = _build_join_spec(
        join_analysis_data,
        selected_columns,
        schema_loader,
        table_types,
        user_input=user_input,
        hint_join_fields=hint_join_fields,
    )

    # ---- Итоговая confidence ----
    if not selected_columns:
        overall = 0.0
        reason = 'Не удалось выбрать колонки ни для одной таблицы'
    elif not per_table_confidence:
        overall = 0.0
        reason = 'Нет данных о колонках'
    else:
        overall = sum(per_table_confidence) / len(per_table_confidence)
        if requested['dimensions']:
            overall += 0.10
        if requested['metric'] and any(
            roles.get('aggregate') for roles in selected_columns.values()
        ):
            overall += 0.10
        # Штраф: мультитабличный запрос без JOIN-ключей
        if len(selected_columns) > 1 and not join_spec:
            overall *= 0.55
            reason = (
                f'{len(selected_columns)} таблиц, но JOIN-ключи не найдены — '
                'снижаем confidence'
            )
        else:
            reason = (
                f'Выбрано таблиц: {len(selected_columns)}, '
                f'join-ключей: {len(join_spec)}, '
                f'avg col confidence: {overall:.2f}'
            )
        overall = min(overall, 0.99)

    logger.info('ColumnSelectorDet: %s', reason)

    return {
        'selected_columns': selected_columns,
        'join_spec': join_spec,
        'confidence': round(overall, 2),
        'reason': reason,
        # Прокидываем HAVING-хинты дальше — sql_planner превратит их в
        # HAVING COUNT(DISTINCT <col>) >= N с подбором колонки по unit_hint.
        'having_hints': hint_having,
    }


# ---------------------------------------------------------------------------
# JOIN-спецификация
# ---------------------------------------------------------------------------

def _norm_col_name(name: str) -> str:
    """Нормализовать имя ключевой колонки: убрать old_/new_/prev_ префиксы."""
    return _PK_NORM_RE.sub("", name.lower())


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
    _DIM = {'dim', 'ref'}
    t1_type = table_types.get(t1, 'unknown')
    t2_type = table_types.get(t2, 'unknown')

    if t2_type in _DIM:
        dim_table, fact_table = t2, t1
    elif t1_type in _DIM:
        dim_table, fact_table = t1, t2
    else:
        return []

    # PK-колонки dim-таблицы
    dim_parts = dim_table.split('.', 1)
    if len(dim_parts) != 2:
        return []
    try:
        dim_cols_df = schema_loader.get_table_columns(dim_parts[0], dim_parts[1])
        if dim_cols_df.empty or 'column_name' not in dim_cols_df.columns:
            return []
        pk_mask = dim_cols_df.get('is_primary_key', pd.Series(dtype=bool)).astype(bool)
        pk_cols: list[str] = dim_cols_df.loc[pk_mask, 'column_name'].tolist()
    except Exception:
        return []

    if len(pk_cols) < 2:
        return []  # Не составной PK

    # Уже покрытая dim-колонка из initial_entry
    initial_left_tbl = '.'.join(initial_entry['left'].split('.')[:2])
    if initial_left_tbl == dim_table:
        covered_dim_col = initial_entry['left'].rsplit('.', 1)[-1]
        fact_is_left = False
    else:
        covered_dim_col = initial_entry['right'].rsplit('.', 1)[-1]
        fact_is_left = True

    # Колонки fact-таблицы
    fact_parts = fact_table.split('.', 1)
    try:
        fact_cols_df = schema_loader.get_table_columns(fact_parts[0], fact_parts[1])
        fact_col_names: list[str] = (
            fact_cols_df['column_name'].tolist() if not fact_cols_df.empty else []
        )
    except Exception:
        fact_col_names = []

    additional: list[dict[str, Any]] = []
    for pk_col in pk_cols:
        if pk_col == covered_dim_col:
            continue  # уже покрыто initial_entry

        # Ищем пару в fact-таблице: exact same name first, затем normalized
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
                'left': f'{fact_table}.{fact_col}',
                'right': f'{dim_table}.{pk_col}',
                'safe': False,
                'strategy': initial_entry.get('strategy', 'fact_dim_join'),
                'risk': f'{fact_col} — composite PK pair, не уникален в {fact_table}',
            }
        else:
            entry = {
                'left': f'{dim_table}.{pk_col}',
                'right': f'{fact_table}.{fact_col}',
                'safe': False,
                'strategy': initial_entry.get('strategy', 'dim_fact_join'),
                'risk': f'{fact_col} — composite PK pair, не уникален в {fact_table}',
            }
        additional.append(entry)

    return additional


def _build_join_spec(
    join_analysis_data: dict[str, Any],
    selected_columns: dict[str, dict],
    schema_loader: Any,
    table_types: dict[str, str],
    user_input: str = "",
    hint_join_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Построить join_spec из pre-computed join_analysis_data (детерминированно).

    Выбирает top-1 кандидата для каждой пары. safe определяется из CSV.
    Если задан hint_join_fields (из user_hints), кандидат с exact-match
    именем из подсказок выбирается с высоким приоритетом.
    """
    join_spec: list[dict[str, Any]] = []
    processed: set[frozenset] = set()

    for _pair_key, data in (join_analysis_data or {}).items():
        if not isinstance(data, dict):
            continue
        t1 = data.get('table1', '')
        t2 = data.get('table2', '')
        if not t1 or not t2:
            continue

        pair = frozenset([t1, t2])
        if pair in processed:
            continue
        processed.add(pair)

        # Обе таблицы должны присутствовать в selected_columns
        if t1 not in selected_columns or t2 not in selected_columns:
            continue

        text = data.get('text', '')
        cand = _pick_join_candidate(
            text, t1, t2, schema_loader,
            user_input=user_input,
            hint_join_fields=hint_join_fields,
        )
        if not cand:
            continue

        col1, col2 = cand['col1'], cand['col2']
        safe = _check_safe(t2, col2, schema_loader)

        t1_type = table_types.get(t1, 'unknown')
        t2_type = table_types.get(t2, 'unknown')
        strategy = _infer_strategy(t1_type, t2_type, safe)

        entry: dict[str, Any] = {
            'left': f'{t1}.{col1}',
            'right': f'{t2}.{col2}',
            'safe': safe,
            'strategy': strategy,
        }
        if not safe:
            entry['risk'] = f'{col2} не уникален в {t2}'

        pair_entries = [entry]
        join_spec.append(entry)

        # Если dim-таблица имеет составной PK — добавляем пары для недостающих колонок
        extra = _complete_composite_join(entry, t1, t2, table_types, schema_loader)
        join_spec.extend(extra)
        pair_entries.extend(extra)
        _apply_composite_safety(pair_entries, schema_loader, table_types)

    return join_spec


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
    candidate_sides.sort(key=lambda x: 0 if x[2] in {"dim", "ref"} else 1)

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


def _pick_join_candidate(
    text: str,
    t1: str,
    t2: str,
    schema_loader: Any,
    user_input: str = "",
    hint_join_fields: list[str] | None = None,
) -> dict[str, str] | None:
    """Выбрать join-кандидат с учётом явного ключа из запроса пользователя.

    Приоритеты:
    1. exact-match имени из user_hints.join_fields, существующего в обеих таблицах
       (даже если не key-like — пользователь явно указал)
    2. exact-match имени, упомянутого в тексте запроса, key-like
    3. normalized-match имени, упомянутого в тексте запроса
    4. fallback на _parse_top_candidate (статистика join_analysis)
    """
    query = _normalize_query_text(user_input)
    hint_join_fields = hint_join_fields or []
    t1_parts = t1.split('.', 1)
    t2_parts = t2.split('.', 1)
    if (query or hint_join_fields) and len(t1_parts) == 2 and len(t2_parts) == 2:
        cols1 = schema_loader.get_table_columns(t1_parts[0], t1_parts[1])
        cols2 = schema_loader.get_table_columns(t2_parts[0], t2_parts[1])
        if cols1.empty or cols2.empty:
            return _parse_top_candidate(text, t1, t2)
        rows1 = {
            str(row.get('column_name', '')).strip(): row
            for _, row in cols1.iterrows()
            if str(row.get('column_name', '')).strip()
        }
        rows2 = {
            str(row.get('column_name', '')).strip(): row
            for _, row in cols2.iterrows()
            if str(row.get('column_name', '')).strip()
        }
        names1 = set(rows1)
        names2 = set(rows2)
        names1_lower = {n.lower(): n for n in names1}
        names2_lower = {n.lower(): n for n in names2}

        # 1. Подсказка пользователя (user_hints.join_fields) — высший приоритет.
        for hf in hint_join_fields:
            hf_lower = hf.lower()
            if hf_lower in names1_lower and hf_lower in names2_lower:
                logger.info(
                    "ColumnSelectorDet: JOIN по '%s' (user_hints.join_fields)", hf,
                )
                return {
                    'col1': names1_lower[hf_lower],
                    'col2': names2_lower[hf_lower],
                }

        def _is_key_like(row: Any) -> bool:
            name = str(row.get('column_name', '')).lower()
            if bool(row.get('is_primary_key', False)):
                return True
            if name.endswith(('_id', '_code', '_num', '_no')):
                return True
            unique = float(row.get('unique_perc', 0) or 0)
            not_null = float(row.get('not_null_perc', 0) or 0)
            return unique >= 50 and not_null >= 50

        query_identifiers = {
            token.lower()
            for token in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query)
        }
        common_exact = sorted(
            names1 & names2,
            key=lambda name: (
                not name.lower().endswith(('_id', '_code')),
                len(name),
                name,
            ),
        )
        for hint in common_exact:
            if (
                hint.lower() in query_identifiers
                and (_is_key_like(rows1[hint]) or _is_key_like(rows2[hint]))
            ):
                return {'col1': hint, 'col2': hint}

        norm_left = {_norm_col_name(name): name for name in names1}
        norm_right = {_norm_col_name(name): name for name in names2}
        common_normalized = sorted(
            set(norm_left) & set(norm_right),
            key=lambda name: (
                not name.endswith(('_id', '_code')),
                len(name),
                name,
            ),
        )
        for hint in common_normalized:
            left_name = norm_left[hint]
            right_name = norm_right[hint]
            if (
                hint in query_identifiers
                and (_is_key_like(rows1[left_name]) or _is_key_like(rows2[right_name]))
            ):
                return {'col1': left_name, 'col2': right_name}

    return _parse_top_candidate(text, t1, t2)


def _parse_top_candidate(text: str, t1: str, t2: str) -> dict[str, str] | None:
    """Извлечь первый JOIN-кандидат из текста join_analysis.

    Форматы, которые понимает парсер:
      «t1.col1 ↔ t2.col2»
      «t1.col1 = t2.col2»
      «Кандидат #1 ... t1.col1 → t2.col2»
    """
    if not text:
        return None

    line_match = re.search(
        r'^\s*\[\d+(?:\.\d+)?\]\s+([a-zA-Z_][a-zA-Z0-9_]*)[^↔\n]*↔\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        text,
        re.MULTILINE,
    )
    if line_match:
        return {'col1': line_match.group(1), 'col2': line_match.group(2)}

    def _escape(s: str) -> str:
        return re.escape(s)

    # Паттерн: t1.col ↔/=/→ t2.col
    for pat in (
        rf'(?:{_escape(t1)})\.(\w+)\s*[↔=→]+\s*(?:{_escape(t2)})\.(\w+)',
        rf'(?:{_escape(t2)})\.(\w+)\s*[↔=→]+\s*(?:{_escape(t1)})\.(\w+)',
    ):
        m = re.search(pat, text, re.I)
        if m:
            if t2 in pat.split('→')[0] if '→' in pat else True:
                # Нужно проверить порядок: col1 для t1, col2 для t2
                if f'{t1}.' in pat[:pat.index('(')] if '(' in pat else True:
                    return {'col1': m.group(1), 'col2': m.group(2)}
                else:
                    return {'col1': m.group(2), 'col2': m.group(1)}

    # Более простой fallback: ищем первую пару word.word ↔ word.word
    simple = re.search(
        r'(\w+)\.(\w+)\s*[↔=→]+\s*(\w+)\.(\w+)',
        text,
    )
    if simple:
        left_tbl, left_col = simple.group(1), simple.group(2)
        right_tbl, right_col = simple.group(3), simple.group(4)
        t1_last = t1.split('.')[-1]
        t2_last = t2.split('.')[-1]
        if left_tbl == t1_last:
            return {'col1': left_col, 'col2': right_col}
        elif left_tbl == t2_last:
            return {'col1': right_col, 'col2': left_col}

    return None


def _check_safe(table_full: str, col_name: str, schema_loader: Any) -> bool:
    """Детерминированная проверка уникальности JOIN-ключа из CSV-метаданных."""
    parts = table_full.split('.', 1)
    if len(parts) != 2:
        return False
    try:
        result = schema_loader.check_key_uniqueness(parts[0], parts[1], [col_name])
        return bool(result.get('is_unique', False))
    except Exception:
        return False


def _infer_strategy(t1_type: str, t2_type: str, safe: bool) -> str:
    """Определить стратегию JOIN по типам таблиц."""
    if t1_type == 'fact' and t2_type in ('dim', 'ref'):
        return 'direct' if safe else 'through_dim'
    if t1_type in ('dim', 'ref') and t2_type == 'fact':
        return 'subquery'
    if t1_type == 'fact' and t2_type == 'fact':
        return 'subquery'
    return 'direct'
