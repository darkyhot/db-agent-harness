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
_JOIN_KEY_TOKEN_HINTS = frozenset({
    'inn', 'kpp', 'ogrn', 'snils', 'okato', 'oktmo', 'bik', 'date',
    'инн', 'кпп', 'огрн', 'снилс', 'окато', 'октмо', 'бик', 'дата',
})
_DIMENSION_SLOT_ALIASES: dict[str, frozenset[str]] = {
    'date': frozenset({
        'date', 'dt', 'data',
        'дата', 'дате', 'дату', 'датой', 'датам', 'датах',
    }),
    'segment_name': frozenset({
        'segment', 'segments',
        'сегмент', 'сегмента', 'сегменту', 'сегментом', 'сегменте', 'сегменты',
    }),
    'region_name': frozenset({
        'region', 'regions',
        'регион', 'региона', 'региону', 'регионом', 'регионе', 'регионы',
    }),
    'channel_name': frozenset({
        'channel', 'channels',
        'канал', 'канала', 'каналу', 'каналом', 'канале', 'каналы',
    }),
}
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


def _looks_like_join_key_token(token: str) -> bool:
    lower = token.lower().strip()
    if not lower:
        return False
    if _looks_like_explicit_column(lower):
        return True
    return lower in _JOIN_KEY_TOKEN_HINTS


def _is_label_slot(slot: str) -> bool:
    return slot.endswith(_LABEL_SUFFIXES) or slot in {'name', 'label'}


def _is_metric_slot(slot: str) -> bool:
    return slot.endswith(_METRIC_SUFFIXES) or slot.endswith(('_code', '_score'))


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
        t for t in re.findall(r'\b[a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*\b', query)
        if _looks_like_explicit_column(t)
    ]

    def _normalize_dimension_slot(raw: str) -> str | None:
        term = (raw or '').strip().strip(" .,:;!?\"'()[]{}")
        if not term:
            return None
        lower = term.lower()
        if _looks_like_explicit_column(lower):
            return lower

        parts = _tokenize(term)
        for part in parts:
            if part in _DATE_HINTS:
                return 'date'
            for slot, aliases in _DIMENSION_SLOT_ALIASES.items():
                if part in aliases:
                    return slot

        concept = _normalize_concept(term)
        if not concept:
            return None
        if any(tok in _DATE_HINTS for tok in concept.split('_')):
            return 'date'
        return f'{concept}_name'

    def _split_dimensions(chunk: str) -> list[str]:
        normalized = re.sub(r'\s+', ' ', chunk or '').strip()
        if not normalized:
            return []
        return [
            part.strip()
            for part in re.split(r'\s*(?:,|;|\s+и\s+)\s*', normalized)
            if part.strip()
        ]
    for col in explicit_cols:
        lower = col.lower()
        if lower.endswith(_LABEL_SUFFIXES) or any(tok in _DATE_HINTS for tok in lower.split('_')):
            dimensions.append(lower)

    for m in re.finditer(
        r'(?:в\s+разбивке\s+по|по)\s+([a-zA-Zа-яА-ЯёЁ0-9_\s,;]+?)(?=(?:\s+(?:в\s+разбивке\s+по|за|с|на|где|через|join|using|ключ)\b|$))',
        query,
    ):
        for part in _split_dimensions(m.group(1)):
            slot = _normalize_dimension_slot(part)
            if slot:
                dimensions.append(slot)

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
    for m in re.finditer(
        r'(?:по|через|join|using|ключ[ауеом]?)\s+(?:ключ[ауеом]?\s+)?([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)',
        query,
    ):
        token = m.group(1).lower()
        if _looks_like_join_key_token(token):
            join_key_hints.append(token)
            translit = _translit(token)
            if translit and translit != token:
                join_key_hints.append(translit)

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
) -> tuple[str, str] | None:
    """Выбрать лучший источник атрибута/метрики по имени, описанию и метаданным."""
    best: tuple[float, str, str] | None = None

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
            if require_numeric and not _is_numeric(dtype):
                continue

            desc = str(row.get('description', '') or '')
            semantic = _semantic_match_score(col_name, desc, slot)
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
            elif _is_metric_slot(slot) and t_type == 'fact':
                score += 25
            if (agg_hint or '').lower() == 'sum' and any(
                marker in lower_name for marker in ('perc', 'pct', 'rate', 'avg')
            ):
                score -= 320

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
) -> dict[str, Any]:
    """Детерминированно выбрать колонки и JOIN-спецификацию.

    Args:
        intent:            dict из intent_classifier (aggregation_hint, entities, …)
        table_structures:  dict schema.table → строка-описание (из table_explorer)
        table_types:       dict schema.table → "fact"/"dim"/"ref"/"unknown"
        join_analysis_data: dict из table_explorer (pre-computed join candidates)
        schema_loader:     SchemaLoader для доступа к колонкам и check_key_uniqueness

    Returns:
        dict с ключами: selected_columns, join_spec, confidence, reason
    """
    entities: list[str] = [str(e) for e in (intent.get('entities') or [])]
    agg_hint: str = str(intent.get('aggregation_hint') or '').lower().strip()
    date_filters: dict = intent.get('date_filters') or {}
    filter_conditions: list[dict] = intent.get('filter_conditions') or []
    requested = _derive_requested_slots(user_input, intent)

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
        if _use_count_distinct:
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
    if requested['dimensions']:
        for slot in requested['dimensions']:
            choice = _choose_best_column(
                table_structures, table_types, schema_loader, slot
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
            roles.setdefault('aggregate', [])
            if '*' not in roles['aggregate']:
                roles['aggregate'].append('*')

    if requested['dimensions']:
        allowed_refs: set[tuple[str, str]] = set()
        for slot in requested['dimensions']:
            choice = _choose_best_column(
                table_structures, table_types, schema_loader, slot
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
) -> list[dict[str, Any]]:
    """Построить join_spec из pre-computed join_analysis_data (детерминированно).

    Выбирает top-1 кандидата для каждой пары. safe определяется из CSV.
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
        cand = _pick_join_candidate(text, t1, t2, schema_loader, user_input=user_input)
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
) -> dict[str, str] | None:
    """Выбрать join-кандидат с учётом явного ключа из запроса пользователя."""
    query = _normalize_query_text(user_input)
    t1_parts = t1.split('.', 1)
    t2_parts = t2.split('.', 1)
    if query and len(t1_parts) == 2 and len(t2_parts) == 2:
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
