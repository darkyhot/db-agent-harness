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


def _is_numeric(dtype: str) -> bool:
    return bool(_NUMERIC_RE.match(dtype.strip()))


def _is_date(dtype: str) -> bool:
    return bool(_DATE_RE.match(dtype.strip()))


def _is_categorical(dtype: str) -> bool:
    return bool(_CATEGORICAL_RE.match(dtype.strip()))


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

    has_date_filter = bool(date_filters.get('from') or date_filters.get('to'))
    wants_aggregation = agg_hint not in ('', 'null', 'list', 'None')

    selected_columns: dict[str, dict] = {}
    per_table_confidence: list[float] = []

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
                    agg_candidates.append((col_name, round(s, 3)))

            # ---- group_by score ----
            # Не кладём PK и числовые метрики в group_by если они не упомянуты в entities
            gb_s = 0.0
            if not is_pk:
                entity_hit = _entity_score(col_name, desc, entities)
                if _is_categorical(dtype):
                    gb_s += 0.25
                if unique_perc > 0 and unique_perc < 30:
                    gb_s += 0.20
                gb_s += entity_hit * 0.55
                # Числовые — только если явно упомянуты в entities
                if _is_numeric(dtype) and entity_hit < 0.3:
                    gb_s = 0.0
                # Дата — слабый кандидат для group_by (обычно это фильтр)
                if _is_date(dtype):
                    gb_s = min(gb_s, 0.2)
            if gb_s > 0:
                gb_candidates.append((col_name, round(gb_s, 3)))

            # ---- filter score ----
            flt_s = 0.0
            if _is_date(dtype) and has_date_filter:
                flt_s = 0.90
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
    join_spec = _build_join_spec(join_analysis_data, selected_columns, schema_loader, table_types)

    # ---- Итоговая confidence ----
    if not selected_columns:
        overall = 0.0
        reason = 'Не удалось выбрать колонки ни для одной таблицы'
    elif not per_table_confidence:
        overall = 0.0
        reason = 'Нет данных о колонках'
    else:
        overall = sum(per_table_confidence) / len(per_table_confidence)
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
        cand = _parse_top_candidate(text, t1, t2)
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

        join_spec.append(entry)

        # Если dim-таблица имеет составной PK — добавляем пары для недостающих колонок
        extra = _complete_composite_join(entry, t1, t2, table_types, schema_loader)
        join_spec.extend(extra)

    return join_spec


def _parse_top_candidate(text: str, t1: str, t2: str) -> dict[str, str] | None:
    """Извлечь первый JOIN-кандидат из текста join_analysis.

    Форматы, которые понимает парсер:
      «t1.col1 ↔ t2.col2»
      «t1.col1 = t2.col2»
      «Кандидат #1 ... t1.col1 → t2.col2»
    """
    if not text:
        return None

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
