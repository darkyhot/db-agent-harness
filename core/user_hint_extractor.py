"""Детерминированный экстрактор явных подсказок пользователя из запроса.

Не использует LLM. Работает по регуляркам + метаданным каталога +
настраиваемой карте синонимов. Собирает:

- must_keep_tables: таблицы, упомянутые явно ("возьми в TABLE", "из таблицы TABLE"),
  которые table_resolver обязан сохранить в финальном списке.
- join_fields: имена колонок, указанных как JOIN-ключ ("по инн", "через customer_id").
- dim_sources: биндинги "измерение X возьми в TABLE по POLE" — подсказка
  column_selector, какую колонку и из какой таблицы брать для измерения.
- having_hints: постагрегатные фильтры «от N человек», «более N клиентов» →
  HAVING COUNT(DISTINCT <unit_col>) >= N в sql_planner/sql_builder.
- group_by_hints: явные оси группировки ("по task_code", "сгруппируй по региону").
- aggregate_hints: явные агрегаты ("посчитай задачи", "сумма по выручке").
- time_granularity: гранулярность времени ("помесячно" → "month", "по кварталам" → "quarter").
- negative_filters: значения для исключения ("не учитывай X", "исключи Y").

Все сущности-идентификаторы валидируются против реального каталога (schema_loader), чтобы
ни одна "мусорная" подсказка не попала в state.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

from core.log_safety import summarize_dict_keys
from core.synonym_map import SYNONYM_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Регулярки
# ---------------------------------------------------------------------------

# "возьми в <word>", "возьми из <word>", "дотяни из <word>", "в таблице <word>"
_TABLE_HINT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"возьм[иите]+\s+(?:из|в)\s+(?:таблиц[ыауе]?\s+)?"
        r"([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\.]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"дотян[иите]+\s+[^\s]*?\s*(?:из|в|по|через)\s+(?:таблиц[ыауе]?\s+)?"
        r"([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\.]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"подтян[иите]+\s+[^\s]*?\s*(?:из|в|по|через)\s+(?:таблиц[ыауе]?\s+)?"
        r"([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\.]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"в\s+таблиц[ауеы]\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\.]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"из\s+таблиц[ыауе]\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\.]*)",
        re.IGNORECASE,
    ),
]

# "по ИНН", "через customer_id", "join по X", "ключ — Y"
# Группы: 1 — connector ("по"/"через"/…), 2 — column token.
# Connector используется для различения "по X" (ambiguous: grouping vs JOIN)
# от явных JOIN-маркеров "через/join/using/ключ".
_JOIN_FIELD_PATTERN = re.compile(
    r"(?:^|\s)(по|через|join|using|ключ[ауеом]?)\s+"
    r"([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
    re.IGNORECASE,
)

# Явные маркеры JOIN-контекста: если в запросе есть один из таких глаголов,
# "по X" трактуется как JOIN-ключ, а не как группировка.
_JOIN_CONTEXT_PATTERN = re.compile(
    r"\b(соедин[иеяю]\w*|объедин[иеяю]\w*|связ[ыьа]\w*|"
    r"join|merge|подтян[иите]+|дотян[иите]+|подкле[йи]\w*)\b",
    re.IGNORECASE,
)

# "от N X", "более N X", "не менее N X", "свыше N X"
_HAVING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:от|более|свыше|не\s+менее|не\s+меньше|минимум)\s+(\d+)\s+"
        r"([a-zA-Zа-яА-ЯёЁ_]+)",
        re.IGNORECASE,
    ),
]

# Явные маркеры группировки: "сгруппируй по X", "группировка по X", "group by X"
_GROUP_BY_EXPLICIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"сгруппир[уоыёе]\w*\s+по\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"группиров\w+\s+по\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bgroup\s+by\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    ),
    # "по X" в начале предложения или после знаков препинания — почти всегда группировка
    re.compile(
        r"(?:^|[.!?,;]\s*)[Пп]о\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    ),
]

# Паттерны агрегатов (русский + английский)
_AGGREGATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # count: "посчитай X", "подсчитай X", "количество X", "число X", "count X"
    (re.compile(
        r"(?:посчита[йи]\w*|подсчита[йи]\w*|количество|число)\s+"
        r"([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    ), "count"),
    (re.compile(
        r"\bcount\s+(?:of\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    ), "count"),
    # sum: "сумма X", "суммируй X", "сумму по X", "sum of X", "sum X"
    (re.compile(
        r"(?:сумм[аыуе]\w*|суммиру\w+)\s+(?:по\s+)?([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    ), "sum"),
    (re.compile(
        r"\bsum\s+(?:of\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    ), "sum"),
    # avg: "среднее X", "средний X", "средняя X", "avg X", "average X"
    (re.compile(
        r"средн[еийяяёая]\w*\s+(?:по\s+)?([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    ), "avg"),
    (re.compile(
        r"\b(?:avg|average)\s+(?:of\s+)?([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    ), "avg"),
]

# Нормализация гранулярности времени.
# Ключ — каноническое значение; список — паттерны (подстроки/regex).
_TIME_GRANULARITY_MAP: dict[str, list[str]] = {
    "day":     [
        r"ежедневн\w*", r"по\s+дням", r"\bdaily\b", r"по\s+дню",
        r"\bby\s+day\b", r"подённо\w*",
    ],
    "week":    [
        r"еженедельн\w*", r"по\s+неделям", r"\bweekly\b", r"по\s+неделе",
        r"\bby\s+week\b", r"понедельн\w*",
    ],
    "month":   [
        r"помесячн\w*", r"по\s+месяц\w+", r"\bmonthly\b",
        r"ежемесячн\w*", r"\bby\s+month\b",
    ],
    "quarter": [
        r"поквартальн\w*", r"по\s+квартал\w+", r"\bquarterly\b",
        r"\bby\s+quarter\b",
    ],
    "year":    [
        r"ежегодн\w*", r"по\s+год\w+", r"\byearly\b", r"\bannually\b",
        r"\bby\s+year\b",
    ],
}

# Компилированные паттерны для гранулярности (строится один раз из _TIME_GRANULARITY_MAP)
_TIME_GRANULARITY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, re.IGNORECASE), granularity)
    for granularity, patterns in _TIME_GRANULARITY_MAP.items()
    for pat in patterns
]

# Паттерны негативных фильтров
_NEGATIVE_FILTER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"не\s+учитыва[йи]\w*\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\s]*?)(?:[,;.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"исключ[иите]\w*\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\s]*?)(?:[,;.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"без\s+учёта\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\s]*?)(?:[,;.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"кроме\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\s]*?)(?:[,;.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexclud(?:e|ing)\s+([a-zA-Z_][a-zA-Z0-9_\s]*?)(?:[,;.!?]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bexcept\s+(?:for\s+)?([a-zA-Z_][a-zA-Z0-9_\s]*?)(?:[,;.!?]|$)",
        re.IGNORECASE,
    ),
]

# "<концепт> возьми (в|из) <таблица> (по <поле>)?"
_DIM_SOURCE_PATTERN = re.compile(
    r"([a-zA-Zа-яА-ЯёЁ_]{3,})\s+возьм[иите]+\s+(?:из|в)\s+"
    r"(?:таблиц[ыауе]?\s+)?([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_\.]*)"
    r"(?:\s+по\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*))?",
    re.IGNORECASE,
)

# schema.table — явная полностью квалифицированная ссылка
_SCHEMA_TABLE_PATTERN = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b"
)

# служебные слова, которые НЕ могут быть именем JOIN-колонки
_STOP_WORDS: frozenset[str] = frozenset({
    "инн", "кпп", "огрн", "бик",  # бизнес-ключи переживут маппинг ниже
    # системные/естественные русские предлоги и частицы
    "и", "или", "но", "как", "этой", "этот", "тот", "из", "в", "на", "за", "по",
    # английские стоп-слова
    "and", "or", "the", "of",
})


# ---------------------------------------------------------------------------
# Вспомогательные
# ---------------------------------------------------------------------------

_TRANSLIT_TABLE = str.maketrans({
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
})


# Русский бизнес-ключ → возможные английские имена JOIN-колонок.
# Используется, когда пользователь пишет "по инн" — в каталоге колонка называется "inn".
_KEY_SYNONYMS: dict[str, list[str]] = {
    "инн": ["inn"],
    "кпп": ["kpp"],
    "огрн": ["ogrn"],
    "огрнип": ["ogrnip"],
    "бик": ["bik"],
    "снилс": ["snils"],
    "дата": ["date", "dt", "report_dt"],
    "сотрудник": ["employee", "emp", "staff"],
    "клиент": ["client", "customer"],
}


def _translit(text: str) -> str:
    return text.lower().translate(_TRANSLIT_TABLE)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _expand_slot_synonyms(slot: str) -> list[str]:
    """Расширить русский термин его английскими синонимами.

    «сегмент» → ["segment", "seg"]; «клиент» → ["client", "customer", ...].
    Если термин уже английский — возвращаем его как есть + транслитерацию.
    """
    slot_lower = slot.lower().strip()
    variants: list[str] = [slot_lower]
    if slot_lower in SYNONYM_MAP:
        variants.extend(SYNONYM_MAP[slot_lower])
    translit = _translit(slot_lower)
    if translit and translit != slot_lower:
        variants.append(translit)
    return list(dict.fromkeys(v for v in variants if v))


# ---------------------------------------------------------------------------
# Валидация подсказок через каталог
# ---------------------------------------------------------------------------

def _resolve_table_hint(
    token: str,
    schema_loader: Any,
) -> tuple[str, str] | None:
    """Сопоставить упоминание таблицы (возможно без схемы) с реальной таблицей.

    Приоритет: точное schema.table → точное table_name → подстрока.
    """
    token_norm = token.strip().lower()
    if not token_norm:
        return None

    tables_df = schema_loader.tables_df
    if tables_df.empty:
        return None

    # schema.table
    if "." in token_norm:
        parts = token_norm.split(".", 1)
        if len(parts) == 2:
            s_mask = tables_df["schema_name"].str.lower() == parts[0]
            t_mask = tables_df["table_name"].str.lower() == parts[1]
            match = tables_df[s_mask & t_mask]
            if not match.empty:
                row = match.iloc[0]
                return (str(row["schema_name"]), str(row["table_name"]))

    # Точное имя таблицы
    exact = tables_df[tables_df["table_name"].str.lower() == token_norm]
    if not exact.empty:
        row = exact.iloc[0]
        return (str(row["schema_name"]), str(row["table_name"]))

    # Подстрока: только если токен достаточно длинный и уникален
    if len(token_norm) >= 5:
        contains = tables_df[
            tables_df["table_name"].str.lower().str.contains(
                re.escape(token_norm), regex=True
            )
        ]
        if len(contains) == 1:
            row = contains.iloc[0]
            return (str(row["schema_name"]), str(row["table_name"]))

    return None


def _resolve_join_field(
    token: str,
    schema_loader: Any,
) -> list[str]:
    """Найти в каталоге реальные имена колонок, соответствующие токену.

    «инн» → ["inn"] (через KEY_SYNONYMS + проверка в attr_list).
    «customer_id» → ["customer_id"] если существует в ≥2 таблицах.
    """
    token_norm = token.strip().lower()
    if not token_norm:
        return []
    # Бизнес-ключи (инн, кпп…) записаны в _STOP_WORDS, но имеют явные синонимы
    # в _KEY_SYNONYMS — такие токены пропускать НЕЛЬЗЯ.
    if token_norm in _STOP_WORDS and token_norm not in _KEY_SYNONYMS:
        return []

    candidates: list[str] = []
    # Русский бизнес-термин → английские варианты
    if token_norm in _KEY_SYNONYMS:
        candidates.extend(_KEY_SYNONYMS[token_norm])
    else:
        candidates.append(token_norm)
        translit = _translit(token_norm)
        if translit and translit != token_norm:
            candidates.append(translit)

    attrs_df = schema_loader.attrs_df
    if attrs_df.empty:
        return []

    resolved: list[str] = []
    for candidate in dict.fromkeys(candidates):
        mask = attrs_df["column_name"].str.lower() == candidate
        hits = attrs_df[mask][["schema_name", "table_name"]].drop_duplicates()
        # JOIN-поле имеет смысл только если есть в двух+ таблицах.
        # Для композитных ключей иногда хватает одной, но там это обрабатывается
        # отдельной логикой; здесь мы фильтруем откровенный мусор.
        if len(hits) >= 1:
            resolved.append(candidate)
    return list(dict.fromkeys(resolved))


# ---------------------------------------------------------------------------
# Публичная функция
# ---------------------------------------------------------------------------

def _column_exists_in_catalog(column: str, schema_loader: Any) -> bool:
    """Проверить, есть ли колонка с таким именем хотя бы в одной таблице каталога."""
    if schema_loader is None:
        return False
    attrs_df = schema_loader.attrs_df
    if attrs_df.empty:
        return False
    return bool((attrs_df["column_name"].str.lower() == column.lower()).any())


def extract_user_hints(
    user_input: str,
    schema_loader: Any,
) -> dict[str, Any]:
    """Извлечь структуру UserHints из запроса пользователя.

    Args:
        user_input: Исходный текст запроса.
        schema_loader: SchemaLoader с загруженным каталогом.

    Returns:
        {
            "must_keep_tables": [(schema, table), ...],
            "join_fields": ["inn", "customer_id", ...],
            "dim_sources": {"segment": {"table": "schema.t", "join_col": "inn"}},
            "having_hints": [{"op": ">=", "value": 3, "unit_hint": "человек"}],
            "group_by_hints": ["task_code", "region", ...],
            "aggregate_hints": [("count", "task"), ("sum", "revenue"), ...],
            "time_granularity": "month" | "quarter" | "year" | "day" | "week" | None,
            "negative_filters": ["канцелярия", ...],
        }
    """
    result: dict[str, Any] = {
        "must_keep_tables": [],
        "join_fields": [],
        "dim_sources": {},
        "having_hints": [],
        "group_by_hints": [],
        "aggregate_hints": [],
        "time_granularity": None,
        "negative_filters": [],
    }

    if not user_input or schema_loader is None:
        return result

    query = _normalize(user_input)
    must_keep: list[tuple[str, str]] = []

    # 1. Явные schema.table (точное совпадение)
    for m in _SCHEMA_TABLE_PATTERN.finditer(user_input):
        resolved = _resolve_table_hint(f"{m.group(1)}.{m.group(2)}", schema_loader)
        if resolved and resolved not in must_keep:
            must_keep.append(resolved)

    # 2. "возьми в TABLE" / "дотяни из TABLE" / "в таблице TABLE"
    for pattern in _TABLE_HINT_PATTERNS:
        for m in pattern.finditer(query):
            token = m.group(1)
            resolved = _resolve_table_hint(token, schema_loader)
            if resolved and resolved not in must_keep:
                must_keep.append(resolved)

    result["must_keep_tables"] = must_keep

    # 3. JOIN-поля: "по <X>", "через <X>", "ключ <X>"
    # Важно: "по X" в аналитическом запросе («посчитай по сегменту») — это
    # НЕ JOIN-hint, а ось группировки. Поэтому "по X" принимаем как JOIN-hint
    # ТОЛЬКО если:
    #   - X — известный бизнес-ключ (_KEY_SYNONYMS: инн, кпп, снилс, …), или
    #   - в запросе есть JOIN-контекст (соедини, объедини, связать, join, …).
    # Токены после "через/join/using/ключ" принимаем всегда — там нет двусмысленности.
    join_context_present = bool(_JOIN_CONTEXT_PATTERN.search(query))
    join_fields: list[str] = []
    for m in _JOIN_FIELD_PATTERN.finditer(query):
        connector = m.group(1).lower()
        token = m.group(2).lower()
        if token in _STOP_WORDS and token not in _KEY_SYNONYMS:
            continue
        # "по X" — принимаем только если X бизнес-ключ или есть JOIN-контекст
        if connector == "по":
            is_business_key = token in _KEY_SYNONYMS
            if not (is_business_key or join_context_present):
                continue
        for resolved in _resolve_join_field(token, schema_loader):
            if resolved not in join_fields:
                join_fields.append(resolved)
    result["join_fields"] = join_fields

    # 4. Dim-sources: "<slot> возьми (в|из) TABLE (по POLE)?"
    dim_sources: dict[str, dict[str, str]] = {}
    for m in _DIM_SOURCE_PATTERN.finditer(query):
        slot_token = m.group(1).lower()
        table_token = m.group(2)
        pole_token = m.group(3)
        if slot_token in _STOP_WORDS:
            continue
        resolved_table = _resolve_table_hint(table_token, schema_loader)
        if not resolved_table:
            continue
        slot_key = _normalize_slot_key(slot_token)
        entry: dict[str, str] = {
            "table": f"{resolved_table[0]}.{resolved_table[1]}",
        }
        if pole_token:
            resolved_fields = _resolve_join_field(pole_token, schema_loader)
            if resolved_fields:
                entry["join_col"] = resolved_fields[0]
                # Дополнительно регистрируем как общий join_fields
                for f in resolved_fields:
                    if f not in result["join_fields"]:
                        result["join_fields"].append(f)
        dim_sources[slot_key] = entry
    result["dim_sources"] = dim_sources

    # 5. HAVING-hints: "от N <слово>", "более N <слово>"
    having_hints: list[dict[str, Any]] = []
    for pattern in _HAVING_PATTERNS:
        for m in pattern.finditer(query):
            try:
                n = int(m.group(1))
            except (ValueError, TypeError):
                continue
            unit = m.group(2).lower() if m.lastindex and m.lastindex >= 2 else ""
            having_hints.append({
                "op": ">=",
                "value": n,
                "unit_hint": unit,
            })
    result["having_hints"] = having_hints

    # 6. Explicit group-by markers: "сгруппируй по X", "group by X", "По X" (после точки)
    group_by_hints: list[str] = []
    for pattern in _GROUP_BY_EXPLICIT_PATTERNS:
        for m in pattern.finditer(user_input):
            token = m.group(1).lower().strip()
            if token in _STOP_WORDS:
                continue
            if not _column_exists_in_catalog(token, schema_loader):
                logger.debug("group_by_hints: колонка '%s' не найдена в каталоге, пропускаем", token)
                continue
            if token not in group_by_hints:
                group_by_hints.append(token)

    result["group_by_hints"] = group_by_hints

    # 7. Aggregate hints: "посчитай X", "сумма по X", etc.
    aggregate_hints: list[tuple[str, str]] = []
    for pattern, agg_func in _AGGREGATE_PATTERNS:
        for m in pattern.finditer(user_input):
            unit = m.group(1).strip().lower()
            if not unit or unit in _STOP_WORDS:
                continue
            pair = (agg_func, unit)
            if pair not in aggregate_hints:
                aggregate_hints.append(pair)
    result["aggregate_hints"] = aggregate_hints

    # 8. Time granularity: "помесячно" → "month", "по кварталам" → "quarter"
    time_granularity: str | None = None
    for pattern, granularity in _TIME_GRANULARITY_PATTERNS:
        if pattern.search(user_input):
            time_granularity = granularity
            break
    result["time_granularity"] = time_granularity

    # 9. Negative filters: "не учитывай X", "исключи Y"
    negative_filters: list[str] = []
    for pattern in _NEGATIVE_FILTER_PATTERNS:
        for m in pattern.finditer(user_input):
            value = m.group(1).strip().lower()
            value = re.sub(r"\s+", " ", value).strip()
            if value and value not in negative_filters:
                negative_filters.append(value)
    result["negative_filters"] = negative_filters

    logger.info(
        "UserHintExtractor: must_keep=%d, join_fields=%d, dim_sources=%s, "
        "having_hints=%d, group_by=%d, aggregate=%d, time_granularity=%s, "
        "negative_filters=%d",
        len(must_keep),
        len(join_fields),
        summarize_dict_keys(dim_sources, label="dim_sources"),
        len(having_hints),
        len(group_by_hints),
        len(aggregate_hints),
        time_granularity,
        len(negative_filters),
    )

    return result


def _normalize_slot_key(token: str) -> str:
    """Привести slot-токен к каноническому ключу через SYNONYM_MAP."""
    token_norm = token.lower().strip()
    # Если термин уже английский — берём первый синоним как представителя.
    variants = _expand_slot_synonyms(token_norm)
    # Канонический ключ — первый английский вариант (если есть), иначе токен.
    for v in variants:
        if re.match(r"^[a-z_][a-z0-9_]*$", v):
            return v
    return token_norm


# ---------------------------------------------------------------------------
# Вспомогательный API для column_selector
# ---------------------------------------------------------------------------

def match_unit_column(
    unit_hint: str,
    table_key: str,
    schema_loader: Any,
) -> str | None:
    """Подобрать имя колонки для HAVING COUNT(DISTINCT ...) по слову-unit.

    «человек» → первая найденная sensible employee-колонка в указанной таблице.
    Используется sql_planner при построении HAVING.
    """
    if not unit_hint or not table_key or schema_loader is None:
        return None

    parts = table_key.split(".", 1)
    if len(parts) != 2:
        return None

    cols_df = schema_loader.get_table_columns(parts[0], parts[1])
    if cols_df.empty:
        return None

    variants = _expand_slot_synonyms(unit_hint)
    # Любая колонка, чьё имя или описание содержит один из вариантов и похожа на ключ.
    best: tuple[float, str] | None = None
    for _, row in cols_df.iterrows():
        name = str(row.get("column_name", "")).lower()
        desc = str(row.get("description", "") or "").lower()
        score = 0.0
        for v in variants:
            if not v:
                continue
            if v in name:
                score += 1.0
            if v in desc:
                score += 0.4
        if score <= 0:
            continue
        # Предпочтительно — ключеподобное имя
        if name.endswith(("_id", "_code", "inn", "_num")):
            score += 0.6
        is_pk = bool(row.get("is_primary_key", False))
        if is_pk:
            score += 0.3
        if best is None or score > best[0]:
            best = (score, str(row.get("column_name", "")))

    return best[1] if best else None


def iter_join_field_candidates(hints: dict[str, Any]) -> Iterable[str]:
    """Итератор по join_fields из UserHints."""
    for field in hints.get("join_fields", []) or []:
        yield field
