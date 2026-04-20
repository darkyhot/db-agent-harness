"""Анализ JOIN-кандидатов: классификация колонок, scoring, ranking, composite keys."""

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# 0. Причины решения (decision_reason) — для агрегации метрик качества.
# `JoinCandidate.match_type` гарантированно принимает одно из этих значений.
# ---------------------------------------------------------------------------

class DecisionReason:
    """Enum-подобный контейнер для `JoinCandidate.match_type`."""

    EXPLICIT_FK = "explicit_fk"
    EXACT_NAME = "exact_name"
    FK_PATTERN = "fk_pattern"
    NORMALIZED_PK = "normalized_pk"
    SUFFIX = "suffix"


DECISION_REASONS: frozenset[str] = frozenset({
    DecisionReason.EXPLICIT_FK,
    DecisionReason.EXACT_NAME,
    DecisionReason.FK_PATTERN,
    DecisionReason.NORMALIZED_PK,
    DecisionReason.SUFFIX,
})


# ---------------------------------------------------------------------------
# 1. Классификация колонок: key / business_key / attribute
# ---------------------------------------------------------------------------

# Паттерны для определения attribute-колонок (НЕ ключи)
_ATTRIBUTE_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(^|_)(name|desc|comment|note|text|fio|login)($|_)", re.I),
    re.compile(r"(^|_)(status|state|flag)($|_)", re.I),
    re.compile(r"^(is|has)_", re.I),
    re.compile(r"_(dttm|timestamp)$", re.I),
    re.compile(r"(^|_)(created|updated|inserted|modified|deleted)(_|$)", re.I),
    re.compile(r"_(perc|pct|amt|qty|cnt|sum|avg|val)$", re.I),
    re.compile(r"(^|_)(salary|amount|total|count|percent|rate)($|_)", re.I),
]

# Паттерны для определения key-колонок
_KEY_SUFFIX_PATTERNS = {"_id", "_code", "_num", "_no"}

# Известные бизнес-ключи (точное совпадение имени)
_BUSINESS_KEY_NAMES = {
    "inn", "kpp", "ogrn", "bik", "snils", "okpo", "oktmo",
    "saphr_id", "tab_num",
}

# Префиксы для нормализации имён PK-колонок при поиске FK-пар в фактовых таблицах.
# Позволяет находить соответствия вида old_gosb_id ↔ gosb_id, new_client_id ↔ client_id.
_PK_NAME_PREFIXES = re.compile(r"^(old|new|prev|cur|current|actual|base|src|tgt)_", re.I)


def _normalize_key_name(name: str) -> str:
    """Нормализовать имя ключевой колонки для сравнения с FK-колонками.

    Удаляет временные/версионные префиксы: old_, new_, prev_, cur_ и т.д.
    Примеры: old_gosb_id → gosb_id, new_client_id → client_id.
    Колонки без таких префиксов возвращаются без изменений.
    """
    return _PK_NAME_PREFIXES.sub("", name.lower())


# Типы данных, характерные для атрибутов
_ATTRIBUTE_DTYPES = re.compile(
    r"^(text|boolean|timestamp|json|jsonb|xml|bytea)", re.I,
)
_LONG_VARCHAR = re.compile(r"varchar\((\d+)\)", re.I)


def classify_column(
    name: str,
    dtype: str,
    unique_perc: float,
    is_pk: bool,
    description: str = "",
) -> str:
    """Классифицировать колонку: ``'key'``, ``'business_key'`` или ``'attribute'``.

    Используется для фильтрации шума в JOIN-анализе.
    """
    lower = name.lower()

    # --- Определённо key ---
    if is_pk:
        return "key"
    if any(lower.endswith(suf) for suf in _KEY_SUFFIX_PATTERNS):
        # _id/_code/_num — key, если НЕ подпадает под attribute-паттерны
        if not _is_attribute_name(lower):
            return "key"

    # --- Определённо attribute ---
    if _is_attribute_name(lower):
        return "attribute"
    if _ATTRIBUTE_DTYPES.match(dtype):
        return "attribute"
    m = _LONG_VARCHAR.search(dtype)
    if m and int(m.group(1)) > 100:
        return "attribute"

    # --- Бизнес-ключ ---
    if lower in _BUSINESS_KEY_NAMES:
        return "business_key"

    # --- Неопределённость → решаем по unique_perc ---
    if unique_perc >= 50:
        return "business_key"
    return "attribute"


def _is_attribute_name(lower_name: str) -> bool:
    """Проверить имя на соответствие attribute-паттернам."""
    return any(p.search(lower_name) for p in _ATTRIBUTE_NAME_PATTERNS)


# ---------------------------------------------------------------------------
# 2. Определение типа таблицы
# ---------------------------------------------------------------------------

def detect_table_type(table_name: str, cols_df: pd.DataFrame) -> str:
    """Определить тип таблицы: ``'fact'``, ``'dim'``, ``'ref'`` или ``'unknown'``."""
    t = table_name.lower()
    if re.search(r"(^|_)fact($|_)", t):
        return "fact"
    if re.search(r"(^|_)dim($|_)", t):
        return "dim"
    if re.search(r"(^|_)(ref|dict|lookup|directory)($|_)", t):
        return "ref"
    if t.endswith("_m") or t.endswith("_funnel"):
        return "fact"
    # Эвристика по структуре колонок
    if not cols_df.empty and "is_primary_key" in cols_df.columns:
        pk_ratio = cols_df["is_primary_key"].astype(bool).mean()
        if pk_ratio > 0.3:
            return "dim"
        col_names = {str(c).lower() for c in cols_df.get("column_name", [])}
        if {"report_dt", "amt"} & col_names:
            return "fact"
        if any(c.endswith("_qty") or c.endswith("_amt") or c.endswith("_perc") for c in col_names):
            return "fact"
    return "unknown"


# ---------------------------------------------------------------------------
# 3. Scoring и ranking join-кандидатов
# ---------------------------------------------------------------------------

@dataclass
class JoinCandidate:
    """Кандидат на JOIN между двумя таблицами."""

    table1: str           # schema.table
    col1: str
    col1_class: str       # key / business_key / attribute

    table2: str           # schema.table
    col2: str
    col2_class: str

    score: float          # 0..1
    match_type: str       # exact_name / fk_pattern / suffix
    reason: str           # человекочитаемое объяснение

    # Безопасность
    safe: bool = True
    risk_detail: str = ""

    # Уникальность сторон
    col1_status: str = ""
    col2_status: str = ""

    # Описания колонок — помогают LLM выбрать семантически верный ключ
    col1_desc: str = ""
    col2_desc: str = ""

    # Оценка кардинальности пары: "1:1", "1:N", "N:1", "N:M", "unknown".
    # Используется sql_planner для принятия решения о необходимости CTE-предагрегации.
    cardinality_rating: str = "unknown"


# Минимальный score для включения в вывод
MIN_CANDIDATE_SCORE = 0.3


def _compute_score(
    col1_class: str, col2_class: str,
    match_type: str,
    u1: float, u2: float,
    is_pk1: bool, is_pk2: bool,
    raw_pk1: bool = False, raw_pk2: bool = False,
    pk_count1: int = 1, pk_count2: int = 1,
) -> float:
    """Вычислить score кандидата на JOIN.

    raw_pk1/raw_pk2 — реальная PK-пометка (до composite-коррекции).
    pk_count1/pk_count2 — сколько PK-колонок в таблице (для composite-штрафа).
    """
    # Обе стороны attribute → шум
    if col1_class == "attribute" and col2_class == "attribute":
        return 0.0

    base = 0.0

    # Match type bonus
    if match_type == "explicit_fk":
        base = 0.9  # Метаданные FK — максимум доверия
    elif match_type == "exact_name":
        base = 0.5
    elif match_type in ("fk_pattern", "normalized_pk"):
        base = 0.55
    elif match_type == "suffix":
        base = 0.35

    # Key classification bonus
    key_bonus = 0.0
    if col1_class == "key" and col2_class == "key":
        key_bonus = 0.45
    elif col1_class == "key" and col2_class == "business_key":
        key_bonus = 0.35
    elif col1_class == "business_key" and col2_class == "key":
        key_bonus = 0.35
    elif col1_class == "business_key" and col2_class == "business_key":
        key_bonus = 0.3
    elif col1_class == "key" or col2_class == "key":
        key_bonus = 0.15
    elif col1_class == "business_key" or col2_class == "business_key":
        key_bonus = 0.1

    # PK bonus — только если PK одиночный (composite не гарантирует уникальность)
    pk_bonus = 0.0
    if (is_pk1 and pk_count1 == 1) or (is_pk2 and pk_count2 == 1):
        pk_bonus = 0.05

    score = min(base + key_bonus + pk_bonus, 1.0)

    # explicit_fk — метаданные прямо говорят о FK-связи.
    # Heuristic-штрафы (attribute, non-PK в fact) здесь неприменимы — метаданные
    # побеждают эвристику. Только penalty за неуникальность dim-стороны остаётся.
    if match_type == "explicit_fk":
        # Только penalty за неуникальность PK-стороны (если dim.pk реально не уникален).
        if raw_pk1 and u1 < 100:
            score *= max(u1 / 100.0, 0.1)
        elif raw_pk2 and u2 < 100:
            score *= max(u2 / 100.0, 0.1)
        return round(score, 2)

    # Штраф за одну attribute-сторону
    if col1_class == "attribute" or col2_class == "attribute":
        score *= 0.4

    # Штраф за non-PK колонку при наличии составного PK в таблице.
    # Если dim имеет составной PK, а join идёт по не-PK колонке — это семантически
    # неверный ключ (например, new_gosb_id вместо tb_id+old_gosb_id).
    # ВАЖНО: штраф НЕ применяется, если другая сторона является PK — это
    # легитимная пара PK(dim)↔FK(fact), где FK не входит в PK fact-таблицы.
    if not raw_pk2 and pk_count2 > 1 and not raw_pk1:
        score *= 0.2  # жёсткий штраф: non-PK в composite-PK таблице (не FK к другому PK)
    elif not raw_pk1 and pk_count1 > 1 and not raw_pk2:
        score *= 0.2
    elif not raw_pk2 and pk_count2 > 0 and u2 < 100:
        # Обычный штраф за non-PK при одиночном PK
        score *= 0.6
    elif not raw_pk1 and not raw_pk2 and pk_count1 > 0 and u1 < 100:
        # Штраф за non-PK только когда обе стороны не являются PK.
        # Если raw_pk2=True — сторона 1 является легитимным FK и штраф не нужен.
        score *= 0.6

    # Штраф за неуникальность.
    # Ключевые соображения:
    # 1. Для члена составного PK: individual unique_perc занижен по природе составного
    #    ключа — используем floor 50% чтобы не обнулять хорошие кандидаты.
    # 2. Для FK-стороны (fact): low unique_perc нормален и ожидаем — в факт-таблице
    #    много строк на один ключ. Релевантна только unique_perc PK-стороны (dim).
    if raw_pk2:
        u_for_penalty = max(u2, 50.0) if pk_count2 > 1 else u2
    elif raw_pk1:
        u_for_penalty = max(u1, 50.0) if pk_count1 > 1 else u1
    else:
        u_for_penalty = min(u1, u2)

    if u_for_penalty < 100:
        score *= max(u_for_penalty / 100.0, 0.1)  # floor 0.1 чтобы не обнулять

    return round(score, 2)


def _col_info(row: Any, pk_count: int) -> dict:
    """Извлечь информацию о колонке из строки DataFrame."""
    is_pk = bool(row.get("is_primary_key", False))
    u = float(row.get("unique_perc", 0)) if pd.notna(row.get("unique_perc")) else 0.0
    dtype = str(row.get("dType", row.get("data_type", "")))
    desc = str(row.get("description", row.get("column_description", "")))
    name = str(row.get("column_name", ""))

    # Composite PK detection: член составного PK НИКОГДА не уникален сам по себе.
    # Уникальность гарантируется только комбинацией всех PK-колонок.
    # Безопасность одиночной колонки определяется только через unique_perc.
    effective_pk = is_pk
    if is_pk and pk_count > 1:
        effective_pk = False  # Часть составного PK — не уникальна сама по себе

    col_class = classify_column(name, dtype, u, effective_pk, desc)

    # Status string
    if is_pk and pk_count > 1:
        status = f"composite-PK unique={u:.0f}%"
    elif is_pk:
        status = "PK"
    elif u >= 100:
        status = f"unique={u:.0f}%"
    else:
        status = f"ДУБЛИ unique={u:.0f}%"

    return {
        "name": name,
        "dtype": dtype,
        "unique_perc": u,
        "is_pk": effective_pk,
        "raw_pk": is_pk,
        "pk_count": pk_count,
        "description": desc,
        "col_class": col_class,
        "status": status,
        "desc_short": desc[:60].strip() if desc else "",
    }


def _make_candidate(
    s1: str, t1: str, info1: dict,
    s2: str, t2: str, info2: dict,
    match_type: str,
) -> JoinCandidate | None:
    """Создать JoinCandidate, вернуть None если score ниже порога."""
    score = _compute_score(
        info1["col_class"], info2["col_class"],
        match_type,
        info1["unique_perc"], info2["unique_perc"],
        info1["is_pk"], info2["is_pk"],
        raw_pk1=info1["raw_pk"], raw_pk2=info2["raw_pk"],
        pk_count1=info1["pk_count"], pk_count2=info2["pk_count"],
    )
    if score < MIN_CANDIDATE_SCORE:
        return None

    # Безопасность
    side1_safe = info1["is_pk"] or info1["unique_perc"] >= 100
    side2_safe = info2["is_pk"] or info2["unique_perc"] >= 100
    safe = side1_safe and side2_safe

    risk_detail = ""
    if not safe:
        problems = []
        if not side1_safe:
            problems.append(f"{info1['name']} ({info1['status']})")
        if not side2_safe:
            problems.append(f"{info2['name']} ({info2['status']})")
        risk_detail = "ДУБЛИ: " + ", ".join(problems)

    # Reason
    reason_parts = {
        "explicit_fk": "метаданные FK (foreign_key_target)",
        "exact_name": "одинаковое имя",
        "fk_pattern": "FK-паттерн",
        "suffix": "suffix-паттерн",
        "normalized_pk": "нормализованное PK-имя (old_/new_-префикс)",
    }
    reason = reason_parts.get(match_type, match_type)

    # Оценка кардинальности по unique_perc каждой стороны:
    # 100% ↔ "1" (уникальная сторона), < 100% ↔ "N".
    def _side(unique: float, raw_pk: bool, pk_count: int) -> str:
        if raw_pk and pk_count == 1:
            return "1"
        if unique >= 99.5:
            return "1"
        return "N"

    left_side = _side(info1["unique_perc"], info1["raw_pk"], info1["pk_count"])
    right_side = _side(info2["unique_perc"], info2["raw_pk"], info2["pk_count"])
    cardinality_rating = f"{left_side}:{right_side}"

    return JoinCandidate(
        table1=f"{s1}.{t1}",
        col1=info1["name"],
        col1_class=info1["col_class"],
        table2=f"{s2}.{t2}",
        col2=info2["name"],
        col2_class=info2["col_class"],
        score=score,
        match_type=match_type,
        reason=reason,
        safe=safe,
        risk_detail=risk_detail,
        col1_status=info1["status"],
        col2_status=info2["status"],
        col1_desc=info1.get("desc_short", ""),
        col2_desc=info2.get("desc_short", ""),
        cardinality_rating=cardinality_rating,
    )


def rank_join_candidates(
    s1: str, t1: str, cols1_df: pd.DataFrame,
    s2: str, t2: str, cols2_df: pd.DataFrame,
    pk_count1: int, pk_count2: int,
) -> list[JoinCandidate]:
    """Найти и ранжировать потенциальные join-ключи между двумя таблицами.

    Возвращает список кандидатов с score >= MIN_CANDIDATE_SCORE, отсортированный
    по убыванию score.

    Если у одной из колонок в `cols*_df` заполнен `foreign_key_target` (формат
    `schema.table.column`) — соответствующая пара добавляется как 0-й этап
    `explicit_fk` с наивысшим базовым score.
    """
    if cols1_df.empty or cols2_df.empty:
        return []

    # Построить info-кэши
    info1_map: dict[str, dict] = {}
    for _, r in cols1_df.iterrows():
        name = str(r["column_name"])
        info1_map[name] = _col_info(r, pk_count1)

    info2_map: dict[str, dict] = {}
    for _, r in cols2_df.iterrows():
        name = str(r["column_name"])
        info2_map[name] = _col_info(r, pk_count2)

    # Собрать explicit FK-пары: колонка с foreign_key_target="s2.t2.col" → пара (col, target)
    def _collect_fk_pairs(
        src_s: str, src_t: str, src_df: pd.DataFrame,
        dst_s: str, dst_t: str,
    ) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        if "foreign_key_target" not in src_df.columns:
            return pairs
        for _, r in src_df.iterrows():
            raw = str(r.get("foreign_key_target", "") or "").strip()
            if not raw:
                continue
            parts = raw.split(".")
            if len(parts) != 3:
                continue
            tgt_schema, tgt_table, tgt_col = (p.strip() for p in parts)
            if tgt_schema == dst_s and tgt_table == dst_t:
                pairs.append((str(r["column_name"]), tgt_col))
        return pairs

    fk_pairs_1to2 = _collect_fk_pairs(s1, t1, cols1_df, s2, t2)
    fk_pairs_2to1 = _collect_fk_pairs(s2, t2, cols2_df, s1, t1)

    candidates: list[JoinCandidate] = []
    seen: set[tuple[str, str]] = set()

    def _add(c1: str, c2: str, match_type: str) -> None:
        key = (c1, c2) if c1 <= c2 else (c2, c1)
        if key in seen:
            return
        i1 = info1_map.get(c1)
        i2 = info2_map.get(c2)
        if not i1 or not i2:
            return
        cand = _make_candidate(s1, t1, i1, s2, t2, i2, match_type)
        if cand:
            seen.add(key)
            candidates.append(cand)

    # --- 0. Explicit FK из метаданных (приоритет над эвристиками) ---
    for c1, c2 in fk_pairs_1to2:
        _add(c1, c2, "explicit_fk")
    for c2, c1 in fk_pairs_2to1:
        _add(c1, c2, "explicit_fk")

    # --- 1. Совпадающие имена ---
    shared = set(info1_map.keys()) & set(info2_map.keys())
    for col in shared:
        _add(col, col, "exact_name")

    # --- 2. FK-паттерн ---
    for (pk_s, pk_t, pk_map, fk_s, fk_t, fk_map) in [
        (s1, t1, info1_map, s2, t2, info2_map),
        (s2, t2, info2_map, s1, t1, info1_map),
    ]:
        pk_cols = [n for n, info in pk_map.items() if info["raw_pk"]]
        t_lower = pk_t.lower()
        fk_names_lower = {n.lower(): n for n in fk_map}

        for pk_col in pk_cols:
            pk_lower = pk_col.lower()
            fk_cands = {f"{t_lower}_id", f"{t_lower}_{pk_lower}"}
            if t_lower.endswith("s") and len(t_lower) > 2:
                fk_cands.add(f"{t_lower[:-1]}_id")
                fk_cands.add(f"{t_lower[:-1]}_{pk_lower}")
            # Entity-based: извлечь значимую часть имени таблицы
            parts = t_lower.split("_")
            skip = {"uzp", "dim", "dwh", "fact", "ref", "ld", "stg", "ods",
                    "salesntwrk", "pcap", "sn", "grnplm", "s"}
            for part in parts:
                if part not in skip and len(part) >= 3:
                    fk_cands.add(f"{part}_id")

            for cand in fk_cands:
                if cand == pk_lower:
                    continue
                if cand in fk_names_lower:
                    fk_actual = fk_names_lower[cand]
                    if pk_s == s1:
                        _add(pk_col, fk_actual, "fk_pattern")
                    else:
                        _add(fk_actual, pk_col, "fk_pattern")

    # --- 3. Нормализованный матчинг PK-колонок (old_X_id ↔ X_id) ---
    # Выполняется ДО суффиксного матчинга, чтобы иметь приоритет.
    # Для составных PK, где колонки имеют временные/версионные префиксы.
    # Например: old_gosb_id (PK в dim) ↔ gosb_id (FK в fact).
    for (pk_s, pk_t, pk_map, fk_s, fk_t, fk_map) in [
        (s1, t1, info1_map, s2, t2, info2_map),
        (s2, t2, info2_map, s1, t1, info1_map),
    ]:
        pk_cols = [n for n, info in pk_map.items() if info["raw_pk"]]
        # Строим индекс: нормализованное_имя → реальное имя колонки в FK-таблице
        fk_normalized: dict[str, str] = {_normalize_key_name(n): n for n in fk_map}

        for pk_col in pk_cols:
            norm_pk = _normalize_key_name(pk_col)
            if norm_pk == pk_col.lower():
                continue  # Нет префикса — уже обработано в шагах 1–2
            if norm_pk in fk_normalized:
                fk_actual = fk_normalized[norm_pk]
                # Сравниваем по (schema, table), а не только по schema —
                # во избежание ложных совпадений при одинаковых схемах.
                if (pk_s, pk_t) == (s1, t1):
                    _add(pk_col, fk_actual, "normalized_pk")
                else:
                    _add(fk_actual, pk_col, "normalized_pk")

    # --- 4. Suffix matching (_id колонки) ---
    # Только если хотя бы одна сторона — key/business_key (не оба attribute).
    # Запускается после normalized_pk — пары уже добавленные туда пропускаются через seen.
    id_cols1 = [n for n in info1_map if n.endswith("_id")]
    id_cols2 = [n for n in info2_map if n.endswith("_id")]
    for c1 in id_cols1:
        for c2 in id_cols2:
            if c1 == c2:
                continue
            if not (c2.endswith(f"_{c1}") or c1.endswith(f"_{c2}")):
                continue
            i1 = info1_map.get(c1)
            i2 = info2_map.get(c2)
            if i1 and i2 and i1["col_class"] == "attribute" and i2["col_class"] == "attribute":
                continue  # Оба attribute — шум, пропускаем
            _add(c1, c2, "suffix")

    # Сортировка по score desc
    candidates.sort(key=lambda c: -c.score)
    return candidates


# ---------------------------------------------------------------------------
# 4. Composite key grouping (для sql_validator)
# ---------------------------------------------------------------------------

@dataclass
class CompositeJoinPair:
    """Составной JOIN-ключ (несколько колонок в ON-условии)."""

    left_schema: str
    left_table: str
    right_schema: str
    right_table: str
    columns: list[tuple[str, str]] = field(default_factory=list)  # [(left_col, right_col)]


def group_composite_keys(
    join_pairs: list[dict[str, Any]],
) -> list[CompositeJoinPair]:
    """Группировать одиночные join-пары в составные ключи по (left_table, right_table).

    Input: список из ``_extract_join_pairs()`` — каждый элемент содержит
    ``left`` и ``right`` с ``schema``, ``table``, ``column``.

    Returns:
        Список ``CompositeJoinPair``, где каждая группа содержит все колонки
        между одной парой таблиц.
    """
    groups: dict[tuple[str, str, str, str], list[tuple[str, str]]] = {}

    for pair in join_pairs:
        if pair.get("type") == "cross_join":
            continue
        left = pair.get("left", {})
        right = pair.get("right", {})
        ls, lt = left.get("schema", ""), left.get("table", "")
        rs, rt = right.get("schema", ""), right.get("table", "")
        lc, rc = left.get("column", ""), right.get("column", "")
        if not all([ls, lt, rs, rt, lc, rc]):
            continue

        key = (ls, lt, rs, rt)
        groups.setdefault(key, []).append((lc, rc))

    result = []
    for (ls, lt, rs, rt), cols in groups.items():
        result.append(CompositeJoinPair(
            left_schema=ls, left_table=lt,
            right_schema=rs, right_table=rt,
            columns=cols,
        ))
    return result


# ---------------------------------------------------------------------------
# 5. Форматирование для LLM
# ---------------------------------------------------------------------------

def _is_fact_type(ttype: str) -> bool:
    """Является ли тип таблицы фактовым (fact или unknown — по умолчанию факт)."""
    return ttype in ("fact", "unknown")


def _is_dim_type(ttype: str) -> bool:
    """Является ли тип таблицы справочником (dim, ref)."""
    return ttype in ("dim", "ref")


def _safe_join_strategy(
    type1: str, type2: str,
    tbl1: str, tbl2: str,
    col1: str, col2: str,
    col1_status: str, col2_status: str,
) -> list[str]:
    """Сгенерировать конкретную стратегию безопасного JOIN на основе типов таблиц.

    Правила:
    1. fact + fact → CTE с GROUP BY для обеих сторон по ключу джойна
    2. fact + dim/ref → уникальная выборка из справочника по ключу джойна
    3. dim/ref + fact → CTE с GROUP BY для таблицы фактов по ключу джойна
    4. dim/ref + dim/ref → уникальные выборки из обоих справочников по ключу джойна
    """
    lines: list[str] = []
    is_fact1 = _is_fact_type(type1)
    is_fact2 = _is_fact_type(type2)
    is_dim1 = _is_dim_type(type1)
    is_dim2 = _is_dim_type(type2)

    # Определяем стороны с дублями
    side1_has_dupes = "ДУБЛИ" in col1_status or "composite-PK" in col1_status
    side2_has_dupes = "ДУБЛИ" in col2_status or "composite-PK" in col2_status

    if is_fact1 and is_fact2:
        # fact + fact: обе стороны — агрегация в CTE, потом соединяем
        lines.append("    → Стратегия: ФАКТ + ФАКТ — предварительная агрегация ОБЕИХ сторон в CTE")
        lines.append(f"    → Паттерн:")
        lines.append(f"      WITH cte1 AS (")
        lines.append(f"        SELECT {col1}, SUM(<метрика1>) AS val1 FROM {tbl1} GROUP BY {col1}")
        lines.append(f"      ), cte2 AS (")
        lines.append(f"        SELECT {col2}, SUM(<метрика2>) AS val2 FROM {tbl2} GROUP BY {col2}")
        lines.append(f"      )")
        lines.append(f"      SELECT * FROM cte1 JOIN cte2 ON cte1.{col1} = cte2.{col2}")

    elif is_fact1 and is_dim2:
        # fact + dim: уникальная выборка из справочника
        lines.append("    → Стратегия: ФАКТ + СПРАВОЧНИК — уникальная выборка из справочника")
        if side2_has_dupes:
            lines.append(f"    → Паттерн:")
            lines.append(f"      SELECT f.*, d.<нужные_колонки>")
            lines.append(f"      FROM {tbl1} f")
            lines.append(f"      JOIN (")
            lines.append(f"        SELECT DISTINCT ON ({col2}) {col2}, <нужные_колонки>")
            lines.append(f"        FROM {tbl2} ORDER BY {col2}, <дата_актуальности> DESC")
            lines.append(f"      ) d ON d.{col2} = f.{col1}")
        else:
            lines.append(f"    → JOIN {tbl2} безопасен по {col2} — прямой JOIN допустим")

    elif is_dim1 and is_fact2:
        # dim + fact: агрегация фактов в CTE/подзапросе
        lines.append("    → Стратегия: СПРАВОЧНИК + ФАКТ — агрегация фактов в CTE/подзапросе")
        if side2_has_dupes:
            lines.append(f"    → Паттерн:")
            lines.append(f"      SELECT d.*, agg.val")
            lines.append(f"      FROM {tbl1} d")
            lines.append(f"      JOIN (")
            lines.append(f"        SELECT {col2}, SUM(<метрика>) AS val FROM {tbl2} GROUP BY {col2}")
            lines.append(f"      ) agg ON agg.{col2} = d.{col1}")
        else:
            lines.append(f"    → JOIN {tbl2} безопасен по {col2} — прямой JOIN допустим")

    elif is_dim1 and is_dim2:
        # dim + dim: уникальные выборки из обоих справочников
        lines.append("    → Стратегия: СПРАВОЧНИК + СПРАВОЧНИК — уникальные выборки из обеих сторон")
        parts: list[str] = []
        if side1_has_dupes:
            parts.append(f"      WITH d1 AS (")
            parts.append(f"        SELECT DISTINCT ON ({col1}) {col1}, <нужные_колонки>")
            parts.append(f"        FROM {tbl1} ORDER BY {col1}, <дата_актуальности> DESC")
            parts.append(f"      )")
        if side2_has_dupes:
            prefix = "      , d2 AS (" if side1_has_dupes else "      WITH d2 AS ("
            parts.append(prefix)
            parts.append(f"        SELECT DISTINCT ON ({col2}) {col2}, <нужные_колонки>")
            parts.append(f"        FROM {tbl2} ORDER BY {col2}, <дата_актуальности> DESC")
            parts.append(f"      )")
        if parts:
            lines.append(f"    → Паттерн:")
            lines.extend(parts)
            a1 = "d1" if side1_has_dupes else tbl1
            a2 = "d2" if side2_has_dupes else tbl2
            lines.append(f"      SELECT * FROM {a1} JOIN {a2} ON {a1}.{col1} = {a2}.{col2}")
    else:
        # Fallback для неизвестных комбинаций
        if side2_has_dupes:
            prob_tbl, prob_col = tbl2, col2
        elif side1_has_dupes:
            prob_tbl, prob_col = tbl1, col1
        else:
            prob_tbl, prob_col = tbl2, col2
        lines.append(f"    → Паттерн: JOIN (SELECT DISTINCT ON ({prob_col}) * "
                      f"FROM {prob_tbl} ORDER BY {prob_col}) sub ON sub.{prob_col} = ...")

    return lines


def suggest_composite_joins(
    candidates: list[JoinCandidate],
    cols1_df: pd.DataFrame,
    cols2_df: pd.DataFrame,
    pk_count1: int,
    pk_count2: int,
    tbl1: str,
    tbl2: str,
) -> list[str]:
    """Найти составные JOIN-ключи и вернуть готовые ON-строки для LLM.

    Логика: если одна из таблиц имеет составной PK (pk_count > 1) и
    среди кандидатов нашлись пары для ВСЕХ колонок этого PK — предлагаем
    составной ON вместо одиночного.

    Returns:
        Список строк для добавления в блок JOIN-анализа.
    """
    lines: list[str] = []

    # Собираем PK-колонки каждой таблицы из DataFrame
    def _pk_cols(df: pd.DataFrame) -> list[str]:
        if df.empty or "is_primary_key" not in df.columns:
            return []
        return df[df["is_primary_key"].astype(bool)]["column_name"].tolist()

    pk1 = _pk_cols(cols1_df)
    pk2 = _pk_cols(cols2_df)

    # Индексы: col2 → col1 и col1 → col2 из найденных кандидатов
    pair_map_2to1: dict[str, str] = {}  # col2 → col1
    pair_map_1to2: dict[str, str] = {}  # col1 → col2
    for c in candidates:
        pair_map_2to1[c.col2] = c.col1
        pair_map_1to2[c.col1] = c.col2

    def _find_normalized_in_df(pk_col: str, df: pd.DataFrame) -> str | None:
        """Найти пару для PK-колонки через нормализацию имени.

        Fallback для случаев, когда пара не попала в candidates из-за штрафов.
        Пример: old_gosb_id → gosb_id → найдено в другой таблице.
        """
        if df.empty or "column_name" not in df.columns:
            return None
        norm = _normalize_key_name(pk_col)
        for col in df["column_name"].tolist():
            if _normalize_key_name(col) == norm:
                return col
        return None

    def _build_composite(pk_side: list[str], pair_map: dict[str, str],
                         alias_pk: str, alias_other: str,
                         pk_tbl: str, other_tbl: str,
                         other_df: pd.DataFrame | None = None) -> list[str]:
        """Строим ON-условие: все PK-колонки pk_side нашли пару в pair_map.

        Если колонка не найдена в pair_map — пробуем нормализованный поиск по other_df
        (fallback для old_/new_-префиксов, не попавших в candidates из-за штрафов).
        """
        if len(pk_side) < 2:
            return []
        matched = []
        other_col_set = (
            set(other_df["column_name"].tolist()) if other_df is not None and not other_df.empty
            else set()
        )
        for pk_col in pk_side:
            # 1) Exact same-name match — highest priority (e.g. dim.tb_id → fact.tb_id)
            if pk_col in other_col_set:
                matched.append((pk_col, pk_col))
            # 2) pair_map from ranked candidates
            elif pk_col in pair_map:
                matched.append((pk_col, pair_map[pk_col]))
            # 3) Normalized name fallback (old_gosb_id → gosb_id)
            elif other_df is not None:
                found = _find_normalized_in_df(pk_col, other_df)
                if found:
                    matched.append((pk_col, found))
        if len(matched) < len(pk_side):
            return []  # Не все PK-колонки нашли пару — не предлагаем

        on_parts = [f"{alias_pk}.{pk_col} = {alias_other}.{other_col}"
                    for pk_col, other_col in matched]
        on_str = " AND ".join(on_parts)
        pk_cols_str = ", ".join(pk_side)
        result = [
            f"  → Составной JOIN (покрывает весь PK {pk_tbl}: {pk_cols_str}):",
            f"    ON {on_str}",
        ]
        return result

    # Проверяем pk2 (правая таблица — чаще справочник)
    composite2 = _build_composite(pk2, pair_map_2to1, "g", "f", tbl2, tbl1,
                                   other_df=cols1_df)
    if composite2:
        lines.append("")
        lines.append("РЕКОМЕНДУЕМЫЙ СОСТАВНОЙ JOIN (покрывает полный PK справочника):")
        lines.extend(composite2)
        lines.append(
            "  Примечание: если пользователь явно не указал иной ключ — "
            "используй составной ON для исключения дублей."
        )

    # Проверяем pk1 (если pk2 не дал результата)
    if not composite2:
        composite1 = _build_composite(pk1, pair_map_1to2, "f", "g", tbl1, tbl2,
                                       other_df=cols2_df)
        if composite1:
            lines.append("")
            lines.append("РЕКОМЕНДУЕМЫЙ СОСТАВНОЙ JOIN (покрывает полный PK):")
            lines.extend(composite1)
            lines.append(
                "  Примечание: если пользователь явно не указал иной ключ — "
                "используй составной ON для исключения дублей."
            )

    return lines


def _non_pk_warnings(
    candidates: list[JoinCandidate],
    cols1_df: pd.DataFrame,
    cols2_df: pd.DataFrame,
    pk_count1: int,
    pk_count2: int,
) -> list[str]:
    """Сформировать фактические предупреждения о non-PK join-ключах в dim-таблицах.

    Не запрещает, а предоставляет факты: is_pk, unique_perc, какой PK у таблицы.
    LLM сам принимает решение с учётом явных инструкций пользователя.
    """
    warnings: list[str] = []

    def _pk_cols(df: pd.DataFrame) -> list[str]:
        if df.empty or "is_primary_key" not in df.columns:
            return []
        return df[df["is_primary_key"].astype(bool)]["column_name"].tolist()

    pk2_cols = _pk_cols(cols2_df)
    pk1_cols = _pk_cols(cols1_df)

    for c in candidates[:5]:
        # Проверяем правую сторону (c.col2) — чаще справочник
        if not c.col2_status.startswith("PK") and not c.col2_status.startswith("composite-PK"):
            if pk_count2 > 0 and pk2_cols:
                pk_str = ", ".join(pk2_cols)
                warnings.append(
                    f"  ИНФО: {c.col2} в {c.table2} — не PK (фактический PK: {pk_str}), "
                    f"{c.col2_status}. "
                    f"Если пользователь не указал иное — предпочти join по PK."
                )
        # Проверяем левую сторону (c.col1)
        if not c.col1_status.startswith("PK") and not c.col1_status.startswith("composite-PK"):
            if pk_count1 > 0 and pk1_cols:
                pk_str = ", ".join(pk1_cols)
                warnings.append(
                    f"  ИНФО: {c.col1} в {c.table1} — не PK (фактический PK: {pk_str}), "
                    f"{c.col1_status}. "
                    f"Если пользователь не указал иное — предпочти join по PK."
                )

    # Дедупликация
    seen: set[str] = set()
    result = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result


def format_join_analysis(
    s1: str, t1: str, cols1_df: pd.DataFrame,
    s2: str, t2: str, cols2_df: pd.DataFrame,
    pk_count1: int, pk_count2: int,
) -> str:
    """Сформировать блок JOIN-анализа для LLM-контекста.

    Возвращает компактный текст с ранжированными кандидатами и рекомендациями.
    Включает:
    - ранжированные одиночные кандидаты с фактической аннотацией (PK/non-PK)
    - составной JOIN-ключ, если PK таблицы составной и все его части нашли пару
    - конкретную стратегию безопасного JOIN на основе типов таблиц
    """
    MAX_DISPLAY_CANDIDATES = 5

    candidates = rank_join_candidates(
        s1, t1, cols1_df, s2, t2, cols2_df, pk_count1, pk_count2,
    )
    if not candidates:
        return ""

    type1 = detect_table_type(t1, cols1_df)
    type2 = detect_table_type(t2, cols2_df)
    type_label1 = f" ({type1})" if type1 != "unknown" else ""
    type_label2 = f" ({type2})" if type2 != "unknown" else ""

    lines = [
        f"\nТаблицы: {s1}.{t1}{type_label1} ↔ {s2}.{t2}{type_label2}",
        "",
        "Наиболее вероятные join keys:",
    ]

    shown = candidates[:MAX_DISPLAY_CANDIDATES]
    has_unsafe = False
    for c in shown:
        safety = "безопасен" if c.safe else f"ОПАСНО: {c.risk_detail}"
        desc1_part = f" '{c.col1_desc}'" if c.col1_desc else ""
        desc2_part = f" '{c.col2_desc}'" if c.col2_desc else ""
        lines.append(
            f"  [{c.score:.2f}] {c.col1}{desc1_part} ({c.col1_status}) ↔ "
            f"{c.col2}{desc2_part} ({c.col2_status}) — {c.reason} → {safety}"
        )
        # Для ОПАСНО — конкретный безопасный SQL-паттерн по типам таблиц
        if not c.safe:
            has_unsafe = True
            strategy_lines = _safe_join_strategy(
                type1, type2,
                c.table1, c.table2,
                c.col1, c.col2,
                c.col1_status, c.col2_status,
            )
            lines.extend(strategy_lines)

    if len(candidates) > MAX_DISPLAY_CANDIDATES:
        lines.append(
            f"  (ещё {len(candidates) - MAX_DISPLAY_CANDIDATES} менее вероятных опущено)"
        )

    # Фактические аннотации non-PK join-ключей (не запрет, а информация)
    non_pk_warns = _non_pk_warnings(
        shown, cols1_df, cols2_df, pk_count1, pk_count2,
    )
    if non_pk_warns:
        lines.append("")
        lines.extend(non_pk_warns)

    # Составной JOIN (если PK составной и все части нашли пару)
    composite_lines = suggest_composite_joins(
        candidates, cols1_df, cols2_df,
        pk_count1, pk_count2,
        f"{s1}.{t1}", f"{s2}.{t2}",
    )
    lines.extend(composite_lines)

    if has_unsafe:
        lines.append("")
        lines.append(
            "КРИТИЧЕСКОЕ ПРАВИЛО: SQL запрос НИКОГДА не должен множить данные.\n"
            "ЗАПРЕЩЕНО использовать прямой JOIN по ключу помеченному ОПАСНО.\n"
            "Используй стратегию из паттерна выше (CTE с GROUP BY или DISTINCT ON в подзапросе).\n"
            "ЗАПРЕЩЕНО добавлять DISTINCT к внешнему SELECT — это маскирует проблему."
        )
    return "\n".join(lines)
