"""WHERE resolver using ranked metadata-driven filter candidates."""

from __future__ import annotations

import logging
import re
from typing import Any

from core.domain_rules import table_can_satisfy_frame
from core.filter_ranking import rank_filter_candidates
from core.log_safety import summarize_dict_keys
from core.text_normalize import (
    normalize_text as _normalize_text,
    stem as _stem,
    tokenize as _tokenize,
)
from core.query_ir import FilterSpec, _parse_calendar_period, _target_looks_calendar

logger = logging.getLogger(__name__)


def _business_event_is_already_encoded_in_table(
    *,
    schema_loader,
    selected_tables: list[str],
    semantic_frame: dict[str, Any] | None,
) -> bool:
    """Определить, что бизнес-смысл запроса уже покрыт выбранной таблицей.

    Сценарий: пользователь или table_resolver выбрал одну витрину по событию
    ("фактический отток"), а filter_intents всё ещё указывают на колонки другой
    таблицы. В таком случае отдельный WHERE по этому событию не нужен.
    """
    if len(selected_tables) != 1:
        return False

    business_event = str((semantic_frame or {}).get("business_event") or "").strip()
    filter_intents = list((semantic_frame or {}).get("filter_intents") or [])
    if not business_event or not filter_intents:
        return False

    selected_table = str(selected_tables[0]).strip().lower()
    if "." not in selected_table:
        return False
    schema, table = selected_table.split(".", 1)

    bound_to_other_tables = []
    for item in filter_intents:
        column_key = str(item.get("column_key") or "").strip().lower()
        if not column_key:
            return False
        table_key = ".".join(column_key.split(".")[:2])
        if table_key == selected_table:
            return False
        bound_to_other_tables.append(table_key)

    if not bound_to_other_tables:
        return False

    table_info = schema_loader.get_table_info(schema, table)
    haystack_stems = {_stem(tok) for tok in _tokenize(table_info)}
    business_stems = {_stem(tok) for tok in _tokenize(business_event) if len(tok) >= 4}
    if not business_stems:
        return False

    # Смысл считается покрытым таблицей, если каждый существенный токен события
    # читается в имени/описании таблицы хотя бы по общему стему.
    return all(
        any(
            cand == token_stem
            or cand.startswith(token_stem)
            or token_stem.startswith(cand)
            for cand in haystack_stems
        )
        for token_stem in business_stems
    )


def _filter_value_encoded_in_table(
    *,
    schema_loader,
    table_key: str,
    value: Any,
) -> bool:
    """True если значение фильтра уже семантически закодировано в имени/описании
    выбранной таблицы (например "фактический отток" + uzp_dwh_fact_outflow).

    Использует _event_stems для расширения значимыми синонимами (отток→outflow).
    Сравнение по стемам, чтобы пройти морфологию русского ("оттока"→"отток").
    """
    if schema_loader is None or value is None or "." not in str(table_key):
        return False
    schema, table = str(table_key).split(".", 1)
    try:
        table_info = schema_loader.get_table_info(schema, table) or ""
    except Exception:  # noqa: BLE001
        return False
    haystack = f"{schema}.{table} {table_info}"
    haystack_stems = {_stem(tok) for tok in _tokenize(haystack) if tok}
    if not haystack_stems:
        return False
    value_stems = _event_stems(str(value))
    # Доп. стемы из самой строки — на случай если value не покрыт _event_stems.
    value_stems = value_stems | {
        _stem(tok) for tok in _tokenize(str(value)) if len(tok) >= 4
    }
    significant = {s for s in value_stems if s and len(s) >= 4}
    if not significant:
        return False
    for token_stem in significant:
        for cand in haystack_stems:
            if cand == token_stem:
                return True
            if (
                min(len(cand), len(token_stem)) >= 5
                and (cand.startswith(token_stem) or token_stem.startswith(cand))
            ):
                return True
    return False


def _event_stems(business_event: str) -> set[str]:
    stems = {_stem(tok) for tok in _tokenize(business_event) if len(tok) >= 4}
    normalized = _normalize_text(business_event)
    if any(tok in normalized for tok in ("outflow", "churn", "attrition", "отток")):
        stems.update({_stem("outflow"), _stem("отток"), _stem("churn"), _stem("attrition")})
    return {stem for stem in stems if stem}


def _text_has_event_stem(text: str, event_stems: set[str]) -> bool:
    if not event_stems:
        return False
    text_stems = {_stem(tok) for tok in _tokenize(text)}
    for left in event_stems:
        for right in text_stems:
            if left == right:
                return True
            if min(len(left), len(right)) >= 5 and (left.startswith(right) or right.startswith(left)):
                return True
    return False


def _business_event_is_aggregate_metric(
    *,
    schema_loader,
    selected_columns: dict[str, dict[str, Any]],
    semantic_frame: dict[str, Any] | None,
) -> bool:
    """Определить, что unresolved filter на самом деле является агрегируемой метрикой."""
    metric_intent = str((semantic_frame or {}).get("metric_intent") or "").lower()
    if metric_intent not in {"sum", "avg", "min", "max"}:
        return False

    business_event = str((semantic_frame or {}).get("business_event") or "").strip()
    filter_intents = list((semantic_frame or {}).get("filter_intents") or [])
    event_stems = _event_stems(business_event)
    if not event_stems or not filter_intents:
        return False

    for item in filter_intents:
        item_text = " ".join(
            str(item.get(key) or "")
            for key in ("request_id", "query_text", "matched_phrase", "column_key")
        )
        if not _text_has_event_stem(item_text, event_stems):
            return False

    for table_key, roles in (selected_columns or {}).items():
        if "." not in table_key:
            continue
        schema, table = table_key.split(".", 1)
        table_info = schema_loader.get_table_info(schema, table)
        for column in roles.get("aggregate", []) or []:
            column = str(column or "").strip()
            if not column or column == "*":
                continue
            description = ""
            cols_df = schema_loader.get_table_columns(schema, table)
            if not cols_df.empty and "column_name" in cols_df.columns:
                match = cols_df[
                    cols_df["column_name"].astype(str).str.lower() == column.lower()
                ]
                if not match.empty:
                    description = str(match.iloc[0].get("description", "") or "")
            if _text_has_event_stem(f"{table_key} {table_info} {column} {description}", event_stems):
                return True
    return False


_CONDITION_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(>=|<=|!=|<>|=|<|>|ILIKE|LIKE|IN|NOT IN|IS NOT|IS)\s*(.*?)\s*$",
    re.IGNORECASE,
)


def _parse_condition(condition: str) -> tuple[str, str, str] | None:
    """Грубо разобрать condition вида `col op value` → (col, op_norm, value_norm)."""
    match = _CONDITION_RE.match(str(condition or "").strip())
    if not match:
        return None
    column = match.group(1).strip().lower()
    op = match.group(2).strip().upper()
    if op == "<>":
        op = "!="
    raw_value = match.group(3).strip()
    # Нормализуем значение: убираем кавычки, ::date и пробелы;
    # 'True'/true/1 для bool сводим к одному виду.
    value_norm = raw_value.replace("::date", "").replace("::timestamp", "").strip()
    if value_norm.startswith("'") and value_norm.endswith("'") and len(value_norm) >= 2:
        value_norm = value_norm[1:-1]
    low = value_norm.lower()
    if low in {"true", "1"}:
        value_norm = "true"
    elif low in {"false", "0"}:
        value_norm = "false"
    return column, op, value_norm


def _dtype_bucket(dtype: str) -> str:
    """Категоризовать dtype колонки в крупные группы для type-check."""
    d = (dtype or "").lower()
    if "bool" in d:
        return "bool"
    if (
        "int" in d
        or "number" in d
        or "decimal" in d
        or "float" in d
        or "numeric" in d
        or "double" in d
    ):
        return "numeric"
    if "timestamp" in d or "time" in d:
        return "datetime"
    if "date" in d:
        return "date"
    return "text"


def _value_bucket(value: Any) -> str:
    """Категоризовать Python-значение в ту же группу, что _dtype_bucket."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "numeric"
    if value is None:
        return "any"
    text = str(value).strip()
    if not text:
        return "text"
    low = text.lower()
    if low in {"true", "false"}:
        return "bool"
    # Числовой литерал
    try:
        float(text)
        return "numeric"
    except (ValueError, TypeError):
        pass
    # ISO-дата YYYY-MM-DD или YYYY-MM
    if re.match(r"^\d{4}-\d{2}(-\d{2})?$", text):
        return "date"
    if re.match(r"^\d{4}-\d{2}-\d{2}[ T]", text):
        return "datetime"
    return "text"


def _dtype_compatible(col_dtype: str, value: Any) -> bool:
    """True если literal-значение совместимо с dtype колонки."""
    col = _dtype_bucket(col_dtype)
    val = _value_bucket(value)
    if val == "any":
        return True
    if col == val:
        return True
    # Допустимые кроссовые комбинации.
    compatible = {
        ("numeric", "text"),  # 42 в text-колонке — допустим (text ≥ numeric)
        ("date", "datetime"),
        ("datetime", "date"),
        ("text", "numeric"),  # '42' в numeric — Postgres сам кастит
    }
    return (col, val) in compatible


def _check_filter_dtype_compatibility(
    *,
    column: str,
    table_key: str,
    value: Any,
    schema_loader,
) -> tuple[bool, str]:
    """Проверить, что литерал значение совместим с dtype колонки.

    Returns:
        (is_compatible, dtype). dtype — пустая строка если каталог не знает колонку.
    """
    if schema_loader is None or "." not in table_key:
        return True, ""
    schema, table = table_key.split(".", 1)
    try:
        dtype = str(schema_loader.get_column_dtype(schema, table, column) or "").strip()
    except Exception:  # noqa: BLE001
        return True, ""
    if not dtype:
        return True, ""
    return _dtype_compatible(dtype, value), dtype


# Регекс для извлечения сырого литерала из condition без bool-нормализации
# (которая ломает type-check `is_outflow = 1` → numeric, не bool).
_CONDITION_LITERAL_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(>=|<=|!=|<>|=|<|>|ILIKE|LIKE|IN|NOT IN|IS NOT|IS)\s*(.*?)\s*$",
    re.IGNORECASE,
)


def _extract_condition_literal(condition: str) -> Any:
    """Извлечь правую часть condition БЕЗ bool/numeric нормализации.

    Возвращает Python-значение пригодное для `_value_bucket`:
    - quoted строка → str без кавычек
    - голый числовой литерал → int/float
    - всё остальное → str.
    """
    match = _CONDITION_LITERAL_RE.match(str(condition or "").strip())
    if not match:
        return None
    raw = match.group(3).strip()
    raw = raw.replace("::date", "").replace("::timestamp", "").strip()
    if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
        return raw[1:-1]
    # Числовой литерал — отдаём как число, чтобы value_bucket=numeric.
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except (ValueError, TypeError):
        return raw


def _add_unique(conditions: list[str], condition: str) -> None:
    """Добавить condition с дедупом по точной строке И по `(column, op, value_norm)`.

    Раньше дедуп шёл только по точной строке — что приводило к парам
    `is_task = true` и `is_task = 'True'` в одном WHERE. Расширяем ключ:
    нормализованная тройка (column, operator, value).
    """
    if condition in conditions:
        return
    parsed = _parse_condition(condition)
    if parsed is not None:
        for existing in conditions:
            existing_parsed = _parse_condition(existing)
            if existing_parsed == parsed:
                return
    conditions.append(condition)


def _explicit_column_choice(user_input: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Если пользователь явно назвал колонку, выбрать соответствующего кандидата."""
    normalized = str(user_input or "").lower().replace("ё", "е")
    for candidate in candidates:
        column = str(candidate.get("column") or "").lower().replace("ё", "е")
        if column and column in normalized:
            return candidate
    return None


def _user_explicitly_named(
    user_input: str,
    best: dict[str, Any],
    second: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any] | None]:
    """Проверить, упоминает ли пользователь явно best.column или second.column.

    Если ровно одна из двух колонок встречается в тексте — пользователь явно указал
    выбор, clarification не нужен. Возвращает (is_explicit, chosen_candidate).
    """
    if second is None:
        return False, None
    normalized = str(user_input or "").lower().replace("ё", "е")
    best_col = str(best.get("column") or "").lower().replace("ё", "е")
    second_col = str(second.get("column") or "").lower().replace("ё", "е")
    best_mentioned = bool(best_col and best_col in normalized)
    second_mentioned = bool(second_col and second_col in normalized)
    if best_mentioned and not second_mentioned:
        return True, best
    if second_mentioned and not best_mentioned:
        return True, second
    return False, None


def candidate_label(candidate: dict[str, Any]) -> str:
    """Построить подпись кандидата для clarification-сообщения.

    Включает имя колонки и описание, при наличии — информативный хвост:
      - если есть matched_example (значение из known_terms, совпавшее с запросом),
        показываем «есть значение "X"»;
      - иначе если у колонки есть примеры значений, показываем «примеры: "A", "B"».
    Без подгона под конкретный кейс: решение принимается по содержимому candidate.
    """
    column = str(candidate.get("column") or "")
    description = str(candidate.get("description") or "")
    label = f"`{column}`"
    if description:
        label += f" ({description}"
    else:
        label += " ("

    extras: list[str] = []
    matched_example = str(candidate.get("matched_example") or "").strip()
    if matched_example:
        extras.append(f"есть значение «{matched_example}»")
    else:
        example_values = [str(v).strip() for v in (candidate.get("example_values") or []) if str(v).strip()]
        if example_values:
            quoted = ", ".join(f"«{v}»" for v in example_values[:2])
            extras.append(f"примеры: {quoted}")

    if description and extras:
        label += ", " + "; ".join(extras) + ")"
    elif description:
        label += ")"
    elif extras:
        label += "; ".join(extras) + ")"
    else:
        # Нет ни описания, ни примеров — убираем пустые скобки.
        label = f"`{column}`"
    return label


def resolve_where(
    *,
    user_input: str,
    intent: dict[str, Any],
    selected_columns: dict[str, dict[str, Any]],
    selected_tables: list[str],
    schema_loader,
    semantic_frame: dict[str, Any] | None,
    base_conditions: list[str] | None = None,
    user_filter_choices: dict[str, str] | None = None,
    rejected_filter_choices: dict[str, list[str]] | None = None,
    filter_tiebreaker=None,
    filter_specs: list[FilterSpec | dict[str, Any]] | None = None,
    time_range: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Достроить WHERE доменными правилами и value profiles.

    user_filter_choices: маппинг request_id → column_name, собранный из явных
    ответов пользователя на предыдущие clarification-вопросы. Если для
    request_id задан выбор — ambiguity-блок пропускается, выбирается
    соответствующий кандидат. Это защищает пайплайн от зацикливания, когда
    CLI перезапускает граф после ответа пользователя.
    """
    conditions: list[str] = list(base_conditions or [])
    reasoning: list[str] = []
    applied_rules: list[str] = []
    unresolved_filters: list[dict[str, Any]] = []
    implicit_filters: list[dict[str, Any]] = []
    clarification_message = ""
    clarification_spec: dict[str, Any] = {}
    user_filter_choices = dict(user_filter_choices or {})
    rejected_filter_choices = {
        str(k): [str(vv).strip().lower() for vv in (v or []) if str(vv).strip()]
        for k, v in dict(rejected_filter_choices or {}).items()
    }
    direct_filter_specs = _normalize_filter_specs(filter_specs)
    calendar_applied, calendar_clarification = _apply_calendar_filter_specs(
        conditions,
        selected_columns,
        direct_filter_specs,
        schema_loader=schema_loader,
    )
    if calendar_clarification:
        return {
            "conditions": conditions,
            "applied_rules": calendar_applied,
            "reasoning": ["calendar_filter_without_time_axis"],
            "filter_candidates": {},
            "needs_clarification": True,
            "clarification_message": calendar_clarification,
            "clarification_spec": {
                "type": "calendar_filter",
                "message": calendar_clarification,
            },
            "user_filter_choices": user_filter_choices,
            "rejected_filter_choices": rejected_filter_choices,
        }
    if calendar_applied:
        direct_filter_specs = [
            spec for idx, spec in enumerate(direct_filter_specs)
            if f"query_spec:{idx}" not in set(calendar_applied)
        ]
    direct_applied = _apply_exact_filter_specs(
        conditions,
        selected_columns,
        direct_filter_specs,
        schema_loader=schema_loader,
        time_range=time_range,
        unresolved=unresolved_filters,
    )
    applied_rules.extend(direct_applied)
    applied_rules.extend(calendar_applied)
    semantic_filter_specs = _filter_specs_for_semantic_frame(
        selected_columns,
        direct_filter_specs,
        schema_loader=schema_loader,
        time_range=time_range,
    )
    effective_semantic_frame = _semantic_frame_with_filter_specs(
        semantic_frame,
        semantic_filter_specs,
    )
    effective_semantic_frame = _semantic_frame_without_redundant_calendar_filters(
        effective_semantic_frame,
        selected_columns,
        schema_loader=schema_loader,
        time_range=time_range,
    )
    effective_intent = _intent_without_redundant_calendar_filters(
        intent,
        selected_columns,
        schema_loader=schema_loader,
        time_range=time_range,
    )

    ranked_candidates = rank_filter_candidates(
        user_input=user_input,
        intent=effective_intent,
        selected_tables=selected_tables,
        schema_loader=schema_loader,
        semantic_frame=effective_semantic_frame,
    )

    # Order request_ids so explicit/user-driven filters resolve first, then
    # text: lexicon rules. This lets us skip a text: rule when an *explicit*
    # rule has already covered the same table, preventing the redundant
    # double-filter the LLM was hedging around (see 2026-05-25 regression).
    # We still allow multiple text: rules on one table — those typically
    # represent independent filter_intents (e.g. subject vs value).
    def _rule_priority(req_id: str) -> int:
        prefix = str(req_id).split(":", 1)[0]
        if prefix in ("explicit", "query_spec"):
            return 0
        if prefix in ("phrase", "flag"):
            return 1
        if prefix == "text":
            return 2
        return 1
    ranked_items = sorted(
        ranked_candidates.items(),
        key=lambda kv: _rule_priority(kv[0]),
    )
    explicit_applied_table_keys: set[str] = set()

    for request_id, candidates in ranked_items:
        if str(request_id) in set(direct_applied):
            continue
        if not candidates:
            continue
        # F6: skip text-lexicon rules when an explicit/query_spec rule already
        # applied to the same table — prevents the LLM from emitting redundant
        # double filters (e.g. task_type ILIKE '%X%' on top of task_subtype = 'X').
        # Multiple text: rules on one table are still allowed because they
        # usually represent distinct intents (subject vs value).
        if str(request_id).startswith("text:") and explicit_applied_table_keys:
            top_table_key = str((candidates[0] or {}).get("table_key") or "")
            if top_table_key and top_table_key in explicit_applied_table_keys:
                reasoning.append(
                    f"skipped_text_rule:{request_id}:already_covered_by_explicit"
                )
                logger.info(
                    "WhereResolver: skip text rule %s — table %s already covered "
                    "by an explicit/query_spec rule",
                    request_id, top_table_key,
                )
                continue
        compatible_candidates = []
        for candidate in candidates:
            schema = candidate.get("schema")
            table = candidate.get("table")
            if schema and table and not table_can_satisfy_frame(schema_loader, schema, table, effective_semantic_frame):
                continue
            compatible_candidates.append(candidate)
        candidates = compatible_candidates or candidates
        rejected_columns = set(rejected_filter_choices.get(str(request_id), []))
        candidates = [
            cand for cand in candidates
            if str(cand.get("column") or "").strip().lower() not in rejected_columns
        ]
        if not candidates:
            reasoning.append(f"rejected_all:{request_id}")
            continue

        # 1) Явный выбор пользователя из предыдущего раунда уточнений побеждает всё.
        chosen_by_user: dict[str, Any] | None = None
        user_column = str(user_filter_choices.get(str(request_id)) or "").strip().lower()
        if user_column:
            for cand in candidates:
                if str(cand.get("column") or "").strip().lower() == user_column:
                    chosen_by_user = cand
                    break

        if chosen_by_user is not None:
            best = chosen_by_user
            second = None
            score_gap = 999.0
            reasoning.append(f"user_choice:{request_id}:{best.get('column')}")
        else:
            explicit_choice = _explicit_column_choice(user_input, candidates)
            if explicit_choice is not None:
                best = explicit_choice
                second = None
                score_gap = 999.0
            else:
                best = candidates[0]
                second = candidates[1] if len(candidates) > 1 else None
                score_gap = float(best.get("score", 0.0)) - float((second or {}).get("score", 0.0))
        if best.get("confidence") == "low":
            continue
        # Триггеры для LLM-tiebreaker: относительный gap < 15 ИЛИ абсолютная
        # уверенность <= medium при низком score (<50). LLM разруливает ничьи
        # только для top-3 кандидатов, не ранжирует весь каталог.
        top_score = float(best.get("score", 0.0) or 0.0)
        absolute_low_confidence = (
            best.get("confidence") == "medium" and top_score < 50.0
        )
        if (
            filter_tiebreaker is not None
            and second is not None
            and (score_gap < 15.0 or absolute_low_confidence)
            and best.get("confidence") != "high"
            and str(request_id) not in user_filter_choices
        ):
            try:
                top_candidates = [c for c in candidates[:3]]
                chosen_column = filter_tiebreaker(
                    request_id=str(request_id),
                    user_input=user_input,
                    candidates=top_candidates,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "filter_tiebreaker упал: %s — fallback на clarification", exc,
                )
                chosen_column = None
            if chosen_column:
                for cand in candidates[:3]:
                    if str(cand.get("column") or "").strip().lower() == str(chosen_column).strip().lower():
                        best = cand
                        second = None
                        reasoning.append(
                            f"tiebreaker_llm:{request_id}:{best.get('column')}"
                        )
                        break

        if second and best.get("confidence") == "medium" and score_gap < 15.0:
            _is_explicit, _chosen = _user_explicitly_named(user_input, best, second)
            if _is_explicit and _chosen is not None:
                # Пользователь явно назвал одну из колонок — берём её без уточнения
                best = _chosen
                second = None
            elif str(request_id) in user_filter_choices:
                # Уже выбрано через user_filter_choices (handled above)
                pass
            else:
                best_label = candidate_label(best)
                second_label = candidate_label(second)
                clarification_message = (
                    "Найдено несколько близких вариантов фильтра. "
                    f"Уточните, пожалуйста, по какому признаку фильтровать: "
                    f"{best_label} или {second_label}?"
                )
                clarification_spec = {
                    "type": "choice",
                    "request_id": str(request_id),
                    "message": clarification_message,
                    "options": [
                        {"column": str(best.get("column") or ""), "label": best_label},
                        {"column": str(second.get("column") or ""), "label": second_label},
                    ],
                }
                reasoning.append(f"ambiguity:{request_id}:gap={score_gap:.1f}")
                break

        best_condition = str(best.get("condition") or "")
        best_column = str(best.get("column") or "")
        best_table_key = str(best.get("table_key") or "")
        # Direction: dtype-check. Извлекаем литерал из condition и проверяем
        # его на совместимость с dtype выбранной колонки. Если не подходит —
        # фильтр уходит в unresolved_filters, condition в SQL не попадает.
        literal_value = _extract_condition_literal(best_condition)
        if literal_value is not None and best_column and best_table_key:
            compatible, dtype = _check_filter_dtype_compatibility(
                column=best_column,
                table_key=best_table_key,
                value=literal_value,
                schema_loader=schema_loader,
            )
            if not compatible:
                logger.info(
                    "WhereResolver: пропущен ranked-фильтр %s (%s) — dtype=%s vs value=%r",
                    best_column, best_table_key, dtype, literal_value,
                )
                unresolved_filters.append({
                    "target": str(best.get("target") or best_column),
                    "value": literal_value,
                    "candidate_column": best_column,
                    "candidate_table": best_table_key,
                    "candidate_dtype": dtype,
                    "reason": "dtype_mismatch",
                    "source": f"ranked:{request_id}",
                })
                reasoning.append(
                    f"dtype_mismatch:{request_id}:{best_column} ({dtype})"
                )
                continue
        _add_unique(conditions, best_condition)
        applied_rules.append(str(request_id))
        rule_prefix = str(request_id).split(":", 1)[0]
        if best_table_key and rule_prefix in ("explicit", "query_spec"):
            explicit_applied_table_keys.add(best_table_key)
        evidence = ", ".join(best.get("evidence", []) or [])
        reasoning.append(
            f"{best.get('table_key')}: {best.get('column')} -> {best.get('condition')}"
            + (f" ({evidence})" if evidence else "")
        )

    filter_intents = list((effective_semantic_frame or {}).get("filter_intents") or [])
    all_rejected_request_ids = {
        str(reason).split(":", 1)[1]
        for reason in reasoning
        if str(reason).startswith("rejected_all:")
    }
    if (
        filter_intents
        and not applied_rules
        and not clarification_message
        and len(all_rejected_request_ids) < len(filter_intents)
    ):
        # G1: short-circuit when an intent's value is already encoded in the
        # chosen table (e.g. "фактический отток" + uzp_dwh_fact_outflow) AND
        # the intent's lexicon binding pointed at a *different* table. The
        # second condition mirrors _business_event_is_already_encoded_in_table:
        # if the filter_intent's column_key is on a selected table, the user
        # still needs candidates from that table — we must not silently fold
        # the filter away.
        selected_lower = {str(t).strip().lower() for t in (selected_tables or []) if t}
        table_encoded_request_ids: set[str] = set()
        # H1: only text-style intents are eligible for the table-encoding fold.
        # boolean_true / flag intents are structural (is_X = true), and
        # explicit_filter intents pin a user-specified column/value — neither
        # can be «semantically covered» by the table choice. Folding them
        # silently drops a filter the user asked for.
        _G1_ELIGIBLE_KINDS = {"text_search", "phrase_filter", ""}
        for intent in filter_intents:
            req_id = str(intent.get("request_id") or "")
            if not req_id or req_id in all_rejected_request_ids:
                continue
            kind = str(intent.get("kind") or "").strip().lower()
            if kind not in _G1_ELIGIBLE_KINDS:
                continue
            value = intent.get("value") or intent.get("query_text")
            if value in (None, ""):
                continue
            intent_column_key = str(intent.get("column_key") or "").strip().lower()
            intent_table_key = ".".join(intent_column_key.split(".")[:2]) if intent_column_key else ""
            if intent_table_key and intent_table_key in selected_lower:
                continue
            covered_by: str | None = None
            for table_key in selected_tables or []:
                if _filter_value_encoded_in_table(
                    schema_loader=schema_loader,
                    table_key=table_key,
                    value=value,
                ):
                    covered_by = table_key
                    break
            if covered_by:
                implicit_filters.append({
                    "target": intent.get("column_hint") or intent.get("query_text"),
                    "value": value,
                    "applied_via": "table_name_encoding",
                    "table": covered_by,
                    "source": f"fallback:{req_id}",
                })
                reasoning.append(f"table_context_covers_value:{covered_by}:{value}")
                logger.info(
                    "WhereResolver: F2 short-circuit — value=%r already encoded "
                    "in table %s; folding into implicit_filters",
                    value, covered_by,
                )
                table_encoded_request_ids.add(req_id)

        # Find the first unbound, non-table-encoded intent. We surface the
        # user's value verbatim + the top candidate columns; the generic
        # «уточните признак» wording (without choices) is the regression we
        # fix here.
        offered_intent: dict[str, Any] | None = None
        offered_candidates: list[dict[str, Any]] = []
        _PLAUSIBLE_CONFIDENCE = {"medium", "high"}
        _IMPLAUSIBLE_SEMANTIC_CLASSES = {"identifier", "join_key"}
        _MIN_OFFER_SCORE = 30.0
        for intent in filter_intents:
            req_id = str(intent.get("request_id") or "")
            if req_id in all_rejected_request_ids:
                continue
            if req_id in table_encoded_request_ids:
                continue
            # G2: keep only plausible candidates — drop low-confidence noise,
            # identifier-class columns (login/fio/id can't hold values like
            # "фактический отток"), and weak-score matches.
            intent_candidates = [
                cand for cand in (ranked_candidates.get(req_id) or [])
                if str(cand.get("confidence") or "").lower() in _PLAUSIBLE_CONFIDENCE
                and str(cand.get("semantic_class") or "").lower() not in _IMPLAUSIBLE_SEMANTIC_CLASSES
                and float(cand.get("score") or 0.0) >= _MIN_OFFER_SCORE
            ]
            if intent_candidates:
                offered_intent = intent
                offered_candidates = intent_candidates[:3]
                break

        # If G1 covered every unbound intent, nothing to clarify — return
        # silently (no message, no spec).
        if (
            offered_intent is None
            and table_encoded_request_ids
            and len(table_encoded_request_ids) + len(all_rejected_request_ids)
                >= len(filter_intents)
        ):
            pass  # all intents accounted for; skip clarification entirely
        elif offered_intent is not None and offered_candidates:
            value_text = str(
                offered_intent.get("value")
                or offered_intent.get("query_text")
                or offered_intent.get("column_hint")
                or ""
            ).strip()
            options = [
                {
                    "column": str(cand.get("column") or ""),
                    "label": candidate_label(cand),
                }
                for cand in offered_candidates
                if cand.get("column")
            ]
            labels = " или ".join(opt["label"] for opt in options)
            value_clause = f" для значения «{value_text}»" if value_text else ""
            clarification_message = (
                f"Нашёл несколько подходящих колонок{value_clause}: {labels}. "
                "Уточните, пожалуйста, по какой фильтровать."
            )
            clarification_spec = {
                "type": "choice",
                "request_id": str(offered_intent.get("request_id") or ""),
                "message": clarification_message,
                "options": options,
            }
            reasoning.append(
                f"fallback_clarify:{offered_intent.get('request_id')}:"
                + ",".join(opt["column"] for opt in options)
            )
            logger.info(
                "WhereResolver: fallback clarification — value=%r, candidates=%s",
                value_text,
                [opt["column"] for opt in options],
            )
        else:
            clarification_message = (
                "Нашёл несколько возможных способов применить фильтр, "
                "но уверенности недостаточно. Уточните, пожалуйста, желаемый признак или значение."
            )
            logger.info(
                "WhereResolver: не удалось привязать filter_intents автоматически "
                "(ranked_candidates=%s, intents=%s)",
                {k: [c.get('column') for c in v[:3]] for k, v in ranked_candidates.items()},
                [
                    {"request_id": i.get("request_id"), "value": i.get("value")}
                    for i in filter_intents
                ],
            )

    if (
        clarification_message
        and not applied_rules
        and not direct_filter_specs
        and _business_event_is_already_encoded_in_table(
            schema_loader=schema_loader,
            selected_tables=selected_tables,
            semantic_frame=effective_semantic_frame,
        )
    ):
        clarification_message = ""
        clarification_spec = {}
        reasoning.append("table_context_covers_business_event")
        logger.info(
            "WhereResolver: suppress clarification — business_event уже покрыт выбранной таблицей %s",
            selected_tables[0] if selected_tables else "",
        )

    if (
        clarification_message
        and not applied_rules
        and not direct_filter_specs
        and _business_event_is_aggregate_metric(
            schema_loader=schema_loader,
            selected_columns=selected_columns,
            semantic_frame=effective_semantic_frame,
        )
    ):
        clarification_message = ""
        clarification_spec = {}
        reasoning.append("aggregate_metric_covers_business_event")
        logger.info(
            "WhereResolver: suppress clarification — business_event покрыт aggregate-метрикой"
        )

    conditions = _drop_system_timestamp_when_time_axis_present(
        conditions, selected_columns, schema_loader=schema_loader,
    )

    # Fix H: значения unresolved_filters, уже семантически закодированные
    # в имени/описании выбранной таблицы (например "фактический отток" +
    # uzp_dwh_fact_outflow), переезжают в implicit_filters. Summarizer
    # упомянет их в финальном ответе ("учтён в таблице X").
    # (G1 may already have populated implicit_filters from the F2 short-circuit.)
    if unresolved_filters and selected_tables:
        survivors: list[dict[str, Any]] = []
        for item in unresolved_filters:
            primary_table = (
                str(item.get("candidate_table") or "")
                or (selected_tables[0] if selected_tables else "")
            )
            value = item.get("value")
            covered = False
            tables_to_check = (
                [primary_table] if primary_table else list(selected_tables)
            )
            for tk in tables_to_check:
                if _filter_value_encoded_in_table(
                    schema_loader=schema_loader,
                    table_key=tk,
                    value=value,
                ):
                    implicit_filters.append({
                        "target": item.get("target"),
                        "value": value,
                        "applied_via": "table_name_encoding",
                        "table": tk,
                    })
                    reasoning.append(
                        f"implicit_filter:{tk}:{value}"
                    )
                    covered = True
                    break
            if not covered:
                survivors.append(item)
        unresolved_filters = survivors

    logger.info(
        "WhereResolver: filter_intents=%d, applied_rules=%s, conditions=%d, "
        "unresolved=%d, implicit=%d, candidates=%s",
        len(filter_intents),
        applied_rules,
        len(conditions),
        len(unresolved_filters),
        len(implicit_filters),
        summarize_dict_keys(
            {k: [c.get('column') for c in v[:3]] for k, v in ranked_candidates.items()},
            label="candidates",
        ),
    )
    return {
        "conditions": conditions,
        "applied_rules": applied_rules,
        "reasoning": reasoning,
        "filter_candidates": ranked_candidates,
        "needs_clarification": bool(clarification_message),
        "clarification_message": clarification_message,
        "clarification_spec": clarification_spec,
        "user_filter_choices": user_filter_choices,
        "rejected_filter_choices": rejected_filter_choices,
        "unresolved_filters": unresolved_filters,
        "implicit_filters": implicit_filters,
    }


def _normalize_filter_specs(raw: list[FilterSpec | dict[str, Any]] | None) -> list[FilterSpec]:
    specs: list[FilterSpec] = []
    for item in raw or []:
        if isinstance(item, FilterSpec):
            specs.append(item)
            continue
        if isinstance(item, dict):
            try:
                specs.append(FilterSpec.model_validate(item))
            except Exception:  # noqa: BLE001
                continue
    return specs


def _apply_calendar_filter_specs(
    conditions: list[str],
    selected_columns: dict[str, dict[str, Any]],
    filter_specs: list[FilterSpec],
    *,
    schema_loader=None,
) -> tuple[list[str], str]:
    applied: list[str] = []
    for idx, spec in enumerate(filter_specs):
        parsed = _parse_calendar_period(spec.value)
        if not parsed or not _target_looks_calendar(spec.target):
            continue
        date_col = _find_time_axis_column(selected_columns, schema_loader=schema_loader)
        if not date_col:
            return applied, (
                "Фильтр задан календарным периодом, но в выбранных таблицах не найдена "
                "подходящая date/time-axis колонка."
            )
        start, end, _grain = parsed
        _add_unique(conditions, f"{date_col} >= '{start}'::date")
        _add_unique(conditions, f"{date_col} < '{end}'::date")
        applied.append(f"query_spec:{idx}")
    return applied, ""


def _find_time_axis_column(
    selected_columns: dict[str, dict[str, Any]],
    *,
    schema_loader=None,
) -> str | None:
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for table_key, roles in (selected_columns or {}).items():
        if "." not in table_key:
            continue
        schema, table = table_key.split(".", 1)
        table_sem = schema_loader.get_table_semantics(schema, table) if schema_loader is not None else {}
        time_axis = [str(v).strip().lower() for v in (table_sem.get("time_axis_columns") or []) if str(v).strip()]
        columns = []
        for role in ("filter", "select", "group_by", "aggregate"):
            columns.extend(
                str(col or "").strip()
                for col in roles.get(role, []) or []
                if str(col or "").strip() and str(col or "").strip() != "*"
            )
        if schema_loader is not None:
            try:
                cols_df = schema_loader.get_table_columns(schema, table)
                for _, row in cols_df.iterrows():
                    col = str(row.get("column_name") or "").strip()
                    if col and col not in columns:
                        columns.append(col)
            except Exception:  # noqa: BLE001
                pass
        for col in columns:
            if not col or col == "*" or col.lower() in seen:
                continue
            priority = _time_axis_priority(schema_loader, schema, table, col, time_axis)
            if priority is None:
                continue
            seen.add(col.lower())
            ranked.append((priority, col))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return ranked[0][1]


def _time_axis_priority(
    schema_loader,
    schema: str,
    table: str,
    col: str,
    time_axis: list[str],
) -> int | None:
    name = col.lower()
    dtype = ""
    semantic_class = ""
    tags: set[str] = set()
    if schema_loader is not None:
        try:
            dtype = str(schema_loader.get_column_dtype(schema, table, col) or "").lower()
        except Exception:  # noqa: BLE001
            dtype = ""
        sem = schema_loader.get_column_semantics(schema, table, col)
        semantic_class = str(sem.get("semantic_class") or "").lower()
        tags = {str(v).lower() for v in (sem.get("semantic_tags") or [])}
    looks_date = (
        semantic_class == "date"
        or "time_axis" in tags
        or dtype.startswith(("date", "timestamp"))
        or name.endswith(("_dt", "_date", "_dttm", "_timestamp", "_ts"))
        or name in {"date", "dttm"}
    )
    if not looks_date:
        return None
    if name in time_axis and name in {"report_dt", "report_date"}:
        return 0
    if name in time_axis or "time_axis" in tags:
        return 1
    if name in {"report_dt", "report_date"}:
        return 2
    if name.startswith(("inserted_", "updated_", "modified_", "created_", "load_", "etl_")):
        return 8
    if dtype.startswith("date") or name.endswith(("_dt", "_date")):
        return 3
    if dtype.startswith("timestamp") or name.endswith(("_dttm", "_timestamp", "_ts")):
        return 5
    return 9


def _filter_specs_for_semantic_frame(
    selected_columns: dict[str, dict[str, Any]],
    filter_specs: list[FilterSpec],
    *,
    schema_loader=None,
    time_range: dict[str, Any] | None = None,
) -> list[FilterSpec]:
    specs: list[FilterSpec] = []
    for spec in filter_specs:
        column_match = _find_selected_column(selected_columns, spec.target)
        if column_match and _should_skip_exact_calendar_date_filter(
            column=column_match[0],
            table_key=column_match[1],
            spec=spec,
            schema_loader=schema_loader,
            time_range=time_range,
        ):
            continue
        specs.append(spec)
    return specs


def _semantic_frame_without_redundant_calendar_filters(
    semantic_frame: dict[str, Any] | None,
    selected_columns: dict[str, dict[str, Any]],
    *,
    schema_loader=None,
    time_range: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not semantic_frame or not time_range:
        return semantic_frame
    frame = dict(semantic_frame)
    filtered = []
    changed = False
    for item in list(frame.get("filter_intents") or []):
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        spec = FilterSpec(
            target=str(item.get("column_hint") or item.get("target") or ""),
            operator=str(item.get("operator") or "="),
            value=item.get("value"),
        )
        column_match = _find_selected_column(selected_columns, spec.target)
        if column_match and _should_skip_exact_calendar_date_filter(
            column=column_match[0],
            table_key=column_match[1],
            spec=spec,
            schema_loader=schema_loader,
            time_range=time_range,
        ):
            changed = True
            continue
        filtered.append(item)
    if not changed:
        return semantic_frame
    frame["filter_intents"] = filtered
    return frame


def _intent_without_redundant_calendar_filters(
    intent: dict[str, Any],
    selected_columns: dict[str, dict[str, Any]],
    *,
    schema_loader=None,
    time_range: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not time_range:
        return intent
    cleaned = []
    changed = False
    for item in (intent or {}).get("filter_conditions", []) or []:
        if not isinstance(item, dict):
            cleaned.append(item)
            continue
        spec = FilterSpec(
            target=str(item.get("column_hint") or ""),
            operator=str(item.get("operator") or "="),
            value=item.get("value"),
        )
        column_match = _find_selected_column(selected_columns, spec.target)
        if column_match and _should_skip_exact_calendar_date_filter(
            column=column_match[0],
            table_key=column_match[1],
            spec=spec,
            schema_loader=schema_loader,
            time_range=time_range,
        ):
            changed = True
            continue
        cleaned.append(item)
    if not changed:
        return intent
    return {**(intent or {}), "filter_conditions": cleaned}


def _find_selected_column(
    selected_columns: dict[str, dict[str, Any]],
    target: str | None,
) -> tuple[str, str] | None:
    target_norm = str(target or "").strip().lower()
    if not target_norm:
        return None
    for table_key, roles in selected_columns.items():
        for role in ("filter", "select", "group_by", "aggregate"):
            for col in roles.get(role, []) or []:
                col_str = str(col or "").strip()
                if col_str and col_str.lower() == target_norm:
                    return col_str, str(table_key)
    return None


def _semantic_frame_with_filter_specs(
    semantic_frame: dict[str, Any] | None,
    filter_specs: list[FilterSpec],
) -> dict[str, Any] | None:
    if not filter_specs:
        return semantic_frame
    frame = dict(semantic_frame or {})
    existing = list(frame.get("filter_intents") or [])
    additions = [
        {
            "request_id": f"query_spec:{idx}",
            "kind": "query_spec_filter",
            "query_text": str(spec.value if spec.value is not None else spec.target),
            "column_hint": spec.target,
            "operator": spec.operator,
            "value": spec.value,
            "match_score": spec.confidence,
            "match_source": "query_spec",
        }
        for idx, spec in enumerate(filter_specs)
    ]
    seen = {str(item.get("request_id") or "") for item in existing}
    frame["filter_intents"] = existing + [
        item for item in additions if str(item.get("request_id") or "") not in seen
    ]
    if filter_specs and not frame.get("business_event"):
        frame["business_event"] = filter_specs[0].target
    return frame


def _apply_exact_filter_specs(
    conditions: list[str],
    selected_columns: dict[str, dict[str, Any]],
    filter_specs: list[FilterSpec],
    *,
    schema_loader=None,
    time_range: dict[str, Any] | None = None,
    unresolved: list[dict[str, Any]] | None = None,
) -> list[str]:
    applied: list[str] = []
    if not filter_specs:
        return applied
    known_columns: dict[str, tuple[str, str]] = {}
    for table_key, roles in selected_columns.items():
        for role in ("filter", "select", "group_by", "aggregate"):
            for col in roles.get(role, []) or []:
                col_str = str(col or "").strip()
                if col_str and col_str != "*":
                    known_columns.setdefault(col_str.lower(), (col_str, str(table_key)))
    for idx, spec in enumerate(filter_specs):
        column_match = known_columns.get(str(spec.target or "").strip().lower())
        if not column_match:
            continue
        column, table_key = column_match
        if _should_skip_exact_calendar_date_filter(
            column=column,
            table_key=table_key,
            spec=spec,
            schema_loader=schema_loader,
            time_range=time_range,
        ):
            continue
        # Direction: dtype-check. Не выпускаем условие, у которого значение
        # несовместимо с типом выбранной колонки (например boolean = 'text').
        compatible, dtype = _check_filter_dtype_compatibility(
            column=column,
            table_key=table_key,
            value=spec.value,
            schema_loader=schema_loader,
        )
        if not compatible:
            logger.info(
                "WhereResolver: пропущен exact-фильтр %s = %r — dtype=%s несовместим",
                column, spec.value, dtype,
            )
            if unresolved is not None:
                unresolved.append({
                    "target": str(spec.target or ""),
                    "value": spec.value,
                    "operator": str(spec.operator or "="),
                    "candidate_column": column,
                    "candidate_table": table_key,
                    "candidate_dtype": dtype,
                    "reason": "dtype_mismatch",
                    "source": f"query_spec:{idx}",
                })
            continue
        condition = _condition_from_filter_spec(column, spec)
        if condition:
            _add_unique(conditions, condition)
            applied.append(f"query_spec:{idx}")
    return applied


def _drop_system_timestamp_when_time_axis_present(
    conditions: list[str],
    selected_columns: dict[str, dict[str, Any]],
    *,
    schema_loader=None,
) -> list[str]:
    """Удалить условия на system_timestamp-колонки, если есть эквивалентные
    условия на time_axis-колонках (semantic_class='date').

    Это устраняет дубль `inserted_dttm >= '2026-02-01'` и `report_dt >= '2026-02-01'`,
    когда LLM сгенерировал филтры по обеим осям. Базис истины — column_semantics:
    semantic_class='system_timestamp' проигрывает любой date-колонке с тем же
    оператором и значением.
    """
    if schema_loader is None or not conditions:
        return conditions

    column_to_table: dict[str, tuple[str, str]] = {}
    for table_key, roles in (selected_columns or {}).items():
        if "." not in table_key:
            continue
        schema, table = table_key.split(".", 1)
        for role in ("filter", "select", "group_by", "aggregate"):
            for col in roles.get(role, []) or []:
                col_str = str(col or "").strip()
                if col_str and col_str != "*":
                    column_to_table.setdefault(col_str.lower(), (schema, table))

    parsed_per_cond: list[tuple[str, str, str] | None] = [
        _parse_condition(cond) for cond in conditions
    ]

    def _semantic_class(column: str) -> str:
        binding = column_to_table.get(column.lower())
        if not binding:
            return ""
        schema, table = binding
        try:
            sem = schema_loader.get_column_semantics(schema, table, column) or {}
        except Exception:  # noqa: BLE001
            sem = {}
        return str(sem.get("semantic_class") or "").lower()

    # Группируем по (op, value_norm) и помечаем колонки в группе.
    groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for idx, parsed in enumerate(parsed_per_cond):
        if parsed is None:
            continue
        col, op, value = parsed
        groups.setdefault((op, value), []).append((idx, col))

    drop_indices: set[int] = set()
    for (_, _), members in groups.items():
        if len(members) < 2:
            continue
        date_axis_present = any(
            _semantic_class(col) == "date" for _, col in members
        )
        if not date_axis_present:
            continue
        for idx, col in members:
            if _semantic_class(col) == "system_timestamp":
                drop_indices.add(idx)
                logger.info(
                    "WhereResolver: drop system_timestamp condition #%d on %s "
                    "— покрыто time_axis колонкой",
                    idx, col,
                )

    if not drop_indices:
        return conditions
    return [cond for idx, cond in enumerate(conditions) if idx not in drop_indices]


def _should_skip_exact_calendar_date_filter(
    *,
    column: str,
    table_key: str,
    spec: FilterSpec,
    schema_loader=None,
    time_range: dict[str, Any] | None = None,
) -> bool:
    if not time_range or not time_range.get("start") or not time_range.get("end"):
        return False
    if str(spec.operator or "=").strip().upper() not in {"=", "=="}:
        return False
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(spec.value or "")):
        return False
    col_lower = str(column or "").lower()
    date_like = any(
        col_lower.endswith(suffix)
        for suffix in ("_dt", "_date", "_dttm", "_timestamp", "_ts", "date", "dttm")
    ) or "date" in col_lower or "dttm" in col_lower or "timestamp" in col_lower
    if not date_like:
        return False

    if schema_loader is not None and "." in table_key:
        schema, table = table_key.split(".", 1)
        semantics = schema_loader.get_column_semantics(schema, table, column)
        semantic_class = str(semantics.get("semantic_class") or "").lower()
        if semantic_class == "system_timestamp":
            return True

    # A calendar period already became a range predicate. A second equality on
    # a date column usually comes from over-binding the natural-language date
    # to an arbitrary physical column and would narrow a month to one day.
    return True


def _condition_from_filter_spec(column: str, spec: FilterSpec) -> str:
    operator = str(spec.operator or "=").strip().upper()
    if operator not in {"=", "!=", "<>", "<", ">", "<=", ">=", "LIKE", "ILIKE", "IN", "NOT IN"}:
        operator = "="
    value = spec.value
    if value is None:
        return f"{column} IS NULL" if operator in {"=", "IS"} else f"{column} IS NOT NULL"
    if operator in {"IN", "NOT IN"}:
        values = value if isinstance(value, list) else [item.strip() for item in str(value).split(",")]
        rendered = ", ".join(_render_literal(item) for item in values)
        return f"{column} {operator} ({rendered})"
    if operator in {"LIKE", "ILIKE"}:
        return f"{column} {operator} {_render_literal(str(value))}"
    return f"{column} {operator} {_render_literal(value)}"


def _render_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"
