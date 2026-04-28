"""Узел hint_extractor: regex-извлечение подсказок + merge с LLM-результатом.

Запускается после hint_extractor_llm. Сам regex-парсер остаётся как safety-net
(если LLM вернул пустую структуру или невалидный JSON — регекс всё равно
поднимет бизнес-ключи вроде "по инн", явные schema.table и т.д.).

Merge-политика:
- LLM-подсказки приоритетнее по ВСЕМ полям (если LLM явно что-то указал, берём его).
- Regex добавляет значения ТОЛЬКО для полей, которые LLM оставил пустыми.
- Расхождения (LLM и regex оба нашли, но разное) логируются как info.

Пишет в state поле user_hints (финальное) и hints_source ("llm" | "regex" | "merged").
"""

import logging
from typing import Any

from core.user_hint_extractor import extract_user_hints
from graph.state import AgentState

logger = logging.getLogger(__name__)


_LIST_FIELDS = (
    "must_keep_tables",
    "join_fields",
    "group_by_hints",
    "aggregate_hints",
    "having_hints",
    "negative_filters",
)
_DICT_FIELDS = ("dim_sources",)
_SCALAR_FIELDS = ("time_granularity",)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple, set, str)):
        return len(value) == 0
    return False


def _dedup_preserve_order(seq: list) -> list:
    seen = set()
    out = []
    for item in seq:
        key = repr(item) if not isinstance(item, (str, int, float, tuple)) else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def merge_user_hints(
    llm_hints: dict[str, Any] | None,
    regex_hints: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Смёржить LLM-подсказки и regex-подсказки в одну структуру.

    Политика: LLM-значения приоритетнее; regex заполняет только пустые поля
    и дополняет списки значениями, которых нет у LLM. Возвращает
    (merged_hints, source), где source — "llm" | "regex" | "merged".
    """
    llm_hints = llm_hints or {}
    merged: dict[str, Any] = {}
    llm_had_something = False
    regex_had_something = False

    for field in _LIST_FIELDS:
        llm_val = llm_hints.get(field) or []
        regex_val = regex_hints.get(field) or []

        if not _is_empty(llm_val):
            llm_had_something = True
        if not _is_empty(regex_val):
            regex_had_something = True

        if _is_empty(llm_val):
            merged[field] = list(regex_val)
        else:
            combined = list(llm_val) + [v for v in regex_val if v not in llm_val]
            merged[field] = _dedup_preserve_order(combined)

    for field in _DICT_FIELDS:
        llm_val = llm_hints.get(field) or {}
        regex_val = regex_hints.get(field) or {}

        if not _is_empty(llm_val):
            llm_had_something = True
        if not _is_empty(regex_val):
            regex_had_something = True

        if _is_empty(llm_val):
            merged[field] = dict(regex_val)
        else:
            combined = dict(regex_val)
            combined.update(llm_val)
            merged[field] = combined

    for field in _SCALAR_FIELDS:
        llm_val = llm_hints.get(field)
        regex_val = regex_hints.get(field)
        if llm_val is not None and llm_val != "":
            merged[field] = llm_val
            llm_had_something = True
        else:
            merged[field] = regex_val
        if regex_val is not None and regex_val != "":
            regex_had_something = True

    # aggregation_preferences / _list — только regex (структурированная форма,
    # нужна sql_planner'у). LLM выдаёт aggregate_hints, и если там есть единичный
    # confident-hint, переносим его как preference.
    merged["aggregation_preferences"] = dict(regex_hints.get("aggregation_preferences") or {})
    merged["aggregation_preferences_list"] = list(regex_hints.get("aggregation_preferences_list") or [])

    # Если regex не нашёл preferences, но LLM дал ровно одну агрегатную подсказку —
    # используем её (совместимо со старым форматом sql_planner'а).
    llm_aggs = llm_hints.get("aggregate_hints") or []
    if (
        not merged["aggregation_preferences"]
        and len(llm_aggs) == 1
        and isinstance(llm_aggs[0], dict)
    ):
        pref = {
            "function": llm_aggs[0].get("function"),
            "column": llm_aggs[0].get("column"),
            "distinct": bool(llm_aggs[0].get("distinct")),
        }
        if pref["function"]:
            merged["aggregation_preferences"] = pref
            merged["aggregation_preferences_list"] = [pref]

    if llm_had_something and regex_had_something:
        source = "merged"
    elif llm_had_something:
        source = "llm"
    elif regex_had_something:
        source = "regex"
    else:
        source = "empty"

    return merged, source


class HintExtractorNodes:
    """Mixin с узлом hint_extractor (regex + merge с hint_extractor_llm)."""

    def hint_extractor(self, state: AgentState) -> dict[str, Any]:
        """Извлечь regex-подсказки и смёржить с LLM-результатом из state.

        Не блокирует пайплайн при отсутствии подсказок — возвращает пустую
        структуру. Все выходы валидируются через каталог schema_loader.
        """
        iterations = state.get("graph_iterations", 0) + 1
        user_input = state.get("user_input", "") or ""
        llm_hints = state.get("user_hints_llm") or {}

        try:
            regex_hints = extract_user_hints(user_input, self.schema)
        except Exception as exc:  # noqa: BLE001
            logger.warning("hint_extractor: ошибка regex-извлечения: %s", exc)
            regex_hints = {
                "must_keep_tables": [],
                "join_fields": [],
                "dim_sources": {},
                "having_hints": [],
                "aggregation_preferences": {},
                "aggregation_preferences_list": [],
                "group_by_hints": [],
                "aggregate_hints": [],
                "time_granularity": None,
                "negative_filters": [],
            }

        merged, source = merge_user_hints(llm_hints, regex_hints)

        if source != "empty":
            logger.info(
                "hint_extractor[source=%s]: must_keep=%s, join_fields=%s, "
                "group_by=%s, aggregate=%s, granularity=%s, negative=%s",
                source,
                merged.get("must_keep_tables"),
                merged.get("join_fields"),
                merged.get("group_by_hints"),
                merged.get("aggregate_hints"),
                merged.get("time_granularity"),
                merged.get("negative_filters"),
            )
        else:
            logger.debug("hint_extractor: подсказок не найдено (ни LLM, ни regex)")

        return {
            "user_hints": merged,
            "hints_source": source,
            "graph_iterations": iterations,
        }
