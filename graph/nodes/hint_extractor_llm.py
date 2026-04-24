"""Узел hint_extractor_llm: LLM-извлечение структурированных подсказок пользователя.

Запускается между intent_classifier и hint_extractor. Возвращает структурированный
JSON с теми же полями, что и регекс-парсер user_hint_extractor, но ловит
вариативные формулировки («в разбивке по», «сколько уникальных», «подтяни»,
«исключи»), которые regex пропускает.

Результат попадает в state["user_hints_llm"]. Merge с regex-результатом
выполняется в существующей ноде hint_extractor (вызывается следом).

Промпт компактный (~70 строк), фокус ТОЛЬКО на структурных якорях —
отдельная семантическая задача, не смешивается с intent_classifier.
"""

from __future__ import annotations

import logging
from typing import Any

from graph.state import AgentState

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "Ты — экстрактор структурных подсказок из аналитического запроса на русском языке. "
    "Твоя задача — найти в запросе пользователя явные указания на таблицы, "
    "ключи JOIN, группировки, агрегаты, гранулярность времени и исключения, "
    "и вернуть их в виде JSON строго по схеме.\n\n"
    "Схема ответа:\n"
    "{\n"
    '  "must_keep_tables": [["schema", "table"], ...],\n'
    '  "join_fields": ["inn", "customer_id", ...],\n'
    '  "dim_sources": {"dimension_key": {"table": "schema.table", "join_col": "col"}},\n'
    '  "having_hints": [{"op": ">=|>|<=|<|=", "value": <number>, "unit_hint": "человек"}],\n'
    '  "group_by_hints": ["column_or_dimension", ...],\n'
    '  "aggregate_hints": [{"function": "count|sum|avg|min|max|list", "column": "col_or_null", "distinct": true|false}],\n'
    '  "time_granularity": "day|week|month|quarter|year" или null,\n'
    '  "negative_filters": ["значение", ...]\n'
    "}\n\n"
    "Правила:\n"
    "- Возвращай ТОЛЬКО JSON-объект, без markdown, без пояснений.\n"
    "- Если поле не выведено из запроса — ставь пустой массив/объект/null (не выдумывай).\n"
    "- must_keep_tables: только если пользователь ЯВНО указал имя таблицы "
    "(«возьми из schema.table», «дотяни из T», «в таблице X»). Имена копируй как есть.\n"
    "- join_fields: берёшь бизнес-ключи из формулировок «по инн», «через customer_id», "
    "«связать через X», «ключ X», «using X». Общие группировки («по региону») — "
    "в group_by_hints, а не сюда.\n"
    "- group_by_hints: «сгруппируй по X», «в разбивке по X», «по X» (когда это ось "
    "отчёта, не JOIN), «распредели по X». Имя оставляй в том виде, в котором "
    "пользователь назвал (регион / region / сегмент).\n"
    "- aggregate_hints: «посчитай», «количество», «сколько», «сумма», «средний», "
    "«минимальный», «максимальный», «список». distinct=true если явно сказано "
    "«уникальных», «distinct», «различных».\n"
    "- time_granularity: «помесячно» → month, «по кварталам» → quarter, «за каждый "
    "день» → day, «понедельно» → week, «по годам» → year. Если не указано — null.\n"
    "- negative_filters: «исключи X», «не учитывай Y», «кроме Z», «без W». Значения.\n"
    "- having_hints: «от N человек», «больше N», «более чем N». op — сравнение.\n"
    "- dim_sources: «измерение X возьми из T по K» — редкая конструкция, обычно пусто.\n"
)


_FEW_SHOTS = (
    "Примеры:\n\n"
    'Запрос: "сколько уникальных клиентов помесячно в разбивке по регионам"\n'
    "JSON: "
    '{"must_keep_tables": [], "join_fields": [], "dim_sources": {}, '
    '"having_hints": [], "group_by_hints": ["регион"], '
    '"aggregate_hints": [{"function": "count", "column": "клиент", "distinct": true}], '
    '"time_granularity": "month", "negative_filters": []}\n\n'
    'Запрос: "подтяни отток из таблицы dm.outflow_fact за последний квартал"\n'
    "JSON: "
    '{"must_keep_tables": [["dm", "outflow_fact"]], "join_fields": [], '
    '"dim_sources": {}, "having_hints": [], "group_by_hints": [], '
    '"aggregate_hints": [], "time_granularity": "quarter", "negative_filters": []}\n\n'
    'Запрос: "посчитай клиентов по ИНН, исключи отменённые заказы"\n'
    "JSON: "
    '{"must_keep_tables": [], "join_fields": ["инн"], "dim_sources": {}, '
    '"having_hints": [], "group_by_hints": [], '
    '"aggregate_hints": [{"function": "count", "column": "клиент", "distinct": false}], '
    '"time_granularity": null, "negative_filters": ["отменённые"]}\n\n'
    'Запрос: "покажи продажи по сегментам с объёмом от 100 заказов"\n'
    "JSON: "
    '{"must_keep_tables": [], "join_fields": [], "dim_sources": {}, '
    '"having_hints": [{"op": ">=", "value": 100, "unit_hint": "заказов"}], '
    '"group_by_hints": ["сегмент"], '
    '"aggregate_hints": [{"function": "sum", "column": "продажи", "distinct": false}], '
    '"time_granularity": null, "negative_filters": []}\n'
)


_EMPTY_HINTS: dict[str, Any] = {
    "must_keep_tables": [],
    "join_fields": [],
    "dim_sources": {},
    "having_hints": [],
    "group_by_hints": [],
    "aggregate_hints": [],
    "time_granularity": None,
    "negative_filters": [],
}


class HintExtractorLLMNodes:
    """Mixin с узлом hint_extractor_llm (LLM-экстрактор подсказок)."""

    def hint_extractor_llm(self, state: AgentState) -> dict[str, Any]:
        """Вызвать LLM и получить структурированные user_hints_llm.

        На невалидный JSON (даже после одного retry) возвращает пустую
        структуру — merge в следующей ноде hint_extractor всё равно
        подтянет regex-результат.
        """
        iterations = state.get("graph_iterations", 0) + 1
        user_input = state.get("user_input", "") or ""

        if not user_input.strip():
            return {
                "user_hints_llm": dict(_EMPTY_HINTS),
                "graph_iterations": iterations,
            }

        user_prompt = (
            _FEW_SHOTS
            + "\nЗапрос: "
            + user_input.strip()
            + "\nJSON:"
        )

        parsed = self._llm_json_with_retry(
            _SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
            failure_tag="hint_extractor_llm",
            expect="object",
        )

        hints = _normalize_llm_hints(parsed) if parsed else dict(_EMPTY_HINTS)

        if any(hints.get(k) for k in (
            "must_keep_tables", "join_fields", "group_by_hints",
            "aggregate_hints", "having_hints", "negative_filters",
        )) or hints.get("time_granularity"):
            logger.info(
                "hint_extractor_llm: must_keep=%s join_fields=%s group_by=%s "
                "aggregate=%s granularity=%s negative=%s",
                hints.get("must_keep_tables"),
                hints.get("join_fields"),
                hints.get("group_by_hints"),
                hints.get("aggregate_hints"),
                hints.get("time_granularity"),
                hints.get("negative_filters"),
            )
        else:
            logger.debug("hint_extractor_llm: LLM не нашёл подсказок")

        return {
            "user_hints_llm": hints,
            "graph_iterations": iterations,
        }


def _normalize_llm_hints(parsed: dict[str, Any]) -> dict[str, Any]:
    """Привести LLM-ответ к каноничной структуре, отбросив мусор."""
    result: dict[str, Any] = dict(_EMPTY_HINTS)

    raw_tables = parsed.get("must_keep_tables") or []
    tables: list[tuple[str, str]] = []
    if isinstance(raw_tables, list):
        for item in raw_tables:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                schema, table = str(item[0]).strip(), str(item[1]).strip()
                if schema and table:
                    tables.append((schema, table))
            elif isinstance(item, str) and "." in item:
                schema, table = item.split(".", 1)
                schema, table = schema.strip(), table.strip()
                if schema and table:
                    tables.append((schema, table))
    result["must_keep_tables"] = tables

    for key in ("join_fields", "group_by_hints", "negative_filters"):
        raw = parsed.get(key) or []
        if isinstance(raw, list):
            result[key] = [str(x).strip().lower() for x in raw if str(x).strip()]

    raw_dims = parsed.get("dim_sources") or {}
    if isinstance(raw_dims, dict):
        clean_dims: dict[str, dict[str, str]] = {}
        for dim, spec in raw_dims.items():
            if not isinstance(spec, dict):
                continue
            table = str(spec.get("table") or "").strip()
            join_col = str(spec.get("join_col") or "").strip()
            if dim and table and join_col:
                clean_dims[str(dim).strip().lower()] = {
                    "table": table,
                    "join_col": join_col.lower(),
                }
        result["dim_sources"] = clean_dims

    raw_having = parsed.get("having_hints") or []
    having: list[dict[str, Any]] = []
    if isinstance(raw_having, list):
        for item in raw_having:
            if not isinstance(item, dict):
                continue
            op = str(item.get("op") or "").strip()
            value = item.get("value")
            if op in {">=", ">", "<=", "<", "="} and isinstance(value, (int, float)):
                entry: dict[str, Any] = {"op": op, "value": value}
                unit = item.get("unit_hint")
                if isinstance(unit, str) and unit.strip():
                    entry["unit_hint"] = unit.strip()
                having.append(entry)
    result["having_hints"] = having

    raw_aggs = parsed.get("aggregate_hints") or []
    aggs: list[dict[str, Any]] = []
    if isinstance(raw_aggs, list):
        for item in raw_aggs:
            if not isinstance(item, dict):
                continue
            fn = str(item.get("function") or "").strip().lower()
            if fn not in {"count", "sum", "avg", "min", "max", "list"}:
                continue
            column = item.get("column")
            column = str(column).strip().lower() if column else None
            distinct = bool(item.get("distinct"))
            aggs.append({
                "function": fn,
                "column": column,
                "distinct": distinct,
            })
    result["aggregate_hints"] = aggs

    raw_gran = parsed.get("time_granularity")
    if isinstance(raw_gran, str):
        raw_gran_lower = raw_gran.strip().lower()
        if raw_gran_lower in {"day", "week", "month", "quarter", "year"}:
            result["time_granularity"] = raw_gran_lower

    return result
