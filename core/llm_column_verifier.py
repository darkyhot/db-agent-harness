"""LLM-верификатор детерминированного выбора колонок.

Запускается ПОСЛЕ детерминированного column_selector в high-confidence-полосе
(>= _DET_CONFIDENCE_THRESHOLD), чтобы поймать ошибки вида:
- для слота "название X" выбран идентификатор `_id` вместо `_name`,
- метрика-агрегация назначена на нечисловое или процентное поле,
- date-фильтр поставлен на технический timestamp вместо отчётной даты.

Архитектурный принцип: non-fatal. Любая ошибка LLM/парсинга → возвращаем
"проблем не найдено", чтобы pipeline не блокировался на флаки-вызовах.

Пример вердикта:
    {
        "issues": [
            {
                "slot": "gosb_name",
                "problem": "выбран идентификатор gosb_id вместо названия",
                "severity": "critical",
                "suggested_column": "dm.uzp_dim_gosb.gosb_name"
            }
        ],
        "should_force_fallback": True,
        "hint": "Слот 'gosb_name': предпочесть dm.uzp_dim_gosb.gosb_name вместо ..."
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Ты валидируешь выбор колонок для аналитического SQL-запроса.\n"
    "Тебе дают слоты запроса и фактически выбранную колонку плюс альтернативы из той же таблицы.\n"
    "Твоя задача — указать, если выбранная колонка очевидно не соответствует слоту по семантике.\n"
    "Типичные ошибки:\n"
    "- для слота вида 'название X' / 'имя X' выбрана колонка-идентификатор '*_id' / '*_code',\n"
    "  при этом среди альтернатив есть '*_name' / '*_label';\n"
    "- метрика SUM/AVG поставлена на процент/долю вместо абсолютной величины;\n"
    "- date-фильтр на техническом timestamp (inserted_/modified_/created_) вместо отчётной даты.\n"
    "НЕ УГАДЫВАЙ. Если у тебя нет уверенных оснований — вернуть пустой issues=[].\n"
    "Верни JSON строго формата:\n"
    '{"issues":[{"slot":"...","problem":"...","severity":"critical|warning",'
    '"suggested_column":"<schema.table.col>"}]}\n'
    "Если всё корректно: {\"issues\":[]}."
)

_MAX_DESC_LEN = 120
_MAX_ALTERNATIVES_PER_SLOT = 3
_EMPTY_VERDICT: dict[str, Any] = {"issues": [], "should_force_fallback": False, "hint": ""}


def _truncate_desc(text: str) -> str:
    s = (text or "").strip().replace("\n", " ")
    return s if len(s) <= _MAX_DESC_LEN else s[: _MAX_DESC_LEN - 1] + "…"


def _collect_chosen_columns(selected_columns: dict[str, dict]) -> set[tuple[str, str]]:
    """Все (table, col), которые попали хотя бы в одну роль."""
    out: set[tuple[str, str]] = set()
    for table_key, roles in selected_columns.items():
        for role in ("select", "aggregate", "group_by", "filter"):
            for col in roles.get(role, []) or []:
                out.add((table_key, col))
    return out


def _build_alternatives(
    schema_loader,
    table_key: str,
    chosen_col: str,
    slot: str,
    semantic_scorer: Callable[[str, str, str], float] | None,
    limit: int = _MAX_ALTERNATIVES_PER_SLOT,
) -> list[dict[str, Any]]:
    """Вернуть до `limit` колонок-кандидатов из той же таблицы, ранжированных
    по семантической близости к slot."""
    if "." not in table_key:
        return []
    schema, table = table_key.split(".", 1)
    try:
        cols_df = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        return []
    if cols_df is None or cols_df.empty:
        return []
    rows: list[tuple[float, dict[str, Any]]] = []
    for _, row in cols_df.iterrows():
        col = str(row.get("column_name") or "").strip()
        if not col or col == chosen_col:
            continue
        desc = str(row.get("description") or "")
        score = semantic_scorer(col, desc, slot) if semantic_scorer else 0.0
        rows.append(
            (
                score,
                {
                    "col": col,
                    "dtype": str(row.get("dType") or "").strip(),
                    "desc": _truncate_desc(desc),
                },
            )
        )
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows[:limit]]


def _slots_to_inspect(requested_slots: dict[str, Any]) -> list[str]:
    """Какие слоты подаём на проверку. Берём metric + dimensions; "date" исключаем
    из dimension-проверки (валидируется отдельно по filter-роли)."""
    slots: list[str] = []
    metric = (requested_slots or {}).get("metric")
    if metric:
        slots.append(str(metric))
    for dim in (requested_slots or {}).get("dimensions", []) or []:
        s = str(dim or "").strip()
        if not s or s in slots:
            continue
        slots.append(s)
    return slots


def _find_chosen_for_slot(
    slot: str,
    selected_columns: dict[str, dict],
    *,
    role_priority: tuple[str, ...] = ("group_by", "aggregate", "select", "filter"),
) -> tuple[str, str] | None:
    """Минимальная эвристика: выбираем первую колонку из приоритетных ролей,
    чьё имя содержит слот или наоборот. Если не нашли — берём первую попавшуюся
    из любой роли. Цель — дать LLM достаточно контекста, не претендуя на
    идеальное соответствие.
    """
    slot_l = slot.lower()
    for role in role_priority:
        for table_key, roles in selected_columns.items():
            for col in roles.get(role, []) or []:
                col_l = col.lower()
                if slot_l in col_l or col_l in slot_l:
                    return table_key, col
    # fallback: первая попавшаяся
    for role in role_priority:
        for table_key, roles in selected_columns.items():
            for col in roles.get(role, []) or []:
                return table_key, col
    return None


def _build_payload(
    *,
    user_input: str,
    requested_slots: dict[str, Any],
    selected_columns: dict[str, dict],
    schema_loader,
    semantic_scorer: Callable[[str, str, str], float] | None,
) -> dict[str, Any]:
    slot_entries: list[dict[str, Any]] = []
    for slot in _slots_to_inspect(requested_slots):
        chosen = _find_chosen_for_slot(slot, selected_columns)
        if not chosen:
            continue
        table_key, col = chosen
        try:
            schema, table = table_key.split(".", 1)
            cols_df = schema_loader.get_table_columns(schema, table)
            row = cols_df[cols_df["column_name"] == col]
            dtype = str(row.iloc[0].get("dType") or "") if not row.empty else ""
            desc = str(row.iloc[0].get("description") or "") if not row.empty else ""
        except Exception:  # noqa: BLE001
            dtype, desc = "", ""

        alternatives = _build_alternatives(
            schema_loader, table_key, col, slot, semantic_scorer
        )
        slot_entries.append(
            {
                "slot": slot,
                "chosen": {
                    "ref": f"{table_key}.{col}",
                    "dtype": dtype.strip(),
                    "desc": _truncate_desc(desc),
                },
                "alternatives": alternatives,
            }
        )

    return {"user_input": user_input, "slots": slot_entries}


def _post_process(verdict: Any) -> dict[str, Any]:
    if not isinstance(verdict, dict):
        return dict(_EMPTY_VERDICT)
    raw_issues = verdict.get("issues")
    if not isinstance(raw_issues, list):
        return dict(_EMPTY_VERDICT)

    cleaned: list[dict[str, Any]] = []
    hint_lines: list[str] = []
    should_force = False
    for issue in raw_issues:
        if not isinstance(issue, dict):
            continue
        slot = str(issue.get("slot") or "").strip()
        problem = str(issue.get("problem") or "").strip()
        severity = str(issue.get("severity") or "warning").strip().lower()
        suggested = str(issue.get("suggested_column") or "").strip()
        if severity not in ("critical", "warning"):
            severity = "warning"
        cleaned.append(
            {
                "slot": slot,
                "problem": problem,
                "severity": severity,
                "suggested_column": suggested,
            }
        )
        if severity == "critical":
            should_force = True
        if slot or suggested or problem:
            target = suggested or "<не указано>"
            hint_lines.append(
                f"Слот '{slot}': предпочесть {target} вместо выбранной — {problem}."
            )

    return {
        "issues": cleaned,
        "should_force_fallback": should_force,
        "hint": "\n".join(hint_lines).strip(),
    }


def verify_column_selection(
    *,
    user_input: str,
    requested_slots: dict[str, Any],
    selected_columns: dict[str, dict],
    schema_loader,
    llm_invoker,
    semantic_scorer: Callable[[str, str, str], float] | None = None,
    failure_tag: str = "column_verifier",
) -> dict[str, Any]:
    """Проверить детерминированный выбор колонок через LLM.

    Args:
        user_input: исходный текст запроса.
        requested_slots: результат `_derive_requested_slots` ({metric, dimensions, ...}).
        selected_columns: то, что вернул детерминированный селектор.
        schema_loader: для извлечения dtype/description колонок.
        llm_invoker: объект с методом `_llm_json_with_retry(...)` (т.е. BaseNodeMixin).
        semantic_scorer: опциональный callable(col_name, desc, slot) -> float
            для ранжирования альтернатив; обычно `_semantic_match_score`.
        failure_tag: тег для журналирования сбоев JSON-парсинга.

    Returns:
        {"issues": [...], "should_force_fallback": bool, "hint": str}.
        На любую ошибку — пустой verdict.
    """
    try:
        if not selected_columns:
            return dict(_EMPTY_VERDICT)
        payload = _build_payload(
            user_input=user_input,
            requested_slots=requested_slots or {},
            selected_columns=selected_columns,
            schema_loader=schema_loader,
            semantic_scorer=semantic_scorer,
        )
        if not payload["slots"]:
            return dict(_EMPTY_VERDICT)

        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        verdict = llm_invoker._llm_json_with_retry(
            _SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
            failure_tag=failure_tag,
            expect="object",
        )
        if verdict is None:
            logger.info("LLMColumnVerifier: LLM не вернул валидный JSON — пропускаем проверку")
            return dict(_EMPTY_VERDICT)
        result = _post_process(verdict)
        logger.info(
            "LLMColumnVerifier: %d issues, force_fallback=%s",
            len(result["issues"]), result["should_force_fallback"],
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLMColumnVerifier: исключение %s — пропускаем проверку", exc)
        return dict(_EMPTY_VERDICT)
