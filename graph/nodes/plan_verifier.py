"""Plan-level LLM-верификатор: проверяет соответствие готового плана запросу.

Запускается между sql_planner и plan_preview, только когда llm_verifier_enabled=True.
В отличие от прежнего column-verifier'а (graph/nodes/explorer.py), этот узел
видит ВЕСЬ план (выбранные колонки, агрегации, GROUP BY, фильтры, ORDER BY)
и может вернуть структурированные правки — список PlanEdit. Free-form rewrite
запрещён, любая правка валидируется по каталогу.

Если verifier одобрил план или вернул только невалидные правки — состояние
не меняется. Если хотя бы одна правка применилась — установлен флаг
`plan_verifier_applied=True`, и downstream sql_planner пересоберёт blueprint
из обновлённого selected_columns.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.plan_edit_applier import apply_plan_edits
from core.plan_verifier_models import PlanEdit, PlanVerdict, empty_verdict
from graph.state import AgentState
from pydantic import ValidationError

logger = logging.getLogger(__name__)

_DEFAULT_ALTERNATIVES = 5
_MAX_DESC = 120


_SYSTEM_PROMPT = (
    "Ты валидируешь готовый SQL-план аналитического агента.\n"
    "Тебе дают пользовательский запрос, итоговый план (главная таблица, "
    "выбранные колонки по ролям, агрегации, GROUP BY, фильтры, сортировка) "
    "и компактный каталог-контекст (только колонки из плана + 3-5 ближайших "
    "альтернатив из тех же таблиц).\n\n"
    "Твоя задача — определить, соответствует ли план запросу пользователя.\n"
    "Типичные ошибки:\n"
    "- Для слота 'название X' в группировке выбран идентификатор '*_id' "
    "вместо 'name/label/title';\n"
    "- В плане нет dim-таблицы, хотя пользователь запрашивает атрибут "
    "(имя/категорию), которого нет в fact-таблице;\n"
    "- COUNT по 'old_*' колонке вместо канонического PK (для запросов "
    "'сколько всего X');\n"
    "- Агрегация SUM/AVG поставлена на процент/долю (`*_perc`, `*_pct`);\n"
    "- WHERE использует system_timestamp ('inserted_dttm', 'modified_dttm') "
    "вместо отчётной даты ('report_dt');\n"
    "- Лишний фильтр, не вытекающий из запроса.\n\n"
    "Если всё корректно — verdict='approved', edits=[].\n"
    "Если есть проблема — verdict='rejected', edits — список структурированных "
    "правок (только whitelisted-операции):\n"
    "  - replace_column: убрать from_ref и поставить to_ref в той же роли\n"
    "    (target_role: select|group_by|filter|aggregate|order_by);\n"
    "  - add_filter: добавить полную WHERE-строку в to_ref;\n"
    "  - drop_filter: удалить условие, упомянутое в from_ref (строкой);\n"
    "  - swap_aggregation: поменять aggregate-функцию (to_ref: SUM|COUNT|AVG|MIN|MAX);\n"
    "  - add_table: подключить отсутствующую dim-таблицу (to_ref: schema.table).\n\n"
    "Правила:\n"
    "- Все ссылки на колонки в формате schema.table.column.\n"
    "- НЕ ПРИДУМЫВАЙ колонки, которых нет в catalog_excerpt.\n"
    "- Если сомневаешься — verdict='approved', edits=[].\n"
    "- Верни строго JSON по схеме: "
    '{"verdict":"approved|rejected","reasons":["..."],'
    '"edits":[{"op":"...","target_role":"...","from_ref":"...","to_ref":"...","reason":"..."}]}'
)


def _truncate(text: str, max_len: int = _MAX_DESC) -> str:
    s = (text or "").strip().replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _column_excerpt(schema_loader, schema: str, table: str, column: str) -> dict[str, str]:
    try:
        cols = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        return {"dtype": "", "semantic_class": "", "description": ""}
    if cols is None or cols.empty:
        return {"dtype": "", "semantic_class": "", "description": ""}
    match = cols[cols["column_name"].astype(str).str.lower() == column.lower()]
    if match.empty:
        return {"dtype": "", "semantic_class": "", "description": ""}
    row = match.iloc[0]
    try:
        sem = schema_loader.get_column_semantics(schema, table, column) or {}
    except Exception:  # noqa: BLE001
        sem = {}
    return {
        "dtype": str(row.get("dType") or "").strip(),
        "semantic_class": str(sem.get("semantic_class") or "").strip(),
        "description": _truncate(str(row.get("description") or "")),
    }


def _alternative_columns(
    schema_loader,
    schema: str,
    table: str,
    used_columns: set[str],
    limit: int = _DEFAULT_ALTERNATIVES,
) -> list[dict[str, str]]:
    try:
        cols = schema_loader.get_table_columns(schema, table)
    except Exception:  # noqa: BLE001
        return []
    if cols is None or cols.empty:
        return []
    out: list[dict[str, str]] = []
    for _, row in cols.iterrows():
        col = str(row.get("column_name") or "").strip()
        if not col or col.lower() in used_columns:
            continue
        try:
            sem = schema_loader.get_column_semantics(schema, table, col) or {}
        except Exception:  # noqa: BLE001
            sem = {}
        out.append({
            "ref": f"{schema}.{table}.{col}",
            "dtype": str(row.get("dType") or "").strip(),
            "semantic_class": str(sem.get("semantic_class") or "").strip(),
            "description": _truncate(str(row.get("description") or "")),
        })
        if len(out) >= limit:
            break
    return out


def _build_plan_payload(
    *,
    user_input: str,
    sql_blueprint: dict[str, Any],
    selected_columns: dict[str, dict[str, list[str]]],
    join_spec: list[dict[str, Any]],
    where_resolution: dict[str, Any],
    schema_loader,
) -> dict[str, Any]:
    main_table = str(sql_blueprint.get("main_table") or "")
    aggregations = sql_blueprint.get("aggregations") or (
        [sql_blueprint.get("aggregation")] if sql_blueprint.get("aggregation") else []
    )
    plan = {
        "main_table": main_table,
        "strategy": str(sql_blueprint.get("strategy") or ""),
        "joins": [
            {"left": j.get("left", ""), "right": j.get("right", "")}
            for j in (join_spec or [])
            if j.get("left") and j.get("right")
        ],
        "select": list(sql_blueprint.get("select_columns") or []),
        "aggregations": [
            {
                "function": str(a.get("function") or "").upper(),
                "column": str(a.get("column") or ""),
                "alias": str(a.get("alias") or ""),
                "distinct": bool(a.get("distinct", False)),
            }
            for a in aggregations
            if isinstance(a, dict) and a.get("function")
        ],
        "group_by": list(sql_blueprint.get("group_by") or []),
        "filters": list(sql_blueprint.get("where_conditions") or []),
        "order_by": str(sql_blueprint.get("order_by") or ""),
    }

    used_refs: set[str] = set()
    catalog_excerpt: dict[str, dict[str, str]] = {}
    for table_key, roles in (selected_columns or {}).items():
        if "." not in table_key:
            continue
        schema, table = table_key.split(".", 1)
        used_in_table: set[str] = set()
        for role_cols in roles.values():
            if not isinstance(role_cols, list):
                continue
            for col in role_cols:
                col_str = str(col or "").strip()
                if not col_str or col_str == "*":
                    continue
                used_in_table.add(col_str.lower())
                ref = f"{table_key}.{col_str}"
                used_refs.add(ref.lower())
                catalog_excerpt[ref] = _column_excerpt(schema_loader, schema, table, col_str)
        for alt in _alternative_columns(
            schema_loader, schema, table, used_in_table, limit=_DEFAULT_ALTERNATIVES,
        ):
            catalog_excerpt.setdefault(alt["ref"], {
                "dtype": alt["dtype"],
                "semantic_class": alt["semantic_class"],
                "description": alt["description"],
            })

    return {
        "user_input": user_input,
        "plan": plan,
        "catalog_excerpt": catalog_excerpt,
    }


def _coerce_verdict(parsed: Any) -> PlanVerdict | None:
    if not isinstance(parsed, dict):
        return None
    try:
        return PlanVerdict.model_validate(parsed)
    except ValidationError as exc:
        logger.warning("plan_verifier: invalid verdict schema — %s", exc.errors())
        return None


class PlanVerifierNodes:
    """Mixin с узлом plan_verifier."""

    def plan_verifier(self, state: AgentState) -> dict[str, Any]:
        """Plan-level LLM-верификатор. Возвращает обновления state.

        - Если llm_verifier_enabled=False → no-op.
        - Если verifier уже отработал в этом запросе (`plan_verifier_done`) → no-op.
        - Если LLM вернул approved или невалидные правки → no-op (с логом).
        - Если применилось хотя бы одна правка → возвращаем обновлённый
          selected_columns + флаг `plan_verifier_applied=True`. Маршрут после
          узла отправит state обратно в sql_planner для пересборки blueprint.
        """
        if not getattr(self, "llm_verifier_enabled", False):
            return {}
        if state.get("plan_verifier_done"):
            return {}

        sql_blueprint = state.get("sql_blueprint") or {}
        if not sql_blueprint:
            return {"plan_verifier_done": True}
        selected_columns = state.get("selected_columns") or {}
        if not selected_columns:
            return {"plan_verifier_done": True}

        payload = _build_plan_payload(
            user_input=str(state.get("user_input") or ""),
            sql_blueprint=sql_blueprint,
            selected_columns=selected_columns,
            join_spec=state.get("join_spec") or [],
            where_resolution=state.get("where_resolution") or {},
            schema_loader=self.schema,
        )
        try:
            user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            return {"plan_verifier_done": True}

        try:
            parsed = self._llm_json_with_retry(
                _SYSTEM_PROMPT,
                user_prompt,
                temperature=0.0,
                failure_tag="plan_verifier",
                expect="object",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("plan_verifier: LLM-вызов упал: %s", exc)
            return {"plan_verifier_done": True}

        verdict = _coerce_verdict(parsed) if parsed is not None else None
        if verdict is None:
            logger.info("plan_verifier: LLM не вернул валидный JSON — пропускаем")
            return {"plan_verifier_done": True, "plan_verdict": empty_verdict()}

        logger.info(
            "plan_verifier: verdict=%s reasons=%s edits=%d",
            verdict.verdict, verdict.reasons, len(verdict.edits),
        )

        if verdict.verdict == "approved" or not verdict.edits:
            return {
                "plan_verifier_done": True,
                "plan_verdict": verdict.model_dump(),
            }

        result = apply_plan_edits(
            selected_columns=selected_columns,
            where_conditions=list(sql_blueprint.get("where_conditions") or []),
            edits=verdict.edits,
            schema_loader=self.schema,
        )
        applied = result.get("applied") or []
        rejected = result.get("rejected") or []
        if not applied:
            logger.info(
                "plan_verifier: ни одна правка не применилась (rejected=%d) — "
                "оставляем план как есть",
                len(rejected),
            )
            return {
                "plan_verifier_done": True,
                "plan_verdict": verdict.model_dump(),
            }

        new_blueprint = dict(sql_blueprint)
        new_blueprint["where_conditions"] = result["where_conditions"]
        return {
            "plan_verifier_done": True,
            "plan_verifier_applied": True,
            "plan_verdict": verdict.model_dump(),
            "selected_columns": result["selected_columns"],
            "sql_blueprint": new_blueprint,
        }
