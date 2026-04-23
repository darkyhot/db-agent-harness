"""Узлы редактирования плана в plan mode.

Поддерживает четыре режима правки:
- patch: локальная правка текущего sql_blueprint
- rebind: изменение источников/таблиц/связей
- rewrite: полная смена смысла запроса
- clarify: правка неоднозначна, нужен короткий вопрос
"""

from __future__ import annotations

import copy
import json
import logging
import re
from datetime import datetime
from typing import Any

from graph.state import AgentState

logger = logging.getLogger(__name__)


def _full_table_name(item: tuple[str, str] | list[str] | str) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return f"{item[0]}.{item[1]}"
    return ""


def _split_table_name(full_name: str) -> tuple[str, str] | None:
    parts = str(full_name or "").strip().split(".", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def _collect_known_columns(selected_columns: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for roles in (selected_columns or {}).values():
        if not isinstance(roles, dict):
            continue
        for role_name in ("select", "filter", "aggregate", "group_by"):
            for col in roles.get(role_name, []) or []:
                if isinstance(col, str) and col:
                    result.add(col)
    return result


def _table_columns_map(schema_loader: Any, main_table: str) -> dict[str, dict[str, Any]]:
    split = _split_table_name(main_table)
    if split is None:
        return {}
    cols_df = schema_loader.get_table_columns(*split)
    if cols_df is None or cols_df.empty:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for _, row in cols_df.iterrows():
        col_name = str(row.get("column_name", "") or "").strip()
        if not col_name:
            continue
        result[col_name.lower()] = {
            "column_name": col_name,
            "description": str(row.get("description", "") or "").strip(),
        }
    return result


def _resolve_column_token(
    schema_loader: Any,
    main_table: str,
    token: str,
    selected_columns: dict[str, Any],
) -> str | None:
    raw = str(token or "").strip()
    if not raw:
        return None
    if raw == "*":
        return "*"

    known_columns = _collect_known_columns(selected_columns)
    raw_lower = raw.lower()
    for col in known_columns:
        if col.lower() == raw_lower:
            return col

    cols_map = _table_columns_map(schema_loader, main_table)
    if raw_lower in cols_map:
        return cols_map[raw_lower]["column_name"]

    for lower_name, meta in cols_map.items():
        if str(meta.get("description") or "").strip().lower() == raw_lower:
            return meta["column_name"]

    desc_matches = [
        meta["column_name"]
        for meta in cols_map.values()
        if raw_lower and raw_lower in str(meta.get("description") or "").strip().lower()
    ]
    if len(desc_matches) == 1:
        return desc_matches[0]

    split = _split_table_name(main_table)
    if split is not None:
        schema_name, table_name = split
        syn_matches: list[str] = []
        for meta in cols_map.values():
            synonyms = schema_loader.get_column_synonyms(schema_name, table_name, meta["column_name"])
            if any(raw_lower == str(syn).strip().lower() for syn in synonyms):
                syn_matches.append(meta["column_name"])
        if len(syn_matches) == 1:
            return syn_matches[0]

    return None


def _render_compact_plan(blueprint: dict[str, Any]) -> str:
    agg = blueprint.get("aggregation") or {}
    distinct_sql = "DISTINCT " if agg.get("distinct") else ""
    agg_str = ""
    if agg:
        agg_str = f"{agg.get('function', '')}({distinct_sql}{agg.get('column', '')})"
    parts = [
        f"main_table={blueprint.get('main_table', '')}",
        f"strategy={blueprint.get('strategy', '')}",
        f"aggregation={agg_str}",
        f"filters={', '.join(blueprint.get('where_conditions') or [])}",
        f"group_by={', '.join(blueprint.get('group_by') or [])}",
        f"order_by={blueprint.get('order_by') or ''}",
        f"limit={blueprint.get('limit')!r}",
    ]
    return "\n".join(parts)


def _parse_json_response(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", text or "")
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    for candidate in re.findall(r"\{.*\}", cleaned, re.DOTALL):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_table_token(text: str) -> str:
    patterns = [
        r"(?:таблиц[ауые]?\s+|витрин[ауые]?\s+)([a-zA-Z_][a-zA-Z0-9_\.]+)",
        r"(?:использовать|возьми|замени|хочу)\s+(?:таблицу\s+)?([a-zA-Z_][a-zA-Z0-9_\.]+)",
        r"\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _ensure_history_entry(
    state: AgentState,
    *,
    kind: str,
    text: str,
    payload: dict[str, Any],
    applied: bool,
) -> list[dict[str, Any]]:
    history = list(state.get("plan_edit_history") or [])
    history.append({
        "iteration": len(history) + 1,
        "text": text,
        "kind": kind,
        "payload": copy.deepcopy(payload),
        "applied": applied,
    })
    return history


class PlanEditNodes:
    """Mixin с узлами plan-edit цикла."""

    _PATCH_SORT_DESC = re.compile(r"(по\s+убывани|descending|\bdesc\b)", re.IGNORECASE)
    _PATCH_SORT_ASC = re.compile(r"(по\s+возрастани|ascending|\basc\b)", re.IGNORECASE)
    _PATCH_LIMIT = re.compile(r"(?:лимит|limit)\s+(\d+)", re.IGNORECASE)
    _PATCH_GROUP_BY = re.compile(
        r"(?:сгруппир\w+|group\s+by|добавь\s+группиров\w+\s+по)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    )
    _PATCH_REMOVE_FILTER = re.compile(
        r"(?:убери|удали)\s+фильтр\s+по\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    )
    _PATCH_COUNT_DISTINCT = re.compile(
        r"\bcount\s*\(\s*distinct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\)|\bcount\s+distinct\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    )
    _PATCH_COUNT_STAR = re.compile(
        r"(?:count\s*\(\s*\*\s*\)|посчита[йи]\w*\s+просто\s+количеств\w*\s+строк|"
        r"прост\w*\s+количеств\w*\s+строк|count\s+all|count\s+rows)",
        re.IGNORECASE,
    )
    _PATCH_NO_DISTINCT = re.compile(
        r"(?:без\s+distinct|не\s+надо\s+distinct|не\s+надо\s+счита\w*\s+по\s+уникальн\w+|"
        r"не\s+счита\w*\s+по\s+уникальн\w+|не\s+по\s+уникальн\w+)",
        re.IGNORECASE,
    )
    _PATCH_COUNT_BY_FIELD = re.compile(
        r"(?:давай\s+)?(?:посчита[йи]\w*|подсчита[йи]\w*|счита[йи]\w*|count)\s+(?:by|по|на)\s+([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    )
    _PATCH_DATE_SHIFT = re.compile(
        r"(?:с|от)\s+(\d{1,2})\s*(?:числ[ао]?)?\s+(?:на|до)\s+(\d{1,2})\s*(?:числ[ао]?)?",
        re.IGNORECASE,
    )

    def _sanitize_patch_operations(
        self,
        operations: list[dict[str, Any]],
        selected_columns: dict[str, Any],
        main_table: str = "",
    ) -> list[dict[str, Any]]:
        allowed_paths = {
            "aggregation.function",
            "aggregation.column",
            "aggregation.distinct",
            "aggregation.alias",
            "order_by.direction",
            "limit",
            "group_by",
            "where.date.from",
            "where.date.to",
        }
        known_columns = _collect_known_columns(selected_columns)
        sanitized: list[dict[str, Any]] = []
        for op in operations:
            if not isinstance(op, dict):
                continue
            action = str(op.get("op") or "").strip()
            path = str(op.get("path") or "").strip()
            if action == "remove_filter":
                column = str(op.get("column") or "").strip()
                if column and column in known_columns:
                    sanitized.append({"op": "remove_filter", "column": column})
                continue
            if path not in allowed_paths or action not in {"replace", "remove", "add"}:
                continue
            cleaned = {"op": action, "path": path}
            if "value" in op:
                cleaned["value"] = op.get("value")
            if path in {"aggregation.column", "group_by"}:
                value = cleaned.get("value")
                if action == "replace" and path == "aggregation.column":
                    resolved = _resolve_column_token(self.schema, main_table, str(value), selected_columns)
                    if resolved is None:
                        continue
                    cleaned["value"] = resolved
                if path == "group_by":
                    if action == "add" and value not in known_columns:
                        continue
                    if action == "replace" and value not in ([], None):
                        values = [item for item in (value if isinstance(value, list) else [value]) if item in known_columns]
                        cleaned["value"] = values
            sanitized.append(cleaned)
        return sanitized

    def _llm_generate_patch_operations(self, blueprint: dict[str, Any], edit_text: str, state: AgentState) -> list[dict[str, Any]]:
        system_prompt = (
            "Ты строишь список операций для правки SQL-плана. "
            'Каждая операция — JSON: {"op":"replace|remove|add|remove_filter","path":"...","value":...}. '
            'Допустимые path: "aggregation.function", "aggregation.column", "aggregation.distinct", '
            '"aggregation.alias", "order_by.direction", "limit", "group_by", "where.date.from", "where.date.to". '
            'Для remove_filter используй поле "column". Верни только JSON {"operations":[...]}.'
        )
        user_prompt = (
            f"Текущий план:\n{_render_compact_plan(blueprint)}\n\n"
            f"Правка пользователя: {edit_text}\n\n"
            "Если пользователь просит считать просто строки, используй aggregation.column='*' и aggregation.distinct=false. "
            "Если просит убрать distinct, верни replace для aggregation.distinct=false."
        )
        try:
            response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.0)
            parsed = _parse_json_response(response) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("plan_edit patch operations LLM fallback failed: %s", exc)
            return []
        operations = parsed.get("operations") if isinstance(parsed, dict) else []
        if not isinstance(operations, list):
            return []
        blueprint = state.get("sql_blueprint") or {}
        return self._sanitize_patch_operations(
            operations,
            state.get("selected_columns") or {},
            str(blueprint.get("main_table") or ""),
        )

    def _resolve_table_name(self, table_token: str) -> str | None:
        token = str(table_token or "").strip()
        if not token:
            return None
        parts = _split_table_name(token)
        if parts is not None:
            schema_name, table_name = parts
            if not self.schema.get_table_columns(schema_name, table_name).empty:
                return f"{schema_name}.{table_name}"
        df = self.schema.tables_df
        if df is None or df.empty:
            return None
        token_lower = token.lower()
        exact = df[
            (df["table_name"].astype(str).str.lower() == token_lower)
            | (
                (df["schema_name"].astype(str) + "." + df["table_name"].astype(str))
                .str.lower()
                == token_lower
            )
        ]
        if not exact.empty:
            row = exact.iloc[0]
            return f"{row['schema_name']}.{row['table_name']}"
        suffix = df[df["table_name"].astype(str).str.lower().str.contains(token_lower, regex=False)]
        if not suffix.empty:
            row = suffix.iloc[0]
            return f"{row['schema_name']}.{row['table_name']}"
        search = self.schema.search_tables(token, top_n=5)
        if not search.empty:
            row = search.iloc[0]
            return f"{row['schema_name']}.{row['table_name']}"
        return None

    def _fallback_route_with_model(
        self,
        state: AgentState,
        edit_text: str,
        blueprint: dict[str, Any],
    ) -> dict[str, Any]:
        system_prompt = (
            "Ты определяешь тип правки плана SQL. "
            "Верни только JSON формата:\n"
            '{'
            '"edit_kind":"patch|rebind|rewrite|clarify",'
            '"confidence":0.0,'
            '"explanation":"...",'
            '"needs_clarification":false,'
            '"payload":{}}'
            "\nНе генерируй SQL. Не придумывай поля вне payload."
        )
        user_prompt = (
            f"Текущий план:\n{_render_compact_plan(blueprint)}\n\n"
            f"Известные таблицы: {', '.join(_full_table_name(t) for t in (state.get('selected_tables') or []))}\n"
            f"Правка пользователя: {edit_text}\n\n"
            "Если пользователь меняет таблицу/источник — это rebind.\n"
            "Если меняет сам смысл запроса — rewrite.\n"
            "Если правка локальная по сортировке/дате/агрегации/фильтру — patch.\n"
            "Если неоднозначно — clarify и положи в payload.question короткий вопрос."
        )
        try:
            response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.0)
            parsed = _parse_json_response(response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("plan_edit_router LLM fallback failed: %s", exc)
            parsed = None
        if isinstance(parsed, dict):
            parsed.setdefault("payload", {})
            parsed.setdefault("confidence", 0.5)
            parsed.setdefault("explanation", "LLM fallback")
            parsed.setdefault("needs_clarification", parsed.get("edit_kind") == "clarify")
            if parsed.get("edit_kind") == "patch" and not (parsed.get("payload") or {}).get("operations"):
                parsed["payload"] = parsed.get("payload") or {}
                parsed["payload"]["operations"] = self._llm_generate_patch_operations(blueprint, edit_text, state)
            return parsed
        return {
            "edit_kind": "clarify",
            "confidence": 0.0,
            "explanation": "Не удалось распознать правку",
            "needs_clarification": True,
            "payload": {"question": "Что именно нужно изменить в плане: фильтр, сортировку, таблицу или сам смысл запроса?"},
        }

    def _deterministic_route(
        self,
        state: AgentState,
        edit_text: str,
        blueprint: dict[str, Any],
    ) -> dict[str, Any] | None:
        text = edit_text.lower().strip()
        if not text:
            return {
                "edit_kind": "clarify",
                "confidence": 0.0,
                "explanation": "Пустая правка",
                "needs_clarification": True,
                "payload": {"question": "Что именно нужно поменять в плане?"},
            }

        if re.search(r"(вообще\s+передумал|не\s+хочу\s+счит|не\s+считать|покажи\s+список|не\s+count,?\s+а|не\s+количеств)", text):
            payload: dict[str, Any] = {"intent_changes": {}}
            if re.search(r"(список|строк|покажи\s+записи|без\s+агрегац)", text):
                payload["intent_changes"]["aggregation_hint"] = "list"
            elif "sum" in text or "сумм" in text:
                payload["intent_changes"]["aggregation_hint"] = "sum"
            else:
                payload["intent_changes"]["aggregation_hint"] = "list"
            return {
                "edit_kind": "rewrite",
                "confidence": 0.95,
                "explanation": "Пользователь меняет сам тип результата",
                "needs_clarification": False,
                "payload": payload,
            }

        table_token = _extract_table_token(edit_text)
        resolved_table = self._resolve_table_name(table_token) if table_token else None
        if re.search(
            r"(добав.*друг.*таблиц|добав.*из\s+другой\s+таблиц|из\s+другой\s+таблиц|"
            r"другую\s+таблиц|замени.*таблиц|использовать.*друг|хочу\s+таблицу|"
            r"передумал\s+использовать\s+эту\s+таблиц)",
            text,
        ):
            ops: list[dict[str, Any]] = []
            if resolved_table and re.search(r"(замени|хочу другую|использовать)", text):
                ops.append({"op": "replace_main_table", "table": resolved_table})
            elif resolved_table:
                ops.append({"op": "add_table", "table": resolved_table})
            else:
                return {
                    "edit_kind": "clarify",
                    "confidence": 0.6,
                    "explanation": "Пользователь хочет сменить источник, но таблица не распознана",
                    "needs_clarification": True,
                    "payload": {"question": "Какую именно таблицу нужно добавить или использовать вместо текущей?"},
                }
            return {
                "edit_kind": "rebind",
                "confidence": 0.92,
                "explanation": "Пользователь меняет источник данных",
                "needs_clarification": False,
                "payload": {"operations": ops},
            }

        operations: list[dict[str, Any]] = []
        if self._PATCH_COUNT_STAR.search(edit_text):
            operations.extend([
                {"op": "replace", "path": "aggregation.function", "value": "COUNT"},
                {"op": "replace", "path": "aggregation.column", "value": "*"},
                {"op": "replace", "path": "aggregation.distinct", "value": False},
            ])
        match = self._PATCH_COUNT_DISTINCT.search(edit_text)
        if match:
            column = (match.group(1) or match.group(2) or "").strip()
            if column:
                operations.extend([
                    {"op": "replace", "path": "aggregation.function", "value": "COUNT"},
                    {"op": "replace", "path": "aggregation.column", "value": column},
                    {"op": "replace", "path": "aggregation.distinct", "value": True},
                ])
        count_by_match = self._PATCH_COUNT_BY_FIELD.search(edit_text)
        if count_by_match:
            column = count_by_match.group(1).strip()
            resolved_column = _resolve_column_token(
                self.schema,
                str(blueprint.get("main_table") or ""),
                column,
                state.get("selected_columns") or {},
            )
            if resolved_column:
                operations.extend([
                    {"op": "replace", "path": "aggregation.function", "value": "COUNT"},
                    {"op": "replace", "path": "aggregation.column", "value": resolved_column},
                ])
            else:
                return {
                    "edit_kind": "clarify",
                    "confidence": 0.75,
                    "explanation": "Пользователь указал колонку для COUNT, но она не распознана",
                    "needs_clarification": True,
                    "payload": {"question": f"Не удалось распознать колонку '{column}'. По какой колонке нужно считать?"},
                }
        if self._PATCH_NO_DISTINCT.search(edit_text):
            operations.append({"op": "replace", "path": "aggregation.distinct", "value": False})

        if self._PATCH_SORT_DESC.search(edit_text):
            operations.append({"op": "replace", "path": "order_by.direction", "value": "DESC"})
        elif self._PATCH_SORT_ASC.search(edit_text):
            operations.append({"op": "replace", "path": "order_by.direction", "value": "ASC"})

        limit_match = self._PATCH_LIMIT.search(edit_text)
        if limit_match:
            operations.append({"op": "replace", "path": "limit", "value": int(limit_match.group(1))})
        elif re.search(r"(убери|удали)\s+(?:лимит|limit)", text):
            operations.append({"op": "remove", "path": "limit"})

        group_by_match = self._PATCH_GROUP_BY.search(edit_text)
        if group_by_match:
            operations.append({"op": "add", "path": "group_by", "value": group_by_match.group(1).strip()})
        elif re.search(r"(убери|удали)\s+группиров", text):
            operations.append({"op": "replace", "path": "group_by", "value": []})

        remove_filter_match = self._PATCH_REMOVE_FILTER.search(edit_text)
        if remove_filter_match:
            operations.append({"op": "remove_filter", "column": remove_filter_match.group(1).strip()})

        date_shift = self._PATCH_DATE_SHIFT.search(edit_text)
        if date_shift:
            old_day = int(date_shift.group(1))
            new_day = int(date_shift.group(2))
            current_from = None
            for cond in blueprint.get("where_conditions") or []:
                m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*>=\s*'(\d{4}-\d{2}-\d{2})'::date", cond)
                if m:
                    current_from = m.group(2)
                    break
            if current_from:
                dt = datetime.strptime(current_from, "%Y-%m-%d")
                if dt.day == old_day:
                    new_from = dt.replace(day=new_day)
                    operations.append({"op": "replace", "path": "where.date.from", "value": new_from.strftime("%Y-%m-%d")})
                    current_to = None
                    date_col = None
                    for cond in blueprint.get("where_conditions") or []:
                        m_to = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*<\s*'(\d{4}-\d{2}-\d{2})'::date", cond)
                        if m_to:
                            date_col = m_to.group(1)
                            current_to = m_to.group(2)
                            break
                        if not date_col:
                            m_col = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*>=\s*'\d{4}-\d{2}-\d{2}'::date", cond)
                            if m_col:
                                date_col = m_col.group(1)
                    try:
                        if current_to and date_col:
                            current_to_dt = datetime.strptime(current_to, "%Y-%m-%d")
                            delta = current_to_dt - dt
                            operations.append({"op": "replace", "path": "where.date.to", "value": (new_from + delta).strftime("%Y-%m-%d")})
                    except ValueError:
                        pass

        if re.fullmatch(r"поменяй\s+порядок", text):
            return {
                "edit_kind": "clarify",
                "confidence": 0.7,
                "explanation": "Неясно, меняется сортировка или порядок колонок",
                "needs_clarification": True,
                "payload": {"question": "Нужно изменить сортировку или порядок колонок в выдаче?"},
            }

        if operations:
            operations = self._sanitize_patch_operations(
                operations,
                state.get("selected_columns") or {},
                str(blueprint.get("main_table") or ""),
            )
            return {
                "edit_kind": "patch",
                "confidence": 0.95,
                "explanation": "Локальная правка текущего плана",
                "needs_clarification": False,
                "payload": {"operations": operations},
            }
        return None

    def plan_edit_router(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        edit_text = str(state.get("plan_edit_text") or "").strip()
        blueprint = state.get("sql_blueprint") or {}
        logger.info("plan_edit_router: edit=%r", edit_text)

        parsed = self._deterministic_route(state, edit_text, blueprint)
        if parsed is None:
            parsed = self._fallback_route_with_model(state, edit_text, blueprint)

        question = str((parsed.get("payload") or {}).get("question") or "").strip()
        update = {
            "plan_edit_kind": parsed.get("edit_kind") or "clarify",
            "plan_edit_confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "plan_edit_payload": parsed.get("payload") or {},
            "plan_edit_explanation": str(parsed.get("explanation") or ""),
            "plan_edit_needs_clarification": bool(parsed.get("needs_clarification")),
            "clarification_message": question if parsed.get("needs_clarification") else "",
            "needs_clarification": bool(parsed.get("needs_clarification")),
            "graph_iterations": iterations,
        }
        logger.info(
            "plan_edit_router: kind=%s confidence=%.2f",
            update["plan_edit_kind"],
            update["plan_edit_confidence"],
        )
        return update

    def _apply_patch_operations(
        self,
        blueprint: dict[str, Any],
        operations: list[dict[str, Any]],
        selected_columns: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bp = copy.deepcopy(blueprint)
        known_columns = _collect_known_columns(selected_columns)
        agg = dict(bp.get("aggregation") or {})
        where_conditions = list(bp.get("where_conditions") or [])
        old_alias = str(agg.get("alias") or "").strip()
        patch_meta: dict[str, Any] = {}
        for op in operations:
            if not isinstance(op, dict):
                continue
            action = op.get("op")
            path = op.get("path", "")
            value = op.get("value")
            if action == "replace" and path == "aggregation.function":
                agg["function"] = str(value)
            elif action == "replace" and path == "aggregation.column":
                agg["column"] = str(value)
                patch_meta["aggregation_column"] = str(value)
                if value:
                    if str(agg.get("function")).upper() == "COUNT" and str(value) == "*":
                        agg["alias"] = "count_all"
                    else:
                        agg["alias"] = (
                            f"count_{value}"
                            if str(agg.get("function")).upper() == "COUNT"
                            else f"{str(agg.get('function')).lower()}_{value}"
                        )
            elif action == "replace" and path == "aggregation.distinct":
                patch_meta["aggregation_distinct"] = bool(value)
                if value:
                    agg["distinct"] = True
                else:
                    agg.pop("distinct", None)
            elif action == "replace" and path == "order_by.direction":
                current = str(bp.get("order_by") or "").strip()
                if current:
                    if re.search(r"\b(ASC|DESC)\b$", current, re.IGNORECASE):
                        bp["order_by"] = re.sub(r"\b(ASC|DESC)\b$", str(value).upper(), current, flags=re.IGNORECASE)
                    else:
                        bp["order_by"] = f"{current} {str(value).upper()}"
                elif agg.get("alias"):
                    bp["order_by"] = f"{agg['alias']} {str(value).upper()}"
            elif action == "replace" and path == "limit":
                bp["limit"] = int(value)
            elif action == "remove" and path == "limit":
                bp["limit"] = None
            elif action == "add" and path == "group_by":
                gb = list(bp.get("group_by") or [])
                if value not in gb:
                    gb.append(str(value))
                bp["group_by"] = gb
            elif action == "replace" and path == "group_by":
                bp["group_by"] = list(value or [])
            elif action == "replace" and path == "where.date.from":
                replaced = False
                new_conditions = []
                for cond in where_conditions:
                    new_cond = re.sub(
                        r"(>=\s*')(\d{4}-\d{2}-\d{2})('::date)",
                        rf"\g<1>{value}\3",
                        cond,
                    )
                    if new_cond != cond and ">=" in cond and "::date" in cond:
                        replaced = True
                    new_conditions.append(new_cond)
                if not replaced:
                    date_col = ""
                    for col in known_columns:
                        if col.lower().endswith(("dt", "date")):
                            date_col = col
                            break
                    if date_col:
                        new_conditions.append(f"{date_col} >= '{value}'::date")
                where_conditions = new_conditions
            elif action == "replace" and path == "where.date.to":
                replaced = False
                new_conditions = []
                for cond in where_conditions:
                    new_cond = re.sub(
                        r"(<\s*')(\d{4}-\d{2}-\d{2})('::date)",
                        rf"\g<1>{value}\3",
                        cond,
                    )
                    if new_cond != cond and "<" in cond and "::date" in cond:
                        replaced = True
                    new_conditions.append(new_cond)
                if not replaced:
                    date_col = ""
                    for col in known_columns:
                        if col.lower().endswith(("dt", "date")):
                            date_col = col
                            break
                    if date_col:
                        new_conditions.append(f"{date_col} < '{value}'::date")
                where_conditions = new_conditions
            elif action == "remove_filter":
                column = str(op.get("column") or "").lower()
                where_conditions = [
                    cond for cond in where_conditions
                    if column not in cond.lower()
                ]
        if agg:
            new_alias = str(agg.get("alias") or "").strip()
            current_order_by = str(bp.get("order_by") or "").strip()
            if old_alias and new_alias and current_order_by.startswith(old_alias):
                bp["order_by"] = current_order_by.replace(old_alias, new_alias, 1)
            bp["aggregation"] = agg
        bp["where_conditions"] = where_conditions
        return bp, patch_meta

    def plan_patcher(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        payload = state.get("plan_edit_payload") or {}
        operations = list(payload.get("operations") or [])
        old_blueprint = copy.deepcopy(state.get("sql_blueprint") or {})
        new_blueprint, patch_meta = self._apply_patch_operations(
            old_blueprint,
            operations,
            state.get("selected_columns") or {},
        )
        user_hints = copy.deepcopy(state.get("user_hints") or {})
        agg_prefs = dict(user_hints.get("aggregation_preferences") or {})
        agg = new_blueprint.get("aggregation") or {}
        if patch_meta.get("aggregation_column") is not None:
            agg_prefs["function"] = str(agg.get("function") or "COUNT").lower()
            agg_prefs["column"] = str(agg.get("column") or "")
            if "distinct" in agg:
                agg_prefs["distinct"] = bool(agg.get("distinct"))
            else:
                agg_prefs["distinct"] = False
            if agg_prefs.get("column") != "*":
                agg_prefs.pop("force_count_star", None)
            else:
                agg_prefs["force_count_star"] = True
        if agg_prefs:
            user_hints["aggregation_preferences"] = agg_prefs
        logger.info("plan_patcher: applied %d operations", len(operations))
        return {
            "previous_sql_blueprint": old_blueprint,
            "sql_blueprint": new_blueprint,
            "user_hints": user_hints,
            "plan_edit_applied": True,
            "plan_edit_history": _ensure_history_entry(
                state,
                kind="patch",
                text=str(state.get("plan_edit_text") or ""),
                payload=payload,
                applied=True,
            ),
            "needs_clarification": False,
            "clarification_message": "",
            "graph_iterations": iterations,
        }

    def _run_plan_from_selected_tables(self, state: AgentState) -> dict[str, Any]:
        temp_state = dict(state)
        temp_state["plan_edit_text"] = ""
        temp_state["plan_edit_kind"] = ""
        temp_state["plan_edit_payload"] = {}
        temp_state["plan_edit_needs_clarification"] = False
        temp_state["needs_clarification"] = False
        temp_state["clarification_message"] = ""

        updates = {}
        for node in (self.table_explorer, self.column_selector, self.sql_planner):
            merged = dict(temp_state)
            merged.update(updates)
            node_update = node(merged)
            updates.update(node_update)
        return updates

    def source_rebinder(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        payload = state.get("plan_edit_payload") or {}
        operations = list(payload.get("operations") or [])
        selected_tables = list(state.get("selected_tables") or [])
        selected_table_names = [_full_table_name(t) for t in selected_tables]
        user_hints = copy.deepcopy(state.get("user_hints") or {})

        for op in operations:
            action = op.get("op")
            if action == "replace_main_table":
                table_name = str(op.get("table") or "")
                split = _split_table_name(table_name)
                if split is not None:
                    selected_tables = [split] + [
                        tuple(t) for t in selected_tables
                        if _full_table_name(t) != table_name
                    ]
                    user_hints["must_keep_tables"] = [split]
            elif action == "add_table":
                table_name = str(op.get("table") or "")
                split = _split_table_name(table_name)
                if split is not None and table_name not in selected_table_names:
                    selected_tables.append(split)
                    user_hints["must_keep_tables"] = list(dict.fromkeys(list(user_hints.get("must_keep_tables", [])) + [split]))
            elif action == "drop_table":
                table_name = str(op.get("table") or "")
                selected_tables = [tuple(t) for t in selected_tables if _full_table_name(t) != table_name]
            elif action in {"bind_dimension", "replace_dim_source"}:
                dim = str(op.get("dimension") or "").strip().lower()
                table_name = str(op.get("table") or "")
                join_key = str(op.get("join_key") or "").strip()
                if dim and table_name:
                    user_hints.setdefault("dim_sources", {})
                    user_hints["dim_sources"][dim] = {"table": table_name}
                    if join_key:
                        user_hints["dim_sources"][dim]["join_col"] = join_key
                        user_hints.setdefault("join_fields", [])
                        if join_key not in user_hints["join_fields"]:
                            user_hints["join_fields"].append(join_key)
                    split = _split_table_name(table_name)
                    if split is not None and split not in selected_tables:
                        selected_tables.append(split)

        rebased_state = dict(state)
        rebased_state.update({
            "selected_tables": selected_tables,
            "allowed_tables": [_full_table_name(t) for t in selected_tables],
            "user_hints": user_hints,
        })
        rebuilt = self._run_plan_from_selected_tables(rebased_state)
        return {
            **rebuilt,
            "selected_tables": selected_tables,
            "allowed_tables": [_full_table_name(t) for t in selected_tables],
            "user_hints": user_hints,
            "previous_sql_blueprint": copy.deepcopy(state.get("sql_blueprint") or {}),
            "plan_edit_applied": True,
            "plan_edit_history": _ensure_history_entry(
                state,
                kind="rebind",
                text=str(state.get("plan_edit_text") or ""),
                payload=payload,
                applied=True,
            ),
            "graph_iterations": max(iterations, rebuilt.get("graph_iterations", 0)),
        }

    def _fallback_rewrite_query(self, state: AgentState) -> str:
        system_prompt = (
            "Ты переписываешь запрос пользователя в новый standalone query после изменения намерения. "
            "Верни только одну строку текста запроса, без пояснений."
        )
        user_prompt = (
            f"Исходный запрос: {state.get('user_input', '')}\n"
            f"Правка пользователя: {state.get('plan_edit_text', '')}\n"
            "Собери новый самостоятельный запрос для аналитического пайплайна."
        )
        try:
            response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("intent_rewriter LLM fallback failed: %s", exc)
            response = ""
        return str(response or "").strip()

    def intent_rewriter(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        payload = state.get("plan_edit_payload") or {}
        intent_changes = dict(payload.get("intent_changes") or {})
        rewrite_text = str(state.get("plan_edit_text") or "").strip()
        rewritten_query = str(payload.get("standalone_query") or "").strip()
        original_query = str(state.get("user_input") or "").strip()

        existing_intent = dict(state.get("intent") or {})
        if intent_changes and set(intent_changes.keys()) <= {"aggregation_hint"} and state.get("selected_tables"):
            existing_intent.update(intent_changes)
            temp_state = dict(state)
            temp_state.update({
                "intent": existing_intent,
                "plan_edit_text": "",
                "plan_edit_kind": "",
                "plan_edit_payload": {},
                "plan_edit_needs_clarification": False,
                "needs_clarification": False,
                "clarification_message": "",
            })
            rebuilt = self._run_plan_from_selected_tables(temp_state)
            return {
                **rebuilt,
                "intent": existing_intent,
                "previous_sql_blueprint": copy.deepcopy(state.get("sql_blueprint") or {}),
                "plan_edit_applied": True,
                "plan_edit_history": _ensure_history_entry(
                    state,
                    kind="rewrite",
                    text=rewrite_text,
                    payload={"intent_changes": intent_changes},
                    applied=True,
                ),
                "graph_iterations": max(iterations, rebuilt.get("graph_iterations", 0)),
            }

        if not rewritten_query:
            if intent_changes.get("aggregation_hint") == "list":
                rewritten_query = original_query
                rewritten_query = re.sub(r"\bсколько\b", "покажи", rewritten_query, flags=re.IGNORECASE)
                rewritten_query = re.sub(r"\bпосчитай\b", "покажи", rewritten_query, flags=re.IGNORECASE)
                rewritten_query = re.sub(r"\bколичеств[ао]\b", "список", rewritten_query, flags=re.IGNORECASE)
            if not rewritten_query:
                rewritten_query = self._fallback_rewrite_query(state)
        if not rewritten_query:
            rewritten_query = f"{original_query}\nУточнение пользователя: {rewrite_text}"

        temp_state = dict(state)
        temp_state.update({
            "user_input": rewritten_query,
            "plan_edit_text": "",
            "sql_blueprint": {},
            "selected_tables": [],
            "table_structures": {},
            "table_samples": {},
            "table_types": {},
            "join_analysis_data": {},
            "selected_columns": {},
            "join_spec": [],
            "allowed_tables": [],
            "needs_clarification": False,
            "clarification_message": "",
            "plan_preview_approved": False,
        })

        updates: dict[str, Any] = {}
        for node in (
            self.intent_classifier,
            self.hint_extractor,
            self.explicit_mode_dispatcher,
            self.table_resolver,
            self.table_explorer,
            self.column_selector,
            self.sql_planner,
        ):
            merged = dict(temp_state)
            merged.update(updates)
            node_update = node(merged)
            updates.update(node_update)
            if node_update.get("needs_clarification") or node_update.get("needs_disambiguation"):
                break

        return {
            **updates,
            "user_input": rewritten_query,
            "previous_sql_blueprint": copy.deepcopy(state.get("sql_blueprint") or {}),
            "plan_edit_applied": True,
            "plan_edit_history": _ensure_history_entry(
                state,
                kind="rewrite",
                text=rewrite_text,
                payload={"intent_changes": intent_changes, "standalone_query": rewritten_query},
                applied=True,
            ),
            "graph_iterations": max(iterations, updates.get("graph_iterations", 0)),
        }

    def plan_edit_validator(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        blueprint = state.get("sql_blueprint") or {}
        selected_columns = state.get("selected_columns") or {}
        main_table = str(blueprint.get("main_table") or "")
        agg = blueprint.get("aggregation") or {}
        known_columns = _collect_known_columns(selected_columns)
        errors: list[str] = []

        if main_table:
            split = _split_table_name(main_table)
            if split is None or self.schema.get_table_columns(*split).empty:
                errors.append(f"Таблица {main_table} не найдена в каталоге")

        agg_col = str(agg.get("column") or "")
        if agg_col == "*" and agg.get("distinct"):
            errors.append("COUNT(DISTINCT *) недопустим")
        if agg_col and agg_col != "*" and agg_col not in known_columns:
            if main_table:
                split = _split_table_name(main_table)
                cols_df = self.schema.get_table_columns(*split) if split is not None else None
                if cols_df is None or cols_df.empty or agg_col not in cols_df["column_name"].astype(str).tolist():
                    errors.append(f"Агрегирующая колонка {agg_col} не найдена")
            else:
                errors.append(f"Агрегирующая колонка {agg_col} не найдена")

        order_by = str(blueprint.get("order_by") or "").strip()
        if order_by:
            order_field = re.sub(r"\s+(ASC|DESC)\s*$", "", order_by, flags=re.IGNORECASE).strip()
            valid_aliases = {
                str(agg.get("alias") or ""),
                *(str(x) for x in (blueprint.get("group_by") or [])),
            }
            if order_field and order_field not in valid_aliases and order_field not in known_columns:
                # допускаем order by alias of aggregation even if not in select columns
                errors.append(f"Поле сортировки {order_field} не найдено")

        limit = blueprint.get("limit")
        if limit is not None:
            try:
                if int(limit) <= 0:
                    errors.append("LIMIT должен быть положительным")
            except (TypeError, ValueError):
                errors.append("LIMIT должен быть целым числом")

        date_from = None
        date_to = None
        for cond in blueprint.get("where_conditions") or []:
            m_from = re.search(r">=\s*'(\d{4}-\d{2}-\d{2})'::date", cond)
            m_to = re.search(r"<\s*'(\d{4}-\d{2}-\d{2})'::date", cond)
            if m_from:
                date_from = m_from.group(1)
            if m_to:
                date_to = m_to.group(1)
        if date_from and date_to:
            try:
                if datetime.strptime(date_from, "%Y-%m-%d") >= datetime.strptime(date_to, "%Y-%m-%d"):
                    errors.append("Диапазон даты некорректен: дата начала должна быть раньше даты конца")
            except ValueError:
                errors.append("Не удалось разобрать диапазон дат")

        for join in state.get("join_spec") or []:
            left = str(join.get("left") or "")
            right = str(join.get("right") or "")
            for side in (left, right):
                match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)$", side)
                if not match:
                    errors.append(f"Некорректный join key: {side}")
                    continue
                schema_name, table_name, col_name = match.groups()
                cols_df = self.schema.get_table_columns(schema_name, table_name)
                if cols_df.empty or col_name not in cols_df["column_name"].astype(str).tolist():
                    errors.append(f"Join key {side} не найден в каталоге")

        if errors:
            logger.info("plan_edit_validator: errors=%s", errors)
            return {
                "plan_edit_applied": False,
                "plan_edit_needs_clarification": True,
                "needs_clarification": True,
                "clarification_message": "Не удалось применить правку: " + "; ".join(errors),
                "graph_iterations": iterations,
            }

        return {
            "plan_edit_applied": True,
            "plan_edit_needs_clarification": False,
            "needs_clarification": False,
            "clarification_message": "",
            "graph_iterations": iterations,
        }

    def plan_diff_renderer(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        before = state.get("previous_sql_blueprint") or {}
        after = state.get("sql_blueprint") or {}
        changes: list[dict[str, str]] = []

        def _record(field: str, left: Any, right: Any) -> None:
            if left != right:
                changes.append({
                    "field": field,
                    "before": str(left),
                    "after": str(right),
                })

        _record("main_table", before.get("main_table"), after.get("main_table"))
        _record("aggregation", before.get("aggregation"), after.get("aggregation"))
        _record("where_conditions", before.get("where_conditions"), after.get("where_conditions"))
        _record("group_by", before.get("group_by"), after.get("group_by"))
        _record("order_by", before.get("order_by"), after.get("order_by"))
        _record("limit", before.get("limit"), after.get("limit"))

        summary_lines: list[str] = []
        for change in changes:
            summary_lines.append(
                f"- {change['field']}: {change['before']} -> {change['after']}"
            )
        summary = "\n".join(summary_lines)

        return {
            "plan_diff": {"changed": changes},
            "plan_diff_summary": summary,
            "graph_iterations": iterations,
        }
