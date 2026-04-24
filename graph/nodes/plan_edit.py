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

from core.query_ir import QuerySpec, query_spec_json_schema
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
    aggs = _iter_blueprint_aggregations(blueprint)
    agg_str = "; ".join(_format_aggregation_expr(agg) for agg in aggs if agg) if aggs else ""
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


def _iter_blueprint_aggregations(blueprint: dict[str, Any]) -> list[dict[str, Any]]:
    raw = blueprint.get("aggregations")
    if isinstance(raw, list) and raw:
        return [dict(item) for item in raw if isinstance(item, dict)]
    agg = blueprint.get("aggregation") or {}
    return [dict(agg)] if agg else []


def _format_aggregation_expr(agg: dict[str, Any]) -> str:
    distinct_sql = "DISTINCT " if agg.get("distinct") else ""
    func = str(agg.get("function") or "")
    col = str(agg.get("column") or "")
    alias = str(agg.get("alias") or "")
    expr = f"{func}({distinct_sql}{col})" if func and col else ""
    if alias and expr:
        expr += f" AS {alias}"
    return expr


def _sync_legacy_aggregation_fields(blueprint: dict[str, Any]) -> dict[str, Any]:
    bp = copy.deepcopy(blueprint)
    aggs = [dict(item) for item in (_iter_blueprint_aggregations(bp)) if item]
    bp["aggregations"] = aggs
    bp["aggregation"] = aggs[0] if aggs else None
    return bp


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
        r"прост\w*\s+количеств\w*\s+строк|счита[йи]\w*\s+просто\s+строк\w*|"
        r"прост\w*\s+строк\w*|count\s+all|count\s+rows)",
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
    _PATCH_ADD_COUNT_FIELD = re.compile(
        r"(?:и\s+ещ[её]|ещ[её]|добав[ььт]\w*|надо\s+ещ[её]|нужно\s+ещ[её])\s+"
        r"(?:count\s+)?(?:(?:by|по|на)\s+)?([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    )
    _PATCH_REPLACE_METRIC = re.compile(
        r"не\s+(?:по\s+)?([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)\s*,?\s+а\s+(?:по\s+)?([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    )
    _PATCH_REPLACE_DIRECT = re.compile(
        r"(?:вместо|замени\s+на)\s+(?:по\s+)?([a-zA-Zа-яА-ЯёЁ_][a-zA-Zа-яА-ЯёЁ0-9_]*)",
        re.IGNORECASE,
    )
    _PATCH_DATE_SHIFT = re.compile(
        r"(?:с|от)\s+(\d{1,2})\s*(?:числ[ао]?)?\s+(?:на|до)\s+(\d{1,2})\s*(?:числ[ао]?)?",
        re.IGNORECASE,
    )

    def _build_plan_edit_resolution(
        self,
        *,
        edit_goal: str,
        requested_changes: list[dict[str, Any]] | None = None,
        candidate_targets: list[dict[str, Any]] | None = None,
        chosen_patch: list[dict[str, Any]] | None = None,
        confidence: float = 0.0,
        clarification_reason: str = "",
    ) -> dict[str, Any]:
        return {
            "edit_goal": edit_goal,
            "requested_changes": list(requested_changes or []),
            "candidate_targets": list(candidate_targets or []),
            "chosen_patch": list(chosen_patch or []),
            "confidence": float(confidence or 0.0),
            "clarification_reason": str(clarification_reason or ""),
        }

    def _patch_result(
        self,
        resolution: dict[str, Any],
        *,
        explanation: str,
    ) -> dict[str, Any]:
        return {
            "edit_kind": "patch",
            "confidence": float(resolution.get("confidence", 0.0) or 0.0),
            "explanation": explanation,
            "needs_clarification": False,
            "payload": {
                "resolution": resolution,
                "chosen_patch": list(resolution.get("chosen_patch") or []),
            },
        }

    def _clarify_result(
        self,
        resolution: dict[str, Any],
        question: str,
        *,
        explanation: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        if confidence is not None:
            resolution["confidence"] = float(confidence)
        return {
            "edit_kind": "clarify",
            "confidence": float(resolution.get("confidence", 0.0) or 0.0),
            "explanation": explanation,
            "needs_clarification": True,
            "payload": {
                "question": question,
                "resolution": resolution,
            },
        }

    def _metric_alias(self, function: str, column: str) -> str:
        func = str(function or "").upper()
        col = str(column or "")
        if func == "COUNT" and col == "*":
            return "count_all"
        return f"{func.lower()}_{col}"

    def _build_metric(self, *, function: str, column: str, distinct: bool = False) -> dict[str, Any]:
        metric = {
            "function": str(function or "COUNT").upper(),
            "column": str(column or ""),
            "alias": self._metric_alias(function, column),
        }
        if distinct and column != "*":
            metric["distinct"] = True
        return metric

    def _build_editable_plan_state(self, blueprint: dict[str, Any]) -> dict[str, Any]:
        bp = _sync_legacy_aggregation_fields(blueprint)
        metrics = [dict(item) for item in (_iter_blueprint_aggregations(bp)) if item]
        return {
            "metrics": metrics,
            "metric_aliases": [str(item.get("alias") or "") for item in metrics if str(item.get("alias") or "")],
            "group_by": list(bp.get("group_by") or []),
            "where_conditions": list(bp.get("where_conditions") or []),
            "order_by": str(bp.get("order_by") or "").strip(),
            "limit": bp.get("limit"),
        }

    def _resolve_column_candidates(
        self,
        token: str,
        *,
        main_table: str,
        selected_columns: dict[str, Any],
    ) -> list[str]:
        raw = str(token or "").strip()
        if not raw:
            return []
        if raw == "*":
            return ["*"]

        raw_lower = raw.lower()
        scores: dict[str, int] = {}

        def _push(column: str, score: int) -> None:
            if not column:
                return
            if score > scores.get(column, -1):
                scores[column] = score

        known_columns = _collect_known_columns(selected_columns)
        for col in known_columns:
            col_lower = col.lower()
            normalized = col_lower.replace("_", " ")
            if raw_lower == col_lower:
                _push(col, 100)
            elif raw_lower == normalized:
                _push(col, 96)
            elif raw_lower in col_lower or raw_lower in normalized:
                _push(col, 70)

        cols_map = _table_columns_map(self.schema, main_table)
        split = _split_table_name(main_table)
        for meta in cols_map.values():
            col_name = str(meta.get("column_name") or "")
            desc = str(meta.get("description") or "").strip().lower()
            col_lower = col_name.lower()
            normalized = col_lower.replace("_", " ")
            if raw_lower == col_lower:
                _push(col_name, 99)
            if raw_lower == normalized:
                _push(col_name, 95)
            if raw_lower and desc == raw_lower:
                _push(col_name, 94)
            elif raw_lower and raw_lower in desc:
                _push(col_name, 72)
            elif raw_lower and (raw_lower in col_lower or raw_lower in normalized):
                _push(col_name, 68)

        if split is not None:
            schema_name, table_name = split
            for meta in cols_map.values():
                col_name = str(meta.get("column_name") or "")
                synonyms = self.schema.get_column_synonyms(schema_name, table_name, col_name)
                for synonym in synonyms:
                    syn = str(synonym or "").strip().lower()
                    if not syn:
                        continue
                    if raw_lower == syn:
                        _push(col_name, 93)
                    elif raw_lower in syn:
                        _push(col_name, 69)

        return [col for col, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]

    def _resolve_metric_target(
        self,
        token: str,
        *,
        blueprint: dict[str, Any],
        selected_columns: dict[str, Any],
        current_metrics: list[dict[str, Any]],
    ) -> tuple[str | None, list[str], str]:
        candidates = self._resolve_column_candidates(
            token,
            main_table=str(blueprint.get("main_table") or ""),
            selected_columns=selected_columns,
        )
        if not candidates:
            return None, [], "missing"

        current_columns = [str(metric.get("column") or "") for metric in current_metrics if str(metric.get("column") or "")]
        unseen_candidates = [col for col in candidates if col not in current_columns]
        if len(unseen_candidates) == 1:
            return unseen_candidates[0], candidates, "ok"
        if len(unseen_candidates) > 1:
            return None, candidates, "ambiguous"

        if len(candidates) == 1:
            return candidates[0], candidates, "ok"
        return None, candidates, "ambiguous"

    def _metric_clarification_question(
        self,
        token: str,
        candidates: list[str],
        *,
        action: str,
        current_metrics: list[dict[str, Any]],
    ) -> str:
        token_norm = str(token or "").strip()
        current_label = ", ".join(
            _format_aggregation_expr(metric)
            for metric in current_metrics
            if metric
        )
        if not candidates:
            prefix = "добавить" if action == "add_metric" else "считать"
            return f"Не удалось распознать колонку '{token_norm}'. По какой колонке нужно {prefix}?"
        options = ", ".join(f"`{item}`" for item in candidates[:3])
        if action == "add_metric" and current_label:
            return f"Добавить вторую метрику к текущему {current_label}? Под `{token_norm}` имеется в виду {options}?"
        if action == "replace_primary_metric" and current_label:
            return f"Заменить текущую метрику {current_label}? Под `{token_norm}` имеется в виду {options}?"
        return f"Под `{token_norm}` имеется в виду {options}?"

    def _extract_shifted_date_range(
        self,
        blueprint: dict[str, Any],
        edit_text: str,
    ) -> tuple[str, str] | None:
        date_shift = self._PATCH_DATE_SHIFT.search(edit_text)
        if not date_shift:
            return None
        old_day = int(date_shift.group(1))
        new_day = int(date_shift.group(2))
        current_from = None
        current_to = None
        for cond in blueprint.get("where_conditions") or []:
            m_from = re.search(r"[a-zA-Z_][a-zA-Z0-9_]*\s*>=\s*'(\d{4}-\d{2}-\d{2})'::date", cond)
            m_to = re.search(r"[a-zA-Z_][a-zA-Z0-9_]*\s*<\s*'(\d{4}-\d{2}-\d{2})'::date", cond)
            if m_from and current_from is None:
                current_from = m_from.group(1)
            if m_to and current_to is None:
                current_to = m_to.group(1)
        if not current_from:
            return None
        dt = datetime.strptime(current_from, "%Y-%m-%d")
        if dt.day != old_day:
            return None
        new_from_dt = dt.replace(day=new_day)
        if not current_to:
            return new_from_dt.strftime("%Y-%m-%d"), ""
        current_to_dt = datetime.strptime(current_to, "%Y-%m-%d")
        delta = current_to_dt - dt
        new_to_dt = new_from_dt + delta
        return new_from_dt.strftime("%Y-%m-%d"), new_to_dt.strftime("%Y-%m-%d")

    def _resolve_patch_route(
        self,
        state: AgentState,
        edit_text: str,
        blueprint: dict[str, Any],
    ) -> dict[str, Any] | None:
        text = str(edit_text or "").lower().strip()
        selected_columns = state.get("selected_columns") or {}
        plan_state = self._build_editable_plan_state(blueprint)
        current_metrics = list(plan_state.get("metrics") or [])
        requested_changes: list[dict[str, Any]] = []
        candidate_targets: list[dict[str, Any]] = []
        chosen_patch: list[dict[str, Any]] = []

        if re.fullmatch(r"поменяй\s+порядок", text):
            resolution = self._build_plan_edit_resolution(
                edit_goal="clarify",
                confidence=0.7,
                clarification_reason="ambiguous_sort_or_columns",
            )
            return self._clarify_result(
                resolution,
                "Нужно изменить сортировку или порядок колонок в выдаче?",
                explanation="Неясно, меняется сортировка или порядок колонок",
            )

        replace_metric_match = self._PATCH_REPLACE_METRIC.search(edit_text)
        replace_direct_match = self._PATCH_REPLACE_DIRECT.search(edit_text)
        add_metric_match = self._PATCH_ADD_COUNT_FIELD.search(edit_text)
        count_by_match = self._PATCH_COUNT_BY_FIELD.search(edit_text)
        distinct_metric_match = self._PATCH_COUNT_DISTINCT.search(edit_text)
        has_count_metric = any(str(item.get("function") or "").upper() == "COUNT" for item in current_metrics)
        explicit_count_context = bool(
            distinct_metric_match
            or self._PATCH_COUNT_STAR.search(edit_text)
            or re.search(r"(счита[йе]\w*|подсчита[йе]\w*|\bcount\b)", text)
        )
        if add_metric_match and not (has_count_metric or explicit_count_context):
            add_metric_match = None

        if self._PATCH_COUNT_STAR.search(edit_text):
            requested_changes.append({"action": "set_count_star"})
            chosen_patch.append({"command": "set_count_star"})

        if replace_metric_match:
            old_token = replace_metric_match.group(1).strip()
            new_token = replace_metric_match.group(2).strip()
            requested_changes.append({"action": "replace_primary_metric", "from": old_token, "to": new_token})
            resolved_column, candidates, status = self._resolve_metric_target(
                new_token,
                blueprint=blueprint,
                selected_columns=selected_columns,
                current_metrics=current_metrics,
            )
            candidate_targets.append({"slot": "metric", "token": new_token, "matches": candidates})
            if status != "ok" or not resolved_column:
                resolution = self._build_plan_edit_resolution(
                    edit_goal="clarify",
                    requested_changes=requested_changes,
                    candidate_targets=candidate_targets,
                    confidence=0.78,
                    clarification_reason=f"replace_metric_{status}",
                )
                return self._clarify_result(
                    resolution,
                    self._metric_clarification_question(new_token, candidates, action="replace_primary_metric", current_metrics=current_metrics),
                    explanation="Не удалось однозначно подобрать замену метрики",
                )
            inherit_distinct = bool(current_metrics[0].get("distinct")) if current_metrics else False
            chosen_patch.append({
                "command": "replace_primary_metric",
                "function": "COUNT",
                "column": resolved_column,
                "distinct": inherit_distinct,
            })
        elif replace_direct_match:
            token = replace_direct_match.group(1).strip()
            requested_changes.append({"action": "replace_primary_metric", "target": token})
            resolved_column, candidates, status = self._resolve_metric_target(
                token,
                blueprint=blueprint,
                selected_columns=selected_columns,
                current_metrics=current_metrics,
            )
            candidate_targets.append({"slot": "metric", "token": token, "matches": candidates})
            if status != "ok" or not resolved_column:
                resolution = self._build_plan_edit_resolution(
                    edit_goal="clarify",
                    requested_changes=requested_changes,
                    candidate_targets=candidate_targets,
                    confidence=0.8,
                    clarification_reason=f"replace_metric_{status}",
                )
                return self._clarify_result(
                    resolution,
                    self._metric_clarification_question(token, candidates, action="replace_primary_metric", current_metrics=current_metrics),
                    explanation="Не удалось однозначно определить заменяющую метрику",
                )
            chosen_patch.append({
                "command": "replace_primary_metric",
                "function": "COUNT",
                "column": resolved_column,
                "distinct": bool(current_metrics[0].get("distinct")) if current_metrics else False,
            })
        elif distinct_metric_match:
            token = (distinct_metric_match.group(1) or distinct_metric_match.group(2) or "").strip()
            requested_changes.append({"action": "replace_primary_metric", "mode": "count_distinct", "target": token})
            resolved_column, candidates, status = self._resolve_metric_target(
                token,
                blueprint=blueprint,
                selected_columns=selected_columns,
                current_metrics=current_metrics,
            )
            candidate_targets.append({"slot": "metric", "token": token, "matches": candidates})
            if status != "ok" or not resolved_column:
                resolution = self._build_plan_edit_resolution(
                    edit_goal="clarify",
                    requested_changes=requested_changes,
                    candidate_targets=candidate_targets,
                    confidence=0.8,
                    clarification_reason=f"distinct_metric_{status}",
                )
                return self._clarify_result(
                    resolution,
                    self._metric_clarification_question(token, candidates, action="replace_primary_metric", current_metrics=current_metrics),
                    explanation="Не удалось однозначно определить колонку для COUNT DISTINCT",
                )
            chosen_patch.append({
                "command": "replace_primary_metric",
                "function": "COUNT",
                "column": resolved_column,
                "distinct": True,
            })
        elif add_metric_match:
            token = add_metric_match.group(1).strip()
            requested_changes.append({"action": "add_metric", "target": token})
            resolved_column, candidates, status = self._resolve_metric_target(
                token,
                blueprint=blueprint,
                selected_columns=selected_columns,
                current_metrics=current_metrics,
            )
            candidate_targets.append({"slot": "metric", "token": token, "matches": candidates})
            if status != "ok" or not resolved_column:
                resolution = self._build_plan_edit_resolution(
                    edit_goal="clarify",
                    requested_changes=requested_changes,
                    candidate_targets=candidate_targets,
                    confidence=0.82,
                    clarification_reason=f"add_metric_{status}",
                )
                return self._clarify_result(
                    resolution,
                    self._metric_clarification_question(token, candidates, action="add_metric", current_metrics=current_metrics),
                    explanation="Не удалось однозначно определить добавляемую метрику",
                )
            inherit_distinct = bool(current_metrics[0].get("distinct")) if current_metrics else False
            command = "add_metric" if current_metrics else "replace_primary_metric"
            chosen_patch.append({
                "command": command,
                "function": "COUNT",
                "column": resolved_column,
                "distinct": inherit_distinct,
            })
        elif count_by_match:
            token = count_by_match.group(1).strip()
            requested_changes.append({"action": "replace_primary_metric", "target": token})
            resolved_column, candidates, status = self._resolve_metric_target(
                token,
                blueprint=blueprint,
                selected_columns=selected_columns,
                current_metrics=current_metrics,
            )
            candidate_targets.append({"slot": "metric", "token": token, "matches": candidates})
            if status != "ok" or not resolved_column:
                resolution = self._build_plan_edit_resolution(
                    edit_goal="clarify",
                    requested_changes=requested_changes,
                    candidate_targets=candidate_targets,
                    confidence=0.8,
                    clarification_reason=f"replace_metric_{status}",
                )
                return self._clarify_result(
                    resolution,
                    self._metric_clarification_question(token, candidates, action="replace_primary_metric", current_metrics=current_metrics),
                    explanation="Не удалось однозначно определить колонку для COUNT",
                )
            chosen_patch.append({
                "command": "replace_primary_metric",
                "function": "COUNT",
                "column": resolved_column,
                "distinct": bool(current_metrics[0].get("distinct")) if current_metrics else False,
            })

        if self._PATCH_NO_DISTINCT.search(edit_text):
            requested_changes.append({"action": "set_distinct", "value": False})
            chosen_patch.append({"command": "set_distinct", "value": False, "target": "primary"})

        if self._PATCH_SORT_DESC.search(edit_text):
            requested_changes.append({"action": "set_order", "direction": "DESC"})
            chosen_patch.append({"command": "set_order", "direction": "DESC"})
        elif self._PATCH_SORT_ASC.search(edit_text):
            requested_changes.append({"action": "set_order", "direction": "ASC"})
            chosen_patch.append({"command": "set_order", "direction": "ASC"})

        limit_match = self._PATCH_LIMIT.search(edit_text)
        if limit_match:
            requested_changes.append({"action": "set_limit", "value": int(limit_match.group(1))})
            chosen_patch.append({"command": "set_limit", "value": int(limit_match.group(1))})
        elif re.search(r"(убери|удали)\s+(?:лимит|limit)", text):
            requested_changes.append({"action": "set_limit", "value": None})
            chosen_patch.append({"command": "set_limit", "value": None})

        group_by_match = self._PATCH_GROUP_BY.search(edit_text)
        if group_by_match:
            token = group_by_match.group(1).strip()
            candidates = self._resolve_column_candidates(
                token,
                main_table=str(blueprint.get("main_table") or ""),
                selected_columns=selected_columns,
            )
            candidate_targets.append({"slot": "group_by", "token": token, "matches": candidates})
            if len(candidates) != 1:
                resolution = self._build_plan_edit_resolution(
                    edit_goal="clarify",
                    requested_changes=requested_changes + [{"action": "add_group_by", "target": token}],
                    candidate_targets=candidate_targets,
                    confidence=0.78,
                    clarification_reason="group_by_ambiguous",
                )
                return self._clarify_result(
                    resolution,
                    self._metric_clarification_question(token, candidates, action="add_group_by", current_metrics=current_metrics),
                    explanation="Не удалось однозначно определить колонку для группировки",
                )
            requested_changes.append({"action": "add_group_by", "target": token})
            chosen_patch.append({"command": "add_group_by", "column": candidates[0]})
        elif re.search(r"(убери|удали)\s+группиров", text):
            requested_changes.append({"action": "remove_group_by", "all": True})
            chosen_patch.append({"command": "remove_group_by", "all": True})

        shifted_range = self._extract_shifted_date_range(blueprint, edit_text)
        if shifted_range:
            date_from, date_to = shifted_range
            requested_changes.append({"action": "set_date_range", "from": date_from, "to": date_to or None})
            chosen_patch.append({"command": "set_date_range", "from": date_from, "to": date_to or None})

        if not chosen_patch:
            return None

        resolution = self._build_plan_edit_resolution(
            edit_goal="patch",
            requested_changes=requested_changes,
            candidate_targets=candidate_targets,
            chosen_patch=chosen_patch,
            confidence=0.95,
        )
        return self._patch_result(
            resolution,
            explanation="Локальная правка текущего плана через state-based resolver",
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
            "aggregations.add",
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
            if path == "aggregations.add":
                value = cleaned.get("value")
                if not isinstance(value, dict):
                    continue
                resolved = _resolve_column_token(
                    self.schema,
                    main_table,
                    str(value.get("column") or ""),
                    selected_columns,
                )
                if resolved is None:
                    continue
                cleaned["value"] = {
                    "function": str(value.get("function") or "COUNT").upper(),
                    "column": resolved,
                    "distinct": bool(value.get("distinct")),
                }
                sanitized.append(cleaned)
                continue
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
            '"aggregation.alias", "aggregations.add", "order_by.direction", "limit", "group_by", "where.date.from", "where.date.to". '
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

    def _llm_classify_edit(
        self,
        state: AgentState,
        edit_text: str,
        blueprint: dict[str, Any],
    ) -> dict[str, Any]:
        """LLM-классификатор edit_kind (PRIMARY путь после формальных regex).

        Возвращает структурированный вердикт (edit_kind + confidence + payload).
        На невалидный JSON (даже после retry) возвращает clarify с низкой
        уверенностью — router тогда переходит к regex-fallback.
        """
        system_prompt = (
            "Ты — классификатор правок SQL-плана. По тексту правки пользователя "
            "определи её тип. Возвращай ТОЛЬКО JSON, без markdown, без пояснений.\n\n"
            "Схема ответа:\n"
            '{"edit_kind": "patch|rebind|rewrite|clarify", '
            '"confidence": 0.0-1.0, '
            '"explanation": "кратко почему", '
            '"needs_clarification": true|false, '
            '"payload": {}}\n\n'
            "Правила классификации:\n"
            "- patch — локальная правка внутри существующего плана: смена сортировки, "
            "добавление/замена метрики, смена группировки, сдвиг дат, лимит, "
            "COUNT(DISTINCT) ↔ COUNT, добавление/снятие фильтра.\n"
            "- rebind — смена источника данных: \"возьми из другой таблицы\", "
            "\"используй X вместо Y\", \"добавь таблицу\".\n"
            "- rewrite — меняется сам смысл запроса: с агрегации на список, с count на sum, "
            "\"передумал\", \"не считать, а показать\".\n"
            "- clarify — правка неоднозначна: в payload.question положи короткий вопрос.\n"
            "Не придумывай конкретные имена колонок/таблиц в payload — это сделает "
            "следующий узел."
        )
        user_prompt = (
            "Текущий план:\n"
            f"{_render_compact_plan(blueprint)}\n\n"
            "Известные таблицы: "
            f"{', '.join(_full_table_name(t) for t in (state.get('selected_tables') or []))}\n"
            f"Правка пользователя: {edit_text}\n\n"
            "JSON:"
        )
        parsed = self._llm_json_with_retry(
            system_prompt, user_prompt,
            temperature=0.0,
            failure_tag="plan_edit_classifier",
            expect="object",
        )
        if not parsed:
            return {
                "edit_kind": "clarify",
                "confidence": 0.0,
                "explanation": "LLM-классификатор не вернул валидный JSON",
                "needs_clarification": True,
                "payload": {},
            }
        edit_kind = str(parsed.get("edit_kind") or "").lower().strip()
        if edit_kind not in {"patch", "rebind", "rewrite", "clarify"}:
            edit_kind = "clarify"
        parsed["edit_kind"] = edit_kind
        parsed.setdefault("payload", {})
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("explanation", "LLM classifier")
        parsed.setdefault("needs_clarification", edit_kind == "clarify")
        return parsed

    def _fallback_route_with_model(
        self,
        state: AgentState,
        edit_text: str,
        blueprint: dict[str, Any],
    ) -> dict[str, Any]:
        """Старый fallback — сохранён для совместимости. В новом порядке
        router вызывает сначала _llm_classify_edit → _deterministic_route,
        и только если оба ничего не дали — этот метод.
        """
        return self._llm_classify_edit(state, edit_text, blueprint)

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
                "payload": {
                    **payload,
                    "resolution": self._build_plan_edit_resolution(
                        edit_goal="rewrite",
                        requested_changes=[{"action": "rewrite_intent", "changes": dict(payload.get("intent_changes") or {})}],
                        confidence=0.95,
                    ),
                },
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
                    "payload": {
                        "question": "Какую именно таблицу нужно добавить или использовать вместо текущей?",
                        "resolution": self._build_plan_edit_resolution(
                            edit_goal="clarify",
                            confidence=0.6,
                            clarification_reason="unresolved_rebind_table",
                        ),
                    },
                }
            resolution = self._build_plan_edit_resolution(
                edit_goal="rebind",
                requested_changes=[{"action": str(item.get("op") or ""), "table": str(item.get("table") or "")} for item in ops],
                confidence=0.92,
            )
            return {
                "edit_kind": "rebind",
                "confidence": 0.92,
                "explanation": "Пользователь меняет источник данных",
                "needs_clarification": False,
                "payload": {"operations": ops, "resolution": resolution},
            }
        return self._resolve_patch_route(state, edit_text, blueprint)

    def _edit_query_spec(self, state: AgentState, edit_text: str) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        raw_spec = state.get("query_spec") or {}
        current_spec, errors = QuerySpec.from_dict(raw_spec)
        if current_spec is None:
            return {
                "plan_edit_kind": "clarify",
                "plan_edit_needs_clarification": True,
                "needs_clarification": True,
                "clarification_message": "Текущий QuerySpec невалиден, поэтому правку нельзя применить безопасно.",
                "query_spec_validation_errors": errors,
                "graph_iterations": iterations,
            }

        system_prompt = (
            "Ты редактируешь QuerySpec аналитического SQL-агента. "
            "Получишь текущий QuerySpec, preview/blueprint и текст правки пользователя. "
            "Верни ПОЛНЫЙ обновлённый QuerySpec JSON по схеме, без markdown. "
            "Не пиши SQL. Не удаляй существующие метрики, фильтры, даты, источники и "
            "измерения, если пользователь явно не попросил удалить или заменить их. "
            "Для сортировки заполняй order_by.target и order_by.direction. "
            "Для просьб вроде 'а где ТБ?' добавляй недостающую metric/dimension, "
            "а не заменяй существующую."
        )
        user_prompt = (
            f"JSON Schema:\n{json.dumps(query_spec_json_schema(), ensure_ascii=False)}\n\n"
            f"Исходный запрос:\n{state.get('user_input', '')}\n\n"
            f"Текущий QuerySpec:\n{json.dumps(current_spec.to_dict(), ensure_ascii=False, indent=2)}\n\n"
            f"Текущий SQL blueprint:\n{json.dumps(state.get('sql_blueprint') or {}, ensure_ascii=False, indent=2)}\n\n"
            f"Правка пользователя:\n{edit_text}\n\n"
            "Верни полный обновлённый QuerySpec JSON:"
        )
        parsed = self._llm_json_with_retry(
            system_prompt,
            user_prompt,
            temperature=0.0,
            failure_tag="query_spec_editor",
            expect="object",
        )
        edited_spec, edit_errors = QuerySpec.from_dict(parsed or {})
        if edited_spec is None:
            return {
                "plan_edit_kind": "clarify",
                "plan_edit_confidence": 0.0,
                "plan_edit_payload": {"errors": edit_errors},
                "plan_edit_needs_clarification": True,
                "needs_clarification": True,
                "clarification_message": "Не удалось применить правку к QuerySpec. Уточните, пожалуйста, что именно изменить.",
                "query_spec_validation_errors": edit_errors,
                "graph_iterations": iterations,
            }

        legacy_intent = edited_spec.to_legacy_intent()
        legacy_hints = edited_spec.to_legacy_user_hints()
        semantic_frame = edited_spec.to_semantic_frame()
        return {
            "query_spec": edited_spec.to_dict(),
            "query_spec_validation_errors": [],
            "intent": legacy_intent,
            "user_hints_llm": legacy_hints,
            "user_hints": legacy_hints,
            "hints_source": "query_spec_edit",
            "semantic_frame": semantic_frame,
            "query_grounding": {},
            "plan_ir": {},
            "selected_tables": [],
            "allowed_tables": [],
            "table_structures": {},
            "table_samples": {},
            "table_types": {},
            "join_analysis_data": {},
            "selected_columns": {},
            "join_spec": [],
            "where_resolution": {},
            "sql_blueprint": {},
            "previous_sql_blueprint": copy.deepcopy(state.get("sql_blueprint") or {}),
            "plan_diff": {},
            "plan_diff_summary": "",
            "plan_edit_text": "",
            "plan_edit_kind": "query_spec",
            "plan_edit_confidence": edited_spec.confidence,
            "plan_edit_payload": {"query_spec": edited_spec.to_dict()},
            "plan_edit_resolution": {"edit_goal": "query_spec_update"},
            "plan_edit_explanation": "QuerySpec updated by LLM editor",
            "plan_edit_needs_clarification": bool(edited_spec.clarification_needed),
            "plan_edit_applied": not bool(edited_spec.clarification_needed),
            "plan_edit_history": _ensure_history_entry(
                state,
                kind="query_spec",
                text=edit_text,
                payload={"query_spec": edited_spec.to_dict()},
                applied=not bool(edited_spec.clarification_needed),
            ),
            "needs_clarification": bool(edited_spec.clarification_needed),
            "clarification_message": edited_spec.clarification.question if edited_spec.clarification else "",
            "graph_iterations": iterations,
        }

    def plan_edit_router(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        edit_text = str(state.get("plan_edit_text") or "").strip()
        blueprint = state.get("sql_blueprint") or {}
        logger.info("plan_edit_router: edit=%r", edit_text)

        if state.get("query_spec"):
            return self._edit_query_spec(state, edit_text)

        # Архитектура: regex-first для быстрых однозначных случаев (экономим 5 сек),
        # LLM подключается только когда regex дал clarify/низкую уверенность/None —
        # т.е. именно для вариативных формулировок, где regex хрупок.

        # 1. Regex: сильные паттерны (sort desc, count star, no distinct, rebind, rewrite)
        #    обрабатываются без LLM. Возвращают confidence ~0.9+ при чётком совпадении.
        parsed: dict[str, Any] | None = self._deterministic_route(state, edit_text, blueprint)

        # 2. LLM-классификатор подключается когда regex:
        #    - вернул None (совсем ничего не подошло), или
        #    - вернул clarify с низкой уверенностью (<0.65) — т.е. regex не уверен.
        #    Это лечит «вариативные формулировки», не замедляя очевидные правки.
        regex_uncertain = (
            parsed is None
            or (
                str(parsed.get("edit_kind") or "") == "clarify"
                and float(parsed.get("confidence", 0.0) or 0.0) < 0.65
            )
        )
        if regex_uncertain:
            llm_result = self._llm_classify_edit(state, edit_text, blueprint)
            llm_kind = str(llm_result.get("edit_kind") or "").lower()
            llm_conf = float(llm_result.get("confidence", 0.0) or 0.0)
            # Используем LLM-результат, только если он confident и не clarify,
            # либо если regex был полностью None.
            if parsed is None or (llm_kind in {"patch", "rebind", "rewrite"} and llm_conf >= 0.5):
                parsed = llm_result
                # Если LLM выбрал patch, но не наполнил payload — дособерём через regex-патчер.
                if llm_kind == "patch":
                    resolved = self._resolve_patch_route(state, edit_text, blueprint)
                    if resolved is not None:
                        parsed = resolved

        # 3. Финальная нормализация: patch без operations → clarify.
        if parsed is None:
            parsed = {
                "edit_kind": "clarify",
                "confidence": 0.0,
                "explanation": "Не удалось распознать правку",
                "needs_clarification": True,
                "payload": {"question": "Что именно нужно изменить в плане: фильтр, сортировку, таблицу или сам смысл запроса?"},
            }
        if str(parsed.get("edit_kind") or "") == "patch" and not (parsed.get("payload") or {}).get("operations"):
            resolved = self._resolve_patch_route(state, edit_text, blueprint)
            if resolved is not None:
                parsed = resolved
            else:
                parsed = {
                    "edit_kind": "clarify",
                    "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                    "explanation": "Недостаточно данных для безопасного patch без уточнения",
                    "needs_clarification": True,
                    "payload": {
                        "question": "Что именно нужно поменять в плане: метрику, сортировку, группировку или фильтр?",
                        "resolution": self._build_plan_edit_resolution(
                            edit_goal="clarify",
                            confidence=float(parsed.get("confidence", 0.0) or 0.0),
                            clarification_reason="fallback_patch_without_resolved_diff",
                        ),
                    },
                }

        question = str((parsed.get("payload") or {}).get("question") or "").strip()
        resolution = dict((parsed.get("payload") or {}).get("resolution") or {})
        update = {
            "plan_edit_kind": parsed.get("edit_kind") or "clarify",
            "plan_edit_confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "plan_edit_payload": parsed.get("payload") or {},
            "plan_edit_resolution": resolution,
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

    def _apply_patch_commands(
        self,
        blueprint: dict[str, Any],
        commands: list[dict[str, Any]],
        selected_columns: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bp = _sync_legacy_aggregation_fields(blueprint)
        known_columns = _collect_known_columns(selected_columns)
        aggs = [dict(item) for item in (bp.get("aggregations") or []) if isinstance(item, dict)]
        where_conditions = list(bp.get("where_conditions") or [])
        patch_meta: dict[str, Any] = {"metrics_changed": False}

        for command in commands:
            if not isinstance(command, dict):
                continue
            name = str(command.get("command") or "").strip()
            if not name:
                continue

            if name == "replace_primary_metric":
                function = str(command.get("function") or "COUNT").upper()
                column = str(command.get("column") or "").strip()
                distinct = bool(command.get("distinct")) and column != "*"
                if not column:
                    continue
                old_alias = str(aggs[0].get("alias") or "").strip() if aggs else ""
                metric = self._build_metric(function=function, column=column, distinct=distinct)
                if aggs:
                    aggs[0] = metric
                else:
                    aggs = [metric]
                current_order_by = str(bp.get("order_by") or "").strip()
                if old_alias and current_order_by.startswith(old_alias):
                    bp["order_by"] = current_order_by.replace(old_alias, metric["alias"], 1)
                patch_meta["metrics_changed"] = True
            elif name == "add_metric":
                function = str(command.get("function") or "COUNT").upper()
                column = str(command.get("column") or "").strip()
                distinct = bool(command.get("distinct")) and column != "*"
                if not column:
                    continue
                metric = self._build_metric(function=function, column=column, distinct=distinct)
                duplicate = any(
                    str(existing.get("function") or "").upper() == metric["function"]
                    and str(existing.get("column") or "") == metric["column"]
                    and bool(existing.get("distinct")) == bool(metric.get("distinct"))
                    for existing in aggs
                )
                if not duplicate:
                    aggs.append(metric)
                    patch_meta["metrics_changed"] = True
            elif name == "remove_metric":
                column = str(command.get("column") or "").strip()
                function = str(command.get("function") or "").upper().strip()
                if not column:
                    continue
                old_alias = str(aggs[0].get("alias") or "").strip() if aggs else ""
                new_aggs = [
                    metric for metric in aggs
                    if not (
                        str(metric.get("column") or "") == column
                        and (not function or str(metric.get("function") or "").upper() == function)
                    )
                ]
                if len(new_aggs) != len(aggs):
                    aggs = new_aggs
                    patch_meta["metrics_changed"] = True
                    current_order_by = str(bp.get("order_by") or "").strip()
                    if old_alias and current_order_by.startswith(old_alias):
                        bp["order_by"] = f"{str(aggs[0].get('alias') or '').strip()} DESC" if aggs else ""
            elif name == "set_distinct":
                value = bool(command.get("value"))
                target = str(command.get("target") or "primary").strip()
                index = 0 if target == "primary" else None
                if index is None or index >= len(aggs):
                    continue
                column = str(aggs[index].get("column") or "")
                if value and column == "*":
                    continue
                if value:
                    aggs[index]["distinct"] = True
                else:
                    aggs[index].pop("distinct", None)
                patch_meta["metrics_changed"] = True
            elif name == "set_count_star":
                old_alias = str(aggs[0].get("alias") or "").strip() if aggs else ""
                metric = self._build_metric(function="COUNT", column="*", distinct=False)
                if aggs:
                    aggs[0] = metric
                else:
                    aggs = [metric]
                current_order_by = str(bp.get("order_by") or "").strip()
                if old_alias and current_order_by.startswith(old_alias):
                    bp["order_by"] = current_order_by.replace(old_alias, metric["alias"], 1)
                patch_meta["metrics_changed"] = True
            elif name == "set_order":
                direction = str(command.get("direction") or "DESC").upper()
                field = str(command.get("field") or "").strip()
                if not field:
                    field = str(aggs[0].get("alias") or "").strip() if aggs else ""
                if field:
                    bp["order_by"] = f"{field} {direction}"
            elif name == "set_limit":
                value = command.get("value")
                bp["limit"] = None if value in ("", None) else int(value)
            elif name == "add_group_by":
                column = str(command.get("column") or "").strip()
                if not column:
                    continue
                gb = list(bp.get("group_by") or [])
                if column not in gb:
                    gb.append(column)
                bp["group_by"] = gb
            elif name == "remove_group_by":
                if command.get("all"):
                    bp["group_by"] = []
                else:
                    column = str(command.get("column") or "").strip()
                    bp["group_by"] = [col for col in (bp.get("group_by") or []) if col != column]
            elif name == "set_date_range":
                date_from = str(command.get("from") or "").strip()
                date_to = str(command.get("to") or "").strip()
                date_col = ""
                for cond in where_conditions:
                    match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:>=|<)\s*'\d{4}-\d{2}-\d{2}'::date", cond)
                    if match:
                        date_col = match.group(1)
                        break
                if not date_col:
                    for col in known_columns:
                        if col.lower().endswith(("dt", "date")):
                            date_col = col
                            break
                new_conditions = [
                    cond for cond in where_conditions
                    if not (
                        date_col
                        and re.search(rf"\b{re.escape(date_col)}\b\s*(?:>=|<)\s*'\d{{4}}-\d{{2}}-\d{{2}}'::date", cond)
                    )
                ]
                if date_col and date_from:
                    new_conditions.append(f"{date_col} >= '{date_from}'::date")
                if date_col and date_to:
                    new_conditions.append(f"{date_col} < '{date_to}'::date")
                where_conditions = new_conditions

        bp["aggregations"] = aggs
        bp["where_conditions"] = where_conditions
        return _sync_legacy_aggregation_fields(bp), patch_meta

    def _apply_legacy_patch_operations(
        self,
        blueprint: dict[str, Any],
        operations: list[dict[str, Any]],
        selected_columns: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        commands: list[dict[str, Any]] = []
        pending_date_range: dict[str, Any] = {"command": "set_date_range", "from": "", "to": ""}
        for op in operations:
            if not isinstance(op, dict):
                continue
            action = str(op.get("op") or "").strip()
            path = str(op.get("path") or "").strip()
            value = op.get("value")
            if action == "replace" and path == "aggregation.column":
                commands.append({"command": "replace_primary_metric", "function": "COUNT", "column": str(value or ""), "distinct": False})
            elif action == "replace" and path == "aggregation.distinct":
                commands.append({"command": "set_distinct", "target": "primary", "value": bool(value)})
            elif action == "add" and path == "aggregations.add" and isinstance(value, dict):
                commands.append({
                    "command": "add_metric",
                    "function": str(value.get("function") or "COUNT").upper(),
                    "column": str(value.get("column") or ""),
                    "distinct": bool(value.get("distinct")),
                })
            elif action == "replace" and path == "order_by.direction":
                commands.append({"command": "set_order", "direction": str(value or "DESC").upper()})
            elif action == "replace" and path == "limit":
                commands.append({"command": "set_limit", "value": value})
            elif action == "remove" and path == "limit":
                commands.append({"command": "set_limit", "value": None})
            elif action == "add" and path == "group_by":
                commands.append({"command": "add_group_by", "column": str(value or "")})
            elif action == "replace" and path == "group_by":
                commands.append({"command": "remove_group_by", "all": True})
            elif action == "replace" and path == "where.date.from":
                pending_date_range["from"] = str(value or "")
            elif action == "replace" and path == "where.date.to":
                pending_date_range["to"] = str(value or "")
            elif action == "remove_filter":
                column = str(op.get("column") or "").lower()
                new_blueprint = copy.deepcopy(blueprint)
                new_blueprint["where_conditions"] = [
                    cond for cond in (new_blueprint.get("where_conditions") or [])
                    if column not in cond.lower()
                ]
                return new_blueprint, {"metrics_changed": False}
        if pending_date_range["from"] or pending_date_range["to"]:
            commands.append({
                "command": "set_date_range",
                "from": pending_date_range["from"] or "",
                "to": pending_date_range["to"] or "",
            })
        return self._apply_patch_commands(blueprint, commands, selected_columns)

    def plan_patcher(self, state: AgentState) -> dict[str, Any]:
        iterations = state.get("graph_iterations", 0) + 1
        payload = state.get("plan_edit_payload") or {}
        resolution = dict(state.get("plan_edit_resolution") or payload.get("resolution") or {})
        commands = list(payload.get("chosen_patch") or resolution.get("chosen_patch") or [])
        operations = list(payload.get("operations") or [])
        old_blueprint = copy.deepcopy(state.get("sql_blueprint") or {})
        if commands:
            new_blueprint, patch_meta = self._apply_patch_commands(old_blueprint, commands, state.get("selected_columns") or {})
        else:
            new_blueprint, patch_meta = self._apply_legacy_patch_operations(old_blueprint, operations, state.get("selected_columns") or {})
        user_hints = copy.deepcopy(state.get("user_hints") or {})
        aggs = _iter_blueprint_aggregations(new_blueprint)
        if patch_meta.get("metrics_changed"):
            agg = new_blueprint.get("aggregation") or {}
            agg_prefs = {}
            if agg:
                agg_prefs = {
                    "function": str(agg.get("function") or "COUNT").lower(),
                    "column": str(agg.get("column") or ""),
                    "distinct": bool(agg.get("distinct")),
                }
                if agg_prefs.get("column") == "*":
                    agg_prefs["force_count_star"] = True
            user_hints["aggregation_preferences"] = agg_prefs
            user_hints["aggregation_preferences_list"] = [
                {
                    "function": str(item.get("function") or "COUNT").lower(),
                    "column": str(item.get("column") or ""),
                    "distinct": bool(item.get("distinct")),
                }
                for item in aggs
                if str(item.get("column") or "")
            ]
        logger.info("plan_patcher: applied %d patch commands", len(commands) if commands else len(operations))
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
        temp_state["plan_edit_resolution"] = {}
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
                "plan_edit_resolution": {},
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
            "plan_edit_resolution": {},
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
        aggs = _iter_blueprint_aggregations(blueprint)
        known_columns = _collect_known_columns(selected_columns)
        errors: list[str] = []

        if main_table:
            split = _split_table_name(main_table)
            if split is None or self.schema.get_table_columns(*split).empty:
                errors.append(f"Таблица {main_table} не найдена в каталоге")

        catalog_columns: set[str] = set()
        if main_table:
            split = _split_table_name(main_table)
            cols_df = self.schema.get_table_columns(*split) if split is not None else None
            if cols_df is not None and not cols_df.empty:
                catalog_columns = set(cols_df["column_name"].astype(str).tolist())

        for agg in aggs:
            agg_col = str(agg.get("column") or "")
            if agg_col == "*" and agg.get("distinct"):
                errors.append("COUNT(DISTINCT *) недопустим")
            if agg_col and agg_col != "*" and agg_col not in known_columns and agg_col not in catalog_columns:
                errors.append(f"Агрегирующая колонка {agg_col} не найдена")

        order_by = str(blueprint.get("order_by") or "").strip()
        if order_by:
            order_field = re.sub(r"\s+(ASC|DESC)\s*$", "", order_by, flags=re.IGNORECASE).strip()
            valid_aliases = {
                *(str(agg.get("alias") or "") for agg in aggs),
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
        _record("aggregations", before.get("aggregations"), after.get("aggregations"))
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
