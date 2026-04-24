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
from core.query_ir import FilterSpec

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


def _add_unique(conditions: list[str], condition: str) -> None:
    if condition not in conditions:
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
    clarification_message = ""
    clarification_spec: dict[str, Any] = {}
    user_filter_choices = dict(user_filter_choices or {})
    rejected_filter_choices = {
        str(k): [str(vv).strip().lower() for vv in (v or []) if str(vv).strip()]
        for k, v in dict(rejected_filter_choices or {}).items()
    }
    direct_filter_specs = _normalize_filter_specs(filter_specs)
    direct_applied = _apply_exact_filter_specs(
        conditions,
        selected_columns,
        direct_filter_specs,
    )
    applied_rules.extend(direct_applied)
    effective_semantic_frame = _semantic_frame_with_filter_specs(
        semantic_frame,
        direct_filter_specs,
    )

    ranked_candidates = rank_filter_candidates(
        user_input=user_input,
        intent=intent,
        selected_tables=selected_tables,
        schema_loader=schema_loader,
        semantic_frame=effective_semantic_frame,
    )

    for request_id, candidates in ranked_candidates.items():
        if str(request_id) in set(direct_applied):
            continue
        if not candidates:
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

        _add_unique(conditions, str(best.get("condition") or ""))
        applied_rules.append(str(request_id))
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
        clarification_message = (
            "Нашёл несколько возможных способов применить фильтр, "
            "но уверенности недостаточно. Уточните, пожалуйста, желаемый признак или значение."
        )
        logger.info("WhereResolver: не удалось привязать filter_intents автоматически")

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
        reasoning.append("aggregate_metric_covers_business_event")
        logger.info(
            "WhereResolver: suppress clarification — business_event покрыт aggregate-метрикой"
        )

    logger.info(
        "WhereResolver: filter_intents=%d, applied_rules=%s, conditions=%d, candidates=%s",
        len(filter_intents),
        applied_rules,
        len(conditions),
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


def _semantic_frame_with_filter_specs(
    semantic_frame: dict[str, Any] | None,
    filter_specs: list[FilterSpec],
) -> dict[str, Any] | None:
    if not filter_specs:
        return semantic_frame
    frame = dict(semantic_frame or {})
    frame["filter_intents"] = [
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
    if filter_specs and not frame.get("business_event"):
        frame["business_event"] = filter_specs[0].target
    return frame


def _apply_exact_filter_specs(
    conditions: list[str],
    selected_columns: dict[str, dict[str, Any]],
    filter_specs: list[FilterSpec],
) -> list[str]:
    applied: list[str] = []
    if not filter_specs:
        return applied
    known_columns: dict[str, str] = {}
    for roles in selected_columns.values():
        for role in ("filter", "select", "group_by", "aggregate"):
            for col in roles.get(role, []) or []:
                col_str = str(col or "").strip()
                if col_str and col_str != "*":
                    known_columns.setdefault(col_str.lower(), col_str)
    for idx, spec in enumerate(filter_specs):
        column = known_columns.get(str(spec.target or "").strip().lower())
        if not column:
            continue
        condition = _condition_from_filter_spec(column, spec)
        if condition:
            _add_unique(conditions, condition)
            applied.append(f"query_spec:{idx}")
    return applied


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
