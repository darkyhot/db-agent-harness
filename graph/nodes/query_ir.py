"""Graph nodes for semantic QuerySpec interpretation and catalog grounding."""

from __future__ import annotations

import json
import logging
from typing import Any

from core.catalog_grounding import ground_query_spec
from core.log_safety import summarize_dict_keys, summarize_text
from core.query_ir import QuerySpec, query_spec_json_schema
from graph.state import AgentState

logger = logging.getLogger(__name__)


class QueryIRNodes:
    """Mixin with the new semantic IR entrypoint nodes."""

    def query_interpreter(self, state: AgentState) -> dict[str, Any]:
        """Interpret user language into a single QuerySpec.

        This is the primary language-understanding node. If the model cannot
        produce a valid QuerySpec after retry, we ask for clarification instead
        of switching to a second interpretation path.
        """
        iterations = state.get("graph_iterations", 0) + 1

        if (state.get("plan_edit_text") and state.get("sql_blueprint")) or (
            state.get("plan_preview_approved") and state.get("sql_blueprint")
        ):
            logger.info("QueryInterpreter: plan-edit/approved fast-path")
            return {"graph_iterations": iterations}

        user_input = state.get("user_input", "") or ""
        logger.info("QueryInterpreter: запрос: %s", summarize_text(user_input, label="user_input"))

        system_prompt = _build_query_interpreter_system_prompt()
        user_prompt = _build_query_interpreter_user_prompt(
            user_input=user_input,
            catalog_context=self._get_query_ir_catalog_context(user_input),
            prev_sql=state.get("prev_sql", ""),
            prev_summary=state.get("prev_result_summary", ""),
        )
        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)

        if self.debug_prompt:
            print(
                f"\n{'='*80}\n[DEBUG PROMPT — query_interpreter]\n{'='*80}\n"
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n"
            )

        parsed = self._llm_json_with_retry(
            system_prompt,
            user_prompt,
            temperature=0.0,
            failure_tag="query_interpreter",
            expect="object",
        )
        if not isinstance(parsed, dict) or "task" not in parsed:
            logger.warning("QueryInterpreter: LLM did not return QuerySpec.task")
            return {
                "needs_clarification": True,
                "clarification_message": "Не удалось разобрать смысл запроса в QuerySpec. Переформулируйте, пожалуйста.",
                "query_spec_validation_errors": ["missing task"],
                "graph_iterations": iterations,
            }

        spec, errors = QuerySpec.from_dict(parsed)
        if spec is None:
            logger.warning(
                "QueryInterpreter: QuerySpec invalid: %s",
                "; ".join(errors),
            )
            return {
                "needs_clarification": True,
                "clarification_message": "QuerySpec получился невалидным. Уточните запрос или укажите нужные метрики/измерения.",
                "query_spec_validation_errors": errors,
                "graph_iterations": iterations,
            }

        _strip_unstated_physical_hints(spec, user_input)

        logger.info(
            "QueryInterpreter: QuerySpec=%s",
            summarize_dict_keys(spec.to_dict(), label="query_spec"),
        )

        legacy_intent = spec.to_legacy_intent()
        legacy_hints = spec.to_legacy_user_hints()

        return {
            "query_spec": spec.to_dict(),
            "query_spec_validation_errors": [],
            "intent": legacy_intent,
            "user_hints_llm": legacy_hints,
            "user_hints": legacy_hints,
            "hints_source": "query_spec",
            "semantic_frame": {},
            "needs_clarification": bool(spec.clarification_needed),
            "clarification_message": spec.clarification.question if spec.clarification else "",
            "messages": state["messages"] + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"QuerySpec: {json.dumps(spec.to_dict(), ensure_ascii=False)}"},
            ],
            "graph_iterations": iterations,
        }

    def catalog_grounder(self, state: AgentState) -> dict[str, Any]:
        """Bind QuerySpec to catalog sources and build the initial PlanIR."""
        iterations = state.get("graph_iterations", 0) + 1
        raw_spec = state.get("query_spec") or {}
        spec, errors = QuerySpec.from_dict(raw_spec)
        if spec is None:
            logger.warning("CatalogGrounder: invalid QuerySpec in state: %s", "; ".join(errors))
            return {
                "needs_clarification": True,
                "clarification_message": "Не удалось разобрать структурированный смысл запроса. Переформулируйте, пожалуйста.",
                "query_spec_validation_errors": errors,
                "graph_iterations": iterations,
            }

        logger.info(
            "CatalogGrounder: input source_constraints=%s, excluded_in_spec=%s, "
            "state.must_keep=%s, state.allowed=%s",
            [(s.schema, s.table) for s in spec.source_constraints],
            [(s.schema, s.table) for s in spec.excluded_source_constraints],
            (state.get("user_hints") or {}).get("must_keep_tables") or [],
            state.get("allowed_tables") or [],
        )
        result = ground_query_spec(
            query_spec=spec,
            schema_loader=self.schema,
            user_input=state.get("user_input", "") or "",
        )
        excluded_tables = {
            str(item).strip().lower()
            for item in (state.get("excluded_tables") or (state.get("user_hints") or {}).get("excluded_tables") or [])
            if str(item).strip()
        }
        if excluded_tables:
            _before = [s.full_name for s in result.sources]
            result.sources = [source for source in result.sources if source.full_name.lower() not in excluded_tables]
            _after = [s.full_name for s in result.sources]
            if _before != _after:
                logger.info(
                    "CatalogGrounder: excluded filter %s → %s (excluded=%s)",
                    _before, _after, sorted(excluded_tables),
                )
            if result.plan_ir is not None:
                result.plan_ir.sources = result.sources
                result.plan_ir.main_source = result.sources[0] if result.sources else None
        logger.info(
            "CatalogGrounder: sources=%s confidence=%.2f",
            [s.full_name for s in result.sources],
            result.confidence,
        )
        prev_planning = state.get("planning_confidence") or {}
        prev_components = prev_planning.get("components", {}) if isinstance(prev_planning, dict) else {}
        table_level = "high" if result.confidence >= 0.8 else ("medium" if result.confidence >= 0.5 else "low")

        update: dict[str, Any] = {
            "query_grounding": result.to_dict(),
            "plan_ir": result.plan_ir.to_dict() if result.plan_ir else {},
            "graph_iterations": iterations,
            "planning_confidence": {
                **prev_planning,
                "query_spec_confidence": spec.confidence,
                "catalog_grounding_confidence": result.confidence,
                "components": {
                    **prev_components,
                    "table_confidence": {
                        "score": round(result.confidence, 3),
                        "level": table_level,
                        "evidence": [source.full_name for source in result.sources],
                    },
                },
            },
            "evidence_trace": {
                **(state.get("evidence_trace") or {}),
                "query_spec": spec.to_dict(),
                "catalog_grounding": result.to_dict(),
            },
        }

        if result.needs_clarification and result.clarification:
            update.update({
                "needs_clarification": True,
                "clarification_message": result.clarification.question,
                "clarification_spec": result.clarification.to_dict(),
            })
            return update

        if result.sources:
            selected = [source.to_tuple() for source in result.sources]
            update.update({
                "selected_tables": selected,
                "allowed_tables": [source.full_name for source in result.sources],
                "excluded_tables": list(excluded_tables),
                "plan": [
                    "Ground QuerySpec against catalog: "
                    + ", ".join(source.full_name for source in result.sources)
                ],
            })

        update.update({
            "needs_clarification": False,
            "clarification_message": "",
        })
        return update

    def _get_query_ir_catalog_context(self, user_input: str, top_n: int = 12) -> str:
        """Compact catalog context for QuerySpec interpretation."""
        lines: list[str] = []
        try:
            df = self.schema.search_tables(user_input, top_n=top_n)
        except Exception as exc:  # noqa: BLE001
            logger.warning("QueryInterpreter catalog search failed: %s", exc)
            df = None

        if df is None or df.empty:
            df = self.schema.tables_df.head(top_n)

        for _, row in df.iterrows():
            schema = str(row.get("schema_name") or "")
            table = str(row.get("table_name") or "")
            desc = str(row.get("description") or "")
            grain = str(row.get("grain") or "")
            if not schema or not table:
                continue
            lines.append(
                f"- {schema}.{table}; grain={grain or 'unknown'}; desc={desc[:220]}"
            )
        return "\n".join(lines)

def _build_query_interpreter_system_prompt() -> str:
    schema = json.dumps(query_spec_json_schema(), ensure_ascii=False)
    return (
        "Ты — query_interpreter аналитического SQL-агента.\n"
        "Понимай естественный язык пользователя и возвращай единый QuerySpec. "
        "Не выбирай SQL и не придумывай несуществующие таблицы/колонки.\n"
        "Если смысл или источник неоднозначен, ставь task='clarify' или "
        "clarification_needed=true и заполняй clarification.\n"
        "Детерминированный код после тебя будет только проверять и связывать "
        "с каталогом, поэтому все смысловые решения должны быть в QuerySpec.\n\n"
        "Правила:\n"
        "- Верни ТОЛЬКО JSON-объект без markdown.\n"
        "- task='answer_data' для расчётов/выборок; 'inspect_schema' для вопросов о каталоге; "
        "'edit_plan' для правки прошлого плана; 'clarify' для неоднозначности.\n"
        "- strategy выбирай явно: 'count_attributes' для вопросов вида "
        "'сколько есть X и Y' / 'сколько всего X и Y', где нужно посчитать "
        "кардинальности нескольких атрибутов; 'aggregate' для обычных агрегатов; "
        "'list' для перечислений.\n"
        "- Для strategy='count_attributes' заполняй entities смысловыми именами "
        "из запроса, например entities=[{'name':'ТБ'}, {'name':'госб'}]. "
        "Не маппь их на физические колонки и не выбирай таблицу, если пользователь "
        "явно не назвал источник.\n"
        "- filters описывают бизнес-смысл, не SQL.\n"
        "- Если пользователь просит несколько обычных показателей, верни несколько "
        "objects в metrics, не один.\n"
        "- В dimensions[*].label можно положить опциональное человеко-читаемое имя оси "
        "(напр. 'Дата отчёта'); поле описательное, downstream его не использует. "
        "Других полей у dimension нет — не добавляй ничего, кроме перечисленных в JSON Schema.\n"
        "- order_by заполняй для просьб о сортировке; direction должен быть ASC или DESC.\n"
        "- source_constraints заполняй только если источник явно назван пользователем; "
        "не угадывай таблицу на этапе QuerySpec.\n"
        "- excluded_source_constraints заполняй только при явном запрете источника в правке плана.\n"
        "- target_column_hint, dimensions[*].source_table и dimensions[*].join_key "
        "заполняй ТОЛЬКО если пользователь явно написал физическое имя колонки/таблицы "
        "(например outflow_qty или schema.table). Не выбирай колонки по каталогу: "
        "физическую привязку выполнит downstream catalog/column resolver.\n"
        "- Каждое важное поле снабжай confidence и evidence.\n\n"
        f"JSON Schema:\n{schema}"
    )


def _strip_unstated_physical_hints(spec: QuerySpec, user_input: str) -> None:
    """Drop physical hints that were inferred from catalog instead of user text."""
    haystack = str(user_input or "").lower()

    def _mentioned(value: str | None) -> bool:
        item = str(value or "").strip().lower()
        return bool(item and item in haystack)

    for entity in spec.entities:
        if entity.target_column_hint and not _mentioned(entity.target_column_hint):
            logger.info(
                "QueryInterpreter: drop inferred target_column_hint=%s for entity=%s",
                entity.target_column_hint,
                entity.name,
            )
            entity.target_column_hint = None
    for dim in spec.dimensions:
        if dim.source_table and not _mentioned(dim.source_table):
            logger.info(
                "QueryInterpreter: drop inferred dimension source_table=%s for target=%s",
                dim.source_table,
                dim.target,
            )
            dim.source_table = None
        if dim.join_key and not _mentioned(dim.join_key):
            logger.info(
                "QueryInterpreter: drop inferred dimension join_key=%s for target=%s",
                dim.join_key,
                dim.target,
            )
            dim.join_key = None


def _apply_query_spec_guardrails(spec: QuerySpec, semantic_frame: dict[str, Any]) -> dict[str, Any]:
    """Keep downstream compatibility flags aligned with the LLM QuerySpec."""
    frame = dict(semantic_frame or {})
    count_metrics = [
        metric for metric in spec.metrics
        if metric.operation == "count" and metric.target
    ]
    if len(count_metrics) > 1:
        frame["requires_single_entity_count"] = False
        frame["subject"] = None
    if spec.dimensions:
        frame["output_dimensions"] = [dim.target for dim in spec.dimensions]
        frame["requires_single_entity_count"] = False
    return frame


def _build_query_interpreter_user_prompt(
    *,
    user_input: str,
    catalog_context: str,
    prev_sql: str,
    prev_summary: str,
) -> str:
    parts = []
    if prev_sql:
        parts.append(f"Предыдущий SQL:\n{prev_sql[:1200]}")
    if prev_summary:
        parts.append(f"Предыдущий результат:\n{prev_summary[:800]}")
    if catalog_context:
        parts.append(f"Компактный каталог-контекст:\n{catalog_context[:12000]}")
    parts.append(f"Запрос пользователя:\n{user_input}")
    parts.append("Верни QuerySpec JSON:")
    return "\n\n".join(parts)
