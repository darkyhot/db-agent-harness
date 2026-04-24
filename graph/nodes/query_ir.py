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

        logger.info(
            "QueryInterpreter: QuerySpec=%s",
            summarize_dict_keys(spec.to_dict(), label="query_spec"),
        )

        legacy_intent = spec.to_legacy_intent()
        legacy_hints = spec.to_legacy_user_hints()
        semantic_frame = spec.to_semantic_frame()

        return {
            "query_spec": spec.to_dict(),
            "query_spec_validation_errors": [],
            "intent": legacy_intent,
            "user_hints_llm": legacy_hints,
            "user_hints": legacy_hints,
            "hints_source": "query_spec",
            "semantic_frame": semantic_frame,
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

        result = ground_query_spec(
            query_spec=spec,
            schema_loader=self.schema,
            user_input=state.get("user_input", "") or "",
        )
        if result.sources and not result.needs_clarification:
            self._review_grounding_with_llm(state, result)
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
            col_bits: list[str] = []
            try:
                cols = self.schema.get_table_columns(schema, table).head(12)
                for _, col in cols.iterrows():
                    name = str(col.get("column_name") or "")
                    dtype = str(col.get("dType") or "")
                    col_desc = str(col.get("description") or "")
                    if name:
                        col_bits.append(f"{name}:{dtype}:{col_desc[:60]}")
            except Exception:  # noqa: BLE001
                col_bits = []
            lines.append(
                f"- {schema}.{table}; grain={grain or 'unknown'}; desc={desc[:160]}; "
                f"cols={'; '.join(col_bits)}"
            )
        return "\n".join(lines)

    def _review_grounding_with_llm(self, state: AgentState, result) -> None:
        """Let LLM reorder/drop only already grounded real source candidates."""
        if len(result.sources) < 2:
            return
        options = [
            {
                "table": source.full_name,
                "reason": source.reason,
                "confidence": source.confidence,
            }
            for source in result.sources
        ]
        system_prompt = (
            "Ты проверяешь выбор таблиц для SQL-агента. Можно выбирать ТОЛЬКО из "
            "переданного списка реальных таблиц. Верни JSON: "
            '{"tables": ["schema.table", ...], "rationale": "кратко"}. '
            "Сохрани все таблицы, которые нужны для метрик, измерений или join."
        )
        user_prompt = (
            "Запрос пользователя:\n"
            f"{state.get('user_input', '')}\n\n"
            "QuerySpec:\n"
            f"{json.dumps(state.get('query_spec') or {}, ensure_ascii=False)}\n\n"
            "Кандидаты:\n"
            f"{json.dumps(options, ensure_ascii=False)}\n\n"
            "JSON:"
        )
        try:
            parsed = self._llm_json_with_retry(
                system_prompt,
                user_prompt,
                temperature=0.0,
                failure_tag="catalog_grounding_reviewer",
                expect="object",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("CatalogGrounder reviewer failed: %s", exc)
            return
        requested = parsed.get("tables") if isinstance(parsed, dict) else None
        if not isinstance(requested, list):
            return
        by_name = {source.full_name.lower(): source for source in result.sources}
        reviewed = []
        for item in requested:
            source = by_name.get(str(item or "").strip().lower())
            if source and source not in reviewed:
                reviewed.append(source)
        if not reviewed:
            return
        for source in result.sources:
            if source not in reviewed:
                reviewed.append(source)
        result.sources = reviewed[: len(result.sources)]
        if result.plan_ir is not None:
            result.plan_ir.sources = result.sources
            result.plan_ir.main_source = result.sources[0] if result.sources else None


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
        "- filters описывают бизнес-смысл, не SQL.\n"
        "- Если пользователь просит несколько показателей (например ТБ и ГОСБ), "
        "верни несколько объектов в metrics, не один.\n"
        "- order_by заполняй для просьб о сортировке; direction должен быть ASC или DESC.\n"
        "- source_constraints заполняй только если источник явно назван или сильно следует из каталога.\n"
        "- Каждое важное поле снабжай confidence и evidence.\n\n"
        f"JSON Schema:\n{schema}"
    )


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
