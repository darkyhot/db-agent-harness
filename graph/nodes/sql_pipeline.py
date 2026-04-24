"""Узлы SQL-конвейера: планирование стратегии, написание SQL, валидация.

Содержит SqlPipelineNodes — миксин для GraphNodes с методами:
- sql_planner: определение стратегии SQL-запроса
- sql_writer: написание SQL по blueprint
- sql_static_checker: детерминированная проверка SQL до БД (без LLM)
- sql_validator_node: валидация, выполнение, проверка результата
"""

import json
import logging
import re
import time
from typing import Any

from core.confidence import (
    build_fallback_policy,
    build_planning_confidence,
    evaluate_filter_confidence,
    evaluate_join_confidence,
)
from core.log_safety import summarize_sql, summarize_text
from core.sql_static_checker import check_sql
from core.sql_planner_deterministic import build_blueprint as _deterministic_blueprint
from core.sql_builder import SqlBuilder as _SqlBuilder
from core.sql_formatter import format_sql_safe as _format_sql
from core.where_resolver import candidate_label as _candidate_label
from graph.nodes.common import (
    BaseNodeMixin,
    ToolResult,
    SQL_FEW_SHOT_EXAMPLES,
    SQL_RULES,
    SQL_CHECKLIST,
    GIGACHAT_COMMON_ERRORS,
)
from graph.state import AgentState

logger = logging.getLogger(__name__)

# === Примеры SQL, отфильтрованные по типу стратегии ===

_STRATEGY_EXAMPLES: dict[str, str] = {
    "simple_select": (
        "Пример — простая выборка с фильтром:\n"
        "SELECT region, COUNT(DISTINCT client_id) AS unique_clients\n"
        "FROM dm.sales\n"
        "WHERE sale_date >= '2024-01-01'::date AND sale_date < '2024-02-01'::date\n"
        "GROUP BY region ORDER BY unique_clients DESC;"
    ),
    "fact_dim_join": (
        "Пример — ФАКТ + СПРАВОЧНИК (уникальная выборка из справочника):\n"
        "SELECT s.sale_id, s.amount, d.name\n"
        "FROM dm.sales s\n"
        "JOIN (\n"
        "    SELECT DISTINCT ON (client_id) client_id, name\n"
        "    FROM dm.clients ORDER BY client_id, updated_at DESC\n"
        ") d ON d.client_id = s.client_id;"
    ),
    "dim_fact_join": (
        "Пример — СПРАВОЧНИК + ФАКТ (агрегация фактов в подзапросе):\n"
        "SELECT c.client_id, c.name, agg.total_sales\n"
        "FROM dm.clients c\n"
        "JOIN (\n"
        "    SELECT client_id, SUM(amount) AS total_sales\n"
        "    FROM dm.sales GROUP BY client_id\n"
        ") agg ON agg.client_id = c.client_id;"
    ),
    "fact_fact_join": (
        "Стратегия ФАКТ + ФАКТ — ОБЯЗАТЕЛЬНЫЕ правила:\n"
        "1. КАЖДАЯ таблица должна быть в ОТДЕЛЬНОМ CTE с агрегацией или DISTINCT ON по join-ключу\n"
        "2. ПРЯМОЙ JOIN без CTE — ЗАПРЕЩЁН (риск размножения строк)\n"
        "3. Колонки берутся ТОЛЬКО из той таблицы, в которой они реально существуют\n"
        "4. SELECT * в финальном запросе — ЗАПРЕЩЁН\n\n"
        "Пример — ФАКТ + ФАКТ (обе стороны агрегированы в CTE):\n"
        "WITH sales_agg AS (\n"
        "    SELECT client_id, SUM(amount) AS total_sales\n"
        "    FROM dm.sales GROUP BY client_id\n"
        "), payments_agg AS (\n"
        "    SELECT client_id, SUM(payment) AS total_paid\n"
        "    FROM dm.payments GROUP BY client_id\n"
        ")\n"
        "SELECT s.client_id, s.total_sales, p.total_paid\n"
        "FROM sales_agg s JOIN payments_agg p ON p.client_id = s.client_id;\n\n"
        "Пример — ФАКТ + таблица с атрибутом (DISTINCT ON по ключу):\n"
        "WITH customer_seg AS (\n"
        "    SELECT DISTINCT ON (customer_id) customer_id, customer_segment\n"
        "    FROM schema.customer_segments ORDER BY customer_id, updated_at DESC\n"
        "), order_agg AS (\n"
        "    SELECT order_dt, customer_id, SUM(order_amount) AS total_amount\n"
        "    FROM schema.fact_orders GROUP BY order_dt, customer_id\n"
        ")\n"
        "SELECT o.order_dt, s.customer_segment, SUM(o.total_amount) AS total_amount\n"
        "FROM order_agg o JOIN customer_seg s ON s.customer_id = o.customer_id\n"
        "GROUP BY o.order_dt, s.customer_segment;"
    ),
    "dim_dim_join": (
        "Пример — СПРАВОЧНИК + СПРАВОЧНИК (уникальные выборки из обеих сторон):\n"
        "WITH d1 AS (\n"
        "    SELECT DISTINCT ON (org_id) org_id, org_name\n"
        "    FROM dm.organizations ORDER BY org_id, effective_date DESC\n"
        "), d2 AS (\n"
        "    SELECT DISTINCT ON (org_id) org_id, region\n"
        "    FROM dm.org_regions ORDER BY org_id, effective_date DESC\n"
        ")\n"
        "SELECT d1.org_id, d1.org_name, d2.region\n"
        "FROM d1 JOIN d2 ON d2.org_id = d1.org_id;"
    ),
}


def _build_join_rule(strategy: str, join_spec: list[dict]) -> str:
    """Сформировать конкретное правило JOIN для sql_writer на основе join_spec.

    Учитывает:
    - safe-флаг каждой пары (проверен post-validation в column_selector)
    - наличие нескольких пар для одной пары таблиц → composite AND
    - стратегию (fact_dim_join, dim_fact_join и др.)
    """
    if not join_spec:
        return ""

    _join_strategies = {"fact_fact_join", "fact_dim_join", "dim_fact_join", "dim_dim_join"}
    if strategy not in _join_strategies:
        return ""

    # Группируем join_spec по парам таблиц для обнаружения composite join
    table_pairs: dict[tuple[str, str], list[dict]] = {}
    for jk in join_spec:
        left_tbl = ".".join(jk.get("left", "").rsplit(".", 1)[:-1])
        right_tbl = ".".join(jk.get("right", "").rsplit(".", 1)[:-1])
        key = (left_tbl, right_tbl)
        table_pairs.setdefault(key, []).append(jk)

    parts: list[str] = []

    for (left_tbl, right_tbl), pairs in table_pairs.items():
        is_composite = len(pairs) > 1

        if is_composite:
            on_conditions = []
            for p in pairs:
                left_col = p["left"].rsplit(".", 1)[-1]
                right_col = p["right"].rsplit(".", 1)[-1]
                on_conditions.append(f"f.{left_col} = g.{right_col}")
            on_str = " AND ".join(on_conditions)
            parts.append(
                f"JOIN по составному ключу между {left_tbl} и {right_tbl}:\n"
                f"  ON {on_str}\n"
                f"  Используй ВСЕ условия через AND — не бери только одно."
            )
        else:
            jk = pairs[0]
            safe = jk.get("safe", False)
            risk = jk.get("risk", "")
            left_col = jk["left"].rsplit(".", 1)[-1]
            right_col = jk["right"].rsplit(".", 1)[-1]

            if safe:
                parts.append(
                    f"JOIN {left_tbl} ↔ {right_tbl} по {left_col} = {right_col}: "
                    f"ключ уникален (safe=True) — прямой JOIN допустим."
                )
            else:
                risk_note = f" ({risk})" if risk else ""
                if strategy == "fact_dim_join":
                    parts.append(
                        f"JOIN {left_tbl} ↔ {right_tbl} по {left_col} = {right_col}: "
                        f"safe=False{risk_note}.\n"
                        f"  Используй DISTINCT ON ({right_col}) в подзапросе для справочника "
                        f"{right_tbl}:\n"
                        f"  JOIN (SELECT DISTINCT ON ({right_col}) {right_col}, <нужные_колонки>\n"
                        f"        FROM {right_tbl} ORDER BY {right_col}) g ON g.{right_col} = f.{left_col}"
                    )
                elif strategy == "dim_fact_join":
                    parts.append(
                        f"JOIN {left_tbl} ↔ {right_tbl} по {left_col} = {right_col}: "
                        f"safe=False{risk_note}.\n"
                        f"  Агрегируй таблицу фактов в подзапросе:\n"
                        f"  JOIN (SELECT {right_col}, SUM(<метрика>) AS val\n"
                        f"        FROM {right_tbl} GROUP BY {right_col}) agg ON agg.{right_col} = d.{left_col}"
                    )
                else:
                    parts.append(
                        f"JOIN {left_tbl} ↔ {right_tbl} по {left_col} = {right_col}: "
                        f"safe=False{risk_note} — используй CTE с агрегацией/DISTINCT ON."
                    )

    if not parts:
        return ""

    return "\nПРАВИЛА JOIN (на основе проверенных ключей):\n" + "\n".join(parts) + "\n"


def _build_specific_clarification(where_resolution: dict[str, Any] | None) -> str:
    """Построить конкретный вопрос по фильтру из кандидатов where_resolution.

    Пропускает request_id, уже закрытые через user_filter_choices, и
    подхватывает примеры значений (matched_example/example_values) из кандидата
    через candidate_label(), чтобы пользователю было видно, почему каждая
    колонка попала в кандидаты.

    Возвращает пустую строку, когда задавать нечего: либо все request_id
    закрыты через user_filter_choices, либо filter_candidates пуст. Caller
    должен интерпретировать это как «уточнение не требуется» и идти дальше.
    """
    where_resolution = where_resolution or {}
    reasoning = {str(item) for item in (where_resolution.get("reasoning", []) or [])}
    if "table_context_covers_business_event" in reasoning:
        return ""
    filter_candidates = where_resolution.get("filter_candidates", {}) or {}
    user_choices = where_resolution.get("user_filter_choices", {}) or {}
    spec = where_resolution.get("clarification_spec", {}) or {}
    if spec.get("message"):
        return str(spec.get("message") or "")
    for request_id, candidates in filter_candidates.items():
        if not candidates:
            continue
        if str(request_id) in user_choices:
            continue
        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        if second:
            left = _candidate_label(top)
            right = _candidate_label(second)
            return (
                "Найдено несколько близких вариантов фильтра. "
                f"Уточните, пожалуйста, по какому признаку фильтровать: {left} или {right}?"
            )
        if top.get("column"):
            label = _candidate_label(top)
            evidence = {str(ev) for ev in (top.get("evidence") or [])}
            if top.get("matched_example") or any(
                ev.startswith("known_term_phrase=") or ev.startswith("value_match=")
                for ev in evidence
            ):
                continue
            return (
                "Уточните, пожалуйста, фильтр: "
                f"нужно отфильтровать именно по полю {label}? Ответьте да/нет."
            )
    return ""


def _build_specific_clarification_spec(where_resolution: dict[str, Any] | None) -> dict[str, Any]:
    """Построить typed clarification spec для CLI."""
    where_resolution = where_resolution or {}
    reasoning = {str(item) for item in (where_resolution.get("reasoning", []) or [])}
    if "table_context_covers_business_event" in reasoning:
        return {}
    existing = where_resolution.get("clarification_spec", {}) or {}
    if existing.get("message"):
        return dict(existing)
    filter_candidates = where_resolution.get("filter_candidates", {}) or {}
    user_choices = where_resolution.get("user_filter_choices", {}) or {}
    for request_id, candidates in filter_candidates.items():
        if not candidates or str(request_id) in user_choices:
            continue
        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        if second:
            left = _candidate_label(top)
            right = _candidate_label(second)
            return {
                "type": "choice",
                "request_id": str(request_id),
                "message": (
                    "Найдено несколько близких вариантов фильтра. "
                    f"Уточните, пожалуйста, по какому признаку фильтровать: {left} или {right}?"
                ),
                "options": [
                    {"column": str(top.get("column") or ""), "label": left},
                    {"column": str(second.get("column") or ""), "label": right},
                ],
            }
        if top.get("column"):
            evidence = {str(ev) for ev in (top.get("evidence") or [])}
            if top.get("matched_example") or any(
                ev.startswith("known_term_phrase=") or ev.startswith("value_match=")
                for ev in evidence
            ):
                continue
            label = _candidate_label(top)
            return {
                "type": "confirm",
                "request_id": str(request_id),
                "message": (
                    "Уточните, пожалуйста, фильтр: "
                    f"нужно отфильтровать именно по полю {label}? Ответьте да/нет."
                ),
                "options": [{"column": str(top.get("column") or ""), "label": label}],
            }
    return {}


class SqlPipelineNodes:
    """Миксин с узлами sql_planner, sql_writer и sql_validator_node для GraphNodes."""

    # --------------------------------------------------------------------------
    # Внутренний LLM-резолвер для выбора identifier-колонки в COUNT(DISTINCT ...)
    # --------------------------------------------------------------------------

    def _resolve_count_identifier_llm(
        self,
        *,
        main_table: str,
        pk_candidates: list[str],
        user_input: str,
    ) -> str | None:
        """Спросить LLM, какая PK-колонка — identifier сущности для COUNT(DISTINCT).

        Срабатывает только при составном PK (≥2 кандидатов). Промпт компактный,
        результат валидируется: возвращаемая колонка должна быть из pk_candidates.
        На невалидный JSON (после одного retry) возвращает None — вызывающая
        сторона применяет детерминированный fallback.
        """
        if not pk_candidates or len(pk_candidates) < 2 or not main_table:
            return None

        descriptions: list[str] = []
        if "." in main_table:
            schema_name, table_name = main_table.split(".", 1)
            cols_df = self.schema.get_table_columns(schema_name, table_name)
            if not cols_df.empty:
                for name in pk_candidates:
                    row = cols_df[cols_df["column_name"].astype(str) == name]
                    if row.empty:
                        continue
                    desc = str(row.iloc[0].get("description", "") or "").strip()
                    semantics = self.schema.get_column_semantics(schema_name, table_name, name)
                    sem_class = str(semantics.get("semantic_class", "") or "").strip()
                    dtype = str(row.iloc[0].get("dType", "") or "").strip()
                    parts = [f"- {name}"]
                    if dtype:
                        parts.append(f"тип: {dtype}")
                    if sem_class:
                        parts.append(f"семантика: {sem_class}")
                    if desc:
                        parts.append(f"описание: {desc[:120]}")
                    descriptions.append("; ".join(parts))

        if not descriptions:
            descriptions = [f"- {name}" for name in pk_candidates]

        system_prompt = (
            "Ты — эксперт по SQL-аналитике. Задача: выбрать, какая из "
            "PK-колонок составного ключа таблицы является identifier'ом "
            "СУЩНОСТИ (клиента / заказа / задачи), которую пользователь "
            "хочет посчитать уникальной через COUNT(DISTINCT ...).\n\n"
            "Правила:\n"
            "- Выбирай колонку, идентифицирующую ОБЪЕКТ подсчёта, не дату и "
            "не срез.\n"
            "- Если среди PK есть дата (_dt, _date, _timestamp, report_dt) "
            "и нет-дата-колонка — выбирай не-дату.\n"
            "- Если непонятно, по какому сущностному ключу считать — верни "
            '"column": null.\n'
            "- Возвращай ТОЛЬКО JSON, без markdown.\n\n"
            "Схема ответа: "
            '{"column": "имя_колонки_из_списка_или_null", "rationale": "кратко"}'
        )
        user_prompt = (
            f"Таблица: {main_table}\n"
            f"Вопрос пользователя: {user_input.strip()}\n"
            "PK-колонки (составной ключ):\n"
            + "\n".join(descriptions)
            + "\n\nJSON:"
        )

        parsed = self._llm_json_with_retry(
            system_prompt, user_prompt,
            temperature=0.0,
            failure_tag="count_identifier_resolver",
            expect="object",
        )
        if not parsed:
            return None
        raw_col = parsed.get("column")
        if not raw_col:
            return None
        chosen = str(raw_col).strip()
        if chosen in pk_candidates:
            return chosen
        lowered = {p.lower(): p for p in pk_candidates}
        if chosen.lower() in lowered:
            return lowered[chosen.lower()]
        logger.warning(
            "_resolve_count_identifier_llm: LLM вернул %r, но это не из pk_candidates=%s",
            chosen, pk_candidates,
        )
        return None

    # --------------------------------------------------------------------------
    # LLM-tiebreaker для ничьи scoring'а фильтр-кандидатов
    # --------------------------------------------------------------------------

    def _tiebreak_filter_candidates_llm(
        self,
        *,
        request_id: str,
        user_input: str,
        candidates: list[dict[str, Any]],
    ) -> str | None:
        """Спросить LLM, какая из близких по score фильтр-колонок подходит лучше.

        Вызывается where_resolver'ом только при ambiguity (узкий gap или низкая
        абсолютная уверенность). Возвращает имя колонки или None (→ fallback на
        clarification-вопрос к пользователю).
        """
        if not candidates or len(candidates) < 2:
            return None

        def _fmt(c: dict[str, Any]) -> str:
            parts = [f"- {c.get('column')}"]
            desc = str(c.get("description") or "").strip()
            if desc:
                parts.append(f"описание: {desc[:150]}")
            sem = str(c.get("semantic_class") or "").strip()
            if sem:
                parts.append(f"семантика: {sem}")
            ex = c.get("example_values") or []
            if ex:
                parts.append(f"примеры: {', '.join(str(v) for v in ex[:2])}")
            score = c.get("score")
            if score is not None:
                parts.append(f"score={score}")
            return "; ".join(parts)

        system_prompt = (
            "Ты — эксперт по SQL-аналитике. Пользователь задал запрос, и "
            "scoring-логика нашла несколько кандидатов-колонок для фильтрации "
            "с близкими оценками. Выбери, какая из них лучше соответствует "
            "смыслу запроса.\n\n"
            "Правила:\n"
            "- Верни ТОЛЬКО JSON: "
            '{"chosen_column": "имя_колонки_из_списка_или_null", "rationale": "кратко"}\n'
            "- Если ни одна не подходит уверенно — верни null.\n"
            "- Выбирай колонку, чья семантика/описание ближе к смыслу запроса, "
            "а не просто чей score выше.\n"
        )
        user_prompt = (
            f"Запрос пользователя: {user_input.strip()}\n"
            f"Request ID: {request_id}\n"
            "Кандидаты (из одной группы scoring'а):\n"
            + "\n".join(_fmt(c) for c in candidates[:3])
            + "\n\nJSON:"
        )
        parsed = self._llm_json_with_retry(
            system_prompt, user_prompt,
            temperature=0.0,
            failure_tag="filter_tiebreaker",
            expect="object",
        )
        if not parsed:
            return None
        chosen = parsed.get("chosen_column")
        if not chosen:
            return None
        chosen_str = str(chosen).strip()
        col_names_lower = {str(c.get("column") or "").strip().lower(): str(c.get("column") or "") for c in candidates}
        if chosen_str.lower() in col_names_lower:
            return col_names_lower[chosen_str.lower()]
        logger.warning(
            "_tiebreak_filter_candidates_llm: LLM вернул %r, но это не из кандидатов",
            chosen_str,
        )
        return None

    # --------------------------------------------------------------------------
    # sql_planner
    # --------------------------------------------------------------------------

    def sql_planner(self, state: AgentState) -> dict[str, Any]:
        """Определение стратегии SQL-запроса: тип JOIN, агрегация, CTE, фильтры.

        Полностью детерминированный — LLM не вызывается.
        Использует core.sql_planner_deterministic.build_blueprint.

        Returns:
            Обновления состояния с sql_blueprint.
        """
        iterations = state.get("graph_iterations", 0) + 1
        intent = state.get("intent", {})
        selected_columns = state.get("selected_columns", {})
        join_spec = state.get("join_spec", [])
        table_types = state.get("table_types", {})
        join_analysis_data = state.get("join_analysis_data", {})

        logger.info("SqlPlanner (deterministic): строю blueprint")

        # --- Проверка согласованности: join_analysis_data vs selected_columns ---
        missing_from_columns: list[str] = []
        if join_analysis_data and selected_columns:
            for _pair_key, data in join_analysis_data.items():
                for tbl_field in ("table1", "table2"):
                    tbl = data.get(tbl_field, "") if isinstance(data, dict) else ""
                    if tbl and tbl not in selected_columns:
                        missing_from_columns.append(tbl)
        missing_from_columns = list(dict.fromkeys(missing_from_columns))

        if missing_from_columns:
            logger.warning(
                "SqlPlanner: таблицы из join_analysis_data отсутствуют в selected_columns: %s",
                missing_from_columns,
            )

        # --- Детерминированное построение blueprint (без LLM) ---
        blueprint = _deterministic_blueprint(
            intent=intent,
            selected_columns=selected_columns,
            join_spec=join_spec,
            table_types=table_types,
            join_analysis_data=join_analysis_data,
            user_input=state.get("user_input", ""),
            user_hints=state.get("user_hints", {}) or {},
            schema_loader=self.schema,
            semantic_frame=state.get("semantic_frame", {}) or {},
            user_filter_choices=state.get("user_filter_choices", {}) or {},
            rejected_filter_choices=state.get("rejected_filter_choices", {}) or {},
            count_identifier_resolver=self._resolve_count_identifier_llm,
            filter_tiebreaker=self._tiebreak_filter_candidates_llm,
            filter_specs=list((state.get("query_spec") or {}).get("filters") or []),
        )

        logger.info("SqlPlanner: стратегия=%s (детерминировано)", blueprint.get("strategy"))

        where_resolution = blueprint.get("where_resolution", {}) or {}
        filter_confidence = evaluate_filter_confidence(
            where_resolution,
            semantic_frame=state.get("semantic_frame", {}) or {},
            intent=intent,
        )
        previous_planning = state.get("planning_confidence", {}) or {}
        previous_components = previous_planning.get("components", {}) if isinstance(previous_planning, dict) else {}
        planning_confidence = build_planning_confidence(
            table_confidence=previous_components.get("table_confidence") or previous_planning,
            filter_confidence=filter_confidence,
            join_confidence=previous_components.get("join_confidence")
            or evaluate_join_confidence(state.get("join_decision", {}) or {}),
            user_hints=state.get("user_hints"),
        )
        evidence_trace = dict(state.get("evidence_trace") or {})
        evidence_trace["where_resolution"] = {
            "applied_rules": where_resolution.get("applied_rules", []),
            "reasoning": where_resolution.get("reasoning", []),
            "filter_candidates": {
                k: [
                    {
                        "column": c.get("column"),
                        "condition": c.get("condition"),
                        "score": c.get("score"),
                        "confidence": c.get("confidence"),
                    }
                    for c in (v or [])[:3]
                ]
                for k, v in (where_resolution.get("filter_candidates", {}) or {}).items()
            },
        }
        evidence_trace["planning_confidence"] = planning_confidence

        # --- Блок C: NEEDS_YEAR guard ---
        # Если intent_classifier уже поставил month_without_year=True, pipeline должен
        # был остановиться на clarification. Если дошли сюда — where_conditions может
        # содержать маркер NEEDS_YEAR от _derive_date_filters_from_text. Прерываем.
        if any("NEEDS_YEAR" in str(w) for w in blueprint.get("where_conditions", [])):
            import re as _re
            _q = (state.get("user_input") or "").lower()
            _ru_stems = {
                'январ': 'январь', 'феврал': 'февраль', 'март': 'март',
                'апрел': 'апрель', 'май': 'май', 'мая': 'май',
                'июн': 'июнь', 'июл': 'июль', 'август': 'август',
                'сентябр': 'сентябрь', 'октябр': 'октябрь',
                'ноябр': 'ноябрь', 'декабр': 'декабрь',
            }
            _month = next((v for k, v in _ru_stems.items() if k in _q), "указанный месяц")
            _clarif = f"За какой год считать данные за {_month}?"
            logger.warning("SqlPlanner: NEEDS_YEAR в where_conditions → clarification: %r", _clarif)
            return {
                "needs_clarification": True,
                "clarification_message": _clarif,
                "sql_blueprint": blueprint,
                "where_resolution": where_resolution,
                "planning_confidence": planning_confidence,
                "evidence_trace": evidence_trace,
                "graph_iterations": iterations,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": _clarif}
                ],
            }

        if where_resolution.get("needs_clarification"):
            _clarif = str(where_resolution.get("clarification_message") or "").strip()
            return {
                "needs_clarification": True,
                "clarification_message": _clarif,
                "sql_blueprint": blueprint,
                "where_resolution": where_resolution,
                "planning_confidence": planning_confidence,
                "evidence_trace": evidence_trace,
                "graph_iterations": iterations,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": _clarif}
                ],
            }

        if planning_confidence.get("action") != "execute":
            clarification_spec = _build_specific_clarification_spec(where_resolution)
            if clarification_spec:
                where_resolution = dict(where_resolution)
                where_resolution["clarification_spec"] = clarification_spec
            _clarif = str(clarification_spec.get("message") or _build_specific_clarification(where_resolution))
            # Пустая строка — задавать нечего: все request_id закрыты через
            # user_filter_choices. Не блокируем пайплайн фейковым clarification.
            if _clarif:
                return {
                    "needs_clarification": True,
                    "clarification_message": _clarif,
                    "sql_blueprint": blueprint,
                    "where_resolution": where_resolution,
                    "planning_confidence": planning_confidence,
                    "evidence_trace": evidence_trace,
                    "graph_iterations": iterations,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": _clarif}
                    ],
                }
            logger.info(
                "SqlPlanner: planning_confidence=%s, но все filter request_id закрыты "
                "через user_filter_choices — продолжаю выполнение",
                planning_confidence.get("action"),
            )

        # Если dim-таблица пропущена — формируем подсказку для повторного column_selector
        new_hint = ""
        if missing_from_columns and not state.get("column_selector_hint", ""):
            miss_str = ", ".join(missing_from_columns)
            name_col_examples: list[str] = []
            for tbl in missing_from_columns:
                parts = tbl.split(".", 1)
                if len(parts) == 2:
                    try:
                        cols_df = self.schema.get_table_columns(parts[0], parts[1])
                        if not cols_df.empty and "column_name" in cols_df.columns:
                            name_cols = [
                                c for c in cols_df["column_name"].tolist()
                                if any(c.endswith(sfx) for sfx in
                                       ("_name", "_short_name", "_full_name"))
                            ]
                            if name_cols:
                                name_col_examples.append(
                                    f"{tbl}: {', '.join(name_cols[:3])}"
                                )
                    except Exception:
                        pass

            hint_lines = [
                f"Таблица {miss_str} была выбрана планировщиком, но ты не включил "
                f"её в selected_columns на предыдущем шаге.",
                f"Пользователь запрашивает данные из этой таблицы — ОБЯЗАТЕЛЬНО "
                f"включи {miss_str} в columns и заполни join_keys для связи с "
                f"основной таблицей.",
            ]
            if name_col_examples:
                hint_lines.append(
                    "Name-колонки доступные в пропущенных таблицах: "
                    + "; ".join(name_col_examples)
                )
            new_hint = " ".join(hint_lines)
            logger.warning(
                "SqlPlanner: dim-таблица пропущена (%s) — отправляю обратно в column_selector",
                miss_str,
            )

        return {
            "sql_blueprint": blueprint,
            "where_resolution": where_resolution,
            "planning_confidence": planning_confidence,
            "evidence_trace": evidence_trace,
            "column_selector_hint": new_hint,
            "graph_iterations": iterations,
            "messages": state["messages"] + [
                {"role": "assistant",
                 "content": f"SQL стратегия: {blueprint.get('strategy', 'unknown')} [детерминировано]"}
            ],
        }

    # --------------------------------------------------------------------------
    # sql_writer
    # --------------------------------------------------------------------------

    def sql_writer(self, state: AgentState) -> dict[str, Any]:
        """Написание SQL по blueprint: получает готовый план и пишет SQL.

        Returns:
            Обновления состояния с sql_to_validate.
        """
        iterations = state.get("graph_iterations", 0) + 1
        blueprint = state.get("sql_blueprint", {})
        selected_columns = state.get("selected_columns", {})
        join_spec_check = state.get("join_spec", [])
        strategy = blueprint.get("strategy", "simple_select")

        logger.info("SqlWriter: пишу SQL, стратегия=%s", strategy)

        # --- Guard: JOIN-стратегия без join_spec и без второй таблицы ---
        _join_strategies = {"fact_fact_join", "fact_dim_join", "dim_fact_join", "dim_dim_join"}
        if strategy in _join_strategies and not join_spec_check and len(selected_columns) < 2:
            err = (
                f"SqlWriter: стратегия '{strategy}' требует JOIN, "
                f"но join_spec пуст и selected_columns содержит только одну таблицу "
                f"({list(selected_columns.keys())}). "
                f"Невозможно построить корректный JOIN — возврат на column_selector."
            )
            logger.error(err)
            return {
                "last_error": err,
                "sql_to_validate": None,
                "pending_sql_tool_call": None,
                "graph_iterations": iterations,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка: {err}"}
                ],
            }

        # --- Попытка детерминированной генерации SQL через SqlBuilder ---
        table_types = state.get("table_types", {})
        allowed_tables: list[str] = state.get("allowed_tables") or []
        planning_confidence = state.get("planning_confidence", {}) or {}
        evidence_trace = dict(state.get("evidence_trace") or {})
        _builder = _SqlBuilder()
        template_sql = _builder.build(
            strategy=strategy,
            selected_columns=selected_columns,
            join_spec=join_spec_check,
            blueprint=blueprint,
            table_types=table_types,
        )
        if template_sql:
            template_sql = _format_sql(template_sql)
            # Проверяем статическим чекером до принятия (включая белый список таблиц)
            check_result = check_sql(
                template_sql,
                schema_loader=self.schema,
                allowed_tables=allowed_tables if allowed_tables else None,
            )
            fallback_policy = build_fallback_policy(
                planning_confidence=planning_confidence,
                deterministic_sql_valid=check_result.is_valid,
                has_template_sql=True,
            )
            if check_result.is_valid:
                logger.info("SqlWriter: SQL сгенерирован детерминированно, минуем LLM")
                step_idx = state["current_step"]
                evidence_trace["sql_generation"] = {
                    "mode": "deterministic",
                    "strategy": strategy,
                    "fallback_policy": fallback_policy,
                }
                return {
                    "sql_to_validate": template_sql,
                    "pending_sql_tool_call": {
                        "tool": "execute_query",
                        "args": {"sql": template_sql},
                        "step_idx": step_idx,
                    },
                    "graph_iterations": iterations,
                    "fallback_policy": fallback_policy,
                    "evidence_trace": evidence_trace,
                    "tool_calls": state.get("tool_calls", []) + [
                        {"tool": "execute_query", "args": {"sql": template_sql},
                         "result": "awaiting_validation"}
                    ],
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": "SQL сгенерирован детерминированно"}
                    ],
                }
            else:
                logger.info(
                    "SqlWriter: шаблонный SQL не прошёл статический чекер (%s) — передаём LLM",
                    check_result.summary()[:200],
                )
                if not fallback_policy.get("allow_llm_fallback"):
                    msg = str(fallback_policy.get("message") or "Генерация SQL остановлена по policy.")
                    evidence_trace["sql_generation"] = {
                        "mode": "blocked_before_llm",
                        "strategy": strategy,
                        "fallback_policy": fallback_policy,
                    }
                    result: dict[str, Any] = {
                        "graph_iterations": iterations,
                        "fallback_policy": fallback_policy,
                        "evidence_trace": evidence_trace,
                        "messages": state["messages"] + [
                            {"role": "assistant", "content": msg}
                        ],
                    }
                    if fallback_policy.get("action") == "clarify":
                        result["needs_clarification"] = True
                        result["clarification_message"] = msg
                    else:
                        result["last_error"] = msg
                    return result
        else:
            fallback_policy = build_fallback_policy(
                planning_confidence=planning_confidence,
                deterministic_sql_valid=False,
                has_template_sql=False,
            )
            if not fallback_policy.get("allow_llm_fallback"):
                msg = str(fallback_policy.get("message") or "Генерация SQL через LLM остановлена по policy.")
                evidence_trace["sql_generation"] = {
                    "mode": "blocked_before_llm",
                    "strategy": strategy,
                    "fallback_policy": fallback_policy,
                }
                result = {
                    "graph_iterations": iterations,
                    "fallback_policy": fallback_policy,
                    "evidence_trace": evidence_trace,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": msg}
                    ],
                }
                if fallback_policy.get("action") == "clarify":
                    result["needs_clarification"] = True
                    result["clarification_message"] = msg
                else:
                    result["last_error"] = msg
                return result

        # --- System prompt (~3K) ---
        # Выбираем 1-2 релевантных примера по стратегии
        relevant_examples = _STRATEGY_EXAMPLES.get(strategy, "")
        if not relevant_examples:
            relevant_examples = _STRATEGY_EXAMPLES["simple_select"]

        # Формируем правило JOIN на основе фактических данных join_spec:
        # - safe=True (проверено схемой) → прямой JOIN допустим
        # - safe=False → DISTINCT ON / агрегация по реальному ключу из join_spec
        # - Composite JOIN: несколько записей для одной пары таблиц → AND в ON
        join_subquery_warning = _build_join_rule(strategy, join_spec_check)

        # Формируем секцию разрешённых таблиц для LLM
        _allowed_section = ""
        if allowed_tables:
            _allowed_section = (
                "\n\n⚠ КРИТИЧНО — РАЗРЕШЁННЫЕ ТАБЛИЦЫ (только они):\n"
                + "\n".join(f"  - {t}" for t in allowed_tables)
                + "\nЛЮБАЯ другая таблица в FROM/JOIN — ЗАПРЕЩЕНА. "
                "Если нужной метрики нет в этих таблицах — используй COUNT(*) или SUM "
                "из доступных колонок, но НЕ выдумывай другие таблицы.\n"
            )

        # Формируем секцию обязательных колонок в SELECT
        _required_section = ""
        _required_output = blueprint.get("required_output") or []
        if _required_output:
            _required_section = (
                "\n\n⚠ ОБЯЗАТЕЛЬНО в SELECT (пользователь явно запросил):\n"
                + "\n".join(f"  - {r}" for r in _required_output)
                + "\nЭти измерения ДОЛЖНЫ присутствовать в SELECT и GROUP BY финального запроса.\n"
            )

        system_prompt = (
            "Ты — SQL-писатель для Greenplum (PostgreSQL-совместимая).\n"
            "Пиши SQL СТРОГО по blueprint. Не меняй стратегию.\n\n"
            "Правила:\n"
            "- Имена таблиц ВСЕГДА в формате schema.table\n"
            "- Алиасы СТРОГО на английском: AS total_cnt, AS outflow\n"
            "- ЗАПРЕЩЕНО: кириллица в алиасах\n"
            "- Даты: используй ::date, ::timestamp, TO_DATE()\n"
            "- NULL: COALESCE() или IS NOT NULL\n"
            "- GROUP BY: все не-агрегированные колонки\n"
            "- DISTINCT на внешнем SELECT — ЗАПРЕЩЁН\n"
            "- Составной JOIN: если join_keys содержит несколько пар для одной пары таблиц — "
            "объединяй ВСЕ условия через AND в одном ON-клаузе\n"
            f"{_allowed_section}"
            f"{_required_section}"
            f"{join_subquery_warning}\n"
            f"{relevant_examples}\n\n"
            "Чеклист:\n"
            "1. Использую ТОЛЬКО таблицы из разрешённого списка?\n"
            "2. Все required_output-атрибуты есть в SELECT и GROUP BY?\n"
            "3. Формат дат соответствует данным?\n"
            "4. NULL обработан для колонок с высоким % NULL?\n"
            "5. JOIN НЕ множит данные (по стратегии из blueprint)?\n"
            "6. GROUP BY содержит все не-агрегированные колонки?\n"
            "7. Алиасы на английском?\n"
            "8. Если join_keys составной (несколько пар) — все условия объединены через AND?\n\n"
            "Верни ТОЛЬКО JSON:\n"
            '{"tool": "execute_query", "args": {"sql": "SELECT ..."}}'
        )

        # --- User prompt (~3-8K) ---
        user_parts = []

        # Blueprint
        bp_str = json.dumps(blueprint, ensure_ascii=False, indent=2)
        user_parts.append(f"SQL Blueprint:\n{bp_str}")

        # Column details for referenced tables
        if selected_columns:
            cols_str = json.dumps(selected_columns, ensure_ascii=False, indent=2)
            user_parts.append(f"Колонки:\n{cols_str}")

        # Join spec
        join_spec = state.get("join_spec", [])
        if join_spec:
            js_str = json.dumps(join_spec, ensure_ascii=False, indent=2)
            user_parts.append(f"JOIN ключи:\n{js_str}")

        # Multi-turn: добавить предыдущий SQL как базу для модификации (followup)
        prev_sql = state.get("prev_sql", "")
        if prev_sql and state.get("intent", {}).get("intent") == "followup":
            user_parts.append(
                f"ПРЕДЫДУЩИЙ SQL (измени его под новый запрос, не переписывай с нуля):\n{prev_sql}"
            )

        # Few-shot примеры из audit log (похожие успешные запросы)
        try:
            semantic_frame = state.get("semantic_frame") or {}
            fact_dim_pair: tuple[str, str] | None = None
            selected_tables = state.get("selected_tables") or []
            table_types = state.get("table_types") or {}
            if selected_tables:
                fact = next(
                    (f"{s}.{t}" for (s, t) in selected_tables if table_types.get(f"{s}.{t}") == "fact"),
                    "",
                )
                dim = next(
                    (f"{s}.{t}" for (s, t) in selected_tables if table_types.get(f"{s}.{t}") == "dim"),
                    "",
                )
                if fact or dim:
                    fact_dim_pair = (fact, dim)
            few_shot_examples = self.few_shot.get_similar(
                state["user_input"],
                strategy=strategy,
                n=2,
                semantic_frame=semantic_frame,
                fact_dim_pair=fact_dim_pair,
            )
            if few_shot_examples:
                user_parts.append(self.few_shot.format_for_prompt(few_shot_examples))
                logger.debug("SqlWriter: добавлено %d few-shot примеров", len(few_shot_examples))
            evidence_trace["few_shot"] = {
                "count": len(few_shot_examples),
                "similarities": self.few_shot.last_similarities,
                "strategy": strategy,
            }
        except Exception as _exc:
            evidence_trace.setdefault("few_shot", {"count": 0, "similarities": [], "error": str(_exc)})

        user_prompt = "\n\n".join(user_parts)

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — sql_writer]\n{'='*80}\n"
                  f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n")

        response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.1)

        # --- Парсинг вызова инструмента ---
        tool_call = self._parse_tool_call(response)

        sql = tool_call.get("args", {}).get("sql")
        if sql:
            sql = _format_sql(sql)
            if isinstance(tool_call.get("args"), dict):
                tool_call["args"]["sql"] = sql
        tool_name = tool_call.get("tool", "execute_query")

        if sql and tool_name in self.SQL_TOOL_NAMES:
            logger.info("SqlWriter: SQL готов, отправляю на валидацию")
            step_idx = state["current_step"]
            evidence_trace["sql_generation"] = {
                "mode": "llm_fallback",
                "strategy": strategy,
                "fallback_policy": fallback_policy,
            }
            return {
                "sql_to_validate": sql,
                "pending_sql_tool_call": {
                    "tool": tool_name,
                    "args": dict(tool_call.get("args", {})),
                    "step_idx": step_idx,
                },
                "graph_iterations": iterations,
                "fallback_policy": fallback_policy,
                "evidence_trace": evidence_trace,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": tool_name, "args": tool_call.get("args", {}),
                     "result": "awaiting_validation"}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant",
                     "content": f"SQL отправлен на валидацию"}
                ],
            }

        # Если tool=none (ответ без SQL)
        if tool_call.get("tool") == "none":
            result_str = tool_call.get("result", response)
            self.memory.add_message("tool", f"[sql_writer] {result_str[:500]}")
            return {
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": "none", "args": {}, "result": result_str}
                ],
                "current_step": state["current_step"] + 1,
                "last_error": None,
                "retry_count": 0,
                "graph_iterations": iterations,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": result_str[:1000]}
                ],
            }

        # Не-SQL инструмент — вызываем напрямую
        tool_result = self._call_tool(tool_call["tool"], tool_call.get("args", {}))
        result_str = str(tool_result)
        if not tool_result.success:
            return {
                "last_error": tool_result.error,
                "graph_iterations": iterations,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": tool_call["tool"], "args": tool_call.get("args", {}),
                     "result": result_str}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка: {result_str}"}
                ],
            }

        self.memory.add_message("tool", f"[{tool_call['tool']}] {result_str[:500]}")
        return {
            "tool_calls": state.get("tool_calls", []) + [
                {"tool": tool_call["tool"], "args": tool_call.get("args", {}),
                 "result": result_str}
            ],
            "current_step": state["current_step"] + 1,
            "last_error": None,
            "retry_count": 0,
            "graph_iterations": iterations,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Результат: {result_str[:1000]}"}
            ],
        }

    # --------------------------------------------------------------------------
    # sql_static_checker
    # --------------------------------------------------------------------------

    def sql_static_checker(self, state: AgentState) -> dict[str, Any]:
        """Детерминированная проверка SQL до отправки в БД (без LLM).

        Проверяет:
        - кириллические алиасы
        - SELECT *
        - галлюцинированные колонки (не существуют в каталоге)

        При ошибке → error_diagnoser. При успехе → sql_validator.
        """
        sql = state.get("sql_to_validate")
        pending_call = state.get("pending_sql_tool_call")
        if not sql and pending_call:
            sql = pending_call.get("args", {}).get("sql")
        if not sql:
            return {}

        iterations = state.get("graph_iterations", 0) + 1
        logger.info("StaticChecker: проверка SQL (%d символов)", len(sql))

        _allowed = state.get("allowed_tables") or None
        check_result = check_sql(sql, schema_loader=self.schema, allowed_tables=_allowed)

        if not check_result.is_valid:
            error_msg = (
                "SQL не выполнен: ошибка статической проверки.\n"
                f"[sql_static_checker] {check_result.summary()}\n"
                "Исправь SQL перед отправкой в БД."
            )
            logger.warning("StaticChecker: найдены ошибки: %s", error_msg[:300])
            return {
                "last_error": error_msg,
                "sql_to_validate": None,
                "pending_sql_tool_call": None,
                "graph_iterations": iterations,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ],
            }

        if check_result.warnings:
            logger.info(
                "StaticChecker: предупреждения: %s",
                "; ".join(check_result.warnings),
            )

        return {"graph_iterations": iterations}

    # --------------------------------------------------------------------------
    # sql_validator_node
    # --------------------------------------------------------------------------

    def sql_validator_node(self, state: AgentState) -> dict[str, Any]:
        """Валидация SQL: проверка синтаксиса, JOIN safety, выполнение, проверка результата.

        В основном код, с одним лёгким LLM-вызовом для семантической проверки.

        Returns:
            Обновления состояния.
        """
        sql = state.get("sql_to_validate")
        pending_call = state.get("pending_sql_tool_call")
        if not sql and pending_call:
            sql = pending_call.get("args", {}).get("sql")
        if not sql:
            return {"sql_to_validate": None, "pending_sql_tool_call": None}

        logger.info("Validator: проверка SQL: %s", summarize_sql(sql))
        result = self.validator.validate(sql)

        # Требуется подтверждение пользователя
        if result.needs_confirmation:
            return {
                "needs_confirmation": True,
                "confirmation_message": result.confirmation_message,
                "sql_to_validate": sql,
                "pending_sql_tool_call": pending_call,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": result.confirmation_message}
                ],
            }

        # JOIN risk info
        join_risk_info = {}
        if result.join_checks:
            join_risk_info = {
                "multiplication_factor": result.multiplication_factor,
                "non_unique_joins": [
                    jc for jc in result.join_checks if not jc["is_unique"]
                ],
                "rewrite_suggestions": result.rewrite_suggestions,
            }

        # Невалидный SQL
        if not result.is_valid:
            error_msg = (
                "SQL не выполнен: ошибка валидации.\n"
                f"{result.summary()}"
            )
            logger.warning("Validator: SQL невалиден: %s", summarize_text(error_msg, label="validation_error"))
            return {
                "last_error": error_msg,
                "sql_to_validate": None,
                "pending_sql_tool_call": None,
                "join_risk_info": join_risk_info,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка валидации:\n{error_msg}"}
                ],
            }

        # SQL валиден — выполняем
        tool_calls = state.get("tool_calls", [])
        last_tool = tool_calls[-1] if tool_calls else {}
        tool_name = (pending_call or {}).get("tool") or last_tool.get("tool", "execute_query")
        tool_args = dict((pending_call or {}).get("args", {}))
        if not tool_args:
            tool_args = {"sql": sql}

        t0 = time.monotonic()
        tool_result = self._call_tool(tool_name, tool_args)
        duration_ms = int((time.monotonic() - t0) * 1000)
        exec_result = str(tool_result)
        structured_payload = None
        if tool_name in self.SQL_TOOL_NAMES:
            structured_payload = self._parse_sql_tool_payload(exec_result)
        rendered_result = self._render_tool_result(exec_result, structured_payload)

        # Ошибка выполнения
        if not tool_result.success:
            self.memory.log_sql_execution(
                state["user_input"], sql, 0, "error", duration_ms,
                retry_count=state.get("retry_count", 0), error_type="execution",
            )
            return {
                "sql_to_validate": None,
                "pending_sql_tool_call": None,
                "last_error": tool_result.error,
                "tool_calls": tool_calls[:-1] + [
                    {**last_tool, "result": rendered_result}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка выполнения SQL:\n{rendered_result}"}
                ],
            }

        # Пустой результат
        empty_result = False
        if tool_name == "execute_query" and structured_payload is not None:
            empty_result = bool(structured_payload.get("is_empty", False))
        elif tool_name == "execute_query" and (
            rendered_result == "Запрос выполнен. Результат пуст."
            or rendered_result.strip() == ""
        ):
            empty_result = True

        # Предупреждения
        warnings_text = ""
        if result.warnings:
            warnings_text = "\nПредупреждения:\n" + "\n".join(
                f"  ⚠ {w}" for w in result.warnings
            )

        if empty_result:
            warnings_text += (
                "\n⚠ Запрос вернул 0 строк. Возможно, условия фильтрации "
                "слишком строгие или данные отсутствуют."
            )

        # Подсчёт строк
        if structured_payload is not None:
            row_count = max(0, int(structured_payload.get("rows_returned", 0)))
        else:
            data_lines = [
                ln for ln in rendered_result.split("\n") if ln.strip().startswith("|")
            ]
            row_count = max(0, len(data_lines) - 2) if data_lines else 0

        # Sanity checks
        if not empty_result and tool_name == "execute_query":
            sanity_source = (
                str(structured_payload.get("preview_markdown", ""))
                if structured_payload is not None
                else rendered_result
            )
            sanity_warnings = self._check_result_sanity(state["user_input"], sanity_source)
            for sw in sanity_warnings:
                warnings_text += f"\n⚠ {sw}"

        # Row-count sanity по истории (Direction 3.3):
        # Сверяем row_count с p95 * 10 по (subject, metric) из semantic_frame.
        # Запускается только для успешных запросов с непустым результатом.
        frame = state.get("semantic_frame") or {}
        subject = str(frame.get("subject") or "")
        metric_intent = str(frame.get("metric_intent") or "")
        if (
            not empty_result
            and tool_name == "execute_query"
            and (subject or metric_intent)
        ):
            suspicion = self.memory.check_row_count_suspicion(
                subject, metric_intent, row_count,
            )
            if suspicion.get("is_suspect"):
                p95 = suspicion.get("p95", 0.0)
                n_hist = suspicion.get("n", 0)
                ratio = suspicion.get("ratio", 0.0)
                suspect_msg = (
                    f"ROW-COUNT SUSPECT: {row_count} строк для (subject={subject or '_'}, "
                    f"metric={metric_intent or '_'}). "
                    f"Исторический p95={p95:.0f} по {n_hist} наблюдениям, "
                    f"текущий результат в {ratio:.1f}× больше p95 (порог: 10×). "
                    "Вероятно, JOIN множит строки или фильтр слишком широкий."
                )
                self.memory.log_sql_execution(
                    state["user_input"], sql, row_count, "row_explosion_suspect",
                    duration_ms, retry_count=state.get("retry_count", 0),
                    error_type="row_count_suspect",
                )
                return {
                    "sql_to_validate": None,
                    "pending_sql_tool_call": None,
                    "last_error": suspect_msg,
                    "retry_count": state.get("retry_count", 0),
                    "join_risk_info": join_risk_info,
                    "tool_calls": tool_calls[:-1] + [
                        {**last_tool, "result": rendered_result}
                    ],
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"⚠ {suspect_msg}"}
                    ],
                }

        # Row explosion detection
        if not empty_result and join_risk_info and tool_name == "execute_query":
            factor = join_risk_info.get("multiplication_factor", 1.0)
            if factor > 1.5 and row_count > 50:
                suggestions = "\n".join(
                    join_risk_info.get("rewrite_suggestions", [])
                )
                explosion_msg = (
                    f"POST-EXECUTION ROW EXPLOSION: {row_count} строк "
                    f"при factor={factor:.1f}x. "
                    f"Дублирование из-за JOIN.\n{suggestions}"
                )
                self.memory.log_sql_execution(
                    state["user_input"], sql, row_count, "row_explosion",
                    duration_ms, retry_count=state.get("retry_count", 0),
                    error_type="join_explosion",
                )
                return {
                    "sql_to_validate": None,
                    "pending_sql_tool_call": None,
                    "last_error": explosion_msg,
                    "retry_count": state.get("retry_count", 0),
                    "join_risk_info": join_risk_info,
                    "tool_calls": tool_calls[:-1] + [
                        {**last_tool, "result": rendered_result}
                    ],
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"⚠ {explosion_msg}"}
                    ],
                }

        # Аудит
        audit_status = "empty" if empty_result else "success"
        self.memory.log_sql_execution(
            state["user_input"], sql, row_count, audit_status, duration_ms,
            retry_count=state.get("retry_count", 0),
        )
        self.memory.add_message("tool", f"[{tool_name}] {rendered_result[:500]}")
        # Сбрасываем few-shot кэш: новый успешный запрос попадёт в следующую сессию
        if audit_status == "success":
            self.few_shot.invalidate_cache()
            # Накопление row_count_stats для будущей row-count sanity (Direction 3.3)
            try:
                self.memory.record_row_count_sample(
                    subject, metric_intent, row_count,
                )
            except Exception as e:
                logger.debug("record_row_count_sample failed: %s", e)

        # Пустой результат → коррекция
        if empty_result:
            return {
                "sql_to_validate": None,
                "pending_sql_tool_call": None,
                "last_error": (
                    f"SQL-запрос выполнен, но вернул 0 строк. SQL: {sql}\n"
                    "Проверь WHERE, формат дат, значения фильтров."
                ),
                "retry_count": state.get("retry_count", 0),
                "join_risk_info": join_risk_info,
                "tool_calls": tool_calls[:-1] + [
                    {**last_tool, "result": rendered_result}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant",
                     "content": f"SQL выполнен, 0 строк.{warnings_text}"}
                ],
            }

        return {
            "sql_to_validate": None,
            "pending_sql_tool_call": None,
            "last_error": None,
            "retry_count": 0,
            "join_risk_info": join_risk_info,
            "current_step": state["current_step"] + 1,
            "tool_calls": tool_calls[:-1] + [
                {**last_tool, "result": rendered_result}
            ],
            "messages": state["messages"] + [
                {"role": "assistant",
                 "content": f"SQL выполнен.{warnings_text}\n{rendered_result[:1000]}"}
            ],
        }

    # --------------------------------------------------------------------------
    # _parse_json_response — общий парсер JSON из ответа LLM
    # --------------------------------------------------------------------------

    def _parse_json_response(self, response: str) -> dict[str, Any] | None:
        """Извлечь JSON-объект из ответа LLM (без требования ключа 'tool')."""
        cleaned = self._clean_llm_json(response)
        for candidate in self._extract_json_objects(cleaned):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue
        return None
