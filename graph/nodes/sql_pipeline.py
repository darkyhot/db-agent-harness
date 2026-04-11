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

from core.sql_static_checker import check_sql
from core.sql_planner_deterministic import build_blueprint as _deterministic_blueprint
from core.sql_builder import SqlBuilder as _SqlBuilder
from core.sql_formatter import format_sql_safe as _format_sql
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


class SqlPipelineNodes:
    """Миксин с узлами sql_planner, sql_writer и sql_validator_node для GraphNodes."""

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
        )

        logger.info("SqlPlanner: стратегия=%s (детерминировано)", blueprint.get("strategy"))

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
            # Проверяем статическим чекером до принятия
            check_result = check_sql(template_sql, schema_loader=self.schema)
            if check_result.is_valid:
                logger.info("SqlWriter: SQL сгенерирован детерминированно, минуем LLM")
                step_idx = state["current_step"]
                return {
                    "sql_to_validate": template_sql,
                    "pending_sql_tool_call": {
                        "tool": "execute_query",
                        "args": {"sql": template_sql},
                        "step_idx": step_idx,
                    },
                    "graph_iterations": iterations,
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
            f"{join_subquery_warning}\n"
            f"{relevant_examples}\n\n"
            "Чеклист:\n"
            "1. Формат дат соответствует данным?\n"
            "2. NULL обработан для колонок с высоким % NULL?\n"
            "3. JOIN НЕ множит данные (по стратегии из blueprint)?\n"
            "4. GROUP BY содержит все не-агрегированные колонки?\n"
            "5. Алиасы на английском?\n"
            "6. Если join_keys составной (несколько пар) — все условия объединены через AND?\n\n"
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
            few_shot_examples = self.few_shot.get_similar(
                state["user_input"], strategy=strategy, n=2
            )
            if few_shot_examples:
                user_parts.append(self.few_shot.format_for_prompt(few_shot_examples))
                logger.debug("SqlWriter: добавлено %d few-shot примеров", len(few_shot_examples))
        except Exception:
            pass  # few-shot не критичен — продолжаем без него

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
            return {
                "sql_to_validate": sql,
                "pending_sql_tool_call": {
                    "tool": tool_name,
                    "args": dict(tool_call.get("args", {})),
                    "step_idx": step_idx,
                },
                "graph_iterations": iterations,
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

        check_result = check_sql(sql, schema_loader=self.schema)

        if not check_result.is_valid:
            error_msg = (
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

        logger.info("Validator: проверка SQL: %s", sql[:200])
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
            error_msg = result.summary()
            logger.warning("Validator: SQL невалиден: %s", error_msg[:200])
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
