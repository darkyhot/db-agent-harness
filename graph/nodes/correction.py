"""Узлы коррекции SQL: диагностика ошибок и исправление запросов.

Содержит CorrectionNodes — миксин для GraphNodes с методами:
- error_diagnoser: классификация ошибки и выбор стратегии исправления
- sql_fixer: переписывание SQL на основе диагноза
"""

import json
import logging
import re
import time
from typing import Any

from graph.nodes.common import BaseNodeMixin, ToolResult, SQL_RULES, GIGACHAT_COMMON_ERRORS
from graph.state import AgentState

logger = logging.getLogger(__name__)

# === Константы для типов ошибок ===

_JOIN_FIX_RULES = (
    "Стратегии безопасного JOIN по типам таблиц:\n\n"
    "1. ФАКТ + ФАКТ → предварительная агрегация ОБЕИХ сторон в CTE:\n"
    "   WITH cte1 AS (\n"
    "     SELECT join_key, SUM(metric1) AS val1 FROM schema.fact1 GROUP BY join_key\n"
    "   ), cte2 AS (\n"
    "     SELECT join_key, SUM(metric2) AS val2 FROM schema.fact2 GROUP BY join_key\n"
    "   )\n"
    "   SELECT * FROM cte1 JOIN cte2 ON cte1.join_key = cte2.join_key\n\n"
    "2. ФАКТ + СПРАВОЧНИК → уникальная выборка из справочника:\n"
    "   JOIN (SELECT DISTINCT ON (key) key, name FROM schema.dim ORDER BY key, date DESC) d ON d.key = f.key\n\n"
    "3. СПРАВОЧНИК + ФАКТ → агрегация фактов в CTE:\n"
    "   JOIN (SELECT key, SUM(amount) AS total FROM schema.fact GROUP BY key) agg ON agg.key = d.key\n\n"
    "4. СПРАВОЧНИК + СПРАВОЧНИК → уникальные выборки из ОБЕИХ сторон:\n"
    "   WITH d1 AS (SELECT DISTINCT ON (key) ... FROM dim1),\n"
    "        d2 AS (SELECT DISTINCT ON (key) ... FROM dim2)\n"
    "   SELECT * FROM d1 JOIN d2 ON d1.key = d2.key\n\n"
    "- DISTINCT на внешнем SELECT — ЗАПРЕЩЁН: маскирует проблему\n"
    "- Составной PK (>1 колонки) — каждая колонка НЕ уникальна сама по себе"
)

_FIX_HINTS: dict[str, str] = {
    "row_explosion": (
        "Исправление row explosion в JOIN:\n" + _JOIN_FIX_RULES
    ),
    "column_not_found": (
        "Используй ТОЛЬКО колонки из списка реальных колонок ниже.\n"
        "ЗАПРЕЩЕНО угадывать имена колонок. Если колонки нет в списке — "
        "используй get_table_columns или search_by_description для поиска."
    ),
    "date_format": (
        "Используй ::date, TO_DATE() для приведения типов дат.\n"
        "Проверь формат даты в образце данных (YYYY-MM-DD, DD.MM.YYYY и т.д.).\n"
        "Пример: WHERE dt >= TO_DATE('01.01.2024', 'DD.MM.YYYY')"
    ),
    "empty_result": (
        "Ослабь условия WHERE. Проверь формат значений в фильтрах.\n"
        "Возможно фильтр по дате или категории слишком узкий.\n"
        "Используй get_sample чтобы посмотреть реальные значения."
    ),
    "syntax_error": (
        "Исправь синтаксис по тексту ошибки. Типичные причины:\n"
        "- Пропущена запятая, скобка, ключевое слово\n"
        "- Кириллица в алиасах или именах колонок\n"
        "- Несовместимый синтаксис (не PostgreSQL/Greenplum)\n\n"
        "ВАЖНО — ошибка «must appear in the GROUP BY clause»:\n"
        "Это НЕ синтаксическая ошибка — это семантическая. Не добавляй GROUP BY механически.\n"
        "Сначала определи: должна ли проблемная колонка быть агрегирована?\n"
        "- Если да (колонка является PK/метрикой и запрос считает итог) — "
        "оберни её в COUNT(DISTINCT ...) или другую агрегацию, а GROUP BY НЕ добавляй.\n"
        "- Если нет (колонка нужна для разбивки) — добавь её в GROUP BY.\n"
        "Добавление GROUP BY к PK-колонке при запросе «сколько всего X» "
        "меняет смысл с «итоговый счётчик» на «разбивка по X» — это НЕВЕРНО."
    ),
    "type_mismatch": (
        "Несовместимые типы данных в сравнении или JOIN.\n"
        "Используй явное приведение типов: ::text, ::integer, ::date.\n"
        "Проверь типы колонок через get_table_columns."
    ),
    "other": (
        "Проанализируй текст ошибки и исправь SQL.\n"
        "Если причина неясна — вызови get_sample или get_table_columns для диагностики."
    ),
}


class CorrectionNodes:
    """Миксин с узлами error_diagnoser и sql_fixer для GraphNodes."""

    # ------------------------------------------------------------------
    # error_diagnoser
    # ------------------------------------------------------------------

    def error_diagnoser(self, state: AgentState) -> dict[str, Any]:
        """Узел диагностики ошибок: классифицирует ошибку и выбирает стратегию.

        Анализирует последнюю ошибку, определяет её тип и решает,
        можно ли исправить тривиальной заменой или нужен sql_fixer.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с диагнозом ошибки.
        """
        error = state.get("last_error", "")
        retry_count = state.get("retry_count", 0)
        step_idx = state["current_step"]
        plan = state["plan"]
        current_step = plan[step_idx] if step_idx < len(plan) else "неизвестный шаг"
        iterations = state.get("graph_iterations", 0) + 1

        logger.info(
            "ErrorDiagnoser: попытка %d/%d, ошибка: %s",
            retry_count + 1, self.MAX_RETRIES, error[:200],
        )

        # --- Проверка лимита попыток ---
        if retry_count >= self.MAX_RETRIES:
            replan_count = state.get("replan_count", 0)
            # Budget-aware: если осталось < 60 секунд до wall-clock timeout — пропускаем replanning
            _budget_ok = True
            start_t = state.get("start_time", 0)
            if start_t:
                from graph.graph import MAX_WALL_CLOCK_SECONDS
                elapsed = time.monotonic() - start_t
                if elapsed > MAX_WALL_CLOCK_SECONDS - 60:
                    _budget_ok = False
                    logger.warning(
                        "ErrorDiagnoser: осталось менее 60с до таймаута (elapsed=%.0fs) — пропускаем replanning",
                        elapsed,
                    )
            # Попробовать replanning один раз перед сдачей (если есть бюджет)
            if replan_count < 1 and _budget_ok:
                logger.info("ErrorDiagnoser: попытки исчерпаны, запрашиваю replanning")
                # Direction 6.5: retry_count сбрасывается (другие ноды используют
                # его как локальный счётчик ретраев текущего шага), но суммарный
                # `total_retry_count` продолжает копить, чтобы replanning не мог
                # «скрыто» удвоить бюджет retry.
                total_retries = int(state.get("total_retry_count", 0)) + int(state.get("retry_count", 0))
                return {
                    "last_error": None,
                    "retry_count": 0,
                    "total_retry_count": total_retries,
                    "replan_count": replan_count + 1,
                    "needs_replan": True,
                    "replan_context": (
                        f"Шаг '{current_step}' не удался после {self.MAX_RETRIES} попыток. "
                        f"Последняя ошибка: {error}"
                    ),
                    "graph_iterations": iterations,
                }
            # Replanning уже был — сдаёмся
            return {
                "last_error": None,
                "current_step": step_idx + 1,
                "retry_count": 0,
                "graph_iterations": iterations,
                "final_answer": (
                    f"Не удалось выполнить шаг '{current_step}' после {self.MAX_RETRIES} попыток. "
                    f"Последняя ошибка: {error}"
                ),
            }

        # --- Извлечение последнего SQL из tool_calls ---
        recent_calls = state.get("tool_calls", [])[-3:]
        last_sql = ""
        if recent_calls:
            last_args = recent_calls[-1].get("args", {})
            if isinstance(last_args, dict):
                last_sql = last_args.get("sql", "")

        # --- Автозагрузка метаданных колонок для таблиц из SQL ---
        column_metadata = ""
        tbl_pattern = re.compile(r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b')
        scan_text = (last_sql or "") + " " + error
        found_tables: set[tuple[str, str]] = set()
        for m in tbl_pattern.finditer(scan_text):
            s, t = m.group(1).lower(), m.group(2).lower()
            cols_df = self.schema.get_table_columns(s, t)
            if not cols_df.empty:
                found_tables.add((s, t))
        if found_tables:
            parts = []
            for s, t in sorted(found_tables):
                info = self.schema.get_table_info(s, t)
                parts.append(f"### {s}.{t}\n{info}")
            column_metadata = "\n\n".join(parts)

        # --- Предыдущие попытки исправления ---
        correction_examples = state.get("correction_examples", [])
        prev_attempts = ""
        if correction_examples:
            recent_fixes = correction_examples[-2:]
            prev_attempts = "\n".join(f"- {ex}" for ex in recent_fixes)

        # --- Системный промпт ---
        system_prompt = (
            "Ты — классификатор SQL-ошибок аналитического агента для Greenplum.\n"
            "Проанализируй ошибку и определи стратегию исправления.\n\n"
            "Категории ошибок:\n"
            "- column_not_found: колонка не найдена в таблице\n"
            "- syntax_error: синтаксическая ошибка SQL\n"
            "- date_format: неправильный формат даты\n"
            "- empty_result: запрос вернул 0 строк\n"
            "- row_explosion: JOIN множит строки\n"
            "- type_mismatch: несовместимые типы данных\n"
            "- other: прочие ошибки\n\n"
            "Формат ответа — строго JSON:\n"
            '{"error_type": "тип_ошибки", "root_cause": "описание причины ошибки",\n'
            ' "fix_strategy": "стратегия_исправления", "replacements": [],\n'
            ' "needs_sample": false, "needs_replan": false}\n\n'
            "Допустимые значения fix_strategy:\n"
            "- replace_column: замена несуществующей колонки на правильную\n"
            "- fix_syntax: исправление синтаксической ошибки\n"
            "- fix_date_format: исправление формата даты\n"
            "- relax_filters: ослабление условий WHERE (0 строк)\n"
            "- fix_join: исправление JOIN (row explosion)\n"
            "- rewrite_sql: полное переписывание запроса\n"
            "- use_tool: вызов диагностического инструмента (get_sample, get_table_columns)\n\n"
            "Если стратегия replace_column — заполни replacements:\n"
            '[{"old": "старое_имя", "new": "правильное_имя"}]\n'
            "Правильное имя бери ТОЛЬКО из метаданных колонок ниже.\n\n"
            "Если причина неясна — установи needs_sample: true для вызова get_sample.\n"
            "Если ошибка неисправима — установи needs_replan: true."
        )

        # --- Авто-сэмпл для empty_result: показываем реальные значения из таблиц ---
        auto_sample_text = ""
        if "0 строк" in error or "empty_result" in error.lower():
            sample_parts = []
            for s, t in list(found_tables)[:2]:  # максимум 2 таблицы чтобы не раздувать промпт
                try:
                    sample_df = self.db.get_sample(s, t, n=5)
                    if not sample_df.empty:
                        sample_md = sample_df.to_markdown(index=False)
                        sample_parts.append(
                            f"Образец {s}.{t} (5 строк — проверь форматы значений для фильтров):\n"
                            f"{sample_md[:600]}"
                        )
                except Exception as e:
                    logger.debug("auto_sample %s.%s: %s", s, t, e)
            if sample_parts:
                auto_sample_text = "\n\n".join(sample_parts)
                logger.info(
                    "ErrorDiagnoser: авто-сэмпл для empty_result: %d таблиц", len(sample_parts)
                )

        # --- Пользовательский промпт ---
        user_parts = []
        if last_sql:
            user_parts.append(f"[SQL КОТОРЫЙ ВЫЗВАЛ ОШИБКУ]\n{last_sql}")
        user_parts.append(f"[СООБЩЕНИЕ ОБ ОШИБКЕ]\n{error}")
        if column_metadata:
            user_parts.append(f"[МЕТАДАННЫЕ КОЛОНОК]\n{column_metadata}")
        if auto_sample_text:
            user_parts.append(f"[ОБРАЗЦЫ ДАННЫХ — для диагностики пустого результата]\n{auto_sample_text}")
        user_parts.append(f"[ПОПЫТКА {retry_count + 1} из {self.MAX_RETRIES}]")
        if prev_attempts:
            user_parts.append(f"[ПРЕДЫДУЩИЕ ПОПЫТКИ ИСПРАВЛЕНИЯ]\n{prev_attempts}")

        user_prompt = "\n\n".join(user_parts)

        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)

        if self.debug_prompt:
            print(
                f"\n{'=' * 80}\n[DEBUG PROMPT — error_diagnoser]\n{'=' * 80}\n"
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'=' * 80}\n"
            )

        response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.1)

        # --- Парсинг диагноза ---
        diagnosis = self._parse_diagnosis(response)

        result: dict[str, Any] = {
            "error_diagnosis": diagnosis,
            "retry_count": retry_count + 1,
            "last_error": None,
            "graph_iterations": iterations,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Диагноз ошибки: {diagnosis.get('error_type', 'unknown')} — {diagnosis.get('root_cause', '')}"}
            ],
        }

        # --- Тривиальное исправление кодом (replace_column) ---
        if (
            diagnosis.get("fix_strategy") == "replace_column"
            and diagnosis.get("replacements")
            and last_sql
        ):
            fixed_sql = last_sql
            all_replacements_valid = True
            for repl in diagnosis["replacements"]:
                old_col = repl.get("old", "")
                new_col = repl.get("new", "")
                if not old_col or not new_col:
                    all_replacements_valid = False
                    break
                # Проверяем что новая колонка реально существует в метаданных
                col_exists = False
                for s, t in found_tables:
                    cols_df = self.schema.get_table_columns(s, t)
                    if not cols_df.empty:
                        col_names = cols_df["column_name"].str.lower().tolist()
                        if new_col.lower() in col_names:
                            col_exists = True
                            break
                if not col_exists:
                    all_replacements_valid = False
                    break
                # Замена с учётом границ слов
                pattern = re.compile(r'\b' + re.escape(old_col) + r'\b', re.IGNORECASE)
                fixed_sql = pattern.sub(new_col, fixed_sql)

            if all_replacements_valid and fixed_sql != last_sql:
                logger.info(
                    "ErrorDiagnoser: тривиальное исправление колонок кодом: %s",
                    diagnosis["replacements"],
                )
                # Определяем инструмент из последнего вызова
                tool_name = "execute_query"
                if recent_calls:
                    tool_name = recent_calls[-1].get("tool", "execute_query")

                result["sql_to_validate"] = fixed_sql
                result["pending_sql_tool_call"] = {
                    "tool": tool_name,
                    "args": {"sql": fixed_sql},
                    "step_idx": step_idx,
                }
                # Обновляем correction_examples
                example = f"Ошибка: {error[:100]} → Замена колонок: {diagnosis['replacements']}"
                result["correction_examples"] = correction_examples + [example]

        return result

    def _parse_diagnosis(self, response: str) -> dict[str, Any]:
        """Извлечь JSON-диагноз ошибки из ответа LLM."""
        cleaned = self._clean_llm_json(response)

        # Попробовать найти JSON в ответе
        for candidate in self._extract_json_objects(cleaned):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "error_type" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue

        # Fallback: попробовать весь ответ
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning("Не удалось распарсить диагноз из ответа LLM, используем fallback")
        return {
            "error_type": "other",
            "root_cause": "Не удалось классифицировать ошибку",
            "fix_strategy": "rewrite_sql",
            "replacements": [],
            "needs_sample": False,
            "needs_replan": False,
        }

    # ------------------------------------------------------------------
    # sql_fixer
    # ------------------------------------------------------------------

    def sql_fixer(self, state: AgentState) -> dict[str, Any]:
        """Узел исправления SQL: переписывает запрос на основе диагноза ошибки.

        Получает диагноз из state['error_diagnosis'] и генерирует
        исправленный SQL-запрос или вызов диагностического инструмента.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с исправленным SQL или результатом вызова инструмента.
        """
        diagnosis = state.get("error_diagnosis", {})
        error_type = diagnosis.get("error_type", "other")
        retry_count = state.get("retry_count", 0)
        step_idx = state["current_step"]
        iterations = state.get("graph_iterations", 0) + 1

        logger.info(
            "SQLFixer: тип ошибки=%s, стратегия=%s",
            error_type, diagnosis.get("fix_strategy", "unknown"),
        )

        # --- Извлечение последнего SQL ---
        recent_calls = state.get("tool_calls", [])[-3:]
        last_sql = ""
        if recent_calls:
            last_args = recent_calls[-1].get("args", {})
            if isinstance(last_args, dict):
                last_sql = last_args.get("sql", "")

        # --- Автозагрузка метаданных колонок ---
        column_context = ""
        tbl_pattern = re.compile(r'\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b')
        scan_text = last_sql or ""
        found_tables: set[tuple[str, str]] = set()
        for m in tbl_pattern.finditer(scan_text):
            s, t = m.group(1).lower(), m.group(2).lower()
            cols_df = self.schema.get_table_columns(s, t)
            if not cols_df.empty:
                found_tables.add((s, t))
        if found_tables:
            parts = []
            for s, t in sorted(found_tables):
                info = self.schema.get_table_info(s, t)
                parts.append(f"### {s}.{t}\n{info}")
            column_context = (
                "[РЕАЛЬНЫЕ КОЛОНКИ ТАБЛИЦ — используй ТОЛЬКО эти колонки!]\n"
                + "\n\n".join(parts)
            )

        # --- Системный промпт (динамический, зависит от error_type) ---
        system_prompt = (
            "Ты — SQL-исправитель для Greenplum/PostgreSQL.\n"
            "Исправь SQL-запрос на основе диагноза ошибки.\n\n"
            f"Доступные инструменты:\n{self.tools_description}\n\n"
            "Формат ответа — СТРОГО один JSON-объект:\n"
            '{"tool": "имя_инструмента", "args": {"sql": "исправленный SQL"}}\n\n'
            "ЗАПРЕЩЕНО вызывать файловые инструменты (create_file, edit_file, delete_file) "
            "для исправления SQL — возвращай только execute_query или диагностические инструменты.\n\n"
        )

        # Добавляем только релевантные правила исправления
        fix_hint = _FIX_HINTS.get(error_type, _FIX_HINTS["other"])
        system_prompt += f"Правила исправления для данного типа ошибки:\n{fix_hint}\n\n"

        # Примеры прошлых исправлений из долгосрочной памяти
        examples = self.memory.get_memory_list("correction_examples")
        if examples:
            recent_examples = examples[-3:]
            system_prompt += (
                "Примеры прошлых исправлений (учитывай эти паттерны):\n"
                + "\n".join(f"  - {ex}" for ex in recent_examples)
                + "\n\n"
            )

        system_prompt += GIGACHAT_COMMON_ERRORS

        # --- Пользовательский промпт ---
        user_parts = []

        if last_sql:
            user_parts.append(f"[ИСХОДНЫЙ SQL]\n{last_sql}")

        # Форматируем диагноз как читаемый текст
        diag_text = (
            f"Тип ошибки: {diagnosis.get('error_type', 'unknown')}\n"
            f"Причина: {diagnosis.get('root_cause', 'неизвестна')}\n"
            f"Стратегия: {diagnosis.get('fix_strategy', 'rewrite_sql')}"
        )
        if diagnosis.get("replacements"):
            repl_text = ", ".join(
                f"{r.get('old', '?')} → {r.get('new', '?')}"
                for r in diagnosis["replacements"]
            )
            diag_text += f"\nЗамены: {repl_text}"
        user_parts.append(f"[ДИАГНОЗ ОШИБКИ]\n{diag_text}")

        if column_context:
            user_parts.append(column_context)

        # Для row_explosion — добавляем join analysis и рекомендации валидатора
        if error_type == "row_explosion":
            join_data = state.get("join_analysis_data", {})
            if join_data:
                user_parts.append(f"[JOIN-АНАЛИЗ]\n{json.dumps(join_data, ensure_ascii=False, indent=2)}")
            # Конкретные предложения по исправлению от валидатора (с реальными именами таблиц/колонок)
            risk = state.get("join_risk_info", {})
            rewrite_suggestions = risk.get("rewrite_suggestions", [])
            if rewrite_suggestions:
                suggestions_text = "\n".join(rewrite_suggestions)
                user_parts.append(
                    f"[РЕКОМЕНДАЦИИ ВАЛИДАТОРА — ИСПОЛЬЗУЙ ЭТОТ ШАБЛОН]\n{suggestions_text}"
                )

        # SQL blueprint — пропускаем для row_explosion: он содержит cte_needed=false,
        # что противоречит требуемому исправлению и вводит LLM в заблуждение
        if error_type != "row_explosion":
            sql_blueprint = state.get("sql_blueprint", "")
            if sql_blueprint:
                user_parts.append(f"[SQL BLUEPRINT]\n{sql_blueprint}")

        user_prompt = "\n\n".join(user_parts)

        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)

        if self.debug_prompt:
            print(
                f"\n{'=' * 80}\n[DEBUG PROMPT — sql_fixer]\n{'=' * 80}\n"
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'=' * 80}\n"
            )

        response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.2)
        tool_call = self._parse_tool_call(response)

        # --- Если исправленный вызов содержит SQL — отправить на валидацию ---
        sql = tool_call.get("args", {}).get("sql")
        if sql and tool_call["tool"] in self.SQL_TOOL_NAMES:
            example = f"Ошибка: {diagnosis.get('root_cause', '')[:100]} → Исправление: {sql[:150]}"
            correction_examples = state.get("correction_examples", []) + [example]
            return {
                "sql_to_validate": sql,
                "pending_sql_tool_call": {
                    "tool": tool_call["tool"],
                    "args": dict(tool_call.get("args", {})),
                    "step_idx": step_idx,
                },
                "retry_count": retry_count,
                "last_error": None,
                "graph_iterations": iterations,
                "correction_examples": correction_examples,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": "awaiting_validation"}
                ],
            }

        # --- Вызов не-SQL инструмента напрямую ---
        _FILE_TOOLS = {"create_file", "edit_file", "delete_file"}
        if tool_call["tool"] in _FILE_TOOLS:
            # Файловые инструменты не имеют смысла при исправлении SQL.
            # Сохраняем оригинальный контекст ошибки и сигнализируем о повторной попытке.
            logger.warning(
                "SQLFixer: вызван файловый инструмент '%s' вместо SQL-инструмента — игнорируем",
                tool_call["tool"],
            )
            original_error = diagnosis.get("root_cause", "row_explosion: прямой JOIN запрещён")
            return {
                "last_error": (
                    f"sql_fixer вызвал недопустимый инструмент '{tool_call['tool']}'. "
                    f"Исходная ошибка: {original_error}"
                ),
                "retry_count": retry_count,
                "graph_iterations": iterations,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка: sql_fixer использовал файловый инструмент. Повтор."}
                ],
            }

        tool_result = self._call_tool(tool_call["tool"], tool_call.get("args", {}))
        result_str = str(tool_result)

        if not tool_result.success:
            return {
                "last_error": tool_result.error,
                "retry_count": retry_count,
                "graph_iterations": iterations,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка инструмента (sql_fixer): {result_str}"}
                ],
            }

        self.memory.add_message("tool", f"[sql_fixer:{tool_call['tool']}] {result_str[:500]}")

        return {
            "tool_calls": state.get("tool_calls", []) + [
                {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": result_str}
            ],
            "current_step": step_idx + 1,
            "last_error": None,
            "retry_count": 0,
            "sql_to_validate": None,
            "pending_sql_tool_call": None,
            "graph_iterations": iterations,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Исправлено (sql_fixer). Шаг {step_idx + 1}: {result_str[:1000]}"}
            ],
        }
