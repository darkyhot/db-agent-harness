"""Узлы классификации интента и выбора таблиц.

Содержит IntentNodes — миксин для GraphNodes с методами:
- intent_classifier: классификация запроса пользователя
- table_resolver: выбор таблиц из каталога на основе интента
- _parse_plan: извлечение плана из ответа LLM
"""

import json
import logging
import re
from typing import Any

from graph.state import AgentState

logger = logging.getLogger(__name__)


class IntentNodes:
    """Миксин с узлами intent_classifier и table_resolver для GraphNodes."""

    # --------------------------------------------------------------------------
    # intent_classifier
    # --------------------------------------------------------------------------

    def intent_classifier(self, state: AgentState) -> dict:
        """Классификация запроса пользователя: определяет интент, сущности, фильтры.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с распознанным интентом.
        """
        user_input = state["user_input"]
        logger.info("IntentClassifier: обработка запроса: %s", user_input[:100])

        self.memory.add_message("user", user_input)

        # --- Системный промпт ---
        system_prompt = (
            "Ты — классификатор запросов аналитического агента для Greenplum (PostgreSQL-совместимая MPP СУБД).\n"
            "Твоя задача — определить тип запроса пользователя и извлечь ключевые сущности.\n\n"
            "Типы интентов:\n"
            "- analytics: требуется SQL-запрос (агрегация, выборка, подсчёт)\n"
            "- followup: продолжение/уточнение предыдущего запроса "
            "(сигналы: 'это', 'ещё раз', 'теперь', 'добавь', 'измени', 'сгруппируй', 'а если', 'а теперь')\n"
            "- schema_question: ответ можно дать из каталога таблиц (что есть в базе, описание таблиц)\n"
            "- table_search: нужен поиск таблиц (пользователь не знает какая таблица нужна)\n"
            "- export: выгрузка данных в CSV/файл\n"
            "- ddl: создание/изменение структуры таблиц (CREATE, ALTER, DROP)\n"
            "- clarification: запрос неоднозначен, нужно уточнение\n\n"
            "Верни ТОЛЬКО JSON:\n"
            "{\n"
            '  "intent": "<тип>",\n'
            '  "entities": ["<сущность1>", "<сущность2>"],\n'
            '  "date_filters": {"from": "<дата или null>", "to": "<дата или null>"},\n'
            '  "aggregation_hint": "<count|sum|avg|min|max|list|null>",\n'
            '  "needs_search": <true|false>,\n'
            '  "complexity": "<single_table|multi_table|join|subquery>"\n'
            "}\n\n"
            "ПРАВИЛО complexity:\n"
            "- single_table: данные нужны только из одной таблицы\n"
            "- join: нужно объединить данные из двух+ таблиц "
            "(сигналы: 'возьми из', 'дотяни из', 'подтяни из', 'по ключу из', 'из таблицы X', "
            "'join', 'связать', упомянуты две разные таблицы)\n"
            "- multi_table: несколько независимых запросов к разным таблицам\n"
            "- subquery: нужен вложенный подзапрос\n\n"
            "=== ПРИМЕРЫ ===\n\n"
            'Запрос: "Сколько клиентов в регионе Москва?"\n'
            '{"intent": "analytics", "entities": ["клиенты", "регион", "Москва"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "count", '
            '"needs_search": false, "complexity": "single_table"}\n\n'
            'Запрос: "Какие таблицы есть по продажам?"\n'
            '{"intent": "schema_question", "entities": ["продажи"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": null, '
            '"needs_search": false, "complexity": "single_table"}\n\n'
            'Запрос: "Есть ли данные по оттоку клиентов?"\n'
            '{"intent": "table_search", "entities": ["отток", "клиенты"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": null, '
            '"needs_search": true, "complexity": "single_table"}\n\n'
            'Запрос: "Посчитай сумму оттока. Сегмент возьми по inn из uzp_data_epk_consolidation"\n'
            '{"intent": "analytics", "entities": ["отток", "сегмент", "inn", "uzp_data_epk_consolidation"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "sum", '
            '"needs_search": false, "complexity": "join"}\n\n'
            'Запрос: "Покажи сумму продаж по менеджерам, сегмент дотяни по inn из справочника клиентов"\n'
            '{"intent": "analytics", "entities": ["продажи", "менеджеры", "сегмент", "inn", "справочник клиентов"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "sum", '
            '"needs_search": false, "complexity": "join"}\n\n'
            'Запрос: "Сколько договоров по каждому сегменту, подтяни сегмент из таблицы clients"\n'
            '{"intent": "analytics", "entities": ["договоры", "сегмент", "clients"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "count", '
            '"needs_search": false, "complexity": "join"}\n'
        )

        # --- Пользовательский промпт ---
        # История сессии (последние 5 сообщений, компактно)
        history_ctx = self._get_session_history_context()
        if history_ctx:
            # Ограничиваем до 5 сообщений по 200 символов
            history_lines = history_ctx.strip().split("\n")
            compact_history = "\n".join(
                line[:200] for line in history_lines[:5]
            )
        else:
            compact_history = ""

        lt_ctx = self._get_long_term_memory_context()

        user_prompt = ""
        if compact_history:
            user_prompt += f"История сессии:\n{compact_history}\n\n"
        if lt_ctx:
            user_prompt += f"Инструкции пользователя из долгосрочной памяти:\n{lt_ctx}\n\n"
        # Multi-turn: предыдущий успешный запрос (для followup-детекции)
        prev_sql = state.get("prev_sql", "")
        prev_summary = state.get("prev_result_summary", "")
        if prev_sql:
            user_prompt += (
                f"Предыдущий успешный SQL-запрос:\n{prev_sql[:300]}\n"
            )
            if prev_summary:
                user_prompt += f"Результат: {prev_summary[:200]}\n"
            user_prompt += "\n"
        user_prompt += f"Запрос пользователя: {user_input}\n"

        # Контекст replanning если есть
        replan_ctx = state.get("replan_context", "")
        if replan_ctx:
            user_prompt += (
                f"\n\n--- REPLANNING ---\n"
                f"Предыдущий план не удался.\n{replan_ctx}\n"
                f"Учти это при классификации."
            )

        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)

        if self.debug_prompt:
            print(
                f"\n{'='*80}\n[DEBUG PROMPT — intent_classifier]\n{'='*80}\n"
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n"
            )

        response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.2)

        # Парсинг JSON-ответа
        cleaned = self._clean_llm_json(response)
        intent: dict[str, Any] = {}
        try:
            intent = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            # Пробуем извлечь JSON из ответа
            try:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    intent = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                logger.warning("IntentClassifier: не удалось распарсить JSON, используем fallback")
                intent = {
                    "intent": "analytics",
                    "entities": [],
                    "date_filters": {"from": None, "to": None},
                    "aggregation_hint": None,
                    "needs_search": True,
                    "complexity": "single_table",
                }

        logger.info("IntentClassifier: intent=%s, entities=%s",
                     intent.get("intent"), intent.get("entities"))

        self.memory.add_message(
            "assistant",
            f"[intent_classifier] Интент: {intent.get('intent')}, "
            f"сущности: {intent.get('entities', [])}"
        )

        return {
            "intent": intent,
            "messages": state["messages"] + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"Интент: {json.dumps(intent, ensure_ascii=False)}"},
            ],
            "graph_iterations": state.get("graph_iterations", 0) + 1,
        }

    # --------------------------------------------------------------------------
    # table_resolver
    # --------------------------------------------------------------------------

    def table_resolver(self, state: AgentState) -> dict:
        """Выбор таблиц из каталога на основе классифицированного интента.

        Args:
            state: Текущее состояние графа (должен содержать intent).

        Returns:
            Обновления состояния с выбранными таблицами и планом.
        """
        user_input = state["user_input"]
        intent = state.get("intent", {})
        logger.info("TableResolver: выбор таблиц для intent=%s", intent.get("intent"))

        # --- Системный промпт ---
        system_prompt = (
            "Ты — селектор таблиц из каталога аналитического агента для Greenplum.\n"
            "Задача: на основе интента пользователя выбрать нужные таблицы из каталога "
            "и составить краткий план выполнения.\n\n"
            "Верни ТОЛЬКО JSON:\n"
            "{\n"
            '  "tables": [\n'
            '    {"schema": "<схема>", "table": "<таблица>", "reason": "<зачем нужна>"},\n'
            "    ...\n"
            "  ],\n"
            '  "plan_steps": ["<шаг 1>", "<шаг 2>", ...]\n'
            "}\n\n"
            "Правила:\n"
            "- Используй ТОЛЬКО реальные таблицы из каталога ниже\n"
            "- Формат: schema.table\n"
            "- Если таблица не найдена в каталоге — НЕ включай её\n"
            "- КАЖДЫЙ шаг плана ДОЛЖЕН содержать ВСЕ задействованные таблицы в формате schema.table\n"
            "- НЕ пиши SQL в плане — только укажи нужные таблицы\n"
            "- Если ответ есть в каталоге — план из одного шага без инструментов\n"
            "- Если таблица неизвестна — используй шаг с search_tables / search_by_description\n"
        )

        # --- Пользовательский промпт ---
        # Компактное представление интента
        intent_repr = json.dumps(intent, ensure_ascii=False, indent=None)

        schema_ctx = self._get_schema_context(user_input)

        user_prompt = (
            f"Интент пользователя:\n{intent_repr}\n\n"
            f"Запрос пользователя: {user_input}\n\n"
            f"Каталог таблиц:\n{schema_ctx}\n"
        )

        # Контекст replanning если есть
        replan_ctx = state.get("replan_context", "")
        if replan_ctx:
            user_prompt += (
                f"\n\n--- REPLANNING ---\n"
                f"Предыдущий план не удался.\n{replan_ctx}\n"
                f"Составь НОВЫЙ план с другим подходом."
            )

        system_prompt, user_prompt = self._trim_to_budget(system_prompt, user_prompt)

        if self.debug_prompt:
            print(
                f"\n{'='*80}\n[DEBUG PROMPT — table_resolver]\n{'='*80}\n"
                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n"
            )

        response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.2)

        # Парсинг JSON-ответа
        cleaned = self._clean_llm_json(response)
        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            try:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                logger.warning("TableResolver: не удалось распарсить JSON, используем fallback")
                parsed = {"tables": [], "plan_steps": [user_input]}

        # Извлекаем таблицы и валидируем против каталога
        raw_tables = parsed.get("tables", [])
        df = self.schema.tables_df
        validated_tables: list[tuple[str, str]] = []
        table_confidences: dict[str, int] = {}  # "schema.table" -> 0..100

        # Предвычисляем: явные упоминания schema.table в user_input
        _explicit_pattern = re.compile(
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b'
        )
        _explicit_mentions = {
            f"{m.group(1).lower()}.{m.group(2).lower()}"
            for m in _explicit_pattern.finditer(user_input)
        }

        for entry in raw_tables:
            schema_name = str(entry.get("schema", "")).strip().lower()
            table_name = str(entry.get("table", "")).strip().lower()
            if not schema_name or not table_name:
                continue

            if not df.empty:
                mask = (
                    df["schema_name"].str.lower() == schema_name
                ) & (
                    df["table_name"].str.lower() == table_name
                )
                if not df[mask].empty:
                    row = df[mask].iloc[0]
                    validated_tables.append((row["schema_name"], row["table_name"]))

                    # Вычисляем confidence
                    full_key = f"{schema_name}.{table_name}"
                    if full_key in _explicit_mentions:
                        confidence = 100  # явное упоминание schema.table
                    elif table_name in user_input.lower() or schema_name in user_input.lower():
                        confidence = 85  # прямое упоминание имени
                    else:
                        # Проверяем через entities
                        entities_lower = [str(e).lower() for e in intent.get("entities", [])]
                        if any(table_name in e or e in table_name for e in entities_lower):
                            confidence = 70  # совпадение через entities
                        else:
                            confidence = 45  # найдено через TF-IDF/синонимы
                    table_confidences[full_key] = confidence
                else:
                    logger.warning(
                        "TableResolver: таблица %s.%s не найдена в каталоге — пропускаем",
                        schema_name, table_name,
                    )

        # Извлекаем план
        plan_steps = parsed.get("plan_steps", [])
        if not plan_steps:
            plan_steps = self._parse_plan(response)

        # Предупреждение при низкой уверенности
        low_confidence_warning = ""
        if table_confidences:
            min_conf = min(table_confidences.values())
            if min_conf < 50 and len(validated_tables) == 1:
                low_table = next(iter(table_confidences))
                low_confidence_warning = (
                    f"\n⚠ Таблица '{low_table}' выбрана с низкой уверенностью ({min_conf}%). "
                    "Если результат неверный — уточните запрос или используйте команду поиска таблиц."
                )
                logger.warning(
                    "TableResolver: низкая уверенность в выборе таблицы %s (%d%%)",
                    low_table, min_conf,
                )

        logger.info(
            "TableResolver: выбрано %d таблиц: %s, план из %d шагов, confidence=%s",
            len(validated_tables),
            ", ".join(f"{s}.{t}" for s, t in validated_tables),
            len(plan_steps),
            table_confidences,
        )

        self.memory.add_message(
            "assistant",
            f"[table_resolver] Выбраны таблицы: "
            f"{', '.join(f'{s}.{t}' for s, t in validated_tables) or 'нет'}, "
            f"план: {len(plan_steps)} шагов"
        )

        return {
            "selected_tables": validated_tables,
            "plan": plan_steps,
            "current_step": 0,
            "retry_count": 0,
            "last_error": None,
            "needs_replan": False,
            "replan_context": "",
            "messages": state["messages"] + [
                {"role": "assistant", "content": (
                    f"Таблицы: {', '.join(f'{s}.{t}' for s, t in validated_tables)}\n"
                    f"Уверенность: {table_confidences}\n"
                    f"План:\n" + "\n".join(plan_steps) + low_confidence_warning
                )},
            ],
        }

    # --------------------------------------------------------------------------
    # _parse_plan (без изменений из оригинала)
    # --------------------------------------------------------------------------

    def _parse_plan(self, response: str) -> list[str]:
        """Извлечь план из ответа LLM."""
        cleaned = self._clean_llm_json(response)

        # Пробуем JSON
        for text in (cleaned, response):
            try:
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    plan = json.loads(match.group())
                    if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
                        return plan
            except (json.JSONDecodeError, ValueError):
                continue

        # Fallback: парсим нумерованный список
        lines = response.strip().split("\n")
        plan = []
        for line in lines:
            line = line.strip()
            if line and re.match(r'^\d+[\.\)]\s*', line):
                plan.append(re.sub(r'^\d+[\.\)]\s*', '', line))
            elif line and line.startswith("Шаг"):
                plan.append(line)
        return plan if plan else [response.strip()]
