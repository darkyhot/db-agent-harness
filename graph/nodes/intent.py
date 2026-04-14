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

from core.column_selector_deterministic import (
    _derive_requested_slots,
    _is_label_slot,
    _is_metric_slot,
    _normalize_query_text,
    _semantic_match_score,
)
from core.join_analysis import detect_table_type
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
            '  "complexity": "<single_table|multi_table|join|subquery>",\n'
            '  "clarification_question": "<короткий вопрос пользователю или пустая строка>",\n'
            '  "filter_conditions": [\n'
            '    {"column_hint": "<ключевое слово для поиска колонки>", '
            '"operator": "<= | >= | = | != | LIKE | IN>", "value": "<литеральное значение>"}\n'
            "  ]\n"
            "}\n\n"
            "ПРАВИЛО clarification:\n"
            "- Выбирай intent=clarification ТОЛЬКО если без ответа пользователя высок риск построить неверный SQL\n"
            "- Сначала пытайся опереться на историю, предыдущий SQL, память и каталог; не спрашивай то, что можно разумно вывести\n"
            "- НЕ используй clarification для косметических деталей, формата вывода, очевидных допущений или случаев, где можно безопасно продолжить\n"
            "- Хорошие поводы для clarification: неоднозначная метрика, неизвестный период, конфликтующие сущности, несколько равноправных трактовок бизнес-термина\n"
            "- Если intent != clarification, верни clarification_question пустой строкой\n"
            "- Если intent = clarification, задай ОДИН короткий конкретный вопрос, который разблокирует следующий шаг\n\n"
            "ПРАВИЛО filter_conditions:\n"
            "- Заполняй ТОЛЬКО если в запросе есть явные фильтры по значениям "
            "(например: 'по региону North', 'product_category = Electronics', 'сумма > 1000')\n"
            "- column_hint — ключевое слово из запроса (регион, сегмент, сумма и т.д.)\n"
            "- value — точное значение из запроса ('North', 'Electronics', '1000')\n"
            "- Если явных фильтров нет — оставь пустым списком []\n\n"
            "ПРАВИЛО complexity:\n"
            "- single_table: данные нужны только из одной таблицы\n"
            "- join: нужно объединить данные из двух+ таблиц "
            "(сигналы: 'возьми из', 'дотяни из', 'подтяни из', 'по ключу из', 'из таблицы X', "
            "'join', 'связать', упомянуты две разные таблицы)\n"
            "- multi_table: несколько независимых запросов к разным таблицам\n"
            "- subquery: нужен вложенный подзапрос\n\n"
            "=== ПРИМЕРЫ ===\n\n"
            'Запрос: "Сколько клиентов в регионе North?"\n'
            '{"intent": "analytics", "entities": ["клиенты", "регион", "North"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "count", '
            '"needs_search": false, "complexity": "single_table", "clarification_question": ""}\n\n'
            'Запрос: "Какие таблицы есть по заказам?"\n'
            '{"intent": "schema_question", "entities": ["заказы"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": null, '
            '"needs_search": false, "complexity": "single_table", "clarification_question": ""}\n\n'
            'Запрос: "Есть ли данные по обращениям пользователей?"\n'
            '{"intent": "table_search", "entities": ["обращения", "пользователи"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": null, '
            '"needs_search": true, "complexity": "single_table", "clarification_question": ""}\n\n'
            'Запрос: "Посчитай сумму платежей. Категорию клиента возьми по customer_id из справочника клиентов"\n'
            '{"intent": "analytics", "entities": ["платежи", "категория клиента", "customer_id", "справочник клиентов"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "sum", '
            '"needs_search": false, "complexity": "join", "clarification_question": ""}\n\n'
            'Запрос: "Покажи сумму заказов по менеджерам, категорию дотяни по customer_id из справочника клиентов"\n'
            '{"intent": "analytics", "entities": ["заказы", "менеджеры", "категория", "customer_id", "справочник клиентов"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "sum", '
            '"needs_search": false, "complexity": "join", "clarification_question": ""}\n\n'
            'Запрос: "Сколько полисов по каждому типу, подтяни тип из таблицы policies"\n'
            '{"intent": "analytics", "entities": ["полисы", "тип", "policies"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "count", '
            '"needs_search": false, "complexity": "join", "clarification_question": ""}\n\n'
            'Запрос: "Покажи заказы"\n'
            '{"intent": "clarification", "entities": ["заказы"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": null, '
            '"needs_search": false, "complexity": "single_table", '
            '"clarification_question": "Что именно показать по заказам: сумму, количество или список записей?"}\n'
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

        needs_clarification = intent.get("intent") == "clarification"
        clarification_message = str(intent.get("clarification_question") or "").strip()
        if needs_clarification and not clarification_message:
            clarification_message = (
                "Не хватает деталей, чтобы корректно продолжить. Уточните, пожалуйста, запрос."
            )

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
            "needs_clarification": needs_clarification,
            "clarification_message": clarification_message,
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

        # --- Детерминированная коррекция по семантике запроса и каталогу ---
        query_norm = _normalize_query_text(user_input)
        requested = _derive_requested_slots(user_input, intent)
        tables_df = self.schema.tables_df

        # Кэш get_table_columns в пределах вызова: каждая таблица загружается ровно один раз,
        # даже если её запрашивают _table_type, _score_table_for_slot и _joinability_score.
        _col_cache: dict[tuple[str, str], object] = {}

        def _get_cols(schema_name: str, table_name: str):
            key = (schema_name, table_name)
            if key not in _col_cache:
                _col_cache[key] = self.schema.get_table_columns(schema_name, table_name)
            return _col_cache[key]

        def _norm_key(name: str) -> str:
            return re.sub(r"^(old|new|prev|cur|current|actual|base|src|tgt)_", "", name.lower())

        def _table_type(schema_name: str, table_name: str) -> str:
            cols = _get_cols(schema_name, table_name)
            return detect_table_type(table_name, cols)

        def _score_table_for_slot(schema_name: str, table_name: str, slot: str) -> float:
            cols = _get_cols(schema_name, table_name)
            if cols.empty:
                return -1.0
            t_type = _table_type(schema_name, table_name)
            best = -1.0
            for _, row in cols.iterrows():
                col_name = str(row.get("column_name", "") or "")
                desc = str(row.get("description", "") or "")
                semantic = _semantic_match_score(col_name, desc, slot)
                if semantic <= 0:
                    continue
                not_null = float(row.get("not_null_perc", 0) or 0)
                unique = float(row.get("unique_perc", 0) or 0)
                score = semantic * 1000 + not_null * 1.5 + min(unique, 100.0) * 0.1
                if _is_label_slot(slot):
                    if t_type in ("dim", "ref", "unknown"):
                        score += 35
                    if t_type == "fact":
                        score -= 25
                elif _is_metric_slot(slot):
                    if t_type == "fact":
                        score += 35
                    else:
                        score -= 20
                best = max(best, score)
            return best

        def _joinability_score(base: tuple[str, str] | None, candidate: tuple[str, str]) -> float:
            if base is None or base == candidate:
                return 0.0
            bs, bt = base
            cs, ct = candidate
            left = _get_cols(bs, bt)
            right = _get_cols(cs, ct)
            if left.empty or right.empty:
                return 0.0

            left_cols = set(left["column_name"].astype(str))
            right_cols = set(right["column_name"].astype(str))
            exact_common = left_cols & right_cols
            norm_left = {_norm_key(c): c for c in left_cols}
            norm_right = {_norm_key(c): c for c in right_cols}
            normalized_common = set(norm_left) & set(norm_right)

            if "is_primary_key" in right.columns:
                right_pks = set(
                    right[right["is_primary_key"].astype(bool)]["column_name"].astype(str).tolist()
                )
            else:
                right_pks = set()
            matched_pk = sum(1 for pk in right_pks if _norm_key(pk) in norm_left)
            return matched_pk * 80 + len(exact_common) * 20 + len(normalized_common) * 12

        explicit_tables: list[tuple[str, str]] = []
        for _, row in tables_df.iterrows():
            schema_name = str(row.get("schema_name", "") or "")
            table_name = str(row.get("table_name", "") or "")
            full = f"{schema_name}.{table_name}".lower()
            if table_name.lower() in query_norm or full in query_norm:
                explicit_tables.append((schema_name, table_name))

        metric_slot = requested.get("metric")
        dimension_slots = list(requested.get("dimensions", []))
        join_requested = (
            str(intent.get("complexity", "")).lower() in {"join", "subquery", "multi_table"}
            or any(phrase in query_norm for phrase in ("подтяни", "дотяни", "возьми из", "join", "по ключу"))
        )

        # Для больших каталогов (>100 таблиц) без явных упоминаний: предварительная фильтрация
        # через TF-IDF/keyword поиск сокращает O(N*cols) scoring до O(top_k*cols).
        _LARGE_CATALOG = 100
        if len(explicit_tables) >= 2:
            candidate_tables = explicit_tables
        elif len(tables_df) > _LARGE_CATALOG and not explicit_tables:
            search_parts = [user_input]
            search_parts += list(intent.get("entities", []))
            if metric_slot:
                search_parts.append(metric_slot)
            search_parts += dimension_slots
            search_query = " ".join(str(p) for p in search_parts if p)
            search_df = self.schema.search_tables(search_query, top_n=50)
            candidate_tables = [
                (str(r["schema_name"]), str(r["table_name"]))
                for _, r in search_df.iterrows()
            ]
            if not candidate_tables:
                candidate_tables = [
                    (str(row["schema_name"]), str(row["table_name"]))
                    for _, row in tables_df.iterrows()
                ]
            logger.info(
                "TableResolver: большой каталог (%d таблиц), кандидаты сужены до %d через поиск",
                len(tables_df), len(candidate_tables),
            )
        else:
            candidate_tables = [
                (str(row["schema_name"]), str(row["table_name"]))
                for _, row in tables_df.iterrows()
            ]

        main_table: tuple[str, str] | None = explicit_tables[0] if explicit_tables else None
        if metric_slot and not explicit_tables:
            metric_candidates: list[tuple[float, tuple[str, str]]] = []
            for st in candidate_tables:
                score = _score_table_for_slot(st[0], st[1], metric_slot)
                if score > 0:
                    if st in explicit_tables:
                        score += 120
                    metric_candidates.append((score, st))
            if metric_candidates:
                metric_candidates.sort(reverse=True)
                main_table = metric_candidates[0][1]

        deterministic_tables: list[tuple[str, str]] = list(dict.fromkeys(explicit_tables))
        if main_table and main_table not in deterministic_tables:
            deterministic_tables.insert(0, main_table)

        preserve_single_explicit = (
            len(explicit_tables) == 1
            and not join_requested
            and str(intent.get("complexity", "")).lower() == "single_table"
        )

        for slot in dimension_slots:
            if preserve_single_explicit:
                continue
            if slot == "date" and main_table:
                if _score_table_for_slot(main_table[0], main_table[1], slot) > 0:
                    if main_table not in deterministic_tables:
                        deterministic_tables.append(main_table)
                    continue
            dim_candidates: list[tuple[float, tuple[str, str]]] = []
            for st in candidate_tables:
                score = _score_table_for_slot(st[0], st[1], slot)
                if score <= 0:
                    continue
                score += _joinability_score(main_table, st)
                if st in explicit_tables:
                    score += 90
                dim_candidates.append((score, st))
            if dim_candidates:
                dim_candidates.sort(reverse=True)
                best_table = dim_candidates[0][1]
                if best_table not in deterministic_tables:
                    deterministic_tables.append(best_table)

        if not deterministic_tables:
            deterministic_tables = validated_tables

        if deterministic_tables:
            deterministic_tables = list(dict.fromkeys(deterministic_tables))
            if deterministic_tables != validated_tables:
                logger.info(
                    "TableResolver: семантически корректирую выбор таблиц: %s -> %s",
                    validated_tables,
                    deterministic_tables,
                )
                validated_tables = deterministic_tables
                table_confidences = {
                    f"{s}.{t}": 100 if (s, t) in explicit_tables else 85
                    for s, t in deterministic_tables
                }

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
