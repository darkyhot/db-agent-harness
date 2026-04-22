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

from core.log_safety import summarize_dict_keys, summarize_text
from core.column_selector_deterministic import (
    _choose_best_column,
    _derive_requested_slots,
    _is_dimension_slot,
    _is_numeric,
    _is_label_slot,
    _is_metric_slot,
    _normalize_query_text,
    _semantic_match_score,
)
from core.join_analysis import detect_table_type
from core.sql_planner_deterministic import _derive_date_filters_from_text
from core.semantic_frame import derive_semantic_frame
from core.semantic_frame import sanitize_user_input_for_semantics
from core.domain_rules import table_bonus_for_frame, table_can_satisfy_frame
from core.join_governor import decide_join_plan
from core.confidence import evaluate_join_confidence, evaluate_table_confidence, build_planning_confidence
from graph.state import AgentState

logger = logging.getLogger(__name__)


def _extract_forced_single_source(
    query_norm: str,
    tables_df,
) -> tuple[str, str] | None:
    """Вытащить таблицу из служебной фразы `использовать таблицу schema.table`.

    CLI уже дописывает такую фразу после выбора витрины. Для planner это должен
    быть жёсткий single-source сигнал, а не просто ещё одно явное упоминание.
    """
    match = re.search(
        r"использовать\s+таблицу\s+([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)",
        query_norm,
    )
    if not match or tables_df is None or tables_df.empty:
        return None
    schema_name = match.group(1).lower()
    table_name = match.group(2).lower()
    mask = (
        tables_df["schema_name"].astype(str).str.lower() == schema_name
    ) & (
        tables_df["table_name"].astype(str).str.lower() == table_name
    )
    if tables_df[mask].empty:
        return None
    row = tables_df[mask].iloc[0]
    return (str(row["schema_name"]), str(row["table_name"]))


class IntentNodes:
    """Миксин с узлами intent_classifier и table_resolver для GraphNodes."""

    @staticmethod
    def _build_disambiguation_message(disambiguation_options: list[dict[str, Any]]) -> str:
        """Собрать человекочитаемое сообщение со списком вариантов витрин."""
        display_lines = ["Для запроса подходят несколько таблиц. Какую использовать?", ""]
        for idx, option in enumerate(disambiguation_options, 1):
            display_lines.append(
                f"{idx}. {option['schema']}.{option['table']} — {option.get('description') or 'без описания'}"
            )
            if option.get("key_columns"):
                display_lines.append(
                    f"   Ключевые колонки: {', '.join(option['key_columns'])}"
                )
        return "\n".join(display_lines)

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
        if (state.get("plan_edit_text") and state.get("sql_blueprint")) or (
            state.get("plan_preview_approved") and state.get("sql_blueprint")
        ):
            logger.info("IntentClassifier: plan-edit/approved fast-path — пропускаю реклассификацию")
            return {
                "graph_iterations": state.get("graph_iterations", 0) + 1,
            }

        user_input = state["user_input"]
        logger.info("IntentClassifier: обработка запроса: %s", summarize_text(user_input, label="user_input"))

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
            '  "date_filters": {"from": "<дата YYYY-MM-DD или null>", "to": "<дата YYYY-MM-DD или null>"},\n'
            '  "aggregation_hint": "<count|sum|avg|min|max|list|null>",\n'
            '  "needs_search": <true|false>,\n'
            '  "complexity": "<single_table|multi_table|join|subquery>",\n'
            '  "clarification_question": "<короткий вопрос пользователю или пустая строка>",\n'
            '  "filter_conditions": [\n'
            '    {"column_hint": "<ключевое слово для поиска колонки>", '
            '"operator": "<= | >= | = | != | LIKE | IN>", "value": "<литеральное значение>"}\n'
            "  ],\n"
            '  "explicit_join": [\n'
            '    {"table_hint": "<часть имени таблицы, где лежит ключ, или null>", '
            '"column_hint": "<имя или смысловая часть ключа-колонки>"}\n'
            "  ],\n"
            '  "required_output": ["<обязательный атрибут в SELECT 1>", "<атрибут 2>"],\n'
            '  "month_without_year": <true|false>\n'
            "}\n\n"
            "=== ПРАВИЛО clarification — МАТРИЦА ===\n\n"
            "УТОЧНЯЙ (intent=clarification) ТОЛЬКО если выполняется хотя бы одно условие:\n"
            "  1. Метрика неоднозначна: несколько равноправных трактовок ("
            "например 'покажи заказы' — сумму? количество? список?)\n"
            "  2. Запрос звучит слишком широко и риск полного сканирования "
            "большой таблицы высок ('выгрузи всё', 'покажи все данные')\n"
            "  3. Конфликт: два разных смысла у ключевого термина в контексте базы\n\n"
            "НЕ УТОЧНЯЙ — продолжай с аналитикой если:\n"
            "  - Пользователь спрашивает о справочных сущностях типа 'сколько ТБ', "
            "'количество ГОСБ', 'список регионов' → это COUNT из dim-таблицы, не clarification\n"
            "  - Агрегация очевидна из глагола: 'посчитай' → count, 'суммируй' → sum, "
            "'сколько' → count\n"
            "  - Таблица явно указана в запросе\n"
            "  - Группировка очевидна: 'по дате', 'по сегменту', 'по региону'\n"
            "  - Формат вывода не задан → выводи таблицей по умолчанию\n"
            "  - Период — 'последний месяц', 'прошлый квартал' → считай от текущей даты\n\n"
            "ПРАВИЛО month_without_year:\n"
            "- Ставь true ТОЛЬКО если в запросе указан месяц (февраль, январь и т.д.) БЕЗ года\n"
            "- При month_without_year=true clarification_question = 'За какой год считать данные за [месяц]?'\n"
            "- При month_without_year=true intent ДОЛЖЕН быть 'clarification'\n\n"
            "ПРАВИЛО explicit_join:\n"
            "- Заполняй ТОЛЬКО если пользователь ЯВНО указал ключ/поле для JOIN\n"
            "- Сигналы: 'по инн', 'по customer_id', 'join по X', 'связать по Y', 'ключ — Z', "
            "'возьми сегмент по инн'\n"
            "- table_hint — часть имени таблицы-источника (может быть null)\n"
            "- column_hint — ключевое слово для имени join-колонки ('инн', 'customer_id', 'дата')\n"
            "- Если явного join-ключа нет — оставь пустым списком []\n\n"
            "ПРАВИЛО required_output:\n"
            "- Перечисли ВСЕ измерения/группировки, которые пользователь явно требует в результате\n"
            "- Сигналы: 'по дате', 'по сегменту', 'по региону', 'с разбивкой по X'\n"
            "- Если пользователь сказал 'по дате и сегменту' → required_output: ['дата', 'сегмент']\n"
            "- Если нет явных требований к выводу — пустой список []\n\n"
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
            "- multi_table: посчитать несколько метрик из разных таблиц и показать вместе\n"
            "- subquery: нужен вложенный подзапрос\n\n"
            "=== ПРИМЕРЫ ===\n\n"
            'Запрос: "Сколько ТБ и ГОСБ есть в базе?"\n'
            '{"intent": "analytics", "entities": ["ТБ", "ГОСБ"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "count", '
            '"needs_search": false, "complexity": "single_table", "clarification_question": "", '
            '"explicit_join": [], "required_output": ["ТБ", "ГОСБ"], "month_without_year": false, '
            '"filter_conditions": []}\n\n'
            'Запрос: "Посчитай сумму оттока по дате и сегменту (сегмент возьми в epk_consolidation по инн)"\n'
            '{"intent": "analytics", "entities": ["отток", "дата", "сегмент", "epk_consolidation", "инн"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "sum", '
            '"needs_search": false, "complexity": "join", "clarification_question": "", '
            '"explicit_join": [{"table_hint": "epk_consolidation", "column_hint": "inn"}], '
            '"required_output": ["дата", "сегмент"], "month_without_year": false, '
            '"filter_conditions": []}\n\n'
            'Запрос: "Посчитай количество задач и количество оттока за февраль по дате"\n'
            '{"intent": "clarification", "entities": ["задачи", "отток", "февраль", "дата"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "count", '
            '"needs_search": false, "complexity": "multi_table", '
            '"clarification_question": "За какой год считать данные за февраль?", '
            '"explicit_join": [{"table_hint": null, "column_hint": "дата"}], '
            '"required_output": ["дата"], "month_without_year": true, "filter_conditions": []}\n\n'
            'Запрос: "Посчитай количество задач и оттока за февраль 2026 по дате"\n'
            '{"intent": "analytics", "entities": ["задачи", "отток", "дата"], '
            '"date_filters": {"from": "2026-02-01", "to": "2026-03-01"}, "aggregation_hint": "count", '
            '"needs_search": false, "complexity": "multi_table", "clarification_question": "", '
            '"explicit_join": [{"table_hint": null, "column_hint": "дата"}], '
            '"required_output": ["дата"], "month_without_year": false, "filter_conditions": []}\n\n'
            'Запрос: "Сколько клиентов в регионе North?"\n'
            '{"intent": "analytics", "entities": ["клиенты", "регион", "North"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "count", '
            '"needs_search": false, "complexity": "single_table", "clarification_question": "", '
            '"explicit_join": [], "required_output": [], "month_without_year": false, '
            '"filter_conditions": [{"column_hint": "регион", "operator": "=", "value": "North"}]}\n\n'
            'Запрос: "Покажи сумму заказов по менеджерам, категорию дотяни по customer_id из справочника"\n'
            '{"intent": "analytics", "entities": ["заказы", "менеджеры", "категория", "customer_id", "справочник"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": "sum", '
            '"needs_search": false, "complexity": "join", "clarification_question": "", '
            '"explicit_join": [{"table_hint": "справочник", "column_hint": "customer_id"}], '
            '"required_output": ["менеджер"], "month_without_year": false, "filter_conditions": []}\n\n'
            'Запрос: "Покажи заказы"\n'
            '{"intent": "clarification", "entities": ["заказы"], '
            '"date_filters": {"from": null, "to": null}, "aggregation_hint": null, '
            '"needs_search": false, "complexity": "single_table", '
            '"clarification_question": "Что именно показать по заказам: сумму, количество или список записей?", '
            '"explicit_join": [], "required_output": [], "month_without_year": false, '
            '"filter_conditions": []}\n'
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

        logger.info(
            "IntentClassifier: intent=%s, entities_count=%d",
            intent.get("intent"),
            len(intent.get("entities") or []),
        )

        # Детерминированная коррекция периода:
        # если пользователь написал "февраль 26" или ответил "Уточнение пользователя: 26",
        # не переспросим год повторно даже если LLM ошибся.
        derived_dates = _derive_date_filters_from_text(user_input)
        if (
            derived_dates.get("from")
            and derived_dates.get("from") != "NEEDS_YEAR"
        ):
            current_dates = dict(intent.get("date_filters") or {})
            if not current_dates.get("from") or intent.get("month_without_year"):
                intent["date_filters"] = derived_dates
                intent["month_without_year"] = False
                if intent.get("intent") == "clarification":
                    intent["intent"] = "analytics"
                logger.info(
                    "IntentClassifier: детерминированно распознан период %s..%s",
                    derived_dates.get("from"), derived_dates.get("to"),
                )

        needs_clarification = intent.get("intent") == "clarification"
        clarification_message = str(intent.get("clarification_question") or "").strip()

        # Если LLM не поставил clarification, но сам поставил month_without_year=True —
        # принудительно переключаем на clarification (защита от несогласованности).
        if intent.get("month_without_year") and not needs_clarification:
            needs_clarification = True
            intent["intent"] = "clarification"
            if not clarification_message:
                # Извлекаем название месяца для подстановки в вопрос
                _month_names = {
                    1: "январь", 2: "февраль", 3: "март", 4: "апрель",
                    5: "май", 6: "июнь", 7: "июль", 8: "август",
                    9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь",
                }
                import re as _re
                _q = (state.get("user_input") or "").lower()
                _ru_months_stems = {
                    'январ': 1, 'феврал': 2, 'март': 3, 'апрел': 4,
                    'май': 5, 'мая': 5, 'июн': 6, 'июл': 7, 'август': 8,
                    'сентябр': 9, 'октябр': 10, 'ноябр': 11, 'декабр': 12,
                }
                _detected_month = None
                for stem, num in _ru_months_stems.items():
                    if stem in _q:
                        _detected_month = _month_names[num]
                        break
                month_label = _detected_month or "указанный месяц"
                clarification_message = f"За какой год считать данные за {month_label}?"
            logger.info(
                "IntentClassifier: month_without_year=True → принудительно clarification (%s)",
                summarize_text(clarification_message, label="clarification_message"),
            )

        if needs_clarification and not clarification_message:
            clarification_message = (
                "Не хватает деталей, чтобы корректно продолжить. Уточните, пожалуйста, запрос."
            )

        semantic_input = sanitize_user_input_for_semantics(user_input)
        semantic_frame = derive_semantic_frame(semantic_input, intent, schema_loader=self.schema)
        logger.info(
            "IntentClassifier: semantic_frame=%s",
            summarize_dict_keys(semantic_frame, label="semantic_frame"),
        )
        logger.info(
            "IntentClassifier: semantic_frame_full=%s",
            semantic_frame,
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
            "semantic_frame": semantic_frame,
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
        semantic_input = sanitize_user_input_for_semantics(user_input)
        requested = _derive_requested_slots(semantic_input, intent)
        semantic_frame = state.get("semantic_frame", {}) or derive_semantic_frame(semantic_input, intent, schema_loader=self.schema)
        logger.info("TableResolver: user_input_full=%r", user_input)
        logger.info("TableResolver: semantic_frame_full=%s", semantic_frame)
        tables_df = self.schema.tables_df
        forced_single_source = _extract_forced_single_source(query_norm, tables_df)

        # === Подсказки пользователя (hint_extractor) ===
        user_hints = state.get("user_hints", {}) or {}
        hint_must_keep: list[tuple[str, str]] = list(
            user_hints.get("must_keep_tables", []) or []
        )
        hint_dim_sources: dict[str, dict[str, str]] = (
            user_hints.get("dim_sources", {}) or {}
        )
        hint_join_fields: list[str] = list(user_hints.get("join_fields", []) or [])

        # Таблицы, упомянутые через dim_sources, тоже считаются must_keep.
        for slot_key, binding in hint_dim_sources.items():
            tbl_full = binding.get("table") if isinstance(binding, dict) else None
            if not tbl_full or "." not in tbl_full:
                continue
            s_part, t_part = tbl_full.split(".", 1)
            tup = (s_part, t_part)
            if tup not in hint_must_keep:
                hint_must_keep.append(tup)

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

        def _table_name_match_score(
            schema_name: str, table_name: str, slot: str,
        ) -> float:
            """Семантическая близость имени/описания таблицы к слоту.

            Используется для штрафа «dim-в-факте»: если таблица типа fact, но
            её имя/описание не пересекаются со словами метрики — это значит,
            что метрика, скорее всего, реализована флагом, и есть более
            подходящая фактовая таблица.
            """
            try:
                row_df = tables_df[
                    (tables_df["schema_name"].astype(str).str.lower() == schema_name.lower())
                    & (tables_df["table_name"].astype(str).str.lower() == table_name.lower())
                ]
                if row_df.empty:
                    return 0.0
                table_descr = str(row_df.iloc[0].get("description", "") or "")
            except Exception:  # noqa: BLE001
                table_descr = ""
            return _semantic_match_score(table_name, table_descr, slot)

        def _table_description(schema_name: str, table_name: str) -> str:
            try:
                row_df = tables_df[
                    (tables_df["schema_name"].astype(str).str.lower() == schema_name.lower())
                    & (tables_df["table_name"].astype(str).str.lower() == table_name.lower())
                ]
                if row_df.empty:
                    return ""
                return str(row_df.iloc[0].get("description", "") or "")
            except Exception:  # noqa: BLE001
                return ""

        def _table_grain(schema_name: str, table_name: str) -> str:
            try:
                return self.schema.get_table_grain(schema_name, table_name)
            except Exception:  # noqa: BLE001
                return ""

        def _grain_bonus(schema_name: str, table_name: str, requested_grain: str | None) -> float:
            if not requested_grain:
                return 0.0
            table_grain = _table_grain(schema_name, table_name)
            if not table_grain:
                return 0.0
            if table_grain == requested_grain:
                return 240.0
            mismatched_but_related = {
                ("event", "snapshot"), ("snapshot", "event"),
                ("client", "task"), ("task", "client"),
                ("dictionary", "organization"), ("organization", "dictionary"),
            }
            if (table_grain, requested_grain) in mismatched_but_related:
                return -60.0
            return -140.0

        def _score_table_for_slot(
            schema_name: str,
            table_name: str,
            slot: str,
            metric_entities: list[str] | None = None,
        ) -> float:
            cols = _get_cols(schema_name, table_name)
            if cols.empty:
                return -1.0
            t_type = _table_type(schema_name, table_name)
            is_dimension_slot = _is_dimension_slot(slot)
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
                elif is_dimension_slot:
                    if t_type in ("dim", "ref", "unknown"):
                        score += 25
                    if t_type == "fact":
                        score -= 20
                elif _is_metric_slot(slot):
                    if t_type == "fact":
                        score += 35
                    else:
                        score -= 20
                    # Штраф «dim-в-факте»: фактовая таблица, чьё имя/описание
                    # не близко к метрике — наверняка метрика тут флаг. Снижаем
                    # вес, чтобы не выиграть у настоящей фактовой таблицы.
                    if t_type == "fact":
                        tbl_match = _table_name_match_score(
                            schema_name, table_name, slot,
                        )
                        if tbl_match <= 0:
                            score *= 0.5
                if is_dimension_slot and metric_entities:
                    competing_fact = False
                    for _, metric_row in cols.iterrows():
                        metric_name = str(metric_row.get("column_name", "") or "")
                        metric_dtype = str(metric_row.get("dType", "") or "").lower()
                        metric_desc = str(metric_row.get("description", "") or "")
                        if (
                            not _is_numeric(metric_dtype)
                            or bool(metric_row.get("is_primary_key", False))
                        ):
                            continue
                        if any(
                            _semantic_match_score(metric_name, metric_desc, entity) > 0.2
                            for entity in metric_entities
                        ):
                            competing_fact = True
                            break
                    if competing_fact:
                        score -= 200
                best = max(best, score)
            return best

        def _best_slot_profile(
            schema_name: str,
            table_name: str,
            slot: str,
        ) -> dict[str, Any] | None:
            cols = _get_cols(schema_name, table_name)
            if cols.empty:
                return None

            best_profile: dict[str, Any] | None = None
            for _, row in cols.iterrows():
                col_name = str(row.get("column_name", "") or "")
                desc = str(row.get("description", "") or "")
                semantic = _semantic_match_score(col_name, desc, slot)
                if semantic <= 0:
                    continue
                not_null = float(row.get("not_null_perc", 0) or 0)
                unique = float(row.get("unique_perc", 0) or 0)
                candidate = {
                    "column_name": col_name,
                    "semantic": semantic,
                    "not_null": not_null,
                    "unique": unique,
                    "score": semantic * 1000 + not_null * 1.5 + min(unique, 100.0) * 0.1,
                }
                if best_profile is None or candidate["score"] > best_profile["score"]:
                    best_profile = candidate
            return best_profile

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
        if forced_single_source is not None:
            explicit_tables = [forced_single_source]
            validated_tables = [forced_single_source]
            table_confidences = {
                f"{forced_single_source[0]}.{forced_single_source[1]}": 100
            }
            logger.info(
                "TableResolver: жёстко фиксирую single-source таблицу %s.%s",
                forced_single_source[0], forced_single_source[1],
            )

        # Hard-lock: must_keep = explicit_tables ∪ user_hints.must_keep_tables.
        # Эти таблицы НЕ могут быть исключены детерминированной коррекцией.
        locked_tables: list[tuple[str, str]] = list(dict.fromkeys(
            explicit_tables + hint_must_keep,
        ))
        if hint_must_keep:
            logger.info(
                "TableResolver: must_keep из user_hints: %s",
                hint_must_keep,
            )

        metric_slot = requested.get("metric")
        dimension_slots = list(dict.fromkeys(
            list(requested.get("dimensions", []))
            + list(semantic_frame.get("output_dimensions", []) or [])
        ))
        requested_grain = (
            semantic_frame.get("requested_grain")
            or self.schema.infer_query_grain(user_input, list(intent.get("entities") or []))
        )
        metric_entities = list(dict.fromkeys(
            [str(metric_slot)] if metric_slot else []
            + [str(entity) for entity in (intent.get("entities") or []) if entity]
        ))
        join_requested = (
            str(intent.get("complexity", "")).lower() in {"join", "subquery", "multi_table"}
            or any(phrase in query_norm for phrase in ("подтяни", "дотяни", "возьми из", "join", "по ключу"))
        )

        # Для больших каталогов (>100 таблиц) без явных упоминаний: предварительная фильтрация
        # через TF-IDF/keyword поиск сокращает O(N*cols) scoring до O(top_k*cols).
        _LARGE_CATALOG = 100
        if forced_single_source is not None:
            candidate_tables = [forced_single_source]
        elif len(explicit_tables) >= 2:
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

        # === Metric-first выбор main_table ===
        # Принцип: main_table выбирается СТРОГО по метрическому слоту с приоритетом
        # фактовых таблиц. Lock'нутые таблицы получают сильный бонус. Димовые
        # скоры (label-слоты) НЕ учитываются — иначе таблица с высоким not_null
        # по измерению может перебить настоящую фактовую таблицу.
        main_table: tuple[str, str] | None = None
        ranked_main_candidates: list[dict[str, Any]] = []
        for st in candidate_tables:
            score = 0.0
            if metric_slot:
                metric_score = _score_table_for_slot(
                    st[0], st[1], metric_slot, metric_entities=metric_entities,
                )
                if metric_score > 0:
                    score += metric_score
            score += _grain_bonus(st[0], st[1], requested_grain)
            score += table_bonus_for_frame(self.schema, st[0], st[1], semantic_frame)
            if st in explicit_tables:
                score += 90
            if st in hint_must_keep:
                score += 150
            if st in validated_tables:
                score += 70
            ranked_main_candidates.append({
                "table": st,
                "score": score,
                "grain": _table_grain(st[0], st[1]),
                "description": _table_description(st[0], st[1]),
            })

        ranked_main_candidates.sort(key=lambda item: item["score"], reverse=True)
        if ranked_main_candidates and ranked_main_candidates[0]["score"] > 0:
            main_table = ranked_main_candidates[0]["table"]
        if main_table is None:
            # Fallback: первая lock'нутая или первая explicit
            if hint_must_keep:
                main_table = hint_must_keep[0]
            elif explicit_tables:
                main_table = explicit_tables[0]
            elif validated_tables:
                main_table = validated_tables[0]

        deterministic_tables: list[tuple[str, str]] = list(dict.fromkeys(locked_tables))
        if main_table and main_table not in deterministic_tables:
            deterministic_tables.insert(0, main_table)

        preserve_single_explicit = (
            len(locked_tables) == 1
            and not join_requested
            and str(intent.get("complexity", "")).lower() == "single_table"
            and not hint_dim_sources
        )

        # Для каждого dim-слота: сначала смотрим в dim_sources binding,
        # потом в main_table, потом ищем joinable dim-партнёра.
        for slot in dimension_slots:
            if preserve_single_explicit:
                continue

            # 1. Если для слота есть прямой биндинг через user_hints.dim_sources —
            #    таблица уже в locked_tables (см. выше), пропускаем поиск.
            slot_lower = slot.lower() if isinstance(slot, str) else ""
            bound_table: tuple[str, str] | None = None
            for binding_key, binding in hint_dim_sources.items():
                if (
                    binding_key.lower() == slot_lower
                    or slot_lower in binding_key.lower()
                    or binding_key.lower() in slot_lower
                ):
                    tbl_full = binding.get("table") if isinstance(binding, dict) else None
                    if tbl_full and "." in tbl_full:
                        s_part, t_part = tbl_full.split(".", 1)
                        bound_table = (s_part, t_part)
                        break
            if bound_table:
                if bound_table not in deterministic_tables:
                    deterministic_tables.append(bound_table)
                logger.info(
                    "TableResolver: dim '%s' → таблица %s.%s (user_hints.dim_sources)",
                    slot, bound_table[0], bound_table[1],
                )
                continue

            # 2. date обычно есть в main_table — не плодим dim-таблицы.
            if slot == "date" and main_table:
                if _score_table_for_slot(main_table[0], main_table[1], slot) > 0:
                    if main_table not in deterministic_tables:
                        deterministic_tables.append(main_table)
                    continue

            # 3. Иначе ищем лучшую dim-партнёршу для main_table.
            main_slot_profile = (
                _best_slot_profile(main_table[0], main_table[1], slot)
                if main_table else None
            )
            local_slot_sparse = (
                _is_dimension_slot(slot)
                and main_slot_profile is not None
                and float(main_slot_profile.get("not_null", 0.0)) < 25.0
            )
            dim_candidates: list[tuple[float, tuple[str, str]]] = []
            for st in candidate_tables:
                score = _score_table_for_slot(
                    st[0], st[1], slot, metric_entities=metric_entities,
                )
                if score <= 0:
                    continue
                candidate_profile = _best_slot_profile(st[0], st[1], slot)
                candidate_not_null = (
                    float(candidate_profile.get("not_null", 0.0))
                    if candidate_profile else 0.0
                )
                candidate_type = _table_type(st[0], st[1])
                score += _joinability_score(main_table, st)
                if st in explicit_tables:
                    score += 90
                if st in hint_must_keep:
                    score += 150
                if _is_dimension_slot(slot):
                    if st != main_table and candidate_type in {"dim", "ref"}:
                        score += 60
                    if st != main_table and candidate_type == "unknown":
                        score += 20
                    if local_slot_sparse:
                        if st == main_table:
                            score -= 220
                        elif candidate_not_null >= 70.0:
                            score += 220
                        elif candidate_not_null >= 45.0:
                            score += 120
                # Бонус за наличие колонки из user_hints.join_fields
                if hint_join_fields:
                    cols = _get_cols(st[0], st[1])
                    if not cols.empty:
                        col_names = {
                            str(c).lower() for c in cols["column_name"].astype(str)
                        }
                        for hf in hint_join_fields:
                            if hf.lower() in col_names:
                                score += 60
                                break
                dim_candidates.append((score, st))
            if dim_candidates:
                dim_candidates.sort(reverse=True)
                best_table = dim_candidates[0][1]
                if best_table not in deterministic_tables:
                    deterministic_tables.append(best_table)

        if not deterministic_tables:
            deterministic_tables = list(validated_tables)

        # Если main_table может закрыть запрос сама, не тянем лишние JOIN.
        if (
            main_table
            and not explicit_tables
            and not hint_must_keep
            and not hint_dim_sources
            and str(intent.get("complexity", "")).lower() in {"single_table", "", "multi_table"}
            and table_can_satisfy_frame(self.schema, main_table[0], main_table[1], semantic_frame)
        ):
            main_ok_for_dims = True
            for slot in dimension_slots:
                if slot == "date":
                    continue
                slot_profile = _best_slot_profile(main_table[0], main_table[1], slot)
                slot_not_null = float(slot_profile.get("not_null", 0.0)) if slot_profile else 0.0
                if (
                    _score_table_for_slot(main_table[0], main_table[1], slot, metric_entities=metric_entities) <= 0
                    or slot_not_null < 50.0
                ):
                    main_ok_for_dims = False
                    break
            if main_ok_for_dims:
                deterministic_tables = [main_table]
                logger.info(
                    "TableResolver: защищаем от лишнего JOIN — оставляем только main_table %s.%s",
                    main_table[0], main_table[1],
                )

        # Hard-lock: гарантируем, что все must_keep-таблицы присутствуют.
        # Они никогда не должны быть удалены детерминированной коррекцией.
        for tup in locked_tables:
            if tup not in deterministic_tables:
                deterministic_tables.append(tup)
                logger.info(
                    "TableResolver: locked-таблица %s.%s принудительно добавлена",
                    tup[0], tup[1],
                )

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
                    f"{s}.{t}": (
                        100 if (s, t) in locked_tables
                        else (85 if (s, t) in explicit_tables else 70)
                    )
                    for s, t in deterministic_tables
                }

        # Универсальный join-governor: single vs multi-table и pruning.
        slot_scores: dict[str, dict[str, float]] = {}
        for st in deterministic_tables:
            table_key = f"{st[0]}.{st[1]}"
            slot_scores[table_key] = {}
            if metric_slot:
                slot_scores[table_key][str(metric_slot)] = _score_table_for_slot(
                    st[0], st[1], str(metric_slot), metric_entities=metric_entities,
                )
            for slot in dimension_slots:
                slot_scores[table_key][str(slot)] = _score_table_for_slot(
                    st[0], st[1], str(slot), metric_entities=metric_entities,
                )

        join_decision = decide_join_plan(
            selected_tables=validated_tables,
            main_table=main_table,
            locked_tables=locked_tables,
            join_requested=join_requested,
            semantic_frame=semantic_frame,
            requested_grain=requested_grain,
            dimension_slots=dimension_slots,
            slot_scores=slot_scores,
            schema_loader=self.schema,
        )
        governed_tables = list(join_decision.get("selected_tables") or validated_tables)
        if governed_tables != validated_tables:
            logger.info(
                "TableResolver: join_governor скорректировал таблицы: %s -> %s (%s)",
                validated_tables,
                governed_tables,
                join_decision.get("reason"),
            )
            validated_tables = governed_tables

        # Если top-кандидаты близки и пользователь не зафиксировал таблицу явно,
        # безопаснее уточнить источник у пользователя вместо молчаливого выбора.
        disambiguation_options: list[dict[str, Any]] = []
        if (
            not explicit_tables
            and not hint_must_keep
            and forced_single_source is None
            and not join_requested
            and len(ranked_main_candidates) >= 2
        ):
            top_1 = ranked_main_candidates[0]
            top_2 = ranked_main_candidates[1]
            score_gap = float(top_1["score"]) - float(top_2["score"])
            top_tables = [top_1, top_2]
            ambiguous = (
                bool(requested_grain)
                and bool(top_1.get("grain"))
                and bool(top_2.get("grain"))
                and top_1.get("grain") != top_2.get("grain")
                and score_gap < 120.0
            )
            if ambiguous:
                for candidate in top_tables:
                    s_name, t_name = candidate["table"]
                    description = candidate.get("description") or "без описания"
                    grain = candidate.get("grain") or "unknown"
                    disambiguation_options.append({
                        "schema": s_name,
                        "table": t_name,
                        "description": f"{description} | grain={grain}",
                        "key_columns": self.schema.get_primary_keys(s_name, t_name),
                    })

        table_confidence_summary = evaluate_table_confidence(
            table_confidences,
            disambiguation_options=disambiguation_options,
        )
        join_confidence_summary = evaluate_join_confidence(join_decision)
        planning_confidence = build_planning_confidence(
            table_confidence=table_confidence_summary,
            filter_confidence=None,
            join_confidence=join_confidence_summary,
            user_hints=state.get("user_hints"),
            explicit_mode=bool(state.get("explicit_mode")),
        )
        evidence_trace = dict(state.get("evidence_trace") or {})
        evidence_trace.update({
            "semantic_frame": semantic_frame,
            "table_selection": {
                "candidates": ranked_main_candidates[:5],
                "winner": f"{validated_tables[0][0]}.{validated_tables[0][1]}" if validated_tables else "",
                "table_confidences": table_confidences,
            },
            "join_decision": join_decision,
            "planning_confidence": planning_confidence,
        })

        logger.info(
            "TableResolver: выбрано %d таблиц: %s, план из %d шагов, confidence=%s, requested_grain=%s, semantic_frame=%s, planning_confidence=%s",
            len(validated_tables),
            ", ".join(f"{s}.{t}" for s, t in validated_tables),
            len(plan_steps),
            table_confidences,
            requested_grain,
            summarize_dict_keys(semantic_frame, label="semantic_frame"),
            summarize_dict_keys(planning_confidence, label="planning_confidence"),
        )

        if disambiguation_options:
            question = self._build_disambiguation_message(disambiguation_options)
            logger.info(
                "TableResolver: запрашиваю disambiguation по таблицам: %s",
                [f"{o['schema']}.{o['table']}" for o in disambiguation_options],
            )
            return {
                "needs_disambiguation": True,
                "disambiguation_options": disambiguation_options,
                "confirmation_message": question,
                "selected_tables": validated_tables,
                "join_decision": join_decision,
                "planning_confidence": planning_confidence,
                "evidence_trace": evidence_trace,
                "allowed_tables": [f"{s}.{t}" for s, t in validated_tables],
                "graph_iterations": state.get("graph_iterations", 0) + 1,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": question},
                ],
            }

        self.memory.add_message(
            "assistant",
            f"[table_resolver] Выбраны таблицы: "
            f"{', '.join(f'{s}.{t}' for s, t in validated_tables) or 'нет'}, "
            f"план: {len(plan_steps)} шагов"
        )

        # Формируем белый список таблиц — сквозной контракт для sql_writer и sql_static_checker.
        # При followup-запросах мёрджим с предыдущим allowed_tables (не перезаписываем).
        new_allowed = [f"{s}.{t}" for s, t in validated_tables]
        prev_allowed = state.get("allowed_tables") or []
        if prev_allowed and state.get("intent", {}).get("intent") == "followup":
            merged = list(dict.fromkeys(prev_allowed + new_allowed))
        else:
            merged = new_allowed

        return {
            "selected_tables": validated_tables,
            "allowed_tables": merged,
            "join_decision": join_decision,
            "planning_confidence": planning_confidence,
            "evidence_trace": evidence_trace,
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
                    f"JOIN decision: {join_decision}\n"
                    f"Planning confidence: {planning_confidence}\n"
                    f"Разрешённые таблицы (белый список): {merged}\n"
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
