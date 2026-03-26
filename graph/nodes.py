"""Узлы графа LangGraph: planner, executor, validator, corrector, summarizer."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.llm import RateLimitedLLM
from core.memory import MemoryManager
from core.schema_loader import SchemaLoader
from core.sql_validator import SQLValidator
from core.database import DatabaseManager
from graph.state import AgentState

logger = logging.getLogger(__name__)


class GraphNodes:
    """Узлы графа агента."""

    MAX_RETRIES = 3

    def __init__(
        self,
        llm: RateLimitedLLM,
        db_manager: DatabaseManager,
        schema_loader: SchemaLoader,
        memory: MemoryManager,
        sql_validator: SQLValidator,
        tools: list,
        debug_prompt: bool = False,
    ) -> None:
        """Инициализация узлов графа.

        Args:
            llm: LLM клиент с rate-limit.
            db_manager: Менеджер БД.
            schema_loader: Загрузчик схемы.
            memory: Менеджер памяти.
            sql_validator: Валидатор SQL.
            tools: Список LangChain tools для агента.
            debug_prompt: Если True, выводить полный промпт в консоль.
        """
        self.llm = llm
        self.db = db_manager
        self.schema = schema_loader
        self.memory = memory
        self.validator = sql_validator
        self.tools = tools
        self.debug_prompt = debug_prompt
        self.tool_map: dict[str, Any] = {t.name: t for t in tools}
        self.tools_description = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )

    def _get_tables_detail_context(self, text: str) -> str:
        """Найти упоминания таблиц в тексте и вернуть полное описание их колонок из CSV.

        Сканирует паттерны schema.table, сверяет с каталогом tables_list.csv
        и для найденных таблиц возвращает get_table_info() из attr_list.csv.

        Args:
            text: Текст для сканирования (текущий шаг + контекст предыдущих).

        Returns:
            Форматированная строка с деталями таблиц или пустая строка.
        """
        df = self.schema.tables_df
        if df.empty:
            return ""

        # Ищем паттерн schema.table в тексте
        pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b')
        found = set()
        for m in pattern.finditer(text):
            schema_name, table_name = m.group(1).lower(), m.group(2).lower()
            mask = (
                df["schema_name"].str.lower() == schema_name
            ) & (
                df["table_name"].str.lower() == table_name
            )
            if not df[mask].empty:
                row = df[mask].iloc[0]
                found.add((row["schema_name"], row["table_name"]))

        if not found:
            return ""

        details = [self.schema.get_table_info(s, t) for s, t in sorted(found)]
        return (
            "Детальное описание задействованных таблиц (колонки из справочника):\n"
            + "\n\n".join(details)
        )

    def _get_schema_context(self) -> str:
        """Сформировать краткий каталог таблиц из SchemaLoader."""
        df = self.schema.tables_df
        if df.empty:
            return "Каталог таблиц пуст. Используй search_tables для поиска."
        lines = ["Доступные таблицы (schema.table — описание):"]
        for _, row in df.iterrows():
            desc = row.get("description", "")
            lines.append(f"  {row['schema_name']}.{row['table_name']} — {desc}")
        return "\n".join(lines)

    def _get_session_history_context(self) -> str:
        """Сформировать контекст истории текущей сессии для системного промпта."""
        messages = self.memory.get_session_messages()
        # Исключаем последнее сообщение — это текущий запрос пользователя,
        # который уже присутствует в промпте явно
        messages = messages[:-1] if messages else []
        # Берём не более 20 последних сообщений
        messages = messages[-20:]
        if not messages:
            return ""

        limits = {"tool": 500, "user": 2000, "assistant": 2000}
        lines = ["История текущей сессии:"]
        for m in messages:
            limit = limits.get(m["role"], 500)
            content = m["content"]
            if len(content) > limit:
                content = content[:limit] + "..."
            lines.append(f"  [{m['role']}] {content}")
        return "\n".join(lines)

    def _get_long_term_memory_context(self) -> str:
        """Сформировать контекст долгосрочной памяти со структурированными слоями."""
        layer_keys = {"user_facts", "behavior_patterns", "user_instructions"}
        sections = []

        # Слой 1: Факты о пользователе
        facts = self.memory.get_memory_list("user_facts")
        if facts:
            sections.append(
                "Факты о пользователе:\n" + "\n".join(f"  - {f}" for f in facts)
            )

        # Слой 2: Паттерны поведения
        patterns = self.memory.get_memory_list("behavior_patterns")
        if patterns:
            sections.append(
                "Паттерны поведения пользователя (учитывай в стиле ответов):\n"
                + "\n".join(f"  - {p}" for p in patterns)
            )

        # Слой 3: Инструкции пользователя
        instructions = self.memory.get_memory_list("user_instructions")
        if instructions:
            sections.append(
                "Инструкции пользователя (ОБЯЗАТЕЛЬНО соблюдай):\n"
                + "\n".join(f"  - {i}" for i in instructions)
            )

        # Прочие ключи долгосрочной памяти (обратная совместимость)
        all_memory = self.memory.get_all_memory()
        other = {k: v for k, v in all_memory.items() if k not in layer_keys}
        if other:
            sections.append(
                "Прочая долгосрочная память:\n"
                + "\n".join(f"  {k}: {v}" for k, v in other.items())
            )

        if not sections:
            return ""
        return "\n\n" + "\n\n".join(sections)

    def _get_system_prompt(self) -> str:
        """Сформировать системный промпт с контекстом."""
        sessions_ctx = self.memory.get_sessions_context()
        lt_ctx = self._get_long_term_memory_context()

        schema_ctx = self._get_schema_context()
        history_ctx = self._get_session_history_context()
        history_section = f"\n\n{history_ctx}" if history_ctx else ""

        return (
            "Ты — аналитический агент для работы с базой данных Greenplum (PostgreSQL-совместимый).\n"
            "Ты помогаешь аналитикам: отвечаешь на вопросы по структуре БД, пишешь и валидируешь SQL,\n"
            "делаешь выгрузки, проектируешь модели данных.\n\n"
            f"{schema_ctx}\n\n"
            "ВАЖНО: Каталог таблиц выше — это твои знания о структуре БД. "
            "Если пользователь спрашивает какие таблицы ты знаешь, какие данные доступны, "
            "что есть в базе — отвечай на основе этого каталога. "
            "Ты ЗНАЕШЬ эти таблицы, это твоя база знаний.\n\n"
            "Доступные инструменты:\n"
            f"{self.tools_description}\n\n"
            "Правила:\n"
            "1. ВСЕГДА используй ТОЛЬКО реальные имена таблиц из каталога выше в формате schema.table. "
            "НЕ придумывай имена схем и таблиц. Если нужной таблицы нет в каталоге — "
            "сначала найди её через search_tables или search_by_description.\n"
            "2. Всегда проверяй SQL через валидатор перед выполнением.\n"
            "3. Для JOIN-ов проверяй уникальность ключей.\n"
            "4. Для деструктивных операций (DELETE, DROP, TRUNCATE) запрашивай подтверждение.\n"
            "5. Сохраняй результаты выгрузок в workspace/.\n"
            "6. Если пользователь просит сгенерировать или сохранить SQL-запрос, текст или отчёт — "
            "используй create_file для сохранения в workspace/ (например, query.sql, report.txt).\n"
            "7. Отвечай на русском языке. НО в SQL-коде алиасы (AS) пиши ТОЛЬКО на английском.\n"
            "8. В SQL-запросах НИКОГДА не используй русские/кириллические алиасы. "
            "Примеры правильно: AS outflow, AS total_revenue, AS client_count. "
            'Примеры НЕПРАВИЛЬНО: AS "отток", AS "выручка", AS "кол_во_клиентов". '
            "Это СТРОГОЕ правило для всех алиасов колонок, подзапросов и таблиц.\n\n"
            f"{sessions_ctx}{lt_ctx}{history_section}"
        )

    def planner(self, state: AgentState) -> dict[str, Any]:
        """Узел планирования: составляет план шагов для выполнения запроса.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с планом.
        """
        user_input = state["user_input"]
        logger.info("Planner: обработка запроса: %s", user_input[:100])

        self.memory.add_message("user", user_input)

        prompt = (
            f"{self._get_system_prompt()}\n\n"
            "Задача: составь пронумерованный план шагов для выполнения запроса пользователя.\n"
            "Для каждого шага укажи какой инструмент нужно вызвать и с какими параметрами.\n"
            "Если задача простая — план может состоять из 1-2 шагов.\n\n"
            "ОБЯЗАТЕЛЬНАЯ СТРАТЕГИЯ ДЛЯ АНАЛИТИЧЕСКИХ ВОПРОСОВ (подсчёты, агрегаты, выборки):\n"
            "1. ПЕРВЫЙ ШАГ — определи какие таблицы нужны. Укажи их явно в формате schema.table.\n"
            "   Если не уверен в таблице — используй search_tables или search_by_description.\n"
            "2. НЕ ПИШИ SQL сразу. После определения таблиц система автоматически подгрузит\n"
            "   структуру колонок и образец 10 строк данных. Ты увидишь их перед выполнением.\n"
            "3. Только ПОСЛЕ изучения структуры и данных — строй SQL-запрос.\n\n"
            "ПОЧЕМУ ЭТО ВАЖНО:\n"
            "- Таблица-справочник может содержать неуникальные строки (например, одна сущность\n"
            "  представлена несколькими строками с разными атрибутами). COUNT(*) без понимания\n"
            "  гранулярности даст неверный результат.\n"
            "- Без знания колонок и типов данных ты не сможешь написать корректный SQL.\n"
            "- Образец данных покажет формат значений, NULL'ы, дубликаты.\n\n"
            "ВАЖНО: Если ответ на вопрос пользователя уже содержится в каталоге таблиц, "
            "долгосрочной памяти или контексте выше — НЕ вызывай инструменты. "
            "Вместо этого верни план из одного шага: "
            '[\"Ответить на основе контекста (без вызова инструментов)\"]\n'
            "Верни ТОЛЬКО JSON-массив строк, без пояснений.\n"
            "Пример аналитического запроса:\n"
            "[\"Шаг 1: Определить нужные таблицы — schema.table_name (система подгрузит структуру и семпл)\",\n"
            " \"Шаг 2: Проанализировать структуру и данные, определить гранулярность таблицы\",\n"
            " \"Шаг 3: Написать SQL-запрос с учётом структуры данных\"]\n"
            "Пример без инструмента: [\"Ответить на основе контекста (без вызова инструментов)\"]\n\n"
            f"Запрос пользователя: {user_input}"
        )

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — planner]\n{'='*80}\n{prompt}\n{'='*80}\n")

        response = self.llm.invoke(prompt)

        # Парсинг плана
        plan = self._parse_plan(response)
        logger.info("Planner: составлен план из %d шагов", len(plan))

        return {
            "plan": plan,
            "current_step": 0,
            "retry_count": 0,
            "last_error": None,
            "messages": state["messages"] + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": "План:\n" + "\n".join(plan)},
            ],
        }

    def table_explorer(self, state: AgentState) -> dict[str, Any]:
        """Узел автоматической разведки таблиц: подгружает структуру и семплы.

        Извлекает упоминания schema.table из плана, загружает описание колонок
        из CSV-справочника и семпл 10 строк из БД для каждой таблицы.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с обогащённым контекстом таблиц.
        """
        plan_text = "\n".join(state.get("plan", []))
        user_input = state.get("user_input", "")
        scan_text = f"{plan_text}\n{user_input}"

        # Извлекаем schema.table из плана и пользовательского запроса
        df = self.schema.tables_df
        if df.empty:
            logger.info("TableExplorer: каталог таблиц пуст, пропускаем")
            return {"tables_context": ""}

        pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b')
        found_tables: set[tuple[str, str]] = set()
        for m in pattern.finditer(scan_text):
            schema_name, table_name = m.group(1).lower(), m.group(2).lower()
            mask = (
                df["schema_name"].str.lower() == schema_name
            ) & (
                df["table_name"].str.lower() == table_name
            )
            if not df[mask].empty:
                row = df[mask].iloc[0]
                found_tables.add((row["schema_name"], row["table_name"]))

        if not found_tables:
            logger.info("TableExplorer: таблицы не найдены в плане, пропускаем")
            return {"tables_context": ""}

        logger.info("TableExplorer: найдено %d таблиц для разведки: %s",
                     len(found_tables),
                     ", ".join(f"{s}.{t}" for s, t in found_tables))

        sections = []
        for schema_name, table_name in sorted(found_tables):
            # 1. Описание колонок из CSV-справочника
            table_info = self.schema.get_table_info(schema_name, table_name)

            # 2. Семпл 10 строк из БД
            try:
                sample_df = self.db.get_sample(schema_name, table_name, 10)
                if sample_df.empty:
                    sample_text = "(таблица пуста)"
                else:
                    sample_text = sample_df.to_markdown(index=False)
            except Exception as e:
                logger.warning("TableExplorer: ошибка семпла %s.%s: %s",
                               schema_name, table_name, e)
                sample_text = f"(ошибка загрузки семпла: {e})"

            sections.append(
                f"### {schema_name}.{table_name}\n\n"
                f"**Структура (из справочника):**\n{table_info}\n\n"
                f"**Образец данных (10 строк):**\n{sample_text}"
            )

        tables_context = (
            "=== РАЗВЕДКА ТАБЛИЦ (автоматически подгруженные данные) ===\n\n"
            "Изучи структуру и образцы данных ПЕРЕД написанием SQL.\n"
            "Обрати внимание на:\n"
            "- Гранулярность: что является одной строкой? Есть ли дубликаты по ключевым полям?\n"
            "- NULL'ы и пустые значения в колонках\n"
            "- Формат дат, числовых значений, кодов\n"
            "- Какие колонки можно использовать для фильтрации и группировки\n\n"
            + "\n\n".join(sections)
        )

        self.memory.add_message(
            "tool",
            f"[table_explorer] Подгружена структура и семплы для: "
            f"{', '.join(f'{s}.{t}' for s, t in sorted(found_tables))}"
        )

        return {
            "tables_context": tables_context,
            "messages": state["messages"] + [
                {"role": "assistant",
                 "content": f"Подгружена структура и семплы таблиц: "
                            f"{', '.join(f'{s}.{t}' for s, t in sorted(found_tables))}"}
            ],
        }

    def _parse_plan(self, response: str) -> list[str]:
        """Извлечь план из ответа LLM."""
        # Пробуем JSON
        try:
            # Ищем JSON-массив в ответе
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                plan = json.loads(match.group())
                if isinstance(plan, list) and all(isinstance(s, str) for s in plan):
                    return plan
        except (json.JSONDecodeError, ValueError):
            pass

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

    def executor(self, state: AgentState) -> dict[str, Any]:
        """Узел выполнения: выполняет текущий шаг плана, вызывая инструменты.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния.
        """
        plan = state["plan"]
        step_idx = state["current_step"]

        if step_idx >= len(plan):
            logger.info("Executor: все шаги выполнены")
            return {}

        current_step = plan[step_idx]
        logger.info("Executor: шаг %d/%d: %s", step_idx + 1, len(plan), current_step[:100])

        recent_calls = state.get("tool_calls", [])[-5:]
        if recent_calls:
            *old_calls, last_call = recent_calls
            prev_context = "\n".join(
                f"  {tc['tool']}: {tc['result'][:1000]}" for tc in old_calls
            )
            prev_context += f"\n  {last_call['tool']} (полный результат):\n{last_call['result']}"
        else:
            prev_context = ""

        # Контекст разведки таблиц (от table_explorer)
        tables_context = state.get("tables_context", "")
        tables_context_section = f"\n\n{tables_context}\n" if tables_context else ""

        # Дополнительно: ищем упоминания таблиц в шаге, которых нет в tables_context
        tables_detail = self._get_tables_detail_context(current_step + " " + prev_context)
        tables_detail_section = f"\n\n{tables_detail}\n" if tables_detail and tables_detail not in tables_context else ""

        prompt = (
            f"{self._get_system_prompt()}\n\n"
            f"{tables_context_section}"
            f"{tables_detail_section}"
            f"Текущий шаг плана: {current_step}\n\n"
            f"Контекст предыдущих шагов:\n{prev_context}\n\n"
            "Выполни этот шаг. Верни JSON с полями:\n"
            '{"tool": "имя_инструмента", "args": {"параметр": "значение"}}\n'
            "Если шаг не требует инструмента (ответ уже есть в каталоге таблиц, "
            "контексте разведки или результатах предыдущих шагов), верни:\n"
            '{"tool": "none", "result": "полный текстовый ответ на вопрос пользователя"}\n'
            "Если нужно выполнить SQL, верни:\n"
            '{"tool": "execute_query", "args": {"sql": "SELECT ... FROM schema.table"}}\n'
            "ВАЖНО: В SQL всегда указывай схему перед таблицей (schema.table).\n"
            "ВАЖНО: Алиасы колонок и таблиц в SQL — только на английском (AS outflow, НЕ AS \"отток\").\n"
            "ВАЖНО: Если выше есть РАЗВЕДКА ТАБЛИЦ — обязательно изучи структуру и образцы данных.\n"
            "Пойми гранулярность таблицы (что является одной строкой) перед написанием запроса.\n"
            "Например, если в справочнике одна сущность представлена несколькими строками — \n"
            "нужен SELECT COUNT(DISTINCT ...), а не простой COUNT(*)."
        )

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — executor, шаг {step_idx + 1}]\n{'='*80}\n{prompt}\n{'='*80}\n")

        response = self.llm.invoke(prompt)

        # Парсим вызов инструмента
        tool_call = self._parse_tool_call(response)

        if tool_call["tool"] == "none":
            result = tool_call.get("result", response)
        else:
            # Проверяем, порождает ли шаг SQL для валидации
            sql = tool_call.get("args", {}).get("sql")
            if sql and tool_call["tool"] in ("execute_query", "execute_write", "execute_ddl"):
                return {
                    "sql_to_validate": sql,
                    "tool_calls": state.get("tool_calls", []) + [
                        {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": "awaiting_validation"}
                    ],
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"Шаг {step_idx + 1}: SQL отправлен на валидацию"}
                    ],
                }

            # Вызов инструмента
            result = self._call_tool(tool_call["tool"], tool_call.get("args", {}))

        # Проверяем, нужна ли disambiguation (несколько таблиц найдено)
        options = self._check_disambiguation_needed(
            tool_call["tool"], str(result), state["user_input"],
        )
        if options is not None:
            display_lines = ["Найдено несколько подходящих витрин данных:", ""]
            for i, opt in enumerate(options, 1):
                display_lines.append(f"  {i}. {opt['schema']}.{opt['table']} — {opt['description']}")
                if opt.get("key_columns"):
                    display_lines.append(f"     Ключевые колонки: {', '.join(opt['key_columns'])}")
            display_msg = "\n".join(display_lines)

            return {
                "needs_disambiguation": True,
                "disambiguation_options": options,
                "confirmation_message": display_msg,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": str(result)}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": display_msg}
                ],
            }

        self.memory.add_message("tool", f"[{tool_call['tool']}] {str(result)[:500]}")

        return {
            "tool_calls": state.get("tool_calls", []) + [
                {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": str(result)}
            ],
            "current_step": step_idx + 1,
            "last_error": None,
            "retry_count": 0,
            "sql_to_validate": None,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Шаг {step_idx + 1}: {str(result)[:1000]}"}
            ],
        }

    def _parse_tool_call(self, response: str) -> dict[str, Any]:
        """Извлечь вызов инструмента из ответа LLM.

        Использует парсер с подсчётом глубины скобок и учётом строковых литералов,
        чтобы корректно обрабатывать вложенные объекты (например, args с SQL внутри).
        """
        # Ищем все JSON-объекты верхнего уровня с учётом вложенности и строк
        for candidate in self._extract_json_objects(response):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue

        return {"tool": "none", "result": response}

    @staticmethod
    def _extract_json_objects(text: str) -> list[str]:
        """Извлечь JSON-объекты из текста с учётом вложенных скобок и строковых литералов.

        Returns:
            Список строк-кандидатов JSON-объектов.
        """
        candidates = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 0
                in_string = False
                escape_next = False
                start = i
                for j in range(i, len(text)):
                    ch = text[j]
                    if escape_next:
                        escape_next = False
                        continue
                    if ch == '\\' and in_string:
                        escape_next = True
                        continue
                    if ch == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[start:j + 1])
                            i = j
                            break
                else:
                    break
            i += 1
        return candidates

    def _check_disambiguation_needed(
        self, tool_name: str, result: str, user_input: str,
    ) -> list[dict[str, Any]] | None:
        """Проверить, вернул ли поиск несколько таблиц, требующих уточнения.

        Returns:
            Список опций для выбора пользователем или None.
        """
        if tool_name not in ("search_tables", "search_by_description"):
            return None

        match = re.search(r'Найдено таблиц:\s*(\d+)', result)
        if not match or int(match.group(1)) <= 1:
            return None

        # Извлекаем schema.table из строк результата
        table_pattern = re.compile(r'^\s+(\w+)\.(\w+)\s*—\s*(.*)$', re.MULTILINE)
        options: list[dict[str, Any]] = []
        for m in table_pattern.finditer(result):
            schema_name, table_name, description = (
                m.group(1), m.group(2), m.group(3).strip(),
            )
            options.append({
                "schema": schema_name,
                "table": table_name,
                "description": description,
                "key_columns": self.schema.get_primary_keys(schema_name, table_name),
            })

        if len(options) <= 1:
            return None

        # Если пользователь уже указал конкретную таблицу — не спрашиваем повторно
        user_lower = user_input.lower()
        for opt in options:
            full_name = f"{opt['schema']}.{opt['table']}"
            if full_name.lower() in user_lower:
                return None

        return options

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Вызвать инструмент по имени с валидацией аргументов.

        Args:
            tool_name: Имя инструмента.
            args: Аргументы.

        Returns:
            Результат выполнения.
        """
        if tool_name not in self.tool_map:
            return f"Инструмент '{tool_name}' не найден."

        tool_fn = self.tool_map[tool_name]

        # Валидация аргументов: проверяем что все обязательные поля присутствуют
        # и нет лишних ключей (LLM может галлюцинировать параметры)
        if hasattr(tool_fn, "args_schema") and tool_fn.args_schema is not None:
            schema = tool_fn.args_schema
            if hasattr(schema, "model_fields"):
                expected_fields = set(schema.model_fields.keys())
                required_fields = {
                    k for k, v in schema.model_fields.items()
                    if v.is_required()
                }
                provided_fields = set(args.keys())
                missing = required_fields - provided_fields
                if missing:
                    return (
                        f"Ошибка инструмента {tool_name}: "
                        f"отсутствуют обязательные параметры: {', '.join(sorted(missing))}"
                    )
                extra = provided_fields - expected_fields
                if extra:
                    logger.warning(
                        "Tool %s: лишние аргументы будут проигнорированы: %s",
                        tool_name, extra,
                    )
                    args = {k: v for k, v in args.items() if k in expected_fields}

        try:
            result = tool_fn.invoke(args)
            logger.info("Tool %s выполнен успешно", tool_name)
            return str(result)
        except Exception as e:
            logger.error("Tool %s ошибка: %s", tool_name, e)
            return f"Ошибка инструмента {tool_name}: {e}"

    def _is_tool_error(self, result: str) -> bool:
        """Проверить, является ли результат ошибкой инструмента.

        Проверяет по префиксу 'Ошибка инструмента' вместо поиска слова 'Ошибка'
        в произвольном месте строки, чтобы избежать ложных срабатываний на данных.
        """
        return result.startswith("Ошибка инструмента ") or result.startswith("Ошибка выполнения запроса:")

    def sql_validator_node(self, state: AgentState) -> dict[str, Any]:
        """Узел валидации SQL: проверяет SQL перед выполнением.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния.
        """
        sql = state.get("sql_to_validate")
        if not sql:
            return {"sql_to_validate": None}

        logger.info("Validator: проверка SQL: %s", sql[:200])
        result = self.validator.validate(sql)

        # Требуется подтверждение пользователя — сохраняем SQL в состоянии
        if result.needs_confirmation:
            return {
                "needs_confirmation": True,
                "confirmation_message": result.confirmation_message,
                "sql_to_validate": sql,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"⚠️ {result.confirmation_message}"}
                ],
            }

        # Есть ошибки — отправить в корректор
        if not result.is_valid:
            error_msg = result.summary()
            logger.warning("Validator: SQL невалиден: %s", error_msg[:200])
            return {
                "last_error": error_msg,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка валидации:\n{error_msg}"}
                ],
            }

        # SQL валиден — выполняем
        tool_calls = state.get("tool_calls", [])
        last_tool = tool_calls[-1] if tool_calls else {}
        tool_name = last_tool.get("tool", "execute_query")

        exec_result = self._call_tool(tool_name, {"sql": sql})

        # Проверка на пустой результат (0 строк) для SELECT-запросов
        empty_result = False
        if tool_name == "execute_query" and (
            exec_result == "Запрос выполнен. Результат пуст."
            or exec_result.strip() == ""
        ):
            empty_result = True
            logger.warning("Validator: запрос вернул 0 строк: %s", sql[:200])

        # Предупреждения
        warnings_text = ""
        if result.warnings:
            warnings_text = "\nПредупреждения:\n" + "\n".join(f"  ⚠ {w}" for w in result.warnings)

        if empty_result:
            warnings_text += "\n⚠ Запрос вернул 0 строк. Возможно, условия фильтрации слишком строгие " \
                             "или данные отсутствуют. Проверь условия WHERE, значения фильтров и формат дат."

        self.memory.add_message("tool", f"[{tool_name}] {exec_result[:500]}")

        # Если пустой результат — отправляем на коррекцию, чтобы LLM пересмотрел запрос
        if empty_result:
            return {
                "sql_to_validate": None,
                "last_error": (
                    f"SQL-запрос выполнен успешно, но вернул 0 строк. SQL: {sql}\n"
                    "Проверь условия WHERE, формат дат, значения фильтров. "
                    "Попробуй ослабить условия или проверить наличие данных в таблице."
                ),
                "retry_count": state.get("retry_count", 0),
                "tool_calls": tool_calls[:-1] + [
                    {**last_tool, "result": exec_result}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"SQL выполнен, но вернул 0 строк.{warnings_text}"}
                ],
            }

        return {
            "sql_to_validate": None,
            "last_error": None,
            "retry_count": 0,
            "current_step": state["current_step"] + 1,
            "tool_calls": tool_calls[:-1] + [
                {**last_tool, "result": exec_result}
            ],
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"SQL выполнен.{warnings_text}\n{exec_result[:1000]}"}
            ],
        }

    def corrector(self, state: AgentState) -> dict[str, Any]:
        """Узел коррекции: анализирует ошибку и исправляет шаг.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния.
        """
        error = state.get("last_error", "")
        retry_count = state.get("retry_count", 0)
        step_idx = state["current_step"]
        plan = state["plan"]
        current_step = plan[step_idx] if step_idx < len(plan) else "неизвестный шаг"

        logger.info("Corrector: попытка %d/%d, ошибка: %s", retry_count + 1, self.MAX_RETRIES, error[:200])

        if retry_count >= self.MAX_RETRIES:
            return {
                "last_error": None,
                "current_step": step_idx + 1,
                "retry_count": 0,
                "final_answer": f"Не удалось выполнить шаг '{current_step}' после {self.MAX_RETRIES} попыток. "
                                f"Последняя ошибка: {error}",
            }

        recent_calls = state.get("tool_calls", [])[-3:]
        if recent_calls:
            *old_calls, last_call = recent_calls
            prev_context = "\n".join(
                f"  {tc['tool']}: {tc['result'][:1000]}" for tc in old_calls
            )
            prev_context += f"\n  {last_call['tool']} (полный результат):\n{last_call['result']}"
        else:
            prev_context = ""

        tables_context = state.get("tables_context", "")
        tables_context_section = f"\n\n{tables_context}\n" if tables_context else ""

        tables_detail = self._get_tables_detail_context(current_step + " " + prev_context)
        tables_detail_section = f"\n\n{tables_detail}\n" if tables_detail and tables_detail not in tables_context else ""

        prompt = (
            f"{self._get_system_prompt()}\n\n"
            f"{tables_context_section}"
            f"{tables_detail_section}"
            f"Текущий шаг: {current_step}\n"
            f"Ошибка: {error}\n\n"
            f"Контекст предыдущих вызовов:\n{prev_context}\n\n"
            "Исправь ошибку. Верни исправленный вызов инструмента в формате JSON:\n"
            '{"tool": "имя_инструмента", "args": {"параметр": "значение"}}\n\n'
            "ПОДСКАЗКИ для исправления:\n"
            "- Если запрос вернул 0 строк — вызови get_sample чтобы посмотреть реальные данные\n"
            "  и понять формат значений, после чего исправь условия WHERE.\n"
            "- Если ошибка в имени колонки — вызови get_table_columns для проверки структуры.\n"
            "- Если COUNT(*) дал неожиданный результат — проверь гранулярность таблицы\n"
            "  (возможно нужен COUNT(DISTINCT ...) вместо COUNT(*))."
        )

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — corrector]\n{'='*80}\n{prompt}\n{'='*80}\n")

        response = self.llm.invoke(prompt)
        tool_call = self._parse_tool_call(response)

        # Если исправленный вызов содержит SQL — отправить на валидацию
        sql = tool_call.get("args", {}).get("sql")
        if sql and tool_call["tool"] in ("execute_query", "execute_write", "execute_ddl"):
            return {
                "sql_to_validate": sql,
                "retry_count": retry_count + 1,
                "last_error": None,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": "awaiting_validation"}
                ],
            }

        # Вызов исправленного инструмента
        result = self._call_tool(tool_call["tool"], tool_call.get("args", {}))

        if self._is_tool_error(result):
            return {
                "last_error": str(result),
                "retry_count": retry_count + 1,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Повторная ошибка (попытка {retry_count + 1}): {result}"}
                ],
            }

        self.memory.add_message("tool", f"[corrector:{tool_call['tool']}] {str(result)[:500]}")

        return {
            "tool_calls": state.get("tool_calls", []) + [
                {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": str(result)}
            ],
            "current_step": step_idx + 1,
            "last_error": None,
            "retry_count": 0,
            "sql_to_validate": None,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Исправлено. Шаг {step_idx + 1}: {str(result)[:1000]}"}
            ],
        }

    def summarizer(self, state: AgentState) -> dict[str, Any]:
        """Узел формирования финального ответа пользователю.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с final_answer.
        """
        # Если уже есть финальный ответ (например, от corrector при исчерпании попыток)
        if state.get("final_answer"):
            self.memory.add_message("assistant", state["final_answer"])
            return {}

        tool_results_parts = []
        for tc in state.get("tool_calls", []):
            sql = tc.get("args", {}).get("sql", "")
            sql_line = f"\n  SQL: {sql}" if sql else ""
            tool_results_parts.append(
                f"- {tc['tool']}{sql_line}\n  Результат: {tc['result'][:5000]}"
            )
        tool_results = "\n".join(tool_results_parts)

        plan_text = "\n".join(state.get("plan", []))

        schema_ctx = self._get_schema_context()

        prompt = (
            "Сформируй краткий и информативный ответ пользователю на основе результатов.\n\n"
            f"Справочник таблиц:\n{schema_ctx}\n\n"
            f"Запрос пользователя: {state['user_input']}\n\n"
            f"План:\n{plan_text}\n\n"
            f"Результаты инструментов:\n{tool_results}\n\n"
            "Дай ответ на русском языке. Если были получены данные — включи их в ответ.\n"
            "Если в ответе есть SQL-код — алиасы (AS) пиши только на английском.\n"
            "Если были предупреждения — упомяни их."
        )

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — summarizer]\n{'='*80}\n{prompt}\n{'='*80}\n")

        answer = self.llm.invoke(prompt)
        self.memory.add_message("assistant", answer)

        logger.info("Summarizer: ответ сформирован")
        return {
            "final_answer": answer,
            "messages": state["messages"] + [
                {"role": "assistant", "content": answer}
            ],
        }
