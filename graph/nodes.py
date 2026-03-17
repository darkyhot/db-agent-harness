"""Узлы графа LangGraph: planner, executor, validator, corrector, summarizer."""

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.llm import RateLimitedLLM
from core.memory import MemoryManager
from core.schema_loader import SchemaLoader
from core.sql_builder import SQLQueryBuilder, SQLBuilderError
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
    ) -> None:
        """Инициализация узлов графа.

        Args:
            llm: LLM клиент с rate-limit.
            db_manager: Менеджер БД.
            schema_loader: Загрузчик схемы.
            memory: Менеджер памяти.
            sql_validator: Валидатор SQL.
            tools: Список LangChain tools для агента.
        """
        self.llm = llm
        self.db = db_manager
        self.schema = schema_loader
        self.memory = memory
        self.validator = sql_validator
        self.tools = tools
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

    def _get_system_prompt(self) -> str:
        """Сформировать системный промпт с контекстом."""
        sessions_ctx = self.memory.get_sessions_context()
        long_term = self.memory.get_all_memory()
        lt_ctx = ""
        if long_term:
            lt_ctx = "\n\nДолгосрочная память:\n" + "\n".join(
                f"  {k}: {v}" for k, v in long_term.items()
            )

        schema_ctx = self._get_schema_context()

        return (
            "Ты — аналитический агент для работы с базой данных Greenplum (PostgreSQL-совместимый).\n"
            "Ты помогаешь аналитикам: отвечаешь на вопросы по структуре БД, пишешь и валидируешь SQL,\n"
            "делаешь выгрузки, проектируешь модели данных.\n\n"
            f"{schema_ctx}\n\n"
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
            "6. Отвечай на русском языке.\n\n"
            f"{sessions_ctx}{lt_ctx}"
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
            "Если задача простая — план может состоять из 1-2 шагов.\n"
            "Верни ТОЛЬКО JSON-массив строк, без пояснений.\n"
            "Пример: [\"Шаг 1: Найти таблицы с помощью search_tables('зарплата')\", "
            "\"Шаг 2: Получить колонки через get_table_columns('hr', 'salary')\"]\n\n"
            f"Запрос пользователя: {user_input}"
        )

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

        # Ищем упоминания таблиц в шаге и контексте — добавляем полные описания колонок
        tables_detail = self._get_tables_detail_context(current_step + " " + prev_context)
        tables_detail_section = f"\n\n{tables_detail}\n" if tables_detail else ""

        prompt = (
            f"{self._get_system_prompt()}\n\n"
            f"{tables_detail_section}"
            f"Текущий шаг плана: {current_step}\n\n"
            f"Контекст предыдущих шагов:\n{prev_context}\n\n"
            "Выполни этот шаг. Верни JSON с полями:\n"
            '{"tool": "имя_инструмента", "args": {"параметр": "значение"}}\n'
            "Если шаг не требует инструмента, верни:\n"
            '{"tool": "none", "result": "текстовый ответ"}\n\n'
            "ВАЖНО: Для SELECT-запросов НИКОГДА не пиши сырой SQL.\n"
            "Вместо этого используй query_spec — структурированное описание запроса.\n"
            "SQL будет сгенерирован автоматически движком, что исключит ошибки умножения строк.\n\n"
            "Формат для SELECT-запроса:\n"
            '{"tool": "execute_query", "args": {"query_spec": {\n'
            '  "from": {"schema": "schema_name", "table": "table_name", "alias": "t"},\n'
            '  "select": [\n'
            '    {"expr": "column", "table": "t", "column": "col_name"},\n'
            '    {"expr": "func", "func": "COUNT", "args": ["*"], "alias": "cnt"}\n'
            '  ],\n'
            '  "joins": [{\n'
            '    "type": "left",\n'
            '    "schema": "schema_name", "table": "other_table", "alias": "o",\n'
            '    "on": {"left": "t.id", "right": "o.t_id"},\n'
            '    "use_subquery": false\n'
            '  }],\n'
            '  "where": [{"column": "t.col", "op": ">", "value": 100}],\n'
            '  "group_by": ["t.col"],\n'
            '  "order_by": [{"expr": "cnt", "direction": "desc"}],\n'
            '  "limit": 100\n'
            '}}}\n\n'
            "Правило use_subquery: если правая таблица JOIN-а может содержать дубликаты\n"
            "по ключу соединения (unique_perc < 100%), установи use_subquery: true.\n"
            "Это предотвратит умножение строк.\n\n"
            "Для оконных функций используй:\n"
            '{"expr": "window", "func": "ROW_NUMBER", "partition_by": ["t.dept_id"],\n'
            ' "order_by": [{"expr": "t.salary", "direction": "desc"}], "alias": "rn"}\n\n'
            "Для CTE (WITH) добавь поле ctes в query_spec:\n"
            '{"ctes": [{"name": "dept_avg", "spec": {...вложенная query_spec...}}], "from": ...}\n\n'
            "Для INSERT/UPDATE/DELETE/DDL используй параметр sql (не query_spec):\n"
            '{"tool": "execute_write", "args": {"sql": "INSERT INTO ..."}}'
        )

        response = self.llm.invoke(prompt)

        # Парсим вызов инструмента
        tool_call = self._parse_tool_call(response)

        if tool_call["tool"] == "none":
            result = tool_call.get("result", response)
        else:
            args = tool_call.get("args", {})
            tool_name = tool_call["tool"]

            # Извлекаем SQL для валидации — либо прямой sql, либо строим из query_spec
            sql = args.get("sql")
            query_spec = args.get("query_spec")

            if query_spec and tool_name == "execute_query":
                # Строим SQL из QuerySpec для передачи валидатору
                try:
                    sql = SQLQueryBuilder().build(query_spec)
                except SQLBuilderError as e:
                    result = f"Ошибка построения SQL из QuerySpec: {e}"
                    self.memory.add_message("tool", f"[{tool_name}] {str(result)[:500]}")
                    return {
                        "tool_calls": state.get("tool_calls", []) + [
                            {"tool": tool_name, "args": args, "result": str(result)}
                        ],
                        "current_step": step_idx + 1,
                        "last_error": str(result),
                        "messages": state["messages"] + [
                            {"role": "assistant", "content": f"Шаг {step_idx + 1}: {result}"}
                        ],
                    }

            if sql and tool_name in ("execute_query", "execute_write", "execute_ddl"):
                return {
                    "sql_to_validate": sql,
                    "tool_calls": state.get("tool_calls", []) + [
                        {"tool": tool_name, "args": args, "result": "awaiting_validation"}
                    ],
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"Шаг {step_idx + 1}: SQL отправлен на валидацию"}
                    ],
                }

            # Вызов инструмента
            result = self._call_tool(tool_name, args)

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

        Использует нежадный regex для корректного извлечения первого JSON-объекта.
        """
        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if "tool" in parsed:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Пробуем найти вложенный JSON (с args внутри)
        try:
            # Ищем от первого { до последнего } — но только если первый простой regex не сработал
            start = response.find('{')
            if start != -1:
                depth = 0
                for i in range(start, len(response)):
                    if response[i] == '{':
                        depth += 1
                    elif response[i] == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = response[start:i + 1]
                            parsed = json.loads(candidate)
                            if "tool" in parsed:
                                return parsed
                            break
        except (json.JSONDecodeError, ValueError):
            pass

        return {"tool": "none", "result": response}

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Вызвать инструмент по имени.

        Args:
            tool_name: Имя инструмента.
            args: Аргументы.

        Returns:
            Результат выполнения.
        """
        if tool_name not in self.tool_map:
            return f"Инструмент '{tool_name}' не найден."
        try:
            tool_fn = self.tool_map[tool_name]
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

        # Предупреждения
        warnings_text = ""
        if result.warnings:
            warnings_text = "\nПредупреждения:\n" + "\n".join(f"  ⚠ {w}" for w in result.warnings)

        self.memory.add_message("tool", f"[{tool_name}] {exec_result[:500]}")

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

        tables_detail = self._get_tables_detail_context(current_step + " " + prev_context)
        tables_detail_section = f"\n\n{tables_detail}\n" if tables_detail else ""

        prompt = (
            f"{self._get_system_prompt()}\n\n"
            f"{tables_detail_section}"
            f"Текущий шаг: {current_step}\n"
            f"Ошибка: {error}\n\n"
            f"Контекст предыдущих вызовов:\n{prev_context}\n\n"
            "Исправь ошибку. Верни исправленный вызов инструмента в формате JSON:\n"
            '{"tool": "имя_инструмента", "args": {"параметр": "значение"}}\n\n'
            "Для SELECT-запросов используй query_spec вместо сырого SQL:\n"
            '{"tool": "execute_query", "args": {"query_spec": {"from": {...}, "select": [...], ...}}}\n'
            "Если правая таблица JOIN может иметь дубликаты — установи use_subquery: true в join."
        )

        response = self.llm.invoke(prompt)
        tool_call = self._parse_tool_call(response)

        # Если исправленный вызов содержит SQL — отправить на валидацию
        args = tool_call.get("args", {})
        tool_name = tool_call["tool"]
        sql = args.get("sql")
        query_spec = args.get("query_spec")

        if query_spec and tool_name == "execute_query":
            try:
                sql = SQLQueryBuilder().build(query_spec)
            except SQLBuilderError as e:
                return {
                    "last_error": f"Ошибка построения SQL из QuerySpec: {e}",
                    "retry_count": retry_count + 1,
                }

        if sql and tool_name in ("execute_query", "execute_write", "execute_ddl"):
            return {
                "sql_to_validate": sql,
                "retry_count": retry_count + 1,
                "last_error": None,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": tool_name, "args": args, "result": "awaiting_validation"}
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

        tool_results = "\n".join(
            f"- {tc['tool']}: {tc['result'][:5000]}"
            for tc in state.get("tool_calls", [])
        )

        plan_text = "\n".join(state.get("plan", []))

        prompt = (
            "Сформируй краткий и информативный ответ пользователю на основе результатов.\n\n"
            f"Запрос пользователя: {state['user_input']}\n\n"
            f"План:\n{plan_text}\n\n"
            f"Результаты инструментов:\n{tool_results}\n\n"
            "Дай ответ на русском языке. Если были получены данные — включи их в ответ.\n"
            "Если были предупреждения — упомяни их."
        )

        answer = self.llm.invoke(prompt)
        self.memory.add_message("assistant", answer)

        logger.info("Summarizer: ответ сформирован")
        return {
            "final_answer": answer,
            "messages": state["messages"] + [
                {"role": "assistant", "content": answer}
            ],
        }
