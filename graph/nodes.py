"""Узлы графа LangGraph: planner, executor, validator, corrector, summarizer."""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import pandas as pd

from core.llm import RateLimitedLLM
from core.memory import MemoryManager
from core.schema_loader import SchemaLoader
from core.sql_validator import SQLValidator
from core.database import DatabaseManager
from graph.state import AgentState

logger = logging.getLogger(__name__)

# === Общие константы для промптов ===

SQL_RULES = (
    "Правила SQL (Greenplum / PostgreSQL):\n"
    "- Имена таблиц ВСЕГДА в формате schema.table\n"
    "- Алиасы СТРОГО на английском: AS outflow, AS total_cnt\n"
    '- ЗАПРЕЩЕНО: AS "отток", AS "выручка", кириллица в алиасах и именах колонок\n'
    "- Изучи РАЗВЕДКУ ТАБЛИЦ (если есть) перед написанием SQL\n"
    "- Пойми гранулярность (что = одна строка) перед COUNT/агрегатами\n"
    "- Для JOIN проверяй уникальность ключей через check_key_uniqueness\n"
    "- GROUP BY: перечисли ВСЕ не-агрегированные колонки из SELECT\n"
    "- Даты: используй приведение типов (::date, ::timestamp, TO_DATE()). "
    "Проверь формат дат в образце данных перед фильтрацией\n"
    "- NULL: используй COALESCE() или IS NOT NULL / IS NULL в WHERE. "
    "Не сравнивай с NULL через = или !=\n"
    "- LIMIT: добавляй LIMIT для exploration-запросов и больших таблиц\n"
    "- Используй стандартный PostgreSQL синтаксис (Greenplum совместим)\n\n"
    "Безопасный SQL (предотвращение дублирования строк):\n"
    "- ПЕРЕД написанием JOIN — изучи JOIN-АНАЛИЗ в разведке таблиц (если есть)\n"
    "- Если unique_perc ключа < 50% — ОБЯЗАТЕЛЬНО подзапрос с DISTINCT\n"
    "- Для lookup-таблиц (справочников) JOIN безопасен по PK\n"
    "- Для fact-таблиц с дублями — предварительная агрегация в подзапросе\n"
    "- Паттерн: JOIN (SELECT DISTINCT key, col FROM table) alias ON ...\n"
    "- Паттерн: JOIN (SELECT key, SUM(val) FROM table GROUP BY key) alias ON ...\n"
    "- DISTINCT на внешнем SELECT — ЗАПРЕЩЁН как маскировка проблемы"
)

SQL_CHECKLIST = (
    "Чеклист перед финализацией SQL:\n"
    "1. Формат дат соответствует данным из образца (YYYY-MM-DD, DD.MM.YYYY, и т.д.)?\n"
    "2. NULL обработан (COALESCE/IS NOT NULL) для колонок с высоким % NULL?\n"
    "3. JOIN-ключи уникальны? Если нет (unique_perc < 50%) — использован подзапрос с DISTINCT?\n"
    "4. GROUP BY содержит все не-агрегированные колонки?\n"
    "5. Алиасы только на английском?\n"
    "6. Есть LIMIT для больших выборок?\n"
    "7. Типы данных совместимы в WHERE/JOIN (нет сравнения text с integer)?"
)

GIGACHAT_COMMON_ERRORS = (
    "Частые ошибки (проверяй и избегай):\n"
    "- Кириллица в алиасах → замени на английские\n"
    "- Несуществующая колонка → проверь через get_table_columns или разведку таблиц\n"
    "- Неправильный формат даты → проверь get_sample для реального формата\n"
    "- Пропущен GROUP BY → добавь все не-агрегированные колонки\n"
    "- COUNT(*) вместо COUNT(DISTINCT ...) → проверь гранулярность\n"
    "- Сравнение с NULL через = → используй IS NULL / IS NOT NULL"
)


@dataclass
class ToolResult:
    """Структурированный результат вызова инструмента."""
    success: bool
    data: str
    error: str | None = None

    def __str__(self) -> str:
        return self.data if self.success else (self.error or self.data)


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
        self.tools_brief = ", ".join(t.name for t in tools)
        # Компактное описание инструментов для planner (имя + первое предложение описания)
        compact_lines = []
        for t in tools:
            first_sentence = t.description.split("\n")[0].split(". ")[0]
            compact_lines.append(f"- {t.name}: {first_sentence}")
        self.tools_compact = "\n".join(compact_lines)

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

    # Стоп-слова для предфильтрации каталога (не несут семантики для поиска таблиц)
    _STOP_WORDS = frozenset({
        "в", "на", "по", "за", "из", "с", "к", "о", "у", "и", "а", "но", "что",
        "как", "все", "это", "так", "уже", "или", "не", "да", "нет", "мне", "мой",
        "ты", "он", "она", "мы", "они", "их", "его", "её", "для", "от", "до",
        "при", "без", "через", "между", "после", "перед", "под", "над", "про",
        "сколько", "какие", "какой", "какая", "какое", "где", "кто", "когда",
        "покажи", "найди", "выведи", "дай", "скажи", "сделай", "можешь",
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
        "is", "are", "was", "were", "be", "been", "have", "has", "had",
        "show", "get", "find", "give", "tell", "make", "how", "many",
    })

    _MAX_TABLES_FOR_FULL_CATALOG = 50

    def _get_schema_context(self, user_input: str = "") -> str:
        """Сформировать краткий каталог таблиц из SchemaLoader.

        При большом каталоге (>50 таблиц) выполняет предфильтрацию по ключевым
        словам из запроса пользователя, чтобы не перегружать контекст LLM.
        """
        df = self.schema.tables_df
        if df.empty:
            return "Каталог таблиц пуст. Используй search_tables для поиска."

        # Маленький каталог — отдаём целиком
        if len(df) <= self._MAX_TABLES_FOR_FULL_CATALOG or not user_input:
            lines = ["Доступные таблицы (schema.table — описание):"]
            for _, row in df.iterrows():
                desc = row.get("description", "")
                lines.append(f"  {row['schema_name']}.{row['table_name']} — {desc}")
            return "\n".join(lines)

        # Большой каталог — предфильтрация по ключевым словам запроса
        words = re.findall(r'[a-zA-Zа-яА-ЯёЁ]{3,}', user_input.lower())
        keywords = [w for w in words if w not in self._STOP_WORDS]

        if not keywords:
            # Нет значимых слов — отдаём весь каталог
            lines = ["Доступные таблицы (schema.table — описание):"]
            for _, row in df.iterrows():
                desc = row.get("description", "")
                lines.append(f"  {row['schema_name']}.{row['table_name']} — {desc}")
            return "\n".join(lines)

        # Ищем таблицы по каждому ключевому слову
        found_indices = set()
        for kw in keywords:
            mask = (
                df["table_name"].str.lower().str.contains(kw, na=False)
                | df["schema_name"].str.lower().str.contains(kw, na=False)
                | df["description"].str.lower().str.contains(kw, na=False)
            )
            found_indices.update(df[mask].index.tolist())

        if not found_indices:
            return (
                f"В каталоге {len(df)} таблиц, но по запросу не найдено прямых совпадений.\n"
                "Используй search_tables или search_by_description для поиска нужных таблиц."
            )

        filtered = df.loc[sorted(found_indices)]
        lines = [
            f"Релевантные таблицы (найдено {len(filtered)} из {len(df)} по запросу):"
        ]
        for _, row in filtered.iterrows():
            desc = row.get("description", "")
            lines.append(f"  {row['schema_name']}.{row['table_name']} — {desc}")
        lines.append(
            f"\nВсего в каталоге {len(df)} таблиц. "
            "Если нужная таблица не в списке — используй search_tables / search_by_description."
        )
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

    def _get_planner_system_prompt(self, user_input: str = "") -> str:
        """Системный промпт для планировщика.

        Включает каталог таблиц (с предфильтрацией при большом каталоге)
        и инструменты — planner должен знать всё.
        """
        lt_ctx = self._get_long_term_memory_context()
        history_ctx = self._get_session_history_context()
        schema_ctx = self._get_schema_context(user_input)
        sessions_ctx = self.memory.get_sessions_context()

        return (
            "Ты — планировщик аналитического агента для Greenplum (PostgreSQL).\n"
            "Задача: составить пошаговый план выполнения запроса пользователя.\n"
            "Верни ТОЛЬКО JSON-массив строк — шаги плана.\n\n"
            f"{sessions_ctx}{lt_ctx}"
            f"{chr(10) + chr(10) + history_ctx if history_ctx else ''}\n\n"
            f"{schema_ctx}\n\n"
            "Это твоя база знаний о таблицах. Если пользователь спрашивает что есть в базе — "
            "отвечай на основе этого каталога.\n\n"
            f"Доступные инструменты:\n{self.tools_compact}\n\n"
            "Правила:\n"
            "- Используй ТОЛЬКО реальные таблицы из каталога в формате schema.table\n"
            "- НЕ пиши SQL в плане — только укажи нужные таблицы, SQL будет на этапе исполнения\n"
            "- НЕ выдумывай таблицы — если не знаешь таблицу, используй search_tables / search_by_description\n"
            "- Если ответ уже есть в каталоге или контексте — план из одного шага без инструментов\n"
            "- Если запрос неоднозначный — начни с search_tables/search_by_description для уточнения\n"
            "- Если search_by_description не нашёл результат — попробуй синонимы или search_tables с английскими ключевыми словами\n"
            "- Если пользователь спрашивает 'какие витрины/таблицы есть' или 'что ты знаешь' — "
            "это вопрос-перечисление, отвечай из каталога БЕЗ вызова search\n\n"
            "=== ПРИМЕРЫ ===\n\n"
            'Аналитика: "Сколько клиентов в регионе X?"\n'
            '["Определить таблицы — dm.clients (подгрузится структура)", '
            '"Написать SQL с учётом гранулярности таблицы"]\n\n'
            'JOIN нескольких таблиц: "Покажи продажи по менеджерам за январь 2024"\n'
            '["Определить таблицы — dm.sales, dm.managers (подгрузится структура)", '
            '"Написать SQL с JOIN по ключу менеджера и фильтром по дате"]\n\n'
            'Вопрос по схеме: "Какие таблицы содержат данные о продажах?"\n'
            '["Ответить на основе каталога (без вызова инструментов)"]\n\n'
            'Вопрос-перечисление: "Какие витрины ты знаешь?" / "Какие таблицы есть?" / "Покажи все витрины"\n'
            '["Ответить на основе каталога — перечислить все витрины с описаниями (без вызова инструментов)"]\n\n'
            'Поиск таблицы: "Есть ли данные по оттоку?"\n'
            '["Найти таблицы через search_by_description по теме оттока"]\n\n'
            'Выгрузка: "Выгрузи клиентов за 2024 в CSV"\n'
            '["Определить таблицы — dm.clients (подгрузится структура)", '
            '"Написать SQL с фильтром по дате", "Экспортировать через export_query в CSV"]\n\n'
            'JOIN с риском дублирования: "Сумма продаж по менеджерам"\n'
            '["Определить таблицы — dm.sales, dm.managers (подгрузится структура и JOIN-анализ)", '
            '"Написать SQL с учётом JOIN-анализа: если ключ не уникален — подзапрос с DISTINCT"]\n\n'
            "=== АНТИПРИМЕРЫ (НЕ ДЕЛАЙ ТАК) ===\n\n"
            "ПЛОХО: [\"SELECT * FROM dm.clients WHERE region = 'X'\"]\n"
            "  → Нельзя писать SQL в плане!\n\n"
            "ПЛОХО: [\"Найти таблицу abc.xyz\"]\n"
            "  → Таблица abc.xyz не существует в каталоге! Используй только реальные таблицы.\n"
        )

    def _get_executor_system_prompt(self) -> str:
        """Системный промпт для исполнителя шагов.

        Компактный: роль + правила SQL + формат ответа. Без каталога таблиц —
        executor получает только релевантные таблицы через tables_context.
        """
        lt_ctx = self._get_long_term_memory_context()

        return (
            "Ты — исполнитель шагов аналитического агента для Greenplum.\n"
            "Получаешь шаг плана и контекст. Вызываешь один инструмент или отвечаешь напрямую.\n\n"
            f"Доступные инструменты:\n{self.tools_description}\n\n"
            "Формат ответа — СТРОГО один JSON-объект (без пояснений, без markdown-обёрток):\n"
            '{"tool": "имя_инструмента", "args": {"параметр": "значение"}}\n'
            'Если инструмент не нужен: {"tool": "none", "result": "текст ответа"}\n\n'
            "Примеры ответов:\n"
            '{"tool": "execute_query", "args": {"sql": "SELECT client_id, SUM(amount) AS total_amount '
            "FROM dm.sales WHERE sale_date >= '2024-01-01'::date GROUP BY client_id LIMIT 100\"}}\n"
            '{"tool": "get_table_columns", "args": {"schema": "dm", "table": "clients"}}\n'
            '{"tool": "none", "result": "В каталоге найдено 3 таблицы с данными о продажах."}\n\n'
            f"{SQL_RULES}\n\n"
            f"{GIGACHAT_COMMON_ERRORS}\n\n"
            "Порядок рассуждений при написании SQL:\n"
            "1. Изучи разведку таблиц: колонки, типы данных, образцы данных\n"
            "2. Определи гранулярность таблицы (что = одна строка)\n"
            "3. Выбери нужные колонки и проверь их наличие в разведке\n"
            "4. Определи условия фильтрации (WHERE) — проверь формат значений в образце\n"
            "5. Если нужен JOIN — ОБЯЗАТЕЛЬНО изучи JOIN-АНАЛИЗ в разведке таблиц. "
            "Если unique_perc < 50% — используй подзапрос с DISTINCT\n"
            "6. Напиши SQL и пройди по чеклисту выше\n"
            f"{lt_ctx}"
        )

    def _get_corrector_system_prompt(self) -> str:
        """Системный промпт для корректора ошибок.

        Компактный: роль + формат ответа + стратегии коррекции + паттерны ошибок.
        Включает примеры прошлых исправлений из долгосрочной памяти.
        """
        lt_ctx = self._get_long_term_memory_context()

        # Примеры прошлых исправлений из долгосрочной памяти
        examples_ctx = ""
        examples = self.memory.get_memory_list("correction_examples")
        if examples:
            recent = examples[-5:]  # последние 5 примеров
            examples_ctx = (
                "\n\nПримеры прошлых исправлений (учитывай эти паттерны):\n"
                + "\n".join(f"  - {ex}" for ex in recent)
            )

        return (
            "Ты — корректор ошибок аналитического агента для Greenplum.\n"
            "Анализируешь ошибку предыдущего шага и выдаёшь исправленный вызов инструмента.\n\n"
            "Порядок действий:\n"
            "1. Прочитай текст ошибки и определи её тип\n"
            "2. Найди причину в контексте (разведка таблиц, предыдущие вызовы)\n"
            "3. Выдай исправленный JSON-вызов инструмента\n\n"
            f"Доступные инструменты:\n{self.tools_description}\n\n"
            "Формат ответа — СТРОГО один JSON-объект:\n"
            '{"tool": "имя_инструмента", "args": {"параметр": "значение"}}\n\n'
            f"{SQL_RULES}\n\n"
            f"{GIGACHAT_COMMON_ERRORS}\n\n"
            "Исправление row explosion в JOIN:\n"
            "Если ошибка содержит 'ROW EXPLOSION' или 'POST-EXECUTION ROW EXPLOSION':\n"
            "1. Оберни НЕуникальную сторону JOIN в подзапрос с DISTINCT\n"
            "2. Или используй предварительную агрегацию в подзапросе (GROUP BY + SUM/COUNT)\n"
            "3. НИКОГДА не добавляй DISTINCT к внешнему SELECT — это маскирует проблему\n"
            "4. Следуй шаблону из текста ошибки\n"
            "Пример исправления:\n"
            "БЫЛО: SELECT a.* FROM t1 a JOIN schema.t2 b ON a.key = b.key\n"
            "СТАЛО: SELECT a.* FROM t1 a JOIN (SELECT DISTINCT key, col FROM schema.t2) b ON a.key = b.key\n"
            f"{lt_ctx}"
            f"{examples_ctx}"
        )

    def _get_summarizer_system_prompt(self) -> str:
        """Системный промпт для формирования финального ответа."""
        return (
            "Ты — аналитический агент для Greenplum. Формируешь финальный ответ пользователю.\n\n"
            "Правила ответа:\n"
            "- Отвечай на русском языке\n"
            "- SQL-алиасы только на английском\n"
            "- Табличные данные оформляй в markdown-таблицу\n"
            "- SQL-код оборачивай в ```sql блок\n"
            "- Не пересказывай шаги плана — только результат\n"
            "- Не повторяй вопрос пользователя\n"
            "- Если были предупреждения — упомяни кратко в конце\n"
            "- Если был выполнен SQL-запрос — покажи его в блоке ```sql и кратко объясни логику\n"
            "- Интерпретируй результат в бизнес-терминах, если это возможно\n"
            "- Если данные обрезаны — укажи это и покажи общее количество строк\n"
            "- Если результат большой — покажи топ-10 строк и общую статистику\n"
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

        system_prompt = self._get_planner_system_prompt(user_input)

        user_prompt = (
            f"Запрос пользователя: {user_input}\n\n"
            "Стратегия планирования:\n"
            "1. Для аналитики (подсчёты, агрегаты, выборки):\n"
            "   - Первый шаг: укажи нужные таблицы в формате schema.table\n"
            "   - Система автоматически подгрузит структуру и 10 строк данных\n"
            "   - SQL пиши только ПОСЛЕ изучения структуры\n"
            "2. Для вопросов по схеме: если ответ есть в каталоге — один шаг без инструментов\n"
            "3. Для выгрузок: определи таблицу → SQL → export_query\n"
            "4. Если таблица неизвестна — используй search_tables / search_by_description\n"
            "5. Для DDL: спроектируй структуру → execute_ddl"
        )

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — planner]\n{'='*80}\n"
                  f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n")

        response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.3)

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
            return {
                "tables_context": (
                    "=== РАЗВЕДКА ТАБЛИЦ ===\n\n"
                    "Таблицы не были определены на этапе планирования.\n"
                    "Используй get_table_columns или search_tables для получения "
                    "структуры нужных таблиц перед написанием SQL."
                )
            }

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

        # JOIN safety analysis: проверяем общие колонки между таблицами
        join_analysis_lines = []
        table_list = sorted(found_tables)
        if len(table_list) >= 2:
            for i, (s1, t1) in enumerate(table_list):
                cols1 = self.schema.get_table_columns(s1, t1)
                for s2, t2 in table_list[i + 1:]:
                    cols2 = self.schema.get_table_columns(s2, t2)
                    if cols1.empty or cols2.empty:
                        continue
                    shared = set(cols1["column_name"]) & set(cols2["column_name"])
                    for col in sorted(shared):
                        r1 = cols1[cols1["column_name"] == col].iloc[0]
                        r2 = cols2[cols2["column_name"] == col].iloc[0]
                        u1 = float(r1.get("unique_perc", 0)) if pd.notna(r1.get("unique_perc")) else 0
                        u2 = float(r2.get("unique_perc", 0)) if pd.notna(r2.get("unique_perc")) else 0
                        pk1 = bool(r1.get("is_primary_key", False))
                        pk2 = bool(r2.get("is_primary_key", False))

                        status1 = "PK" if pk1 else (f"unique={u1:.0f}%" if u1 >= 95 else f"ДУБЛИ unique={u1:.0f}%")
                        status2 = "PK" if pk2 else (f"unique={u2:.0f}%" if u2 >= 95 else f"ДУБЛИ unique={u2:.0f}%")

                        safe = (pk1 or u1 >= 95) and (pk2 or u2 >= 95)
                        recommendation = "безопасен" if safe else "ОПАСНО — нужен подзапрос с DISTINCT"

                        join_analysis_lines.append(
                            f"  {s1}.{t1}.{col} ({status1}) ↔ {s2}.{t2}.{col} ({status2})"
                            f" → {recommendation}"
                        )

        join_analysis = ""
        if join_analysis_lines:
            join_analysis = (
                "\n\n=== JOIN-АНАЛИЗ (автоматическая оценка безопасности JOIN) ===\n"
                + "\n".join(join_analysis_lines)
                + "\n\nЕсли ключ помечен как ДУБЛИ — используй подзапрос с DISTINCT или агрегацию."
            )

        tables_context = (
            "=== РАЗВЕДКА ТАБЛИЦ (автоматически подгруженные данные) ===\n\n"
            "Изучи структуру и образцы данных ПЕРЕД написанием SQL.\n"
            "Обрати внимание на:\n"
            "- Гранулярность: что является одной строкой? Есть ли дубликаты по ключевым полям?\n"
            "- NULL'ы и пустые значения в колонках\n"
            "- Формат дат, числовых значений, кодов\n"
            "- Какие колонки можно использовать для фильтрации и группировки\n"
            "- Связь колонок с запросом пользователя: какие колонки соответствуют терминам из вопроса?\n\n"
            + "\n\n".join(sections)
            + join_analysis
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

    def executor(self, state: AgentState) -> dict[str, Any]:
        """Узел выполнения: выполняет текущий шаг плана, вызывая инструменты.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния.
        """
        iterations = state.get("graph_iterations", 0) + 1
        plan = state["plan"]
        step_idx = state["current_step"]

        if step_idx >= len(plan):
            logger.info("Executor: все шаги выполнены")
            return {"graph_iterations": iterations}

        current_step = plan[step_idx]
        logger.info("Executor: шаг %d/%d: %s", step_idx + 1, len(plan), current_step[:100])

        recent_calls = state.get("tool_calls", [])[-3:]
        if recent_calls:
            *old_calls, last_call = recent_calls
            prev_context = "\n".join(
                f"  {tc['tool']}: {tc['result'][:500]}" for tc in old_calls
            )
            prev_context += f"\n  {last_call['tool']} (полный результат):\n{last_call['result']}"
        else:
            prev_context = ""

        # Контекст разведки таблиц (от table_explorer)
        tables_context = state.get("tables_context", "")

        # Дополнительно: ищем упоминания таблиц в шаге, которых нет в tables_context
        tables_detail = self._get_tables_detail_context(current_step + " " + prev_context)
        if tables_detail and tables_detail in tables_context:
            tables_detail = ""

        system_prompt = self._get_executor_system_prompt()

        # Краткий контекст полного плана
        plan_summary = " → ".join(
            f"{'[✓] ' if i < step_idx else '[→] ' if i == step_idx else ''}{s[:60]}"
            for i, s in enumerate(plan)
        )

        user_parts = []
        user_parts.append(f"[ЗАПРОС ПОЛЬЗОВАТЕЛЯ]\n{state['user_input']}")
        user_parts.append(f"[ПЛАН] {plan_summary}")
        if tables_context:
            user_parts.append(tables_context)
        if tables_detail:
            user_parts.append(tables_detail)
        user_parts.append(f"{SQL_CHECKLIST}")
        user_parts.append(f"[ТЕКУЩИЙ ШАГ {step_idx + 1}/{len(plan)}]\n{current_step}")
        if prev_context:
            user_parts.append(f"[КОНТЕКСТ ПРЕДЫДУЩИХ ШАГОВ]\n{prev_context}")

        user_prompt = "\n\n".join(user_parts)

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — executor, шаг {step_idx + 1}]\n{'='*80}\n"
                  f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n")

        response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.1)

        # Парсим вызов инструмента
        tool_call = self._parse_tool_call(response)

        if tool_call["tool"] == "none":
            result_str = tool_call.get("result", response)
        else:
            # Проверяем, порождает ли шаг SQL для валидации
            sql = tool_call.get("args", {}).get("sql")
            if sql and tool_call["tool"] in ("execute_query", "execute_write", "execute_ddl", "export_query"):
                return {
                    "sql_to_validate": sql,
                    "graph_iterations": iterations,
                    "tool_calls": state.get("tool_calls", []) + [
                        {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": "awaiting_validation"}
                    ],
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"Шаг {step_idx + 1}: SQL отправлен на валидацию"}
                    ],
                }

            # Вызов инструмента
            tool_result = self._call_tool(tool_call["tool"], tool_call.get("args", {}))
            result_str = str(tool_result)

            # Ошибка инструмента — отправить в корректор
            if not tool_result.success:
                return {
                    "last_error": tool_result.error,
                    "graph_iterations": iterations,
                    "tool_calls": state.get("tool_calls", []) + [
                        {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": result_str}
                    ],
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"Ошибка на шаге {step_idx + 1}: {result_str}"}
                    ],
                }

        # Проверяем, нужна ли disambiguation (несколько таблиц найдено)
        options = self._check_disambiguation_needed(
            tool_call["tool"], result_str, state["user_input"],
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
                "graph_iterations": iterations,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": result_str}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": display_msg}
                ],
            }

        self.memory.add_message("tool", f"[{tool_call['tool']}] {result_str[:500]}")

        return {
            "tool_calls": state.get("tool_calls", []) + [
                {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": result_str}
            ],
            "current_step": step_idx + 1,
            "last_error": None,
            "retry_count": 0,
            "sql_to_validate": None,
            "graph_iterations": iterations,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Шаг {step_idx + 1}: {result_str[:1000]}"}
            ],
        }

    @staticmethod
    def _clean_llm_json(text: str) -> str:
        """Очистить ответ LLM от markdown-обёрток и типичных ошибок GigaChat.

        GigaChat часто оборачивает JSON в ```json ... ```, добавляет trailing commas,
        или вставляет пояснения до/после JSON.
        """
        # Убираем markdown code block обёртки
        text = re.sub(r'```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        # Убираем trailing commas перед } или ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text.strip()

    def _parse_tool_call(
        self, response: str, retry_on_fail: bool = True, _original: str | None = None,
    ) -> dict[str, Any]:
        """Извлечь вызов инструмента из ответа LLM.

        Использует парсер с подсчётом глубины скобок и учётом строковых литералов,
        чтобы корректно обрабатывать вложенные объекты (например, args с SQL внутри).
        При неудаче — retry с уточняющим промптом.
        """
        original_response = _original or response
        cleaned = self._clean_llm_json(response)

        # Ищем все JSON-объекты верхнего уровня с учётом вложенности и строк
        for candidate in self._extract_json_objects(cleaned):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue

        # Также пробуем из оригинального ответа (на случай если clean сломал что-то)
        if cleaned != response:
            for candidate in self._extract_json_objects(response):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue

        # Retry: повторный запрос к LLM с уточнением формата
        if retry_on_fail:
            logger.warning("JSON не найден в ответе LLM, retry с уточнением формата")
            retry_system = (
                "Ты — форматировщик ответов. Преобразуй текст в JSON-объект.\n"
                "Верни ТОЛЬКО один JSON-объект, без пояснений, без markdown-обёрток."
            )
            retry_user = (
                "Предыдущий ответ не содержит валидный JSON.\n"
                f"Ответ: {response[:1500]}\n\n"
                "Верни ТОЛЬКО один JSON-объект в одном из форматов:\n"
                '{"tool": "execute_query", "args": {"sql": "SELECT ..."}}\n'
                '{"tool": "get_table_columns", "args": {"schema": "dm", "table": "clients"}}\n'
                '{"tool": "none", "result": "текстовый ответ"}'
            )
            retry_response = str(self.llm.invoke_with_system(
                retry_system, retry_user, temperature=0.0,
            ))
            return self._parse_tool_call(
                retry_response, retry_on_fail=False, _original=original_response,
            )

        return {"tool": "none", "result": original_response}

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

        # Вопросы-перечисления ("какие витрины есть?") не требуют disambiguation —
        # пользователь хочет увидеть ВСЕ результаты, а не выбрать один
        listing_patterns = [
            r'какие\s+(витрины|таблицы|данные)',
            r'покажи\s+(все|витрины|таблицы)',
            r'что\s+(ты\s+)?знаешь',
            r'список\s+(витрин|таблиц)',
            r'перечисли',
            r'какие\s+есть',
        ]
        user_lower = user_input.lower()
        if any(re.search(p, user_lower) for p in listing_patterns):
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

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Вызвать инструмент по имени с валидацией аргументов.

        Args:
            tool_name: Имя инструмента.
            args: Аргументы.

        Returns:
            ToolResult с результатом выполнения.
        """
        if tool_name not in self.tool_map:
            return ToolResult(success=False, data="", error=f"Инструмент '{tool_name}' не найден.")

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
                    return ToolResult(
                        success=False, data="",
                        error=f"Ошибка инструмента {tool_name}: "
                              f"отсутствуют обязательные параметры: {', '.join(sorted(missing))}",
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
            data = str(result)
            logger.info("Tool %s выполнен успешно", tool_name)
            # Проверяем ошибки, возвращённые самим инструментом как строки
            if data.startswith("Ошибка инструмента ") or data.startswith("Ошибка выполнения запроса:"):
                return ToolResult(success=False, data=data, error=data)
            return ToolResult(success=True, data=data)
        except Exception as e:
            error_msg = f"Ошибка инструмента {tool_name}: {e}"
            logger.error("Tool %s ошибка: %s", tool_name, e)
            return ToolResult(success=False, data="", error=error_msg)

    @staticmethod
    def _check_result_sanity(user_input: str, exec_result: str) -> list[str]:
        """Эвристические проверки осмысленности результата SQL-запроса.

        Возвращает список предупреждений (пустой если всё ок).
        """
        warnings = []
        user_lower = user_input.lower()

        # Считаем строки в markdown-таблице (строки с | в начале, кроме заголовка и разделителя)
        result_lines = [
            line for line in exec_result.split("\n")
            if line.strip().startswith("|") and not line.strip().startswith("|---") and not line.strip().startswith("| ---")
        ]
        # Первая строка — заголовок, вторая — разделитель, остальные — данные
        data_row_count = max(0, len(result_lines) - 1)  # -1 для заголовка

        # Проверка: вопрос «сколько/количество» но результат — несколько строк (вероятно забыли агрегацию)
        count_patterns = re.compile(r'сколько|количество|общее\s+число|итого|всего', re.IGNORECASE)
        if count_patterns.search(user_lower) and data_row_count > 5:
            warnings.append(
                f"Вопрос содержит 'сколько/количество', но результат содержит {data_row_count} строк. "
                "Возможно, пропущена агрегация (COUNT, SUM) или GROUP BY."
            )

        # Проверка: результат упирается в LIMIT при наличии JOIN
        if "показано" in exec_result and "из" in exec_result:
            limit_match = re.search(r'показано\s+\d+\s+из\s+(\d+)', exec_result)
            if limit_match and int(limit_match.group(1)) >= 1000:
                warnings.append(
                    f"Результат содержит {limit_match.group(1)}+ строк (упирается в LIMIT). "
                    "Если SQL содержит JOIN — возможен row explosion."
                )

        # Проверка: не-списочный вопрос, но очень много строк
        listing_patterns = re.compile(
            r'список|перечисли|все\s+(строки|записи|данные)|покажи\s+все', re.IGNORECASE,
        )
        if data_row_count > 100 and not listing_patterns.search(user_lower):
            warnings.append(
                f"Результат содержит {data_row_count} строк, хотя вопрос не предполагает список. "
                "Проверь, нет ли дублирования из-за JOIN."
            )

        return warnings

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

        # Сохраняем информацию о рисках JOIN для post-execution проверки
        join_risk_info = {}
        if result.join_checks:
            join_risk_info = {
                "multiplication_factor": result.multiplication_factor,
                "non_unique_joins": [
                    jc for jc in result.join_checks if not jc["is_unique"]
                ],
                "rewrite_suggestions": result.rewrite_suggestions,
            }

        # Есть ошибки — отправить в корректор
        if not result.is_valid:
            error_msg = result.summary()
            logger.warning("Validator: SQL невалиден: %s", error_msg[:200])
            return {
                "last_error": error_msg,
                "join_risk_info": join_risk_info,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка валидации:\n{error_msg}"}
                ],
            }

        # SQL валиден — выполняем
        tool_calls = state.get("tool_calls", [])
        last_tool = tool_calls[-1] if tool_calls else {}
        tool_name = last_tool.get("tool", "execute_query")

        t0 = time.monotonic()
        tool_result = self._call_tool(tool_name, {"sql": sql})
        duration_ms = int((time.monotonic() - t0) * 1000)
        exec_result = str(tool_result)

        # Ошибка выполнения — отправляем в корректор
        if not tool_result.success:
            self.memory.log_sql_execution(state["user_input"], sql, 0, "error", duration_ms)
            return {
                "sql_to_validate": None,
                "last_error": tool_result.error,
                "tool_calls": tool_calls[:-1] + [
                    {**last_tool, "result": exec_result}
                ],
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Ошибка выполнения SQL:\n{exec_result}"}
                ],
            }

        # Проверка на пустой результат (0 строк) для SELECT-запросов
        empty_result = False
        if tool_name == "execute_query" and (
            exec_result == "Запрос выполнен. Результат пуст."
            or exec_result.strip() == ""
        ):
            empty_result = True
            logger.warning("Validator: запрос вернул 0 строк: %s", sql[:200])

        # Предупреждения из валидатора
        warnings_text = ""
        if result.warnings:
            warnings_text = "\nПредупреждения:\n" + "\n".join(f"  ⚠ {w}" for w in result.warnings)

        if empty_result:
            warnings_text += "\n⚠ Запрос вернул 0 строк. Возможно, условия фильтрации слишком строгие " \
                             "или данные отсутствуют. Проверь условия WHERE, значения фильтров и формат дат."

        # Подсчёт строк из markdown-таблицы (нужен для sanity checks и post-exec detection)
        data_lines = [l for l in exec_result.split("\n") if l.strip().startswith("|")]
        row_count = max(0, len(data_lines) - 2) if data_lines else 0  # -2: заголовок + разделитель

        # Sanity checks результата (бизнес-уровень)
        if not empty_result and tool_name == "execute_query":
            sanity_warnings = self._check_result_sanity(state["user_input"], exec_result)
            for sw in sanity_warnings:
                warnings_text += f"\n⚠ {sw}"

        # Post-execution row explosion detection
        if not empty_result and join_risk_info and tool_name == "execute_query":
            factor = join_risk_info.get("multiplication_factor", 1.0)
            if factor > 1.0 and row_count > 100:
                suggestions = "\n".join(join_risk_info.get("rewrite_suggestions", []))
                explosion_msg = (
                    f"POST-EXECUTION ROW EXPLOSION: Результат содержит {row_count} строк "
                    f"при multiplication_factor={factor:.1f}x. "
                    f"Вероятно дублирование строк из-за JOIN.\n{suggestions}"
                )
                self.memory.log_sql_execution(
                    state["user_input"], sql, row_count, "row_explosion", duration_ms,
                )
                return {
                    "sql_to_validate": None,
                    "last_error": explosion_msg,
                    "retry_count": state.get("retry_count", 0),
                    "join_risk_info": join_risk_info,
                    "tool_calls": tool_calls[:-1] + [
                        {**last_tool, "result": exec_result}
                    ],
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"⚠ {explosion_msg}"}
                    ],
                }

        # Аудит-лог SQL
        audit_status = "empty" if empty_result else "success"
        self.memory.log_sql_execution(state["user_input"], sql, row_count, audit_status, duration_ms)

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
                "join_risk_info": join_risk_info,
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
            "join_risk_info": join_risk_info,
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
        iterations = state.get("graph_iterations", 0) + 1

        logger.info("Corrector: попытка %d/%d, ошибка: %s", retry_count + 1, self.MAX_RETRIES, error[:200])

        if retry_count >= self.MAX_RETRIES:
            return {
                "last_error": None,
                "current_step": step_idx + 1,
                "retry_count": 0,
                "graph_iterations": iterations,
                "final_answer": f"Не удалось выполнить шаг '{current_step}' после {self.MAX_RETRIES} попыток. "
                                f"Последняя ошибка: {error}",
            }

        recent_calls = state.get("tool_calls", [])[-3:]
        if recent_calls:
            *old_calls, last_call = recent_calls
            prev_context = "\n".join(
                f"  {tc['tool']}: {tc['result'][:500]}" for tc in old_calls
            )
            prev_context += f"\n  {last_call['tool']} (полный результат):\n{last_call['result']}"
        else:
            prev_context = ""

        tables_context = state.get("tables_context", "")

        tables_detail = self._get_tables_detail_context(current_step + " " + prev_context)
        if tables_detail and tables_detail in tables_context:
            tables_detail = ""

        system_prompt = self._get_corrector_system_prompt()

        # Эскалация стратегии в зависимости от номера попытки
        is_row_explosion = "ROW EXPLOSION" in error
        if retry_count == 0:
            strategy_hint = (
                "Стратегии исправления:\n"
                "- 0 строк → вызови get_sample, проверь формат данных, ослабь WHERE\n"
                "- Ошибка колонки → вызови get_table_columns\n"
                "- Синтаксическая ошибка → исправь SQL по тексту ошибки\n"
                "- Неожиданный COUNT → проверь гранулярность (COUNT(DISTINCT ...) вместо COUNT(*))\n"
                "- ROW EXPLOSION → используй подзапрос с DISTINCT для неуникальной таблицы. "
                "НЕ добавляй DISTINCT к внешнему SELECT. Следуй шаблону из текста ошибки."
            )
        else:
            strategy_hint = (
                f"ВНИМАНИЕ: Это попытка {retry_count + 1} из {self.MAX_RETRIES}.\n"
                "Предыдущий подход не сработал — СМЕНИ СТРАТЕГИЮ:\n"
                "- Попробуй другой инструмент для диагностики (get_sample, get_table_columns)\n"
                "- Переформулируй SQL с нуля, не патчи старый\n"
                "- Проверь, правильная ли таблица используется"
            )
            if is_row_explosion:
                strategy_hint += (
                    "\n- ROW EXPLOSION не исправлен! Оберни неуникальную таблицу в "
                    "подзапрос: JOIN (SELECT DISTINCT key, col FROM table) alias ON ...\n"
                    "- Или используй предварительную агрегацию: "
                    "JOIN (SELECT key, SUM(val) FROM table GROUP BY key) alias ON ..."
                )

        user_parts = []
        user_parts.append(f"[ЗАПРОС ПОЛЬЗОВАТЕЛЯ]\n{state['user_input']}")
        user_parts.append(f"[ПОПЫТКА {retry_count + 1} из {self.MAX_RETRIES}]")
        user_parts.append(f"[ШАГ]\n{current_step}")
        user_parts.append(f"[ОШИБКА]\n{error}")
        user_parts.append(strategy_hint)
        # Передаём join_risk_info для row explosion ошибок
        risk = state.get("join_risk_info", {})
        if risk and is_row_explosion:
            non_unique = risk.get("non_unique_joins", [])
            risk_details = "\n".join(
                f"  {jc.get('table', '?')}.{jc.get('columns', '?')} — дубли: {jc.get('duplicate_pct', '?')}%"
                for jc in non_unique
            )
            user_parts.append(
                f"[JOIN RISK INFO]\n"
                f"Multiplication factor: {risk.get('multiplication_factor', '?')}x\n"
                f"Проблемные ключи:\n{risk_details}"
            )
        if tables_context:
            user_parts.append(f"[КОНТЕКСТ ТАБЛИЦ]\n{tables_context}")
        if tables_detail:
            user_parts.append(tables_detail)
        if prev_context:
            user_parts.append(f"[ПРЕДЫДУЩИЕ ВЫЗОВЫ]\n{prev_context}")

        user_prompt = "\n\n".join(user_parts)

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — corrector]\n{'='*80}\n"
                  f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n")

        response = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.2)
        tool_call = self._parse_tool_call(response)

        # Если исправленный вызов содержит SQL — отправить на валидацию
        sql = tool_call.get("args", {}).get("sql")
        if sql and tool_call["tool"] in ("execute_query", "execute_write", "execute_ddl"):
            # Сохраняем пример исправления для обучения
            example = f"Ошибка: {error[:100]} → Исправление: {sql[:150]}"
            correction_examples = state.get("correction_examples", []) + [example]
            return {
                "sql_to_validate": sql,
                "retry_count": retry_count + 1,
                "last_error": None,
                "graph_iterations": iterations,
                "correction_examples": correction_examples,
                "tool_calls": state.get("tool_calls", []) + [
                    {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": "awaiting_validation"}
                ],
            }

        # Вызов исправленного инструмента
        tool_result = self._call_tool(tool_call["tool"], tool_call.get("args", {}))
        result_str = str(tool_result)

        if not tool_result.success:
            return {
                "last_error": tool_result.error,
                "retry_count": retry_count + 1,
                "graph_iterations": iterations,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Повторная ошибка (попытка {retry_count + 1}): {result_str}"}
                ],
            }

        self.memory.add_message("tool", f"[corrector:{tool_call['tool']}] {result_str[:500]}")

        return {
            "tool_calls": state.get("tool_calls", []) + [
                {"tool": tool_call["tool"], "args": tool_call.get("args", {}), "result": result_str}
            ],
            "current_step": step_idx + 1,
            "last_error": None,
            "retry_count": 0,
            "sql_to_validate": None,
            "graph_iterations": iterations,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Исправлено. Шаг {step_idx + 1}: {result_str[:1000]}"}
            ],
        }

    def summarizer(self, state: AgentState) -> dict[str, Any]:
        """Узел формирования финального ответа пользователю.

        Args:
            state: Текущее состояние графа.

        Returns:
            Обновления состояния с final_answer.
        """
        # Сохраняем примеры исправлений в долгосрочную память
        new_examples = state.get("correction_examples", [])
        if new_examples:
            existing = self.memory.get_memory_list("correction_examples")
            combined = (existing + new_examples)[-20:]  # храним последние 20
            self.memory.set_memory("correction_examples", json.dumps(combined, ensure_ascii=False))

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

        system_prompt = self._get_summarizer_system_prompt()

        # Краткий контекст таблиц (описания колонок без семплов)
        tables_ctx = state.get("tables_context", "")
        tables_summary = ""
        if tables_ctx:
            # Извлекаем только строки с описаниями колонок (без семплов данных)
            ctx_lines = tables_ctx.split("\n")
            summary_lines = []
            skip_sample = False
            for line in ctx_lines:
                if "Образец данных" in line or "Sample" in line:
                    skip_sample = True
                    continue
                if skip_sample and (line.strip() == "" or line.startswith("[")):
                    skip_sample = False
                if not skip_sample:
                    summary_lines.append(line)
            tables_summary = "\n".join(summary_lines).strip()

        user_parts_sum = [f"Запрос пользователя: {state['user_input']}"]
        user_parts_sum.append(f"План:\n{plan_text}")
        if tables_summary:
            user_parts_sum.append(f"Контекст таблиц (для интерпретации колонок):\n{tables_summary}")
        user_parts_sum.append(f"Результаты инструментов:\n{tool_results}")
        user_prompt = "\n\n".join(user_parts_sum)

        if self.debug_prompt:
            print(f"\n{'='*80}\n[DEBUG PROMPT — summarizer]\n{'='*80}\n"
                  f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}\n{'='*80}\n")

        answer = self.llm.invoke_with_system(system_prompt, user_prompt, temperature=0.5)
        self.memory.add_message("assistant", answer)

        logger.info("Summarizer: ответ сформирован")
        return {
            "final_answer": answer,
            "messages": state["messages"] + [
                {"role": "assistant", "content": answer}
            ],
        }
