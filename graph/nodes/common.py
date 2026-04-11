"""Общие константы, утилиты и базовый mixin для узлов графа.

Содержит SQL-промпты, правила, чеклисты, ToolResult и BaseNodeMixin
с разделяемыми методами (парсинг JSON, вызов инструментов, управление контекстом).
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from core.few_shot_retriever import FewShotRetriever
from core.llm import RateLimitedLLM
from core.memory import MemoryManager
from core.schema_loader import SchemaLoader
from core.sql_validator import SQLValidator
from core.database import DatabaseManager
from graph.state import AgentState

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

try:
    from core.synonym_map import expand_with_synonyms
except ImportError:
    expand_with_synonyms = None

logger = logging.getLogger(__name__)


# === Общие константы для промптов ===

SQL_FEW_SHOT_EXAMPLES = (
    "=== ПРИМЕРЫ SQL (используй как образец стиля) ===\n\n"
    "1. Агрегация + фильтр по дате:\n"
    "   SELECT customer_segment, COUNT(DISTINCT customer_id) AS customer_cnt\n"
    "   FROM dm.orders\n"
    "   WHERE order_date >= '2024-01-01'::date AND order_date < '2024-02-01'::date\n"
    "   GROUP BY customer_segment\n"
    "   ORDER BY customer_cnt DESC;\n\n"
    "2. СПРАВОЧНИК + ФАКТ (агрегация фактов в подзапросе):\n"
    "   SELECT c.customer_id, c.customer_name, agg.total_amount\n"
    "   FROM dm.customers c\n"
    "   JOIN (\n"
    "       SELECT customer_id, SUM(order_amount) AS total_amount\n"
    "       FROM dm.orders GROUP BY customer_id\n"
    "   ) agg ON agg.customer_id = c.customer_id;\n\n"
    "3. ФАКТ + СПРАВОЧНИК (уникальная выборка из справочника):\n"
    "   SELECT o.order_id, o.order_amount, p.product_name\n"
    "   FROM dm.orders o\n"
    "   JOIN (\n"
    "       SELECT DISTINCT ON (product_id) product_id, product_name\n"
    "       FROM dm.products ORDER BY product_id, updated_at DESC\n"
    "   ) p ON p.product_id = o.product_id;\n\n"
    "4. ФАКТ + ФАКТ (обе стороны агрегированы в CTE):\n"
    "   WITH orders_agg AS (\n"
    "       SELECT customer_id, SUM(order_amount) AS total_orders\n"
    "       FROM dm.orders GROUP BY customer_id\n"
    "   ), payments_agg AS (\n"
    "       SELECT customer_id, SUM(payment_amount) AS total_paid\n"
    "       FROM dm.payments GROUP BY customer_id\n"
    "   )\n"
    "   SELECT o.customer_id, o.total_orders, p.total_paid\n"
    "   FROM orders_agg o\n"
    "   JOIN payments_agg p ON p.customer_id = o.customer_id;\n\n"
    "5. СПРАВОЧНИК + СПРАВОЧНИК (уникальные выборки из обеих сторон):\n"
    "   WITH d1 AS (\n"
    "       SELECT DISTINCT ON (contract_id) contract_id, contract_type\n"
    "       FROM dm.contracts ORDER BY contract_id, effective_date DESC\n"
    "   ), d2 AS (\n"
    "       SELECT DISTINCT ON (contract_id) contract_id, risk_level\n"
    "       FROM dm.contract_risk ORDER BY contract_id, effective_date DESC\n"
    "   )\n"
    "   SELECT d1.contract_id, d1.contract_type, d2.risk_level\n"
    "   FROM d1 JOIN d2 ON d2.contract_id = d1.contract_id;\n\n"
    "6. NULL-обработка и COALESCE:\n"
    "   SELECT customer_id, COALESCE(phone, email, 'нет контакта') AS contact\n"
    "   FROM dm.customers\n"
    "   WHERE status IS NOT NULL AND region_name = 'North';\n\n"
    "7. Поиск через search_by_description (диагностика):\n"
    '   {"tool": "search_by_description", "args": {"query": "заказы клиентов"}}\n'
)

SQL_RULES = (
    "Правила SQL (Greenplum / PostgreSQL):\n"
    "- Имена таблиц ВСЕГДА в формате schema.table\n"
    "- Алиасы СТРОГО на английском: AS outflow, AS total_cnt\n"
    '- ЗАПРЕЩЕНО: AS "сумма", AS "выручка", кириллица в алиасах и именах колонок\n'
    "- Изучи РАЗВЕДКУ ТАБЛИЦ (если есть) перед написанием SQL\n"
    "- Пойми гранулярность (что = одна строка) перед COUNT/агрегатами\n"
    "- GROUP BY: перечисли ВСЕ не-агрегированные колонки из SELECT\n"
    "- Даты: используй приведение типов (::date, ::timestamp, TO_DATE()). "
    "Проверь формат дат в образце данных перед фильтрацией\n"
    "- NULL: используй COALESCE() или IS NOT NULL / IS NULL в WHERE. "
    "Не сравнивай с NULL через = или !=\n"
    "- LIMIT: добавляй LIMIT для exploration-запросов и больших таблиц\n"
    "- Используй стандартный PostgreSQL синтаксис (Greenplum совместим)\n\n"
    "КРИТИЧЕСКОЕ ПРАВИЛО: SQL запрос НИКОГДА не должен множить данные!\n"
    "ПЕРЕД написанием JOIN — изучи JOIN-АНАЛИЗ и определи типы таблиц.\n\n"
    "Стратегии безопасного JOIN по типам таблиц:\n\n"
    "1. ФАКТ + ФАКТ → предварительная агрегация ОБЕИХ сторон в CTE:\n"
    "   WITH cte1 AS (\n"
    "     SELECT join_key, SUM(metric1) AS val1 FROM schema.fact1 GROUP BY join_key\n"
    "   ), cte2 AS (\n"
    "     SELECT join_key, SUM(metric2) AS val2 FROM schema.fact2 GROUP BY join_key\n"
    "   )\n"
    "   SELECT * FROM cte1 JOIN cte2 ON cte1.join_key = cte2.join_key\n\n"
    "2. ФАКТ + СПРАВОЧНИК → уникальная выборка из справочника по ключу джойна:\n"
    "   SELECT f.*, d.name\n"
    "   FROM schema.fact f\n"
    "   JOIN (\n"
    "     SELECT DISTINCT ON (key) key, name FROM schema.dim ORDER BY key, date DESC\n"
    "   ) d ON d.key = f.key\n\n"
    "3. СПРАВОЧНИК + ФАКТ → агрегация фактов в CTE по ключу джойна:\n"
    "   SELECT d.*, agg.total\n"
    "   FROM schema.dim d\n"
    "   JOIN (\n"
    "     SELECT key, SUM(amount) AS total FROM schema.fact GROUP BY key\n"
    "   ) agg ON agg.key = d.key\n\n"
    "4. СПРАВОЧНИК + СПРАВОЧНИК → уникальные выборки из ОБЕИХ сторон:\n"
    "   WITH d1 AS (\n"
    "     SELECT DISTINCT ON (key) key, col FROM schema.dim1 ORDER BY key, date DESC\n"
    "   ), d2 AS (\n"
    "     SELECT DISTINCT ON (key) key, col FROM schema.dim2 ORDER BY key, date DESC\n"
    "   )\n"
    "   SELECT * FROM d1 JOIN d2 ON d1.key = d2.key\n\n"
    "- DISTINCT на внешнем SELECT — ЗАПРЕЩЁН: маскирует проблему, не решает её\n"
    "- DISTINCT как ПЕРВОЕ решение — ЗАПРЕЩЁН без понимания природы дублей\n"
    "- Составной PK (несколько колонок с is_primary_key=True) — каждая колонка "
    "НЕ уникальна сама по себе, уникальна только их комбинация"
)

SQL_CHECKLIST = (
    "Чеклист перед финализацией SQL:\n"
    "1. Формат дат соответствует данным из образца (YYYY-MM-DD, DD.MM.YYYY, и т.д.)?\n"
    "2. NULL обработан (COALESCE/IS NOT NULL) для колонок с высоким % NULL?\n"
    "3. JOIN НИКОГДА не множит данные? Проверь по типам таблиц:\n"
    "   - факт+факт → обе стороны агрегированы в CTE по ключу джойна?\n"
    "   - факт+справочник → уникальная выборка из справочника?\n"
    "   - справочник+факт → факты агрегированы в CTE/подзапросе?\n"
    "   - справочник+справочник → обе стороны уникальны по ключу джойна?\n"
    "   - Составной PK (>1 колонки с is_primary_key) — каждая колонка НЕ уникальна сама по себе!\n"
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
    "- Сравнение с NULL через = → используй IS NULL / IS NOT NULL\n"
    "- DISTINCT как первое решение для дублей → НЕВЕРНО: сначала изучи образец и пойми причину дублей;\n"
    "  справочник с двумя статусами — это WHERE, факты — это GROUP BY, только идентичные строки — DISTINCT"
)


@dataclass
class ToolResult:
    """Структурированный результат вызова инструмента."""
    success: bool
    data: str
    error: str | None = None

    def __str__(self) -> str:
        return self.data if self.success else (self.error or self.data)


class BaseNodeMixin:
    """Базовый mixin с общими методами для всех узлов графа.

    Содержит: инициализацию, парсинг JSON, вызов инструментов,
    управление контекстом и бюджетом промптов.
    """

    MAX_RETRIES = 4
    SQL_TOOL_NAMES = {"execute_query", "execute_write", "execute_ddl", "export_query"}
    MAX_PROMPT_CHARS = 100_000
    SAMPLE_CACHE_TTL = 600
    # Лимиты на размер списков в state — предотвращают раздувание промпта summarizer'а
    MAX_MESSAGES = 30
    MAX_TOOL_CALLS = 15

    # Бюджет символов для каждой ноды (переопределяет MAX_PROMPT_CHARS если указан)
    # Маленький бюджет для простых нод, большой — для тех, где нужен полный контекст таблиц
    NODE_BUDGET: dict[str, int] = {
        "intent_classifier": 20_000,
        "table_resolver": 40_000,
        "table_explorer": 60_000,
        "column_selector": 80_000,   # нужен полный контекст всех таблиц
        "sql_planner": 40_000,
        "sql_writer": 60_000,
        "sql_static_checker": 20_000,
        "error_diagnoser": 30_000,
        "sql_fixer": 50_000,
        "summarizer": 60_000,
    }

    # Стоп-слова для предфильтрации каталога
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

    _MAX_TABLES_FOR_FULL_CATALOG = 100

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
        self.llm = llm
        self.db = db_manager
        self.schema = schema_loader
        self.memory = memory
        self.validator = sql_validator
        self.tools = tools
        self.debug_prompt = debug_prompt
        self.tool_map: dict[str, Any] = {t.name: t for t in tools}
        self.few_shot = FewShotRetriever(memory)
        self._sample_cache: dict[tuple[str, str], tuple[float, str]] = {}
        self.tools_description = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )
        self.tools_brief = ", ".join(t.name for t in tools)
        compact_lines = []
        for t in tools:
            first_sentence = t.description.split("\n")[0].split(". ")[0]
            compact_lines.append(f"- {t.name}: {first_sentence}")
        self.tools_compact = "\n".join(compact_lines)

    # ------------------------------------------------------------------
    # Ограничение роста state-списков
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_messages(messages: list, max_size: int | None = None) -> list:
        """Обрезать список сообщений до последних max_size элементов.

        Предотвращает раздувание промпта summarizer'а при длинных сессиях с retry.
        """
        limit = max_size or BaseNodeMixin.MAX_MESSAGES
        if len(messages) > limit:
            return messages[-limit:]
        return messages

    @staticmethod
    def _cap_tool_calls(tool_calls: list, max_size: int | None = None) -> list:
        """Обрезать список tool_calls до последних max_size элементов."""
        limit = max_size or BaseNodeMixin.MAX_TOOL_CALLS
        if len(tool_calls) > limit:
            return tool_calls[-limit:]
        return tool_calls

    # ------------------------------------------------------------------
    # Бюджет промптов
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_to_budget(
        system_prompt: str,
        user_prompt: str,
        max_chars: int | None = None,
    ) -> tuple[str, str]:
        """Обрезать промпты при превышении бюджета символов."""
        if max_chars is None:
            max_chars = BaseNodeMixin.MAX_PROMPT_CHARS

        total = len(system_prompt) + len(user_prompt)
        if total <= max_chars:
            return system_prompt, user_prompt

        logger.warning(
            "Промпт превышает бюджет: %d символов (лимит %d), обрезаю",
            total, max_chars,
        )

        sys_budget = int(max_chars * 0.4)
        usr_budget = max_chars - sys_budget

        def _trim_md_tables(text: str, max_data_rows: int = 3) -> str:
            lines_out: list[str] = []
            in_table = False
            data_row_count = 0
            for line in text.split('\n'):
                if line.strip().startswith('|'):
                    if not in_table:
                        in_table = True
                        data_row_count = 0
                    data_row_count += 1
                    if data_row_count <= max_data_rows + 2:
                        lines_out.append(line)
                    elif data_row_count == max_data_rows + 3:
                        lines_out.append('| ... (rows trimmed) |')
                else:
                    in_table = False
                    data_row_count = 0
                    lines_out.append(line)
            return '\n'.join(lines_out)

        if len(system_prompt) + len(user_prompt) > max_chars:
            user_prompt = _trim_md_tables(user_prompt)

        if len(user_prompt) > usr_budget:
            tables_marker = "=== РАЗВЕДКА ТАБЛИЦ"
            if tables_marker in user_prompt:
                start_idx = user_prompt.index(tables_marker)
                end_markers = [
                    "Чеклист перед финализацией", "[ТЕКУЩИЙ ШАГ", "[ШАГ]",
                    "[ОШИБКА]", "[КОНТЕКСТ ТАБЛИЦ]", "[РЕАЛЬНЫЕ КОЛОНКИ",
                ]
                end_idx = len(user_prompt)
                for marker in end_markers:
                    pos = user_prompt.find(marker, start_idx + len(tables_marker))
                    if pos != -1:
                        end_idx = min(end_idx, pos)

                tables_section = user_prompt[start_idx:end_idx]
                other_len = len(user_prompt) - len(tables_section)
                available = usr_budget - other_len

                if available >= 500:
                    trimmed = BaseNodeMixin._smart_trim_recon(tables_section, available)
                else:
                    trimmed = (
                        "[РАЗВЕДКА ТАБЛИЦ ПРОПУЩЕНА из-за лимита токенов"
                        " — используй get_table_columns]\n\n"
                    )
                user_prompt = user_prompt[:start_idx] + trimmed + user_prompt[end_idx:]

        if len(user_prompt) > usr_budget:
            user_prompt = (
                user_prompt[:usr_budget]
                + "\n\n[...контекст обрезан из-за лимита токенов]"
            )

        if len(system_prompt) > sys_budget:
            system_prompt = (
                system_prompt[:sys_budget]
                + "\n\n[...системный промпт обрезан из-за лимита токенов]"
            )

        return system_prompt, user_prompt

    @staticmethod
    def _smart_trim_recon(section: str, max_chars: int) -> str:
        """Умная обрезка секции разведки таблиц."""
        trimmed = re.sub(
            r'(\*\*Образец данных[^*]*\*\*:?\s*\n)(\|.*\n)+',
            r'\1[...образцы обрезаны — используй get_sample]\n',
            section,
        )
        if len(trimmed) <= max_chars:
            return trimmed

        trimmed = re.sub(
            r'\n*=== JOIN-АНАЛИЗ.*',
            '\n[...JOIN-анализ обрезан]\n',
            trimmed,
            flags=re.DOTALL,
        )
        if len(trimmed) <= max_chars:
            return trimmed

        return (
            trimmed[:max_chars]
            + "\n\n[...структура таблиц обрезана — используй get_table_columns]\n\n"
        )

    # ------------------------------------------------------------------
    # Контекст таблиц
    # ------------------------------------------------------------------

    def _get_tables_detail_context(self, text: str) -> str:
        """Найти упоминания таблиц в тексте и вернуть описание их колонок."""
        df = self.schema.tables_df
        if df.empty:
            return ""

        pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b')
        found: set[tuple[str, str]] = set()
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

    def _get_schema_context(self, user_input: str = "") -> str:
        """Сформировать краткий каталог таблиц из SchemaLoader."""
        df = self.schema.tables_df
        if df.empty:
            return "Каталог таблиц пуст. Используй search_tables для поиска."

        if len(df) <= self._MAX_TABLES_FOR_FULL_CATALOG or not user_input:
            lines = ["Доступные таблицы (schema.table — описание):"]
            for _, row in df.iterrows():
                desc = row.get("description", "")
                lines.append(f"  {row['schema_name']}.{row['table_name']} — {desc}")
            return "\n".join(lines)

        # Используем TF-IDF поиск из SchemaLoader (включает synonym expansion + keyword match)
        filtered = self.schema.search_tables(user_input, top_n=30)

        if filtered.empty:
            return (
                f"В каталоге {len(df)} таблиц, но по запросу не найдено прямых совпадений.\n"
                "Используй search_tables или search_by_description для поиска нужных таблиц."
            )

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

    # ------------------------------------------------------------------
    # Контекст сессии и памяти
    # ------------------------------------------------------------------

    def _get_session_history_context(self, max_messages: int = 20) -> str:
        """Сформировать контекст истории текущей сессии."""
        messages = self.memory.get_session_messages()
        messages = messages[:-1] if messages else []
        messages = messages[-max_messages:]
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
        """Сформировать контекст долгосрочной памяти."""
        layer_keys = {"user_facts", "behavior_patterns", "user_instructions"}
        sections = []

        facts = self.memory.get_memory_list("user_facts")
        if facts:
            sections.append(
                "Факты о пользователе:\n" + "\n".join(f"  - {f}" for f in facts)
            )

        patterns = self.memory.get_memory_list("behavior_patterns")
        if patterns:
            sections.append(
                "Паттерны поведения пользователя (учитывай в стиле ответов):\n"
                + "\n".join(f"  - {p}" for p in patterns)
            )

        instructions = self.memory.get_memory_list("user_instructions")
        if instructions:
            sections.append(
                "Инструкции пользователя (ОБЯЗАТЕЛЬНО соблюдай):\n"
                + "\n".join(f"  - {i}" for i in instructions)
            )

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

    # ------------------------------------------------------------------
    # Парсинг JSON из ответов LLM
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_llm_json(text: str) -> str:
        """Очистить ответ LLM от markdown-обёрток и типичных ошибок GigaChat."""
        text = re.sub(r'```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text.strip()

    def _parse_tool_call(
        self, response: str, retry_on_fail: bool = True, _original: str | None = None,
    ) -> dict[str, Any]:
        """Извлечь вызов инструмента из ответа LLM."""
        original_response = _original or response
        cleaned = self._clean_llm_json(response)

        for candidate in self._extract_json_objects(cleaned):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "tool" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue

        if cleaned != response:
            for candidate in self._extract_json_objects(response):
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    continue

        if repair_json is not None:
            try:
                repaired = repair_json(cleaned)
                parsed = json.loads(repaired)
                if isinstance(parsed, dict) and "tool" in parsed:
                    logger.info("json_repair recovered valid tool call")
                    return parsed
            except (json.JSONDecodeError, ValueError, Exception):
                pass

        # Эвристика: попробовать извлечь SQL из свободного текста перед LLM retry
        # Экономит 5с задержки и один LLM-вызов когда GigaChat объясняет SQL вместо JSON
        sql_in_text = re.search(
            r'(?:^|\n)\s*(SELECT\s.{10,})',
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if sql_in_text:
            extracted = sql_in_text.group(1).strip().rstrip(';')
            # Обрезаем на первом явном закрытии запроса (пустая строка после SQL)
            first_break = re.search(r'\n\s*\n', extracted)
            if first_break:
                extracted = extracted[:first_break.start()].strip()
            if len(extracted) > 20:
                logger.info(
                    "_parse_tool_call: SQL извлечён через regex (без LLM retry), len=%d",
                    len(extracted),
                )
                return {"tool": "execute_query", "args": {"sql": extracted}}

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
        """Извлечь JSON-объекты из текста с учётом вложенных скобок."""
        candidates: list[str] = []
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

    # ------------------------------------------------------------------
    # Вызов инструментов
    # ------------------------------------------------------------------

    def _call_tool(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Вызвать инструмент по имени с валидацией аргументов."""
        if tool_name not in self.tool_map:
            return ToolResult(success=False, data="", error=f"Инструмент '{tool_name}' не найден.")

        tool_fn = self.tool_map[tool_name]

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
            if data.startswith("Ошибка"):
                return ToolResult(success=False, data=data, error=data)
            return ToolResult(success=True, data=data)
        except Exception as e:
            error_msg = f"Ошибка инструмента {tool_name}: {e}"
            logger.error("Tool %s ошибка: %s", tool_name, e)
            return ToolResult(success=False, data="", error=error_msg)

    @staticmethod
    def _parse_sql_tool_payload(raw: str) -> dict[str, Any] | None:
        """Parse structured SQL tool payload if present."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None

        required = {"message", "preview_markdown", "is_empty", "saved_file", "mode"}
        has_new_shape = {"rows_returned", "rows_saved", "is_truncated"}.issubset(set(data.keys()))
        has_legacy_shape = "total_rows" in data
        if not required.issubset(set(data.keys())) or not (has_new_shape or has_legacy_shape):
            return None

        try:
            rows_returned = int(data.get("rows_returned", data.get("total_rows", 0)))
            rows_saved = int(data.get("rows_saved", data.get("total_rows", 0)))
            is_truncated = bool(data.get("is_truncated", False))
            return {
                "message": str(data.get("message", "")),
                "preview_markdown": str(data.get("preview_markdown", "")),
                "rows_returned": rows_returned,
                "rows_saved": rows_saved,
                "is_empty": bool(data.get("is_empty", False)),
                "is_truncated": is_truncated,
                "saved_file": data.get("saved_file"),
                "mode": str(data.get("mode", "")),
            }
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _render_tool_result(raw: str, payload: dict[str, Any] | None = None) -> str:
        """Render tool output for logs/messages (human-readable)."""
        p = payload if payload is not None else BaseNodeMixin._parse_sql_tool_payload(raw)
        if not p:
            return raw

        parts = [p.get("message", "")]
        preview = p.get("preview_markdown", "")
        if preview:
            parts.append(preview)
        if p.get("mode") == "preview" and p.get("is_truncated"):
            parts.append("Результат усечён до preview-режима.")
        saved_file = p.get("saved_file")
        if saved_file:
            parts.append(f"Файл: {saved_file}")
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def _check_result_sanity(user_input: str, exec_result: str) -> list[str]:
        """Эвристические проверки осмысленности результата SQL-запроса."""
        warnings: list[str] = []
        user_lower = user_input.lower()

        result_lines = [
            line for line in exec_result.split("\n")
            if line.strip().startswith("|") and not line.strip().startswith("|---") and not line.strip().startswith("| ---")
        ]
        data_row_count = max(0, len(result_lines) - 1)

        count_patterns = re.compile(r'сколько|количество|общее\s+число|итого|всего', re.IGNORECASE)
        if count_patterns.search(user_lower) and data_row_count > 5:
            warnings.append(
                f"Вопрос содержит 'сколько/количество', но результат содержит {data_row_count} строк. "
                "Возможно, пропущена агрегация (COUNT, SUM) или GROUP BY."
            )

        if "показано" in exec_result and "из" in exec_result:
            limit_match = re.search(r'показано\s+\d+\s+из\s+(\d+)', exec_result)
            if limit_match and int(limit_match.group(1)) >= 1000:
                warnings.append(
                    f"Результат содержит {limit_match.group(1)}+ строк (упирается в LIMIT). "
                    "Если SQL содержит JOIN — возможен row explosion."
                )

        listing_patterns = re.compile(
            r'список|перечисли|все\s+(строки|записи|данные)|покажи\s+все', re.IGNORECASE,
        )
        if data_row_count > 100 and not listing_patterns.search(user_lower):
            warnings.append(
                f"Результат содержит {data_row_count} строк, хотя вопрос не предполагает список. "
                "Проверь, нет ли дублирования из-за JOIN."
            )

        return warnings

    def _check_disambiguation_needed(
        self, tool_name: str, result: str, user_input: str,
    ) -> list[dict[str, Any]] | None:
        """Проверить, вернул ли поиск несколько таблиц, требующих уточнения."""
        if tool_name not in ("search_tables", "search_by_description"):
            return None

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

        for opt in options:
            full_name = f"{opt['schema']}.{opt['table']}"
            if full_name.lower() in user_lower:
                return None

        return options
