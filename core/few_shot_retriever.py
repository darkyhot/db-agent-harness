"""Few-shot retrieval успешных SQL-запросов из audit log.

Находит 2-3 похожих прошлых успешных запроса и возвращает их как
few-shot примеры для sql_writer. Работает по keyword overlap с user_input.
"""

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.memory import MemoryManager

logger = logging.getLogger(__name__)

# Стоп-слова для keyword overlap (те же что в BaseNodeMixin._STOP_WORDS)
_STOP_WORDS = frozenset({
    "в", "на", "по", "за", "из", "с", "к", "о", "у", "и", "а", "но", "что",
    "как", "все", "это", "так", "уже", "или", "не", "да", "нет", "мне", "мой",
    "ты", "он", "она", "мы", "они", "их", "его", "её", "для", "от", "до",
    "при", "без", "через", "между", "после", "перед", "под", "над", "про",
    "сколько", "какие", "какой", "какая", "какое", "где", "кто", "когда",
    "покажи", "найди", "выведи", "дай", "скажи", "сделай", "можешь",
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
    "is", "are", "was", "were", "be", "have", "has", "had", "show", "get",
})


def _tokenize(text: str) -> set[str]:
    """Разбить текст на значимые токены."""
    words = re.findall(r'[a-zA-Zа-яА-ЯёЁ_]{3,}', text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Коэффициент Жаккара между двумя множествами токенов."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


class FewShotRetriever:
    """Извлечение похожих успешных SQL-запросов из истории для few-shot примеров."""

    def __init__(self, memory: "MemoryManager") -> None:
        self._memory = memory
        self._cache: list[dict] | None = None  # Кэш успешных запросов

    def _load_successful_queries(self, limit: int = 100) -> list[dict]:
        """Загрузить успешные запросы из audit log."""
        if self._cache is not None:
            return self._cache

        try:
            from datetime import datetime, timezone, timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

            with self._memory._connect() as conn:
                cursor = conn.execute(
                    "SELECT user_input, sql FROM sql_audit "
                    "WHERE status = 'success' AND retry_count = 0 "
                    "AND timestamp >= ? "
                    "AND length(sql) > 20 "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (cutoff, limit),
                )
                rows = cursor.fetchall()

            self._cache = [
                {"user_input": r[0], "sql": r[1]}
                for r in rows
                if r[0] and r[1]
            ]
            logger.info("FewShotRetriever: загружено %d успешных запросов", len(self._cache))
        except Exception as e:
            logger.warning("FewShotRetriever: ошибка загрузки истории: %s", e)
            self._cache = []

        return self._cache

    def invalidate_cache(self) -> None:
        """Сбросить кэш (вызвать после новых успешных запросов)."""
        self._cache = None

    def get_similar(
        self,
        user_input: str,
        strategy: str = "",
        n: int = 2,
        min_similarity: float = 0.15,
    ) -> list[dict]:
        """Найти n наиболее похожих успешных запросов.

        Args:
            user_input: Текущий запрос пользователя.
            strategy: SQL-стратегия (simple_select, fact_dim_join и т.д.) — для фильтрации.
            n: Количество примеров.
            min_similarity: Минимальный порог similarity (Jaccard).

        Returns:
            Список {"user_input": str, "sql": str} отсортированный по релевантности.
        """
        candidates = self._load_successful_queries()
        if not candidates:
            return []

        query_tokens = _tokenize(user_input)
        if not query_tokens:
            return []

        scored: list[tuple[float, dict]] = []
        for entry in candidates:
            entry_tokens = _tokenize(entry["user_input"])
            score = _jaccard_similarity(query_tokens, entry_tokens)
            if score >= min_similarity:
                # Небольшой бонус если стратегия соответствует паттернам в SQL
                if strategy and strategy in ("fact_dim_join", "dim_fact_join"):
                    sql_lower = entry["sql"].lower()
                    if "distinct on" in sql_lower or "cte" in sql_lower or "with " in sql_lower:
                        score += 0.05
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [entry for _, entry in scored[:n]]

    def format_for_prompt(self, examples: list[dict]) -> str:
        """Форматировать примеры для вставки в промпт sql_writer."""
        if not examples:
            return ""

        lines = ["=== ПОХОЖИЕ ЗАПРОСЫ ИЗ ИСТОРИИ (используй как образец) ==="]
        for i, ex in enumerate(examples, 1):
            user_q = ex["user_input"][:150]
            sql = ex["sql"].strip()
            # Обрезаем очень длинные SQL
            if len(sql) > 800:
                sql = sql[:800] + "\n-- ... (обрезано)"
            lines.append(f"\nПример {i}:")
            lines.append(f'  Запрос: "{user_q}"')
            lines.append(f"  SQL:\n{sql}")

        return "\n".join(lines)
