"""Few-shot retrieval успешных SQL-запросов из audit log.

Находит 2-3 похожих прошлых успешных запроса и возвращает их как
few-shot примеры для sql_writer. Работает по keyword overlap с user_input,
с бонусами за совпадение strategy / subject / metric_intent из semantic_frame.
"""

import logging
import re
from typing import TYPE_CHECKING, Any

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
    words = re.findall(r'[a-zA-Zа-яА-ЯёЁ_]{3,}', (text or "").lower())
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
        self._last_similarities: list[float] = []

    def _load_successful_queries(self, limit: int = 100) -> list[dict]:
        """Загрузить успешные запросы из audit log через публичный API memory."""
        if self._cache is not None:
            return self._cache

        try:
            rows = self._memory.iter_sql_audit(
                status="success",
                max_retry_count=0,
                min_sql_length=20,
                min_row_count=1,
                days=90,
                limit=limit,
            )
            self._cache = [
                {
                    "user_input": r.get("user_input") or "",
                    "sql": r.get("sql") or "",
                    "row_count": int(r.get("row_count", 0) or 0),
                    "timestamp": r.get("timestamp") or "",
                }
                for r in rows
                if r.get("user_input") and r.get("sql")
            ]
            logger.info("FewShotRetriever: загружено %d успешных запросов", len(self._cache))
        except Exception as e:
            logger.warning("FewShotRetriever: ошибка загрузки истории: %s", e)
            self._cache = []

        return self._cache

    def invalidate_cache(self) -> None:
        """Сбросить кэш (вызвать после новых успешных запросов)."""
        self._cache = None

    def _scoring_bonus(
        self,
        sql_lower: str,
        *,
        strategy: str,
        semantic_frame: dict[str, Any] | None,
        fact_dim_pair: tuple[str, str] | None,
    ) -> float:
        """Дополнительные бонусы поверх Jaccard."""
        bonus = 0.0
        # Структурный бонус по стратегии (CTE / DISTINCT ON / JOIN-паттерны)
        if strategy:
            if strategy in ("fact_dim_join", "dim_fact_join"):
                if "distinct on" in sql_lower or "with " in sql_lower or " join " in sql_lower:
                    bonus += 0.05
            elif strategy == "simple_select":
                if " join " not in sql_lower and "with " not in sql_lower:
                    bonus += 0.03
            elif strategy == "cte_preaggregation":
                if "with " in sql_lower and " join " in sql_lower:
                    bonus += 0.05

        # Бонус за совпадение пары (fact_table, dim_table)
        if fact_dim_pair:
            fact, dim = fact_dim_pair
            if fact and fact.lower() in sql_lower:
                bonus += 0.04
            if dim and dim.lower() in sql_lower:
                bonus += 0.03

        # Бонус за семантический фрейм
        if semantic_frame:
            subject = str(semantic_frame.get("subject") or "").lower()
            metric = str(semantic_frame.get("metric_intent") or "").lower()
            if subject and subject in sql_lower:
                bonus += 0.03
            if metric == "count" and re.search(r"\bcount\s*\(", sql_lower):
                bonus += 0.03
            elif metric == "sum" and re.search(r"\bsum\s*\(", sql_lower):
                bonus += 0.03
            elif metric == "avg" and re.search(r"\bavg\s*\(", sql_lower):
                bonus += 0.03
        return bonus

    def get_similar(
        self,
        user_input: str,
        strategy: str = "",
        n: int = 2,
        min_similarity: float = 0.15,
        *,
        semantic_frame: dict[str, Any] | None = None,
        fact_dim_pair: tuple[str, str] | None = None,
    ) -> list[dict]:
        """Найти n наиболее похожих успешных запросов.

        Args:
            user_input: Текущий запрос пользователя.
            strategy: SQL-стратегия (simple_select, fact_dim_join и т.д.) — для фильтрации.
            n: Количество примеров.
            min_similarity: Минимальный порог similarity (Jaccard).
            semantic_frame: Семантический фрейм запроса (subject/metric_intent) для бонусов.
            fact_dim_pair: Пара (fact_table, dim_table) для бонусов.

        Returns:
            Список {"user_input": str, "sql": str, "similarity": float} по релевантности.
        """
        self._last_similarities = []
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
            if score < min_similarity:
                continue
            sql_lower = entry["sql"].lower()
            score += self._scoring_bonus(
                sql_lower,
                strategy=strategy,
                semantic_frame=semantic_frame,
                fact_dim_pair=fact_dim_pair,
            )
            scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        top = scored[:n]
        self._last_similarities = [round(s, 3) for s, _ in top]
        return [
            {**entry, "similarity": round(s, 3)}
            for s, entry in top
        ]

    @property
    def last_similarities(self) -> list[float]:
        """Similarities последнего вызова get_similar — для evidence_trace."""
        return list(self._last_similarities)

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
