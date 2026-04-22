"""Узел hint_extractor: детерминированное извлечение явных подсказок пользователя.

Запускается между intent_classifier и table_resolver. Без LLM — только regex
+ валидация по каталогу. Не нарушает 5-секундный лимит GigaChat и
правило компактных промптов.

Пишет в state поле user_hints со структурой:
    {
        "must_keep_tables": [(schema, table), ...],
        "join_fields": ["inn", ...],
        "dim_sources": {"segment": {"table": "schema.t", "join_col": "inn"}},
        "having_hints": [{"op": ">=", "value": 3, "unit_hint": "человек"}],
    }
"""

import logging
from typing import Any

from core.user_hint_extractor import extract_user_hints
from graph.state import AgentState

logger = logging.getLogger(__name__)


class HintExtractorNodes:
    """Mixin с узлом hint_extractor (детерминированный, без LLM)."""

    def hint_extractor(self, state: AgentState) -> dict[str, Any]:
        """Извлечь подсказки пользователя из user_input.

        Не блокирует пайплайн при отсутствии подсказок — возвращает пустую
        структуру. Все выходы валидируются через каталог schema_loader.
        """
        iterations = state.get("graph_iterations", 0) + 1
        user_input = state.get("user_input", "") or ""

        try:
            hints = extract_user_hints(user_input, self.schema)
        except Exception as exc:  # noqa: BLE001
            # Парсер не должен ронять граф — на ошибке возвращаем пустую структуру.
            logger.warning("hint_extractor: ошибка извлечения подсказок: %s", exc)
            hints = {
                "must_keep_tables": [],
                "join_fields": [],
                "dim_sources": {},
                "having_hints": [],
                "aggregation_preferences": {},
            }

        if (
            hints.get("must_keep_tables")
            or hints.get("join_fields")
            or hints.get("dim_sources")
            or hints.get("having_hints")
            or hints.get("aggregation_preferences")
        ):
            logger.info(
                "hint_extractor: must_keep=%s, join_fields=%s, "
                "dim_sources=%s, having_hints=%s, aggregation_preferences=%s",
                hints.get("must_keep_tables"),
                hints.get("join_fields"),
                list(hints.get("dim_sources", {}).keys()),
                hints.get("having_hints"),
                hints.get("aggregation_preferences"),
            )
        else:
            logger.debug("hint_extractor: подсказок не найдено")

        return {
            "user_hints": hints,
            "graph_iterations": iterations,
        }
