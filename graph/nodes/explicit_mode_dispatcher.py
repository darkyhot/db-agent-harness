"""Узел explicit_mode_dispatcher: определяет режим power-user по числу явных хинтов.

Запускается между hint_extractor и table_resolver. Без LLM — только подсчёт
непустых ключей в user_hints. Если пользователь задал ≥2 параметра явно —
выставляет explicit_mode=True.

В explicit_mode:
- plan_preview показывается принудительно (игнорируя config.show_plan)
- hint-boost в confidence применяется строже (max(score, 0.95))
- column_selector и sql_planner трактуют хинты как жёсткие ограничения
"""

import logging
from typing import Any

from graph.state import AgentState

logger = logging.getLogger(__name__)

# Ключи, наличие которых считается «явным параметром» от пользователя
_EXPLICIT_HINT_KEYS = (
    "must_keep_tables",
    "join_fields",
    "group_by_hints",
    "time_granularity",
)


class ExplicitModeDispatcherNodes:
    """Mixin с узлом explicit_mode_dispatcher (детерминированный, без LLM)."""

    def explicit_mode_dispatcher(self, state: AgentState) -> dict[str, Any]:
        """Определить, является ли запрос power-user-ским.

        Если ≥2 из ключей must_keep_tables, join_fields, group_by_hints,
        time_granularity непусты — устанавливает explicit_mode=True.
        """
        hints = state.get("user_hints", {}) or {}
        non_empty_count = sum(
            1 for key in _EXPLICIT_HINT_KEYS
            if hints.get(key)
        )
        explicit = non_empty_count >= 2

        if explicit:
            logger.info(
                "explicit_mode_dispatcher: explicit_mode=True "
                "(non_empty_hint_keys=%d, keys=%s)",
                non_empty_count,
                [k for k in _EXPLICIT_HINT_KEYS if hints.get(k)],
            )
        else:
            logger.debug(
                "explicit_mode_dispatcher: explicit_mode=False (non_empty=%d)",
                non_empty_count,
            )

        return {"explicit_mode": explicit}
