"""Pydantic-модели контракта plan-level LLM-верификатора.

PlanVerdict — выход верификатора. Либо `approved` без правок, либо
`rejected` со списком структурированных правок (PlanEdit). Free-form rewrite
плана запрещён: единственный путь изменить SQL — это набор whitelisted
операций над selected_columns/blueprint.
"""

from __future__ import annotations

from typing import Any, Literal

from core.query_ir import StrictModel
from pydantic import Field


PlanEditOp = Literal[
    "replace_column",
    "add_filter",
    "drop_filter",
    "swap_aggregation",
    "add_table",
]

PlanEditRole = Literal["select", "group_by", "filter", "aggregate", "order_by"]


class PlanEdit(StrictModel):
    """Атомарная whitelisted-правка плана.

    Семантика по `op`:
    - replace_column: убрать `from_ref` из роли `target_role` и поставить `to_ref`.
        `from_ref` и `to_ref` обязательны, оба формата `schema.table.column`.
    - add_filter: добавить `to_ref` (полная WHERE-строка `col op value`)
        в conditions, если нет эквивалента.
    - drop_filter: удалить условие, текстово или семантически совпадающее
        с `from_ref` (фильтр-строка целиком или просто `schema.table.column`).
    - swap_aggregation: поменять SQL-функцию на колонке `from_ref`
        на функцию из `to_ref` (значение — имя функции SUM/COUNT/AVG/MIN/MAX).
    - add_table: гарантировать наличие `to_ref` (`schema.table`) в selected_columns.
    """

    op: PlanEditOp
    target_role: PlanEditRole | None = None
    from_ref: str | None = None
    to_ref: str | None = None
    reason: str = ""


class PlanVerdict(StrictModel):
    """Результат plan-level верификации."""

    verdict: Literal["approved", "rejected"]
    reasons: list[str] = Field(default_factory=list)
    edits: list[PlanEdit] = Field(default_factory=list)


def empty_verdict() -> dict[str, Any]:
    """Безопасный «всё ок» в формате dict — для no-op путей."""
    return {"verdict": "approved", "reasons": [], "edits": []}
