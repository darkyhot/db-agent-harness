"""Единый enrichment pipeline для таблиц, колонок и value profiles."""

from __future__ import annotations

from typing import Any


class EnrichmentPipeline:
    """Оркестратор enrichment-слоя."""

    def __init__(self, schema_loader, llm: Any | None = None, db_manager: Any | None = None) -> None:
        self.schema = schema_loader
        self.llm = llm
        self.db = db_manager

    def run(self, *, sample_cache: dict[tuple[str, str], Any] | None = None) -> None:
        """Запустить полный enrichment pipeline.

        Порядок: сначала детерминированное заполнение grain (без LLM) — это
        экономит rate-limit. Затем LLM-добор для оставшихся таблиц.
        """
        self.schema.fill_deterministic_grains()
        self.schema.ensure_table_grains(self.llm)
        self.schema.ensure_column_semantics()
        self.schema.ensure_table_semantics()
        self.schema.ensure_value_profiles(self.db, sample_cache=sample_cache)
        self.schema.ensure_semantic_registry()
