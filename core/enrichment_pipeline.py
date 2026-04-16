"""Единый enrichment pipeline для таблиц, колонок и value profiles."""

from __future__ import annotations

from typing import Any


class EnrichmentPipeline:
    """Оркестратор enrichment-слоя."""

    def __init__(self, schema_loader, llm: Any | None = None, db_manager: Any | None = None) -> None:
        self.schema = schema_loader
        self.llm = llm
        self.db = db_manager

    def run(self) -> None:
        """Запустить полный enrichment pipeline."""
        self.schema.ensure_table_grains(self.llm)
        self.schema.ensure_column_semantics()
        self.schema.ensure_table_semantics()
        self.schema.ensure_value_profiles(self.db)
        self.schema.ensure_semantic_registry()
