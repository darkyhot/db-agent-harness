"""Фабрика agent-контекста: общая точка сборки для Jupyter и CLI.

Cобирает стандартный набор компонентов (db, llm, schema, memory, validator,
query_cache) — чтобы Jupyter (`agent.ipynb`) и `cli/interface.py` использовали
одну и ту же инициализацию без дрейфа версий.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Набор компонентов, необходимых для работы графа."""

    llm: Any
    db: Any
    schema: Any
    memory: Any
    validator: Any
    query_cache: Any
    tools: list


def build_agent_context(
    *,
    memory_dir: Path | None = None,
    debug: bool = False,
    run_enrichment: bool = True,
) -> AgentContext:
    """Собрать стандартный agent-контекст.

    Args:
        memory_dir: Путь к директории JSON-памяти. None — стандартная `memory/`.
        debug: Включить debug_prompt во всех нодах.
        run_enrichment: Запускать ли EnrichmentPipeline при инициализации.

    Returns:
        AgentContext с готовыми к использованию компонентами.
    """
    # Импорты выполняются внутри функции, чтобы фабрика могла быть импортирована
    # без тяжёлых зависимостей в тестах.
    from core.database import DatabaseManager
    from core.exceptions import KerberosAuthError
    from core.llm import RateLimitedLLM
    from core.memory import MemoryManager
    from core.query_cache import QueryCache
    from core.schema_loader import SchemaLoader
    from core.sql_validator import SQLValidator
    from core.enrichment_pipeline import EnrichmentPipeline
    from tools.db_tools import create_db_tools
    from tools.fs_tools import FS_TOOLS
    from tools.schema_tools import create_schema_tools

    db = DatabaseManager()
    llm = RateLimitedLLM()
    memory = MemoryManager(memory_dir=memory_dir)
    schema = SchemaLoader()
    if run_enrichment:
        try:
            EnrichmentPipeline(schema, llm=llm, db_manager=db).run()
        except KerberosAuthError:
            raise
        except Exception as e:
            logger.warning("EnrichmentPipeline: %s — продолжаем без enrichment", e)

    validator = SQLValidator(db, schema_loader=schema)
    db_tools = create_db_tools(db, validator, schema)
    schema_tools = create_schema_tools(schema)
    all_tools = FS_TOOLS + db_tools + schema_tools
    query_cache = QueryCache(memory)

    logger.info(
        "AgentContext собран: debug=%s, tools=%d, memory_dir=%s",
        debug, len(all_tools), memory._memory_dir,
    )
    return AgentContext(
        llm=llm,
        db=db,
        schema=schema,
        memory=memory,
        validator=validator,
        query_cache=query_cache,
        tools=all_tools,
    )
