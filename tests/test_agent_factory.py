"""Тесты core.agent_factory (Direction 6.2).

Тестируется только сборка AgentContext из mock-компонентов: реальные модули
core.database / core.llm / etc. требуют sqlalchemy / langchain — стабим их
через sys.modules до вызова build_agent_context.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _install_stub_modules(spec: dict):
    """Установить заглушки в sys.modules для тяжёлых импортов.

    spec: {"module.path": {"AttrName": value}, ...}
    Возвращает список сохранённых оригиналов, чтобы их можно было восстановить.
    """
    saved = []
    for mod_path, attrs in spec.items():
        original = sys.modules.get(mod_path)
        saved.append((mod_path, original))
        m = ModuleType(mod_path)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[mod_path] = m
    return saved


def _restore_stub_modules(saved):
    for mod_path, original in saved:
        if original is None:
            sys.modules.pop(mod_path, None)
        else:
            sys.modules[mod_path] = original


@pytest.fixture
def factory_stubs(tmp_path):
    """Подменить тяжёлые модули заглушками."""
    fake_db = MagicMock(name="DatabaseManager")
    fake_llm = MagicMock(name="RateLimitedLLM")
    fake_memory = MagicMock(name="MemoryManager")
    fake_memory._memory_dir = tmp_path
    fake_schema = MagicMock(name="SchemaLoader")
    fake_validator = MagicMock(name="SQLValidator")
    fake_query_cache = MagicMock(name="QueryCache")
    fake_db_tools = [MagicMock(name="db_tool_a"), MagicMock(name="db_tool_b")]
    fake_schema_tools = [MagicMock(name="schema_tool")]
    fake_fs_tools = [MagicMock(name="fs_tool")]
    fake_pipeline_run = MagicMock()

    spec = {
        "core.database": {"DatabaseManager": MagicMock(return_value=fake_db)},
        "core.llm": {"RateLimitedLLM": MagicMock(return_value=fake_llm)},
        "core.memory": {"MemoryManager": MagicMock(return_value=fake_memory)},
        "core.schema_loader": {"SchemaLoader": MagicMock(return_value=fake_schema)},
        "core.sql_validator": {"SQLValidator": MagicMock(return_value=fake_validator)},
        "core.query_cache": {"QueryCache": MagicMock(return_value=fake_query_cache)},
        "core.enrichment_pipeline": {
            "EnrichmentPipeline": MagicMock(return_value=MagicMock(run=fake_pipeline_run)),
        },
        "tools.db_tools": {"create_db_tools": MagicMock(return_value=fake_db_tools)},
        "tools.schema_tools": {"create_schema_tools": MagicMock(return_value=fake_schema_tools)},
        "tools.fs_tools": {"FS_TOOLS": fake_fs_tools},
    }

    # Сохраним agent_factory если уже импортирован (чтобы не отравить кэш)
    saved_af = sys.modules.pop("core.agent_factory", None)
    saved = _install_stub_modules(spec)

    yield {
        "db": fake_db, "llm": fake_llm, "memory": fake_memory,
        "schema": fake_schema, "validator": fake_validator,
        "query_cache": fake_query_cache,
        "db_tools": fake_db_tools, "schema_tools": fake_schema_tools,
        "fs_tools": fake_fs_tools,
        "pipeline_run": fake_pipeline_run,
        "pipeline_cls": spec["core.enrichment_pipeline"]["EnrichmentPipeline"],
    }

    _restore_stub_modules(saved)
    sys.modules.pop("core.agent_factory", None)
    if saved_af is not None:
        sys.modules["core.agent_factory"] = saved_af


def test_build_agent_context_wires_components(tmp_path, factory_stubs):
    from core import agent_factory as af

    ctx = af.build_agent_context(memory_dir=tmp_path, run_enrichment=True)

    assert ctx.db is factory_stubs["db"]
    assert ctx.llm is factory_stubs["llm"]
    assert ctx.memory is factory_stubs["memory"]
    assert ctx.schema is factory_stubs["schema"]
    assert ctx.validator is factory_stubs["validator"]
    assert ctx.query_cache is factory_stubs["query_cache"]
    assert ctx.tools == (
        factory_stubs["fs_tools"]
        + factory_stubs["db_tools"]
        + factory_stubs["schema_tools"]
    )
    factory_stubs["pipeline_run"].assert_called_once()


def test_build_agent_context_skip_enrichment(tmp_path, factory_stubs):
    from core import agent_factory as af

    af.build_agent_context(memory_dir=tmp_path, run_enrichment=False)

    factory_stubs["pipeline_cls"].assert_not_called()


def test_build_agent_context_enrichment_errors_swallowed(tmp_path, factory_stubs):
    from core import agent_factory as af

    factory_stubs["pipeline_run"].side_effect = RuntimeError("boom")

    ctx = af.build_agent_context(memory_dir=tmp_path, run_enrichment=True)
    assert ctx is not None
    assert ctx.db is factory_stubs["db"]
