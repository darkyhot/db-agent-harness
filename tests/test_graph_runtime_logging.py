"""Проверки runtime-логирования при сборке графа."""

import logging
from unittest.mock import MagicMock


def test_build_graph_logs_runtime_module_paths(caplog):
    from graph.graph import build_graph

    caplog.set_level(logging.INFO)

    llm = MagicMock()
    db_manager = MagicMock()
    schema_loader = MagicMock()
    memory = MagicMock()
    sql_validator = MagicMock()

    build_graph(
        llm=llm,
        db_manager=db_manager,
        schema_loader=schema_loader,
        memory=memory,
        sql_validator=sql_validator,
        tools=[],
    )

    runtime_logs = [
        rec.getMessage() for rec in caplog.records
        if "Runtime modules:" in rec.getMessage()
    ]
    assert runtime_logs, "Ожидали startup-лог с путями модулей рантайма"
    msg = runtime_logs[0]
    assert "column_selector=" in msg
    assert "intent=" in msg
    assert "sql_planner=" in msg


def test_sql_planner_branch_allows_end():
    from langgraph.graph import END

    from graph.graph import build_graph

    graph = build_graph(
        llm=MagicMock(),
        db_manager=MagicMock(),
        schema_loader=MagicMock(),
        memory=MagicMock(),
        sql_validator=MagicMock(),
        tools=[],
    )

    branch = graph.builder.branches["sql_planner"]["_route_after_sql_planner"]
    assert END in branch.ends
