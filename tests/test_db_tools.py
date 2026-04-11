"""Regression tests for SQL tool payload semantics."""

import importlib
import json
import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock


@contextmanager
def _stub_langchain_tool_module():
    original_core = sys.modules.get("langchain_core")
    original_tools = sys.modules.get("langchain_core.tools")

    tools_module = ModuleType("langchain_core.tools")

    def tool(fn):
        fn.name = fn.__name__
        fn.invoke = lambda args: fn(**args)
        return fn

    tools_module.tool = tool
    core_module = ModuleType("langchain_core")
    core_module.tools = tools_module

    sys.modules["langchain_core"] = core_module
    sys.modules["langchain_core.tools"] = tools_module
    try:
        yield
    finally:
        if original_core is not None:
            sys.modules["langchain_core"] = original_core
        else:
            sys.modules.pop("langchain_core", None)

        if original_tools is not None:
            sys.modules["langchain_core.tools"] = original_tools
        else:
            sys.modules.pop("langchain_core.tools", None)


def _tool_map(db_manager):
    with _stub_langchain_tool_module():
        db_tools = importlib.import_module("tools.db_tools")
        db_tools = importlib.reload(db_tools)
        return {tool.name: tool for tool in db_tools.create_db_tools(db_manager)}


def test_execute_query_payload_is_honest_preview_metadata():
    db = MagicMock()
    preview_df = MagicMock()
    preview_df.empty = False
    preview_df.__len__.return_value = 1000
    preview_df.head.return_value.to_markdown.return_value = "| id |\n| --- |\n| 1 |"
    preview_df.to_markdown.return_value = "| id |\n| --- |\n| 1 |"
    db.preview_query.return_value = preview_df

    payload = json.loads(_tool_map(db)["execute_query"].invoke({"sql": "SELECT * FROM dm.orders"}))

    assert payload["mode"] == "preview"
    assert payload["rows_returned"] == 1000
    assert payload["rows_saved"] == 1000
    assert payload["is_truncated"] is True
    assert payload["saved_file"] == "last_query_result.csv"
    assert "только preview-результат" in payload["message"]
    assert "полный результат сохран" not in payload["message"].lower()


def test_export_query_payload_is_separate_full_export_contract():
    db = MagicMock()

    class _FakePath:
        def relative_to(self, _):
            return "exports/orders.csv"

    db.export_query_to_file.return_value = (_FakePath(), 321)

    payload = json.loads(
        _tool_map(db)["export_query"].invoke(
            {"sql": "SELECT * FROM dm.orders", "filename": "exports/orders.csv"}
        )
    )

    assert payload["mode"] == "export"
    assert payload["rows_returned"] == 321
    assert payload["rows_saved"] == 321
    assert payload["is_truncated"] is False
    assert payload["saved_file"] == "exports/orders.csv"

