"""Тесты для core/llm_column_verifier.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from core.llm_column_verifier import verify_column_selection


def _build_loader(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame(
        [{"schema_name": "dm", "table_name": "uzp_dim_gosb", "description": "Справочник ГОСБ"}]
    )
    attrs_df = pd.DataFrame(
        [
            {
                "schema_name": "dm", "table_name": "uzp_dim_gosb",
                "column_name": "gosb_id", "dType": "int4",
                "description": "Идентификатор ГОСБ. Наименование ГОСБ.",
                "is_primary_key": True, "unique_perc": 100.0, "not_null_perc": 100.0,
            },
            {
                "schema_name": "dm", "table_name": "uzp_dim_gosb",
                "column_name": "gosb_name", "dType": "varchar",
                "description": "Название ГОСБ",
                "is_primary_key": False, "unique_perc": 100.0, "not_null_perc": 99.0,
            },
        ]
    )
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def _make_invoker(verdict):
    """Builds a mock invoker exposing `_llm_json_with_retry`."""
    invoker = MagicMock()
    invoker._llm_json_with_retry = MagicMock(return_value=verdict)
    return invoker


def test_verifier_returns_empty_on_correct_selection(tmp_path):
    loader = _build_loader(tmp_path)
    invoker = _make_invoker({"issues": []})

    result = verify_column_selection(
        user_input="по названию госб",
        requested_slots={"metric": None, "dimensions": ["gosb_name"]},
        selected_columns={"dm.uzp_dim_gosb": {"select": ["gosb_name"], "group_by": ["gosb_name"]}},
        schema_loader=loader,
        llm_invoker=invoker,
    )
    assert result["issues"] == []
    assert result["should_force_fallback"] is False
    assert result["hint"] == ""


def test_verifier_forces_fallback_on_critical_issue(tmp_path):
    loader = _build_loader(tmp_path)
    verdict = {
        "issues": [
            {
                "slot": "gosb_name",
                "problem": "выбран идентификатор вместо названия",
                "severity": "critical",
                "suggested_column": "dm.uzp_dim_gosb.gosb_name",
            }
        ]
    }
    invoker = _make_invoker(verdict)

    result = verify_column_selection(
        user_input="по названию госб",
        requested_slots={"metric": None, "dimensions": ["gosb_name"]},
        selected_columns={"dm.uzp_dim_gosb": {"select": ["gosb_id"], "group_by": ["gosb_id"]}},
        schema_loader=loader,
        llm_invoker=invoker,
    )
    assert result["should_force_fallback"] is True
    assert "gosb_name" in result["hint"]
    assert any(issue["severity"] == "critical" for issue in result["issues"])


def test_verifier_returns_default_on_llm_error(tmp_path):
    loader = _build_loader(tmp_path)
    invoker = MagicMock()
    invoker._llm_json_with_retry = MagicMock(side_effect=RuntimeError("boom"))

    result = verify_column_selection(
        user_input="по названию госб",
        requested_slots={"metric": None, "dimensions": ["gosb_name"]},
        selected_columns={"dm.uzp_dim_gosb": {"select": ["gosb_id"]}},
        schema_loader=loader,
        llm_invoker=invoker,
    )
    assert result == {"issues": [], "should_force_fallback": False, "hint": ""}


def test_verifier_returns_default_when_llm_returns_none(tmp_path):
    loader = _build_loader(tmp_path)
    invoker = _make_invoker(None)

    result = verify_column_selection(
        user_input="по названию госб",
        requested_slots={"metric": None, "dimensions": ["gosb_name"]},
        selected_columns={"dm.uzp_dim_gosb": {"select": ["gosb_id"]}},
        schema_loader=loader,
        llm_invoker=invoker,
    )
    assert result["should_force_fallback"] is False
    assert result["issues"] == []


def test_verifier_returns_default_on_empty_selection(tmp_path):
    """Empty selected_columns → ничего не валидируем, LLM не зовётся."""
    loader = _build_loader(tmp_path)
    invoker = _make_invoker({"issues": [{"slot": "x", "severity": "critical"}]})

    result = verify_column_selection(
        user_input="…",
        requested_slots={"metric": None, "dimensions": ["gosb_name"]},
        selected_columns={},
        schema_loader=loader,
        llm_invoker=invoker,
    )
    assert result["issues"] == []
    invoker._llm_json_with_retry.assert_not_called()


def test_verifier_returns_default_when_no_inspectable_slots(tmp_path):
    """Если нет ни metric, ни dimensions — не зовём LLM."""
    loader = _build_loader(tmp_path)
    invoker = _make_invoker({"issues": [{"slot": "x", "severity": "critical"}]})

    result = verify_column_selection(
        user_input="…",
        requested_slots={"metric": None, "dimensions": []},
        selected_columns={"dm.uzp_dim_gosb": {"select": ["gosb_id"]}},
        schema_loader=loader,
        llm_invoker=invoker,
    )
    assert result["issues"] == []
    invoker._llm_json_with_retry.assert_not_called()
