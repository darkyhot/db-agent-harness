"""Тесты на count_identifier_resolver callback в build_blueprint.

Проверяет:
- срабатывает только при составном PK (≥2 PK-кандидата);
- LLM-выбор подставляется в aggregate, если колонка валидна;
- невалидная колонка → fallback на детерминистику.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pandas as pd


def _ensure_mock_modules():
    for mod_name in ("langchain_gigachat", "langchain_gigachat.chat_models"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_ensure_mock_modules()

from core.sql_planner_deterministic import (  # noqa: E402
    _list_pk_candidates,
    build_blueprint,
)


def _build_composite_pk_loader(tmp_path):
    """Snapshot-таблица с составным PK (report_dt, gosb_id, inn) + dim."""
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["outflow_snapshot", "gosb_dim"],
        "description": ["Снимок фактического оттока", "Справочник ГОСБ"],
        "grain": ["snapshot", "organization"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": [
            "outflow_snapshot", "outflow_snapshot", "outflow_snapshot", "outflow_snapshot",
            "gosb_dim", "gosb_dim",
        ],
        "column_name": ["report_dt", "gosb_id", "inn", "task_code", "gosb_id", "gosb_name"],
        "dType": ["date", "text", "text", "text", "text", "text"],
        "description": [
            "Отчетная дата", "Код ГОСБ", "ИНН клиента", "Код задачи",
            "Код ГОСБ", "Название ГОСБ",
        ],
        "is_primary_key": [True, True, True, False, True, False],
        "unique_perc": [100.0, 100.0, 100.0, 80.0, 100.0, 20.0],
        "not_null_perc": [100.0] * 6,
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def _build_single_pk_loader(tmp_path):
    """Fact-таблица с одиночным PK."""
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["orders"],
        "description": ["Заказы"],
        "grain": ["transaction"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 2,
        "table_name": ["orders"] * 2,
        "column_name": ["order_id", "amount"],
        "dType": ["bigint", "numeric"],
        "description": ["ID заказа", "Сумма"],
        "is_primary_key": [True, False],
        "unique_perc": [100.0, 50.0],
        "not_null_perc": [100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


class TestListPkCandidates:
    def test_composite_pk(self, tmp_path):
        loader = _build_composite_pk_loader(tmp_path)
        result = _list_pk_candidates("dm.outflow_snapshot", loader)
        assert set(result) == {"report_dt", "gosb_id", "inn"}

    def test_single_pk(self, tmp_path):
        loader = _build_single_pk_loader(tmp_path)
        result = _list_pk_candidates("dm.orders", loader)
        assert result == ["order_id"]

    def test_invalid_table(self, tmp_path):
        loader = _build_composite_pk_loader(tmp_path)
        assert _list_pk_candidates("no_dot", loader) == []
        assert _list_pk_candidates("", loader) == []


def _simple_args(loader, user_input: str = "Сколько задач по оттоку по ГОСБ"):
    """Аргументы для fact_dim_join — safety-net не пропускается force_count_star'ом."""
    return {
        "intent": {"aggregation_hint": "count"},
        "selected_columns": {
            "dm.outflow_snapshot": {
                "select": ["report_dt"],
                "filter": ["report_dt"],
                "aggregate": ["*"],
                "group_by": ["report_dt"],
            },
            "dm.gosb_dim": {
                "select": ["gosb_name"],
                "filter": [],
                "aggregate": [],
                "group_by": ["gosb_name"],
            },
        },
        "join_spec": [
            {"left": "dm.outflow_snapshot.gosb_id", "right": "dm.gosb_dim.gosb_id"}
        ],
        "table_types": {"dm.outflow_snapshot": "fact", "dm.gosb_dim": "dim"},
        "join_analysis_data": {},
        "user_input": user_input,
        "user_hints": {},
        "schema_loader": loader,
        "semantic_frame": {"requires_single_entity_count": True, "subject": "task"},
    }


class TestCountIdentifierResolverCallback:
    def test_resolver_called_for_composite_pk(self, tmp_path):
        loader = _build_composite_pk_loader(tmp_path)
        calls = []

        def resolver(*, main_table, pk_candidates, user_input):
            calls.append((main_table, sorted(pk_candidates)))
            return "inn"

        bp = build_blueprint(
            **_simple_args(loader),
            count_identifier_resolver=resolver,
        )
        assert len(calls) == 1
        assert calls[0][0] == "dm.outflow_snapshot"
        assert "inn" in calls[0][1]
        assert bp["aggregation"]["column"] == "inn"

    def test_resolver_not_called_for_single_pk(self, tmp_path):
        loader = _build_single_pk_loader(tmp_path)
        args = _simple_args(loader, "сколько заказов")
        args["selected_columns"] = {
            "dm.orders": {
                "select": [],
                "aggregate": ["*"],
                "filter": [],
                "group_by": [],
            }
        }
        args["table_types"] = {"dm.orders": "fact"}

        calls = []

        def resolver(**kw):
            calls.append(kw)
            return "order_id"

        build_blueprint(**args, count_identifier_resolver=resolver)
        assert calls == []

    def test_resolver_returns_none_falls_back(self, tmp_path):
        loader = _build_composite_pk_loader(tmp_path)

        def resolver(**kw):
            return None

        bp = build_blueprint(
            **_simple_args(loader),
            count_identifier_resolver=resolver,
        )
        # Детерминистика должна выбрать task_code (subject=task) или не-date колонку
        assert bp["aggregation"]["column"] != "report_dt"

    def test_resolver_invalid_column_falls_back(self, tmp_path):
        loader = _build_composite_pk_loader(tmp_path)

        def resolver(**kw):
            return "definitely_not_a_real_column"

        bp = build_blueprint(
            **_simple_args(loader),
            count_identifier_resolver=resolver,
        )
        assert bp["aggregation"]["column"] != "report_dt"

    def test_resolver_exception_does_not_break(self, tmp_path):
        loader = _build_composite_pk_loader(tmp_path)

        def resolver(**kw):
            raise RuntimeError("LLM down")

        bp = build_blueprint(
            **_simple_args(loader),
            count_identifier_resolver=resolver,
        )
        assert bp["aggregation"] is not None

    def test_no_resolver_matches_old_behavior(self, tmp_path):
        loader = _build_composite_pk_loader(tmp_path)
        bp = build_blueprint(**_simple_args(loader))
        assert bp["aggregation"]["column"] != "report_dt"
