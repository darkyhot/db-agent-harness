import pandas as pd

from core.catalog_grounding import ground_query_spec
from core.query_ir import FilterSpec, QuerySpec, query_spec_json_schema
from core.schema_loader import SchemaLoader
from core.where_resolver import resolve_where


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["orders"],
        "description": ["Фактовая таблица заказов"],
        "grain": ["transaction"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "dm", "dm"],
        "table_name": ["orders", "orders", "orders"],
        "column_name": ["order_id", "order_dt", "amount"],
        "dType": ["bigint", "date", "numeric"],
        "description": ["ID заказа", "Дата заказа", "Сумма заказа"],
        "is_primary_key": [True, False, False],
        "unique_perc": [100.0, 2.0, 50.0],
        "not_null_perc": [100.0, 99.0, 95.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_query_spec_accepts_valid_payload():
    payload = {
        "task": "answer_data",
        "metrics": [
            {
                "operation": "sum",
                "target": "amount",
                "distinct_policy": "auto",
                "confidence": 0.9,
                "evidence": [{"source": "user", "text": "сумма заказов", "confidence": 0.9}],
            }
        ],
        "dimensions": [{"target": "order_dt", "confidence": 0.8}],
        "filters": [],
        "time_range": {"start": "2024-01-01", "end": "2024-02-01", "confidence": 0.8},
        "source_constraints": [{"schema": "dm", "table": "orders", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.88,
    }

    spec, errors = QuerySpec.from_dict(payload)

    assert errors == []
    assert spec is not None
    assert spec.to_legacy_intent()["aggregation_hint"] == "sum"
    assert spec.to_legacy_user_hints()["group_by_hints"] == ["order_dt"]


def test_query_spec_rejects_invalid_metric_operation():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "median", "confidence": 0.5}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.5,
    })

    assert spec is None
    assert any("metrics[0].operation" in err for err in errors)


def test_query_spec_schema_exposes_required_contract():
    schema = query_spec_json_schema()

    assert schema["title"] == "QuerySpec"
    assert "metrics" in schema["required"]
    assert "source_constraints" in schema["properties"]
    assert schema["additionalProperties"] is False


def test_catalog_grounding_binds_explicit_source(tmp_path):
    loader = _loader(tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "target": "orders", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [{"schema": "dm", "table": "orders", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    result = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="сколько заказов")

    assert result.needs_clarification is False
    assert [s.full_name for s in result.sources] == ["dm.orders"]
    assert result.plan_ir is not None
    assert result.plan_ir.main_source.full_name == "dm.orders"


def test_catalog_grounding_clarifies_when_no_source(tmp_path):
    loader = _loader(tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "sum", "target": "unknown_metric", "distinct_policy": "auto", "confidence": 0.7}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.7,
    })
    assert spec is not None, errors

    result = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="")

    assert result.needs_clarification is True
    assert result.clarification is not None
    assert result.clarification.field == "source_constraints"


def test_query_spec_rejects_extra_fields():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.5,
        "unexpected": True,
    })

    assert spec is None
    assert any("unexpected" in err for err in errors)


def test_where_resolver_accepts_filter_specs_directly(tmp_path):
    loader = _loader(tmp_path)
    loader._rule_registry = {
        "rules": [
            {
                "rule_id": "text:dm.orders.amount",
                "column_key": "dm.orders.amount",
                "semantic_class": "metric",
                "match_kind": "numeric_compare",
                "match_phrases": ["amount"],
                "value_candidates": [],
            }
        ]
    }

    result = resolve_where(
        user_input="",
        intent={"filter_conditions": []},
        selected_columns={"dm.orders": {"select": ["amount"], "filter": ["amount"], "aggregate": []}},
        selected_tables=["dm.orders"],
        schema_loader=loader,
        semantic_frame={},
        base_conditions=[],
        filter_specs=[
            FilterSpec(target="amount", operator=">=", value=100, confidence=0.9),
        ],
    )

    assert result["needs_clarification"] is False
    assert any("amount" in cond and ">=" in cond for cond in result["conditions"])
