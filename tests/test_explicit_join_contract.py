import pandas as pd

from graph.nodes.explorer import (
    _apply_explicit_join_override,
    _enforce_explicit_join_columns,
)


class _SchemaStub:
    def __init__(self, columns_by_table: dict[str, list[str]]):
        self.columns_by_table = columns_by_table

    def get_table_columns(self, schema: str, table: str) -> pd.DataFrame:
        cols = self.columns_by_table.get(f"{schema}.{table}", [])
        return pd.DataFrame({"column_name": cols})


def test_enforce_explicit_join_columns_adds_select_filter_for_both_tables():
    schema = _SchemaStub(
        {
            "dm.fact_sales": ["sale_id", "inn"],
            "dm.dim_customer": ["customer_id", "inn", "customer_name"],
        }
    )
    selected_columns = {
        "dm.fact_sales": {"aggregate": ["sale_id"]},
        "dm.dim_customer": {"select": ["customer_name"]},
    }
    intent = {"explicit_join": [{"table_hint": "dim_customer", "column_hint": "inn"}]}

    patched, diagnostics = _enforce_explicit_join_columns(
        selected_columns=selected_columns,
        intent=intent,
        schema_loader=schema,
        selected_tables=[("dm", "fact_sales"), ("dm", "dim_customer")],
    )

    assert diagnostics == []
    assert "inn" in patched["dm.fact_sales"]["select"]
    assert "inn" in patched["dm.fact_sales"]["filter"]
    assert "inn" in patched["dm.dim_customer"]["select"]
    assert "inn" in patched["dm.dim_customer"]["filter"]


def test_apply_override_uses_metadata_fallback_when_column_not_in_roles():
    schema = _SchemaStub(
        {
            "dm.fact_sales": ["sale_id", "inn"],
            "dm.dim_customer": ["customer_id", "inn"],
        }
    )
    selected_columns = {
        "dm.fact_sales": {"aggregate": ["sale_id"]},
        "dm.dim_customer": {"select": ["customer_id"]},
    }
    join_spec = []
    intent = {"explicit_join": [{"table_hint": "dim_customer", "column_hint": "inn"}]}

    overridden = _apply_explicit_join_override(
        join_spec=join_spec,
        selected_columns=selected_columns,
        intent=intent,
        schema_loader=schema,
        selected_tables=[("dm", "fact_sales"), ("dm", "dim_customer")],
    )

    assert len(overridden) == 1
    left = overridden[0]["left"]
    right = overridden[0]["right"]
    assert left.endswith(".inn")
    assert right.endswith(".inn")
    assert left.split(".")[:2] != right.split(".")[:2]


def test_enforce_returns_diagnostic_when_second_table_missing():
    schema = _SchemaStub({"dm.fact_sales": ["sale_id", "inn"]})
    selected_columns = {"dm.fact_sales": {"aggregate": ["sale_id"]}}
    intent = {"explicit_join": [{"table_hint": "dim_customer", "column_hint": "inn"}]}

    patched, diagnostics = _enforce_explicit_join_columns(
        selected_columns=selected_columns,
        intent=intent,
        schema_loader=schema,
        selected_tables=[("dm", "fact_sales")],
    )

    assert patched["dm.fact_sales"]["aggregate"] == ["sale_id"]
    assert diagnostics
    assert "нет второй таблицы" in diagnostics[0]
