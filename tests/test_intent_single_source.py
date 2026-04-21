import pandas as pd

from graph.nodes.intent import _extract_forced_single_source


def test_extract_forced_single_source_from_augmented_query():
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["sale_funnel", "outflow"],
    })

    forced = _extract_forced_single_source(
        "сколько задач (использовать таблицу dm.sale_funnel)",
        tables_df,
    )

    assert forced == ("dm", "sale_funnel")


def test_extract_forced_single_source_returns_none_without_marker():
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
    })

    forced = _extract_forced_single_source(
        "сколько задач из dm.sale_funnel",
        tables_df,
    )

    assert forced is None
