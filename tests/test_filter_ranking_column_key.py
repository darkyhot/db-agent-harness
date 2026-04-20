import pandas as pd

from core.filter_ranking import rank_filter_candidates
from core.schema_loader import SchemaLoader


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 4,
        "table_name": ["sale_funnel"] * 4,
        "column_name": ["task_subtype", "task_category", "task_type", "task_code"],
        "dType": ["text", "text", "text", "text"],
        "description": [
            "Подтип задачи",
            "Категория задачи",
            "Тип задачи",
            "Код задачи",
        ],
        "is_primary_key": [False, False, False, False],
        "unique_perc": [10.0, 0.02, 0.11, 90.0],
        "not_null_perc": [100.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader._value_profiles = {
        "dm.sale_funnel.task_subtype": {
            "known_terms": ["фактический отток"],
            "top_values": [],
            "value_mode": "enum_like",
        },
        "dm.sale_funnel.task_category": {
            "known_terms": ["задача"],
            "top_values": [],
            "value_mode": "enum_like",
        },
        "dm.sale_funnel.task_type": {
            "known_terms": ["сервисная задача по юр-лицу"],
            "top_values": [],
            "value_mode": "enum_like",
        },
    }
    return loader


def test_filter_ranking_restricts_rule_to_declared_column_key(tmp_path):
    loader = _loader(tmp_path)
    semantic_frame = {
        "filter_intents": [
            {
                "request_id": "text:dm.sale_funnel.task_subtype",
                "kind": "text_search",
                "query_text": "фактический отток",
                "column_key": "dm.sale_funnel.task_subtype",
            },
            {
                "request_id": "text:dm.sale_funnel.task_category",
                "kind": "text_search",
                "query_text": "задача",
                "column_key": "dm.sale_funnel.task_category",
            },
        ]
    }

    ranked = rank_filter_candidates(
        user_input="Сколько задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_tables=["dm.sale_funnel"],
        schema_loader=loader,
        semantic_frame=semantic_frame,
    )

    subtype_candidates = ranked["text:dm.sale_funnel.task_subtype"]
    category_candidates = ranked["text:dm.sale_funnel.task_category"]

    assert [cand["column"] for cand in subtype_candidates] == ["task_subtype"]
    assert [cand["column"] for cand in category_candidates] == ["task_category"]
