from core.semantic_registry import find_matching_rules


def test_find_matching_rules_drops_generic_subset_when_specific_phrase_present():
    registry = {
        "rules": [
            {
                "rule_id": "text:dm.sale_funnel.task_subtype",
                "column_key": "dm.sale_funnel.task_subtype",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["подтип задачи"],
                "value_candidates": ["фактический отток"],
            },
            {
                "rule_id": "text:dm.sale_funnel.task_type",
                "column_key": "dm.sale_funnel.task_type",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["тип задачи"],
                "value_candidates": ["отток"],
            },
        ]
    }

    matched = find_matching_rules("Сколько задач по фактическому оттоку", registry)

    assert [item["rule_id"] for item in matched] == ["text:dm.sale_funnel.task_subtype"]


def test_find_matching_rules_keeps_generic_match_when_specific_phrase_absent():
    registry = {
        "rules": [
            {
                "rule_id": "text:dm.sale_funnel.task_subtype",
                "column_key": "dm.sale_funnel.task_subtype",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["подтип задачи"],
                "value_candidates": ["фактический отток"],
            },
            {
                "rule_id": "text:dm.sale_funnel.task_type",
                "column_key": "dm.sale_funnel.task_type",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["тип задачи"],
                "value_candidates": ["отток"],
            },
        ]
    }

    matched = find_matching_rules("Сколько задач по оттоку", registry)

    assert any(item["rule_id"] == "text:dm.sale_funnel.task_type" for item in matched)
