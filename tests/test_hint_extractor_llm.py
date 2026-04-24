"""Тесты на hint_extractor_llm: нормализация LLM-ответа и merge с regex."""

import sys
from unittest.mock import MagicMock


def _ensure_mock_modules():
    for mod_name in ("langchain_gigachat", "langchain_gigachat.chat_models"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_ensure_mock_modules()

from graph.nodes.hint_extractor import merge_user_hints  # noqa: E402
from graph.nodes.hint_extractor_llm import _normalize_llm_hints  # noqa: E402


class TestNormalizeLlmHints:
    def test_happy_path_full(self):
        parsed = {
            "must_keep_tables": [["dm", "outflow"]],
            "join_fields": ["инн", "Customer_id"],
            "group_by_hints": ["регион"],
            "aggregate_hints": [
                {"function": "count", "column": "клиент", "distinct": True}
            ],
            "time_granularity": "Month",
            "negative_filters": ["отменённые"],
            "having_hints": [{"op": ">=", "value": 100, "unit_hint": "заказов"}],
            "dim_sources": {
                "segment": {"table": "dm.segments", "join_col": "INN"}
            },
        }
        result = _normalize_llm_hints(parsed)
        assert result["must_keep_tables"] == [("dm", "outflow")]
        assert result["join_fields"] == ["инн", "customer_id"]
        assert result["group_by_hints"] == ["регион"]
        assert result["aggregate_hints"] == [
            {"function": "count", "column": "клиент", "distinct": True}
        ]
        assert result["time_granularity"] == "month"
        assert result["negative_filters"] == ["отменённые"]
        assert result["having_hints"] == [
            {"op": ">=", "value": 100, "unit_hint": "заказов"}
        ]
        assert result["dim_sources"] == {
            "segment": {"table": "dm.segments", "join_col": "inn"}
        }

    def test_schema_table_as_string(self):
        parsed = {"must_keep_tables": ["dm.users"]}
        result = _normalize_llm_hints(parsed)
        assert result["must_keep_tables"] == [("dm", "users")]

    def test_invalid_function_dropped(self):
        parsed = {
            "aggregate_hints": [
                {"function": "sum", "column": "x"},
                {"function": "median", "column": "y"},
            ]
        }
        result = _normalize_llm_hints(parsed)
        assert result["aggregate_hints"] == [
            {"function": "sum", "column": "x", "distinct": False}
        ]

    def test_invalid_granularity_dropped(self):
        parsed = {"time_granularity": "hourly"}
        result = _normalize_llm_hints(parsed)
        assert result["time_granularity"] is None

    def test_empty_parsed_gives_empty_structure(self):
        result = _normalize_llm_hints({})
        assert result["must_keep_tables"] == []
        assert result["join_fields"] == []
        assert result["time_granularity"] is None

    def test_invalid_having_dropped(self):
        parsed = {
            "having_hints": [
                {"op": "<>", "value": 5},  # нет такого op
                {"op": ">=", "value": "ten"},  # не число
                {"op": ">", "value": 3, "unit_hint": "шт"},
            ]
        }
        result = _normalize_llm_hints(parsed)
        assert result["having_hints"] == [{"op": ">", "value": 3, "unit_hint": "шт"}]


class TestMergeUserHints:
    def _regex_empty(self):
        return {
            "must_keep_tables": [],
            "join_fields": [],
            "dim_sources": {},
            "having_hints": [],
            "group_by_hints": [],
            "aggregate_hints": [],
            "time_granularity": None,
            "negative_filters": [],
            "aggregation_preferences": {},
            "aggregation_preferences_list": [],
        }

    def test_both_empty(self):
        merged, source = merge_user_hints({}, self._regex_empty())
        assert source == "empty"
        assert merged["join_fields"] == []
        assert merged["time_granularity"] is None

    def test_llm_only(self):
        llm = {"join_fields": ["inn"], "time_granularity": "month"}
        merged, source = merge_user_hints(llm, self._regex_empty())
        assert source == "llm"
        assert merged["join_fields"] == ["inn"]
        assert merged["time_granularity"] == "month"

    def test_regex_only(self):
        regex = self._regex_empty()
        regex["join_fields"] = ["customer_id"]
        regex["time_granularity"] = "quarter"
        merged, source = merge_user_hints({}, regex)
        assert source == "regex"
        assert merged["join_fields"] == ["customer_id"]
        assert merged["time_granularity"] == "quarter"

    def test_llm_wins_for_granularity(self):
        regex = self._regex_empty()
        regex["time_granularity"] = "quarter"
        llm = {"time_granularity": "month"}
        merged, source = merge_user_hints(llm, regex)
        assert source == "merged"
        assert merged["time_granularity"] == "month"

    def test_lists_are_unioned(self):
        regex = self._regex_empty()
        regex["join_fields"] = ["kpp"]
        llm = {"join_fields": ["inn"]}
        merged, _ = merge_user_hints(llm, regex)
        assert "inn" in merged["join_fields"]
        assert "kpp" in merged["join_fields"]
        assert merged["join_fields"].index("inn") < merged["join_fields"].index("kpp")

    def test_aggregate_promotes_to_preferences_if_regex_empty(self):
        regex = self._regex_empty()
        llm = {
            "aggregate_hints": [
                {"function": "count", "column": "client", "distinct": True}
            ]
        }
        merged, _ = merge_user_hints(llm, regex)
        assert merged["aggregation_preferences"] == {
            "function": "count", "column": "client", "distinct": True,
        }
        assert merged["aggregation_preferences_list"] == [
            {"function": "count", "column": "client", "distinct": True}
        ]

    def test_regex_preferences_win_over_llm_list(self):
        regex = self._regex_empty()
        regex["aggregation_preferences"] = {
            "function": "sum", "column": "revenue", "distinct": False,
        }
        llm = {"aggregate_hints": [{"function": "count", "column": "x", "distinct": False}]}
        merged, _ = merge_user_hints(llm, regex)
        assert merged["aggregation_preferences"]["function"] == "sum"

    def test_dim_sources_merged_llm_wins(self):
        regex = self._regex_empty()
        regex["dim_sources"] = {"segment": {"table": "dm.s1", "join_col": "a"}}
        llm = {"dim_sources": {"segment": {"table": "dm.s2", "join_col": "b"}}}
        merged, _ = merge_user_hints(llm, regex)
        assert merged["dim_sources"]["segment"]["table"] == "dm.s2"
