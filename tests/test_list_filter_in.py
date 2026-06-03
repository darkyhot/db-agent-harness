"""Регрессии: список-фильтр (=any → IN) и гашение спуриозных фильтр-интентов.

Контекст: запрос «сумма зачислений с типами 1,2,16,…» не строил
`enrollment_type IN (...)` (оператор =any дропался) и вставал на ложной
кларификации, порождённой фразой-метрикой «зарплатных зачислений».
"""

from core.filter_ranking import _build_condition
from core.query_ir import FilterSpec
from core.semantic_frame import _derive_filter_intents
from core.where_resolver import _condition_from_filter_spec


class TestBuildConditionListMembership:
    def test_any_operator_numeric_list_to_in(self):
        cond = _build_condition("enrollment_type", "=any", [1, 2, 16], {"dType": "int4"}, {})
        assert cond == "enrollment_type IN (1, 2, 16)"

    def test_any_operator_text_list_quoted(self):
        cond = _build_condition("status", "=any", ["a", "b"], {}, {})
        assert cond == "status IN ('a', 'b')"

    def test_string_list_value_to_in(self):
        cond = _build_condition("x", "IN", "1,2,3", {}, {})
        assert cond == "x IN (1, 2, 3)"

    def test_not_any_to_not_in(self):
        cond = _build_condition("x", "!=any", [1, 2], {}, {})
        assert cond == "x NOT IN (1, 2)"

    def test_scalar_equality_unaffected(self):
        cond = _build_condition("x", "=", "5", {"dType": "int4"}, {})
        assert cond == "x = '5'"


class TestConditionFromFilterSpecMembership:
    def test_any_with_numeric_list(self):
        spec = FilterSpec(target="enrollment_type", operator="=any", value=[1, 2, 16])
        assert _condition_from_filter_spec("enrollment_type", spec) == "enrollment_type IN (1, 2, 16)"

    def test_list_value_forces_in_even_with_eq_operator(self):
        spec = FilterSpec(target="t", operator="=", value=["x", "y"])
        assert _condition_from_filter_spec("t", spec) == "t IN ('x', 'y')"

    def test_scalar_unaffected(self):
        spec = FilterSpec(target="t", operator="=", value="5")
        assert _condition_from_filter_spec("t", spec) == "t = '5'"


class TestDeriveFilterIntentsSkipsMetricEntity:
    def test_phrase_matching_entity_is_not_a_filter(self):
        intent = {"entities": ["отток"], "filter_conditions": []}
        intents = _derive_filter_intents(
            user_input="покажи сумму по фактическому оттоку",
            intent=intent,
            schema_loader=None,
        )
        phrase_texts = [
            str(i.get("query_text") or "")
            for i in intents
            if i.get("kind") == "phrase_filter"
        ]
        assert all("отток" not in t.lower() for t in phrase_texts)

    def test_control_phrase_kept_without_matching_entity(self):
        intent = {"entities": [], "filter_conditions": []}
        intents = _derive_filter_intents(
            user_input="покажи сумму по фактическому оттоку",
            intent=intent,
            schema_loader=None,
        )
        phrase_texts = [
            str(i.get("query_text") or "")
            for i in intents
            if i.get("kind") == "phrase_filter"
        ]
        assert any("отток" in t.lower() for t in phrase_texts)
