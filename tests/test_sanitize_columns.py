"""Тесты sanitize_selected_columns (Direction 5.2): fuzzy-match и отсев галлюцинаций."""

from __future__ import annotations

from core.column_selector_deterministic import (
    _levenshtein_le2,
    sanitize_selected_columns,
)


class TestLevenshtein:
    def test_equal_zero(self):
        assert _levenshtein_le2("abc", "abc") == 0

    def test_one_insert(self):
        assert _levenshtein_le2("abc", "abcd") == 1

    def test_one_delete(self):
        assert _levenshtein_le2("abcd", "abc") == 1

    def test_one_substitute(self):
        assert _levenshtein_le2("abcd", "abce") == 1

    def test_two_edits(self):
        assert _levenshtein_le2("abcd", "axcy") == 2

    def test_three_plus_returns_marker(self):
        # far apart — early-exit marker 3
        assert _levenshtein_le2("abcdef", "xyz1234") >= 3

    def test_length_gap_gt_2(self):
        assert _levenshtein_le2("ab", "abcdef") >= 3


class TestSanitizeSelectedColumns:
    def test_exact_match_preserved(self):
        res = sanitize_selected_columns(
            ["customer_id", "order_amount"],
            ["customer_id", "order_amount", "order_date"],
        )
        assert res["columns"] == ["customer_id", "order_amount"]
        assert res["coerced"] == []
        assert res["rejected"] == []

    def test_case_insensitive_exact(self):
        res = sanitize_selected_columns(
            ["CUSTOMER_ID"],
            ["customer_id"],
        )
        assert res["columns"] == ["customer_id"]
        assert res["coerced"] == []

    def test_fuzzy_coerces_typo(self):
        # typo on single char → dist=1 → fuzzy coercion
        res = sanitize_selected_columns(
            ["custmer_id"],
            ["customer_id", "order_amount"],
        )
        assert res["columns"] == ["customer_id"]
        assert res["coerced"] == [("custmer_id", "customer_id")]
        assert res["rejected"] == []
        assert res["warnings"], "ожидается warning о fuzzy-коррекции"

    def test_hallucinated_rejected(self):
        res = sanitize_selected_columns(
            ["totally_fake_column"],
            ["customer_id", "order_amount"],
        )
        assert res["columns"] == []
        assert res["rejected"] == ["totally_fake_column"]

    def test_duplicates_dropped(self):
        res = sanitize_selected_columns(
            ["customer_id", "customer_id", "CUSTOMER_ID"],
            ["customer_id"],
        )
        assert res["columns"] == ["customer_id"]

    def test_empty_and_whitespace_ignored(self):
        res = sanitize_selected_columns(
            ["", "  ", "customer_id"],
            ["customer_id"],
        )
        assert res["columns"] == ["customer_id"]
