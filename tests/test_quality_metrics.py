"""Тесты per-node quality metrics (Direction 4.3)."""

from __future__ import annotations

import pytest

from core.memory import MemoryManager


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(memory_dir=tmp_path)


def _seed(memory: MemoryManager, entries: list[dict]) -> None:
    memory.start_session("user")
    for e in entries:
        memory.log_sql_execution(
            user_input=e["user_input"],
            sql=e.get("sql", "SELECT 1"),
            row_count=e.get("row_count", 3),
            status=e.get("status", "success"),
            duration_ms=e.get("duration_ms", 100),
            retry_count=e.get("retry_count", 0),
            error_type=e.get("error_type", ""),
        )


class TestTopProblemQueries:
    def test_filters_out_retry_zero(self, memory):
        _seed(memory, [
            {"user_input": "clean query", "retry_count": 0},
            {"user_input": "clean query", "retry_count": 0},
            {"user_input": "hard query", "retry_count": 2, "status": "error",
             "error_type": "column_not_found"},
        ])
        top = memory.get_top_problem_queries(days=30, limit=10)
        assert len(top) == 1
        assert top[0]["user_input"] == "hard query"
        assert top[0]["total_retries"] == 2
        assert top[0]["last_error_type"] == "column_not_found"

    def test_aggregates_multiple_attempts(self, memory):
        _seed(memory, [
            {"user_input": "Q1", "retry_count": 1},
            {"user_input": "Q1", "retry_count": 3},
            {"user_input": "Q2", "retry_count": 1},
        ])
        top = memory.get_top_problem_queries(days=30, limit=10)
        q1 = next(r for r in top if r["user_input"] == "Q1")
        assert q1["attempts"] == 2
        assert q1["total_retries"] == 4

    def test_sorted_by_total_retries_desc(self, memory):
        _seed(memory, [
            {"user_input": "small", "retry_count": 1},
            {"user_input": "big", "retry_count": 5},
            {"user_input": "mid", "retry_count": 3},
        ])
        top = memory.get_top_problem_queries(days=30, limit=10)
        assert [r["user_input"] for r in top] == ["big", "mid", "small"]

    def test_limit_respected(self, memory):
        _seed(memory, [
            {"user_input": f"Q{i}", "retry_count": i + 1} for i in range(5)
        ])
        top = memory.get_top_problem_queries(days=30, limit=2)
        assert len(top) == 2


class TestErrorTypeBreakdown:
    def test_buckets_and_median(self, memory):
        _seed(memory, [
            {"user_input": "q1", "retry_count": 1, "status": "error",
             "error_type": "column_not_found"},
            {"user_input": "q2", "retry_count": 3, "status": "error",
             "error_type": "column_not_found"},
            {"user_input": "q3", "retry_count": 1, "status": "error",
             "error_type": "type_mismatch"},
        ])
        br = memory.get_error_type_breakdown(days=30)
        assert "column_not_found" in br
        assert br["column_not_found"]["count"] == 2
        assert br["column_not_found"]["median_retries"] in (1, 3)
        assert br["type_mismatch"]["count"] == 1

    def test_empty_error_type_filtered(self, memory):
        _seed(memory, [
            {"user_input": "clean", "retry_count": 0},
        ])
        br = memory.get_error_type_breakdown(days=30)
        assert br == {}

    def test_sample_user_input_captured(self, memory):
        _seed(memory, [
            {"user_input": "длинный вопрос от пользователя", "retry_count": 2,
             "status": "error", "error_type": "join_explosion"},
        ])
        br = memory.get_error_type_breakdown(days=30)
        assert br["join_explosion"]["sample_user_input"].startswith("длинный")
