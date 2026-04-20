"""Тесты row-count sanity в MemoryManager (Direction 3.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.memory import MemoryManager


def _new_manager(tmp_path: Path) -> MemoryManager:
    return MemoryManager(memory_dir=tmp_path)


class TestPercentile:
    def test_single_value(self):
        assert MemoryManager._percentile([100], 0.95) == 100.0

    def test_empty(self):
        assert MemoryManager._percentile([], 0.95) == 0.0

    def test_monotone(self):
        vals = list(range(1, 101))  # 1..100
        # p95 должен быть около 95-96 (линейная интерполяция: (99)*0.95 = 94.05 → [94]=95, [95]=96)
        p95 = MemoryManager._percentile(vals, 0.95)
        assert 94.0 <= p95 <= 96.0


class TestRecordAndCheck:
    def test_not_suspect_below_threshold(self, tmp_path):
        mgr = _new_manager(tmp_path)
        for _ in range(10):
            mgr.record_row_count_sample("client", "count", 100)
        result = mgr.check_row_count_suspicion("client", "count", 200)
        assert not result["is_suspect"]
        assert result["p95"] == 100.0
        assert result["n"] == 10

    def test_suspect_above_threshold(self, tmp_path):
        mgr = _new_manager(tmp_path)
        for _ in range(10):
            mgr.record_row_count_sample("client", "count", 100)
        # 100 * 10 = 1000 — порог; 1001 превышает
        result = mgr.check_row_count_suspicion("client", "count", 1001)
        assert result["is_suspect"]
        assert result["ratio"] > 10.0

    def test_not_suspect_when_too_few_samples(self, tmp_path):
        mgr = _new_manager(tmp_path)
        for _ in range(4):  # порог min_samples=5
            mgr.record_row_count_sample("client", "count", 10)
        result = mgr.check_row_count_suspicion("client", "count", 1_000_000)
        assert not result["is_suspect"]
        assert result["n"] == 4

    def test_bucket_isolation(self, tmp_path):
        mgr = _new_manager(tmp_path)
        for _ in range(10):
            mgr.record_row_count_sample("client", "count", 100)
        # Другой subject — не должно использоваться распределение client/count
        result = mgr.check_row_count_suspicion("payment", "sum", 5000)
        assert result["n"] == 0
        assert not result["is_suspect"]

    def test_sliding_window_trim(self, tmp_path):
        mgr = _new_manager(tmp_path)
        for i in range(MemoryManager._ROW_COUNT_MAX_SAMPLES + 50):
            mgr.record_row_count_sample("s", "m", i)
        result = mgr.check_row_count_suspicion("s", "m", 0)
        assert result["n"] == MemoryManager._ROW_COUNT_MAX_SAMPLES

    def test_empty_frame_uses_underscore_bucket(self, tmp_path):
        mgr = _new_manager(tmp_path)
        for _ in range(10):
            mgr.record_row_count_sample(None, None, 5)
        result = mgr.check_row_count_suspicion("", "", 100)
        # Тот же bucket _|_
        assert result["n"] == 10

    def test_negative_row_count_ignored(self, tmp_path):
        mgr = _new_manager(tmp_path)
        mgr.record_row_count_sample("x", "y", -1)
        result = mgr.check_row_count_suspicion("x", "y", 1)
        assert result["n"] == 0
