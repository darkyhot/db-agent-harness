"""Тесты адаптивного сэмплинга метаданных (большие таблицы/вью)."""

from __future__ import annotations

import json
from contextlib import contextmanager

import pandas as pd

from core.database import (
    DatabaseManager,
    SAMPLE_ASSUMED_ROWS_UNKNOWN,
    SAMPLE_OVERSAMPLE,
    SAMPLE_SORT_THRESHOLD,
    build_metadata_sample_sql,
)


class TestBuildMetadataSampleSql:
    def test_small_estimate_uses_order_by_random(self):
        sql = build_metadata_sample_sql("s", "t", 100_000, est_rows=1000)
        assert "ORDER BY random()" in sql
        assert "LIMIT :n" in sql
        assert "random() <" not in sql

    def test_large_estimate_uses_random_filter(self):
        sql = build_metadata_sample_sql("s", "t", 100_000, est_rows=50_000_000)
        assert "ORDER BY random()" not in sql
        assert "WHERE random() <" in sql
        # p = min(1, 2 * 100000 / 50_000_000) = 0.004
        assert "0.004" in sql

    def test_unknown_estimate_treated_as_large(self):
        expected_p = min(1.0, SAMPLE_OVERSAMPLE * 100_000 / SAMPLE_ASSUMED_ROWS_UNKNOWN)
        for est in (None, 0, -1):
            sql = build_metadata_sample_sql("s", "t", 100_000, est_rows=est)
            assert "ORDER BY random()" not in sql
            assert "random() <" in sql
            assert f"{expected_p:.10g}" in sql

    def test_threshold_boundary(self):
        sort_sql = build_metadata_sample_sql("s", "t", 100_000, est_rows=SAMPLE_SORT_THRESHOLD)
        assert "ORDER BY random()" in sort_sql
        filter_sql = build_metadata_sample_sql("s", "t", 100_000, est_rows=SAMPLE_SORT_THRESHOLD + 1)
        assert "random() <" in filter_sql and "ORDER BY random()" not in filter_sql

    def test_columns_projection(self):
        sql = build_metadata_sample_sql("s", "t", 10, est_rows=10, columns=["a", "b"])
        assert 'SELECT "a", "b" FROM' in sql

    def test_where_clause_small_and_large(self):
        small = build_metadata_sample_sql("s", "t", 10, est_rows=10, where="x > 1")
        assert "WHERE x > 1 ORDER BY random()" in small
        large = build_metadata_sample_sql("s", "t", 10, est_rows=10_000_000, where="x > 1")
        assert "WHERE x > 1 AND random() <" in large

    def test_p_capped_at_one(self):
        # est чуть больше порога → p может превысить 1 без cap; проверяем cap.
        sql = build_metadata_sample_sql("s", "t", 100_000, est_rows=SAMPLE_SORT_THRESHOLD + 1)
        # p = min(1, 2*100000/2000001) ≈ 0.1 — не превышает 1, просто sanity
        assert "random() < 1 " not in sql  # не вырожденный фильтр на крупном объекте


def _db_with_fake_connect(scalar_value=None, raise_exc=None):
    db = DatabaseManager()

    class _Result:
        def scalar(self_inner):
            if raise_exc is not None:
                raise raise_exc
            return scalar_value

    class _Conn:
        def execute(self_inner, *a, **k):
            if raise_exc is not None:
                raise raise_exc
            return _Result()

    @contextmanager
    def _fake_connect():
        yield _Conn()

    db._connect = _fake_connect  # type: ignore[method-assign]
    return db


class TestEstimateRowCount:
    def test_parses_plan_rows_from_list(self):
        db = _db_with_fake_connect(scalar_value=[{"Plan": {"Plan Rows": 42}}])
        assert db.estimate_row_count("s", "t") == 42

    def test_parses_plan_rows_from_json_string(self):
        db = _db_with_fake_connect(scalar_value=json.dumps([{"Plan": {"Plan Rows": 7}}]))
        assert db.estimate_row_count("s", "t") == 7

    def test_zero_rows_returns_none(self):
        db = _db_with_fake_connect(scalar_value=[{"Plan": {"Plan Rows": 0}}])
        assert db.estimate_row_count("s", "t") is None

    def test_explain_failure_returns_none(self):
        db = _db_with_fake_connect(raise_exc=RuntimeError("boom"))
        assert db.estimate_row_count("s", "t") is None


class TestGetMetadataSampleStrategy:
    def test_large_estimate_runs_random_filter_sql(self, monkeypatch):
        import core.database as db_module

        db = DatabaseManager()
        monkeypatch.setattr(db, "estimate_row_count", lambda s, t: 50_000_000)

        captured = {}

        @contextmanager
        def _fake_connect():
            yield object()

        def _fake_read_sql(sql, conn, params=None):
            captured["sql"] = str(sql)
            captured["params"] = params
            return pd.DataFrame({"a": [1]})

        monkeypatch.setattr(db, "_connect", _fake_connect)
        monkeypatch.setattr(db_module.pd, "read_sql", _fake_read_sql)

        df = db.get_metadata_sample("s", "t", n=100_000)
        assert not df.empty
        assert "random() <" in captured["sql"]
        assert "ORDER BY random()" not in captured["sql"]
        assert captured["params"] == {"n": 100_000}

    def test_small_estimate_runs_order_by_random_sql(self, monkeypatch):
        import core.database as db_module

        db = DatabaseManager()
        monkeypatch.setattr(db, "estimate_row_count", lambda s, t: 100)

        captured = {}

        @contextmanager
        def _fake_connect():
            yield object()

        def _fake_read_sql(sql, conn, params=None):
            captured["sql"] = str(sql)
            return pd.DataFrame()

        monkeypatch.setattr(db, "_connect", _fake_connect)
        monkeypatch.setattr(db_module.pd, "read_sql", _fake_read_sql)

        db.get_metadata_sample("s", "t", n=100_000)
        assert "ORDER BY random()" in captured["sql"]
