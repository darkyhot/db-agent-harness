"""Тесты для explicit_mode_dispatcher (задача 2.2)."""

from unittest.mock import MagicMock

import pytest

from graph.nodes.explicit_mode_dispatcher import ExplicitModeDispatcherNodes


def _make_node() -> ExplicitModeDispatcherNodes:
    """Создать экземпляр узла с заглушками."""

    class _Node(ExplicitModeDispatcherNodes):
        pass

    return _Node()


def _state(**user_hints_kwargs):
    """Собрать минимальный state с user_hints."""
    return {
        "user_hints": {
            "must_keep_tables": [],
            "join_fields": [],
            "group_by_hints": [],
            "time_granularity": None,
            **user_hints_kwargs,
        }
    }


# ─────────────────────────────────────────────────────────────────
# Тест: запрос без явных указаний → explicit_mode=False
# ─────────────────────────────────────────────────────────────────

def test_no_hints_gives_false():
    node = _make_node()
    result = node.explicit_mode_dispatcher(_state())
    assert result["explicit_mode"] is False


def test_single_hint_gives_false():
    node = _make_node()
    result = node.explicit_mode_dispatcher(
        _state(must_keep_tables=[("dm", "fact_churn")])
    )
    assert result["explicit_mode"] is False


# ─────────────────────────────────────────────────────────────────
# Тест: ≥2 явных хинтов → explicit_mode=True
# ─────────────────────────────────────────────────────────────────

def test_table_and_granularity_gives_true():
    node = _make_node()
    result = node.explicit_mode_dispatcher(
        _state(
            must_keep_tables=[("dm", "fact_churn")],
            time_granularity="month",
        )
    )
    assert result["explicit_mode"] is True


def test_table_and_join_fields_gives_true():
    node = _make_node()
    result = node.explicit_mode_dispatcher(
        _state(
            must_keep_tables=[("dm", "fact_churn")],
            join_fields=["reason_code"],
        )
    )
    assert result["explicit_mode"] is True


def test_group_by_and_granularity_gives_true():
    node = _make_node()
    result = node.explicit_mode_dispatcher(
        _state(
            group_by_hints=["task_code"],
            time_granularity="month",
        )
    )
    assert result["explicit_mode"] is True


def test_all_four_hints_gives_true():
    node = _make_node()
    result = node.explicit_mode_dispatcher(
        _state(
            must_keep_tables=[("dm", "fact_churn")],
            join_fields=["reason_code"],
            group_by_hints=["task_code"],
            time_granularity="month",
        )
    )
    assert result["explicit_mode"] is True


# ─────────────────────────────────────────────────────────────────
# Тест: power-user запрос из roadmap acceptance criteria
# «возьми dm.fact_churn, соедини с dm.dim_reason по reason_code,
#  посчитай помесячно» → explicit_mode=True
# ─────────────────────────────────────────────────────────────────

def test_roadmap_acceptance_example():
    """Acceptance: запрос с таблицей + JOIN + гранулярность → explicit_mode=True."""
    node = _make_node()
    result = node.explicit_mode_dispatcher(
        _state(
            must_keep_tables=[("dm", "fact_churn"), ("dm", "dim_reason")],
            join_fields=["reason_code"],
            time_granularity="month",
        )
    )
    assert result["explicit_mode"] is True


def test_roadmap_acceptance_no_explicit():
    """Acceptance: запрос без явных указаний → explicit_mode=False."""
    node = _make_node()
    result = node.explicit_mode_dispatcher(_state())
    assert result["explicit_mode"] is False


# ─────────────────────────────────────────────────────────────────
# Тест: user_hints отсутствует в state → graceful fallback
# ─────────────────────────────────────────────────────────────────

def test_missing_user_hints_gives_false():
    node = _make_node()
    result = node.explicit_mode_dispatcher({})
    assert result["explicit_mode"] is False


def test_none_user_hints_gives_false():
    node = _make_node()
    result = node.explicit_mode_dispatcher({"user_hints": None})
    assert result["explicit_mode"] is False
