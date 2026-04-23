"""Тесты для plan_preview (задача 2.1)."""

import pytest

from graph.nodes.plan_preview import PlanPreviewNodes, _render_plan


# ─────────────────────────────────────────────────────────────────
# Вспомогательные утилиты
# ─────────────────────────────────────────────────────────────────

def _make_node(show_plan: bool = False) -> PlanPreviewNodes:
    """Создать экземпляр узла с заданным show_plan."""

    class _Node(PlanPreviewNodes):
        pass

    node = _Node()
    node.show_plan = show_plan
    return node


def _blueprint(**kwargs):
    return {
        "strategy": "fact_dim_join",
        "main_table": "dm.fact_churn",
        "where_conditions": ["reason_code = 'actual_churn'"],
        "aggregation": {"function": "COUNT", "column": "task_id", "alias": "cnt"},
        "group_by": ["DATE_TRUNC('month', report_dt)", "task_code"],
        "having": [],
        "order_by": "cnt DESC",
        "limit": None,
        **kwargs,
    }


def _state(
    *,
    show_plan: bool = False,
    explicit_mode: bool = False,
    plan_preview_approved: bool = False,
    blueprint=None,
    user_hints=None,
):
    return {
        "explicit_mode": explicit_mode,
        "plan_preview_approved": plan_preview_approved,
        "plan_preview_pending": False,
        "plan_preview_iteration": 0,
        "sql_blueprint": blueprint or _blueprint(),
        "selected_columns": {},
        "join_spec": [],
        "where_resolution": {},
        "user_hints": user_hints or {},
    }


# ─────────────────────────────────────────────────────────────────
# Тест 1: транзит (show_plan=False, explicit_mode=False)
# ─────────────────────────────────────────────────────────────────

def test_transit_when_inactive():
    """show_plan=False и explicit_mode=False → транзит, pending не выставляется."""
    node = _make_node(show_plan=False)
    result = node.plan_preview(_state(show_plan=False, explicit_mode=False))
    assert not result.get("plan_preview_pending")


def test_transit_when_approved():
    """plan_preview_approved=True → транзит, pending сбрасывается."""
    node = _make_node(show_plan=True)
    result = node.plan_preview(_state(show_plan=True, plan_preview_approved=True))
    assert result.get("plan_preview_pending") is False
    assert result.get("plan_preview_approved") is False


# ─────────────────────────────────────────────────────────────────
# Тест 2: confirmation (show_plan=True)
# ─────────────────────────────────────────────────────────────────

def test_confirmation_when_show_plan():
    """show_plan=True → plan_preview_pending=True и confirmation_message заполнен."""
    node = _make_node(show_plan=True)
    result = node.plan_preview(_state(show_plan=True, explicit_mode=False))
    assert result.get("plan_preview_pending") is True
    msg = result.get("confirmation_message", "")
    assert "dm.fact_churn" in msg
    assert "COUNT" in msg


def test_no_confirmation_when_explicit_mode_only():
    """explicit_mode=True без show_plan → транзит (plan_preview требует show_plan=True)."""
    node = _make_node(show_plan=False)
    result = node.plan_preview(_state(show_plan=False, explicit_mode=True))
    assert not result.get("plan_preview_pending")


def test_confirmation_when_show_plan_and_explicit_mode():
    """explicit_mode=True + show_plan=True → план показывается."""
    node = _make_node(show_plan=True)
    result = node.plan_preview(_state(show_plan=True, explicit_mode=True))
    assert result.get("plan_preview_pending") is True
    assert result.get("confirmation_message")


# ─────────────────────────────────────────────────────────────────
# Тест 3: правка плана — итерация счётчика
# ─────────────────────────────────────────────────────────────────

def test_iteration_counter_preserved():
    """plan_preview_iteration из state попадает в результат без изменений."""
    node = _make_node(show_plan=True)
    state = _state(show_plan=True)
    state["plan_preview_iteration"] = 2
    result = node.plan_preview(state)
    assert result.get("plan_preview_iteration") == 2


# ─────────────────────────────────────────────────────────────────
# Тест 4: пустой blueprint → транзит
# ─────────────────────────────────────────────────────────────────

def test_empty_blueprint_transits():
    """Если sql_blueprint пуст — транзит (нечего показывать)."""
    node = _make_node(show_plan=True)
    state = _state(show_plan=True)
    state["sql_blueprint"] = {}
    result = node.plan_preview(state)
    assert not result.get("plan_preview_pending")


# ─────────────────────────────────────────────────────────────────
# Тест 5: _render_plan рендерит ожидаемые секции
# ─────────────────────────────────────────────────────────────────

def test_render_plan_contains_main_table():
    md = _render_plan(
        sql_blueprint=_blueprint(),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={},
    )
    assert "dm.fact_churn" in md


def test_render_plan_contains_aggregation():
    md = _render_plan(
        sql_blueprint=_blueprint(),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={},
    )
    assert "COUNT" in md


def test_render_plan_contains_distinct_aggregation():
    md = _render_plan(
        sql_blueprint=_blueprint(
            aggregation={"function": "COUNT", "column": "task_code", "alias": "cnt", "distinct": True},
        ),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={},
    )
    assert "COUNT(DISTINCT task_code)" in md


def test_render_plan_contains_multiple_aggregations():
    md = _render_plan(
        sql_blueprint=_blueprint(
            aggregation={"function": "COUNT", "column": "tb_id", "alias": "count_tb_id", "distinct": True},
            aggregations=[
                {"function": "COUNT", "column": "tb_id", "alias": "count_tb_id", "distinct": True},
                {"function": "COUNT", "column": "gosb_id", "alias": "count_gosb_id", "distinct": True},
            ],
            order_by="count_tb_id DESC",
        ),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={},
    )
    assert "COUNT(DISTINCT tb_id)" in md
    assert "COUNT(DISTINCT gosb_id)" in md


def test_render_plan_contains_diff_summary():
    md = _render_plan(
        sql_blueprint=_blueprint(),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={},
        plan_diff_summary="- order_by: cnt DESC -> cnt ASC",
    )
    assert "Изменения после правки" in md
    assert "cnt ASC" in md


def test_render_plan_contains_group_by():
    md = _render_plan(
        sql_blueprint=_blueprint(),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={},
    )
    assert "task_code" in md or "Группировка" in md


def test_render_plan_contains_filters():
    md = _render_plan(
        sql_blueprint=_blueprint(),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={},
    )
    assert "actual_churn" in md or "Фильтры" in md


def test_render_plan_contains_join_spec():
    md = _render_plan(
        sql_blueprint=_blueprint(),
        selected_columns={},
        join_spec=[{"left": "dm.fact_churn.inn", "right": "dm.dim_reason.inn"}],
        where_resolution={},
        user_hints={},
    )
    assert "dm.fact_churn.inn" in md


def test_render_plan_negative_filters():
    md = _render_plan(
        sql_blueprint=_blueprint(),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={"negative_filters": ["канцелярия"]},
    )
    assert "канцелярия" in md


def test_render_plan_time_granularity():
    md = _render_plan(
        sql_blueprint=_blueprint(),
        selected_columns={},
        join_spec=[],
        where_resolution={},
        user_hints={"time_granularity": "week"},
    )
    assert "week" in md
