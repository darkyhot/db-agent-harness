"""Приоритеты выбора колонок для COUNT-метрик в _resolve_count_pref_column.

Регрессии:
- target='gosb_id' → должен возвращать gosb_id, не old_gosb_id (текущий PK).
- target='tb' → tb_id, не isu_branch_id.
"""

import pandas as pd

from core.column_selector_deterministic import _resolve_count_pref_column


def _df_with_columns(columns: list[tuple[str, str, bool]]) -> pd.DataFrame:
    """Helper: построить cols_df с (column_name, description, is_primary_key)."""
    return pd.DataFrame(
        [
            {"column_name": name, "description": desc, "is_primary_key": pk}
            for name, desc, pk in columns
        ]
    )


def test_gosb_target_prefers_canonical_id_over_legacy_old():
    cols = _df_with_columns([
        ("old_gosb_id", "Историческое значение ID ГОСБ", False),
        ("gosb_id", "Текущий ID ГОСБ", True),
        ("isu_branch_id", "ИСУ branch_id", False),
    ])
    assert _resolve_count_pref_column(cols, "gosb_id") == "gosb_id"
    assert _resolve_count_pref_column(cols, "gosb") == "gosb_id"
    assert _resolve_count_pref_column(cols, "госб") == "gosb_id"


def test_tb_target_prefers_tb_id_over_isu_branch():
    cols = _df_with_columns([
        ("isu_branch_id", "ИСУ-идентификатор территориального банка", False),
        ("tb_id", "Идентификатор территориального банка", True),
    ])
    assert _resolve_count_pref_column(cols, "tb") == "tb_id"
    assert _resolve_count_pref_column(cols, "тб") == "tb_id"
    assert _resolve_count_pref_column(cols, "tb_id") == "tb_id"


def test_falls_back_to_legacy_when_canonical_missing():
    cols = _df_with_columns([
        ("old_gosb_id", "Старый id ГОСБ", False),
    ])
    assert _resolve_count_pref_column(cols, "gosb") == "old_gosb_id"


def test_returns_none_when_no_match():
    cols = _df_with_columns([
        ("unrelated_col", "ничего", False),
    ])
    assert _resolve_count_pref_column(cols, "gosb_id") is None
