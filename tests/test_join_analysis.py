"""Тесты для core/join_analysis.py: классификация, scoring, noise filtering, composite keys."""

import pandas as pd
import pytest

from core.join_analysis import (
    classify_column,
    detect_table_type,
    rank_join_candidates,
    group_composite_keys,
    format_join_analysis,
    suggest_composite_joins,
    _normalize_key_name,
    JoinCandidate,
    CompositeJoinPair,
    MIN_CANDIDATE_SCORE,
)


# ---------------------------------------------------------------------------
# classify_column
# ---------------------------------------------------------------------------

class TestClassifyColumn:
    """Тесты классификации колонок."""

    def test_pk_is_key(self):
        assert classify_column("id", "int", 100.0, is_pk=True) == "key"

    def test_id_suffix_is_key(self):
        assert classify_column("client_id", "int8", 50.0, is_pk=False) == "key"

    def test_code_suffix_is_key(self):
        assert classify_column("region_code", "varchar(10)", 80.0, is_pk=False) == "key"

    def test_status_is_attribute(self):
        assert classify_column("status", "varchar(20)", 5.0, is_pk=False) == "attribute"

    def test_name_is_attribute(self):
        assert classify_column("client_name", "varchar(200)", 90.0, is_pk=False) == "attribute"

    def test_dttm_is_attribute(self):
        assert classify_column("inserted_dttm", "timestamp", 99.0, is_pk=False) == "attribute"

    def test_flag_is_attribute(self):
        assert classify_column("is_active", "boolean", 50.0, is_pk=False) == "attribute"

    def test_amount_is_attribute(self):
        assert classify_column("salary_amt", "numeric", 40.0, is_pk=False) == "attribute"

    def test_inn_is_business_key(self):
        assert classify_column("inn", "varchar(12)", 99.0, is_pk=False) == "business_key"

    def test_snils_is_business_key(self):
        assert classify_column("snils", "varchar(14)", 99.0, is_pk=False) == "business_key"

    def test_high_unique_unknown_is_business_key(self):
        """Колонка без паттерна, но с высоким unique_perc → business_key."""
        assert classify_column("region", "varchar(50)", 70.0, is_pk=False) == "business_key"

    def test_low_unique_unknown_is_attribute(self):
        """Колонка без паттерна, с низким unique_perc → attribute."""
        assert classify_column("region", "varchar(50)", 10.0, is_pk=False) == "attribute"

    def test_text_type_is_attribute(self):
        assert classify_column("bio", "text", 99.0, is_pk=False) == "attribute"

    def test_long_varchar_is_attribute(self):
        assert classify_column("description", "varchar(500)", 90.0, is_pk=False) == "attribute"

    def test_created_at_is_attribute(self):
        assert classify_column("created_at", "timestamp", 99.0, is_pk=False) == "attribute"


# ---------------------------------------------------------------------------
# detect_table_type
# ---------------------------------------------------------------------------

class TestDetectTableType:
    """Тесты определения типа таблицы."""

    def test_fact_table(self):
        df = pd.DataFrame({"column_name": ["id"], "is_primary_key": [True]})
        assert detect_table_type("uzp_dwh_fact_outflow", df) == "fact"

    def test_dim_table(self):
        df = pd.DataFrame({"column_name": ["id"], "is_primary_key": [True]})
        assert detect_table_type("uzp_dim_gosb", df) == "dim"

    def test_ref_table(self):
        df = pd.DataFrame({"column_name": ["id"], "is_primary_key": [True]})
        assert detect_table_type("ref_regions", df) == "ref"

    def test_unknown_table(self):
        df = pd.DataFrame({"column_name": ["a", "b", "c"], "is_primary_key": [False, False, False]})
        assert detect_table_type("some_table", df) == "unknown"

    def test_dict_table_is_ref(self):
        df = pd.DataFrame({"column_name": ["id"], "is_primary_key": [True]})
        assert detect_table_type("dict_currencies", df) == "ref"

    def test_high_pk_ratio_is_dim(self):
        """Таблица с >30% PK-колонок → dim (эвристика)."""
        df = pd.DataFrame({
            "column_name": ["a", "b", "c"],
            "is_primary_key": [True, True, False],
        })
        assert detect_table_type("some_link_table", df) == "dim"


# ---------------------------------------------------------------------------
# rank_join_candidates: scoring и noise filtering
# ---------------------------------------------------------------------------

def _make_cols_df(cols: list[dict]) -> pd.DataFrame:
    """Helper для создания DataFrame колонок."""
    return pd.DataFrame(cols)


class TestRankJoinCandidates:
    """Тесты scoring и ранжирования."""

    def test_exact_name_key_columns_both_unique(self):
        """client_id ↔ client_id (оба key, оба 100% unique) → высокий score."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 100.0, "description": ""},
        ])
        candidates = rank_join_candidates("s", "t1", df1, "s", "t2", df2, 1, 1)
        assert len(candidates) >= 1
        top = candidates[0]
        assert top.col1 == "client_id"
        assert top.col2 == "client_id"
        assert top.score >= 0.9

    def test_low_unique_penalizes_score(self):
        """client_id ↔ client_id, одна сторона 50% unique → score ниже из-за штрафа."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 50.0, "description": ""},
        ])
        candidates = rank_join_candidates("s", "t1", df1, "s", "t2", df2, 1, 1)
        assert len(candidates) >= 1
        top = candidates[0]
        assert top.score < 0.7, "50% unique should penalize score significantly"
        assert not top.safe, "50% unique should be unsafe"

    def test_attribute_columns_filtered_out(self):
        """status ↔ status (оба attribute) → score 0, отфильтровано."""
        df1 = _make_cols_df([
            {"column_name": "status", "dType": "varchar(20)", "is_primary_key": False,
             "unique_perc": 5.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "status", "dType": "varchar(20)", "is_primary_key": False,
             "unique_perc": 5.0, "description": ""},
        ])
        candidates = rank_join_candidates("s", "t1", df1, "s", "t2", df2, 1, 1)
        assert len(candidates) == 0, "attribute↔attribute should be filtered"

    def test_fk_pattern_detected(self):
        """PK 'id' в t1 → 't1_id' в t2 (FK-паттерн)."""
        df1 = _make_cols_df([
            {"column_name": "id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "t1_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 50.0, "description": ""},
        ])
        candidates = rank_join_candidates("s", "t1", df1, "s", "t2", df2, 1, 1)
        assert len(candidates) >= 1
        assert any(c.match_type == "fk_pattern" for c in candidates)

    def test_suffix_pattern_detected(self):
        """gosb_id ↔ new_gosb_id (suffix-паттерн)."""
        df1 = _make_cols_df([
            {"column_name": "gosb_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 80.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "new_gosb_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 80.0, "description": ""},
        ])
        candidates = rank_join_candidates("s", "t1", df1, "s", "t2", df2, 1, 1)
        assert len(candidates) >= 1
        assert any(c.match_type == "suffix" for c in candidates)

    def test_composite_pk_member_not_safe(self):
        """Колонка из составного PK с unique<90% → not safe."""
        df1 = _make_cols_df([
            {"column_name": "emp_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 40.0, "description": ""},
            {"column_name": "dept_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 20.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "emp_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        candidates = rank_join_candidates("s", "t1", df1, "s", "t2", df2, 2, 1)
        assert len(candidates) >= 1
        emp_cand = [c for c in candidates if c.col1 == "emp_id"][0]
        assert not emp_cand.safe, "composite PK member with low unique should be unsafe"

    def test_sorted_by_score_desc(self):
        """Результаты отсортированы по score desc."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
            {"column_name": "region_code", "dType": "varchar(10)", "is_primary_key": False,
             "unique_perc": 30.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 50.0, "description": ""},
            {"column_name": "region_code", "dType": "varchar(10)", "is_primary_key": False,
             "unique_perc": 40.0, "description": ""},
        ])
        candidates = rank_join_candidates("s", "t1", df1, "s", "t2", df2, 1, 1)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_empty_tables_return_empty(self):
        df_empty = pd.DataFrame()
        df_non_empty = _make_cols_df([
            {"column_name": "id", "dType": "int", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        assert rank_join_candidates("s", "t1", df_empty, "s", "t2", df_non_empty, 1, 1) == []


# ---------------------------------------------------------------------------
# group_composite_keys
# ---------------------------------------------------------------------------

class TestGroupCompositeKeys:
    """Тесты группировки составных ключей."""

    def test_single_pair(self):
        pairs = [
            {"left": {"schema": "s", "table": "t1", "column": "a"},
             "right": {"schema": "s", "table": "t2", "column": "a"}},
        ]
        groups = group_composite_keys(pairs)
        assert len(groups) == 1
        assert groups[0].columns == [("a", "a")]

    def test_composite_grouping(self):
        """Две пары между одними таблицами → один CompositeJoinPair."""
        pairs = [
            {"left": {"schema": "s", "table": "t1", "column": "a"},
             "right": {"schema": "s", "table": "t2", "column": "a"}},
            {"left": {"schema": "s", "table": "t1", "column": "b"},
             "right": {"schema": "s", "table": "t2", "column": "b"}},
        ]
        groups = group_composite_keys(pairs)
        assert len(groups) == 1
        assert len(groups[0].columns) == 2

    def test_different_table_pairs_separate(self):
        """Пары между разными таблицами → отдельные группы."""
        pairs = [
            {"left": {"schema": "s", "table": "t1", "column": "a"},
             "right": {"schema": "s", "table": "t2", "column": "a"}},
            {"left": {"schema": "s", "table": "t1", "column": "b"},
             "right": {"schema": "s", "table": "t3", "column": "b"}},
        ]
        groups = group_composite_keys(pairs)
        assert len(groups) == 2

    def test_cross_join_skipped(self):
        pairs = [{"type": "cross_join", "right": {"schema": "s", "table": "t"}}]
        groups = group_composite_keys(pairs)
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# format_join_analysis
# ---------------------------------------------------------------------------

class TestFormatJoinAnalysis:
    """Тесты форматирования для LLM."""

    def test_no_candidates_returns_empty(self):
        """Нет совпадающих колонок → пустая строка."""
        df1 = _make_cols_df([
            {"column_name": "aaa", "dType": "int", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "bbb", "dType": "int", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        assert format_join_analysis("s", "t1", df1, "s", "t2", df2, 1, 1) == ""

    def test_includes_table_type(self):
        """fact/dim типы показываются в заголовке."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 50.0, "description": ""},
        ])
        output = format_join_analysis("s", "fact_sales", df1, "s", "dim_clients", df2, 1, 1)
        assert "(fact)" in output
        assert "(dim)" in output

    def test_includes_score_and_safety(self):
        """Вывод содержит score и оценку безопасности."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 50.0, "description": ""},
        ])
        output = format_join_analysis("s", "t1", df1, "s", "t2", df2, 1, 1)
        assert "ОПАСНО" in output
        assert "[" in output  # score в квадратных скобках
        assert "Стратегия" in output  # конкретная стратегия для ОПАСНО

    def test_safe_join_no_pattern(self):
        """Безопасный join (100% unique обе стороны) — нет паттерна."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        output = format_join_analysis("s", "t1", df1, "s", "t2", df2, 1, 1)
        assert "безопасен" in output
        assert "Стратегия" not in output

    def test_fact_fact_strategy(self):
        """fact + fact → стратегия с агрегацией обеих сторон в CTE."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 60.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 60.0, "description": ""},
        ])
        output = format_join_analysis("s", "fact_sales", df1, "s", "fact_payments", df2, 0, 0)
        assert "ФАКТ + ФАКТ" in output
        assert "CTE" in output or "cte" in output.lower()
        assert "GROUP BY" in output

    def test_fact_dim_strategy(self):
        """fact + dim → стратегия с уникальной выборкой из справочника."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 60.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 60.0, "description": ""},
        ])
        output = format_join_analysis("s", "fact_sales", df1, "s", "dim_clients", df2, 0, 0)
        assert "ФАКТ + СПРАВОЧНИК" in output
        assert "DISTINCT ON" in output

    def test_dim_fact_strategy(self):
        """dim + fact → стратегия с агрегацией фактов."""
        df1 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "client_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 30.0, "description": ""},
        ])
        output = format_join_analysis("s", "dim_clients", df1, "s", "fact_sales", df2, 1, 0)
        assert "СПРАВОЧНИК + ФАКТ" in output
        assert "GROUP BY" in output

    def test_dim_dim_strategy(self):
        """dim + dim → стратегия с уникальными выборками из обеих сторон."""
        df1 = _make_cols_df([
            {"column_name": "org_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 50.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "org_id", "dType": "int8", "is_primary_key": False,
             "unique_perc": 60.0, "description": ""},
        ])
        output = format_join_analysis("s", "dim_orgs", df1, "s", "ref_regions", df2, 0, 0)
        assert "СПРАВОЧНИК + СПРАВОЧНИК" in output
        assert "DISTINCT ON" in output

    def test_composite_pk_high_unique_still_unsafe(self):
        """Composite PK member with unique_perc=95% should still be unsafe.

        Before the fix, threshold was < 90%, so 95% was treated as safe PK.
        Now ANY composite PK member is not treated as PK for join safety.
        """
        df1 = _make_cols_df([
            {"column_name": "emp_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 95.0, "description": ""},
            {"column_name": "dept_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 20.0, "description": ""},
        ])
        df2 = _make_cols_df([
            {"column_name": "emp_id", "dType": "int8", "is_primary_key": True,
             "unique_perc": 100.0, "description": ""},
        ])
        candidates = rank_join_candidates("s", "t1", df1, "s", "t2", df2, 2, 1)
        assert len(candidates) >= 1
        emp_cand = [c for c in candidates if c.col1 == "emp_id"][0]
        # emp_id in t1 is composite PK member (pk_count=2), so effective_pk=False.
        # unique_perc=95% < 100%, so it's NOT safe.
        assert not emp_cand.safe, (
            "composite PK member with 95% unique should be unsafe "
            "(only unique_perc=100% makes it safe)"
        )


# ---------------------------------------------------------------------------
# _normalize_key_name
# ---------------------------------------------------------------------------

class TestNormalizeKeyName:
    """Тесты нормализации имён PK-колонок."""

    def test_old_prefix_stripped(self):
        assert _normalize_key_name("old_gosb_id") == "gosb_id"

    def test_new_prefix_stripped(self):
        assert _normalize_key_name("new_gosb_id") == "gosb_id"

    def test_prev_prefix_stripped(self):
        assert _normalize_key_name("prev_client_id") == "client_id"

    def test_cur_prefix_stripped(self):
        assert _normalize_key_name("cur_tb_id") == "tb_id"

    def test_no_prefix_unchanged(self):
        assert _normalize_key_name("gosb_id") == "gosb_id"

    def test_no_prefix_unchanged_tb(self):
        assert _normalize_key_name("tb_id") == "tb_id"

    def test_case_insensitive(self):
        assert _normalize_key_name("OLD_GOSB_ID") == "gosb_id"

    def test_no_over_strip(self):
        """Не стрипаем 'old' если он не является префиксом (часть имени)."""
        # 'old_gosb' → 'gosb', но 'folder_id' → 'folder_id' (нет совпадения)
        assert _normalize_key_name("folder_id") == "folder_id"

    def test_current_prefix(self):
        assert _normalize_key_name("current_tb_id") == "tb_id"


# ---------------------------------------------------------------------------
# Умное определение составного JOIN-ключа: uzp_dim_gosb + uzp_dwh_fact_outflow
# ---------------------------------------------------------------------------

def _make_dim_gosb_df() -> pd.DataFrame:
    """Структура uzp_dim_gosb (составной PK: tb_id + old_gosb_id)."""
    return _make_cols_df([
        {"column_name": "tb_id",         "dType": "int4",  "is_primary_key": True,
         "unique_perc": 4.0,  "description": "Номер ТБ"},
        {"column_name": "old_gosb_id",   "dType": "int4",  "is_primary_key": True,
         "unique_perc": 72.0, "description": "Старый номер ГОСБ"},
        {"column_name": "old_gosb_name", "dType": "text",  "is_primary_key": False,
         "unique_perc": 72.0, "description": "Старое наименование ГОСБ"},
        {"column_name": "new_gosb_id",   "dType": "int4",  "is_primary_key": False,
         "unique_perc": 58.0, "description": "Новый номер ГОСБ"},
        {"column_name": "new_gosb_name", "dType": "text",  "is_primary_key": False,
         "unique_perc": 58.0, "description": "Новое наименование ГОСБ"},
        {"column_name": "tb_short_name", "dType": "text",  "is_primary_key": False,
         "unique_perc": 4.0,  "description": "Краткое наименование ТБ"},
    ])


def _make_fact_outflow_df() -> pd.DataFrame:
    """Структура uzp_dwh_fact_outflow (FK: tb_id + gosb_id)."""
    return _make_cols_df([
        {"column_name": "report_dt",   "dType": "date",    "is_primary_key": False,
         "unique_perc": 0.2,  "description": "Отчетная дата"},
        {"column_name": "tb_id",       "dType": "int2",    "is_primary_key": False,
         "unique_perc": 1.5,  "description": "Идентификатор ТБ"},
        {"column_name": "gosb_id",     "dType": "int4",    "is_primary_key": False,
         "unique_perc": 2.0,  "description": "Идентификатор ГОСБ"},
        {"column_name": "inn",         "dType": "int8",    "is_primary_key": True,
         "unique_perc": 100.0,"description": "ИНН"},
        {"column_name": "outflow_qty", "dType": "int4",    "is_primary_key": False,
         "unique_perc": 30.0, "description": "Кол-во ФЛ оттока"},
    ])


class TestSmartJoinKeyDetection:
    """Умное определение ключей JOIN между фактовой таблицей и dim с составным PK.

    Кейс: uzp_dwh_fact_outflow (fact) ↔ uzp_dim_gosb (dim, composite PK: tb_id + old_gosb_id).
    Правильный JOIN: f.tb_id = g.tb_id AND f.gosb_id = g.old_gosb_id.
    Неправильный JOIN: f.gosb_id = g.new_gosb_id (new_gosb_id — не PK, 42% дублей).
    """

    def test_normalized_pk_pair_found(self):
        """old_gosb_id (PK dim) ↔ gosb_id (fact) находится через нормализацию."""
        df_dim = _make_dim_gosb_df()
        df_fact = _make_fact_outflow_df()
        candidates = rank_join_candidates(
            "s", "uzp_dwh_fact_outflow", df_fact,
            "s", "uzp_dim_gosb", df_dim,
            pk_count1=1, pk_count2=2,
        )
        pairs = {(c.col1, c.col2) for c in candidates}
        assert ("gosb_id", "old_gosb_id") in pairs, (
            "gosb_id (fact) должен матчиться с old_gosb_id (dim PK) через normalized_pk"
        )

    def test_normalized_pk_match_type(self):
        """Пара gosb_id ↔ old_gosb_id должна иметь match_type=normalized_pk."""
        df_dim = _make_dim_gosb_df()
        df_fact = _make_fact_outflow_df()
        candidates = rank_join_candidates(
            "s", "uzp_dwh_fact_outflow", df_fact,
            "s", "uzp_dim_gosb", df_dim,
            pk_count1=1, pk_count2=2,
        )
        gosb_cand = next(
            (c for c in candidates if c.col1 == "gosb_id" and c.col2 == "old_gosb_id"), None
        )
        assert gosb_cand is not None
        assert gosb_cand.match_type == "normalized_pk"

    def test_tb_id_exact_match_found(self):
        """tb_id ↔ tb_id (точное совпадение, оба PK-члена составного ключа) присутствует."""
        df_dim = _make_dim_gosb_df()
        df_fact = _make_fact_outflow_df()
        candidates = rank_join_candidates(
            "s", "uzp_dwh_fact_outflow", df_fact,
            "s", "uzp_dim_gosb", df_dim,
            pk_count1=1, pk_count2=2,
        )
        pairs = {(c.col1, c.col2) for c in candidates}
        assert ("tb_id", "tb_id") in pairs, (
            "tb_id ↔ tb_id должен присутствовать как exact_name кандидат"
        )

    def test_composite_pk_candidates_above_threshold(self):
        """Оба члена составного PK (tb_id, old_gosb_id) имеют score выше MIN_CANDIDATE_SCORE."""
        df_dim = _make_dim_gosb_df()
        df_fact = _make_fact_outflow_df()
        candidates = rank_join_candidates(
            "s", "uzp_dwh_fact_outflow", df_fact,
            "s", "uzp_dim_gosb", df_dim,
            pk_count1=1, pk_count2=2,
        )
        tb_cand = next((c for c in candidates if c.col1 == "tb_id" and c.col2 == "tb_id"), None)
        gosb_cand = next(
            (c for c in candidates if c.col1 == "gosb_id" and c.col2 == "old_gosb_id"), None
        )
        assert tb_cand is not None and tb_cand.score >= MIN_CANDIDATE_SCORE
        assert gosb_cand is not None and gosb_cand.score >= MIN_CANDIDATE_SCORE

    def test_new_gosb_id_penalized(self):
        """new_gosb_id (не PK, при наличии составного PK) получает низкий score."""
        df_dim = _make_dim_gosb_df()
        df_fact = _make_fact_outflow_df()
        candidates = rank_join_candidates(
            "s", "uzp_dwh_fact_outflow", df_fact,
            "s", "uzp_dim_gosb", df_dim,
            pk_count1=1, pk_count2=2,
        )
        new_gosb_cand = next(
            (c for c in candidates if c.col2 == "new_gosb_id"), None
        )
        gosb_pk_cand = next(
            (c for c in candidates if c.col2 == "old_gosb_id"), None
        )
        # new_gosb_id либо отфильтрован, либо имеет score значительно ниже PK-кандидата
        if new_gosb_cand is not None and gosb_pk_cand is not None:
            assert gosb_pk_cand.score > new_gosb_cand.score, (
                "old_gosb_id (PK) должен иметь score выше new_gosb_id (не PK)"
            )

    def test_composite_join_suggested(self):
        """suggest_composite_joins предлагает составной ON по полному PK dim_gosb."""
        df_dim = _make_dim_gosb_df()
        df_fact = _make_fact_outflow_df()
        candidates = rank_join_candidates(
            "s", "uzp_dwh_fact_outflow", df_fact,
            "s", "uzp_dim_gosb", df_dim,
            pk_count1=1, pk_count2=2,
        )
        lines = suggest_composite_joins(
            candidates, df_fact, df_dim,
            pk_count1=1, pk_count2=2,
            tbl1="s.uzp_dwh_fact_outflow",
            tbl2="s.uzp_dim_gosb",
        )
        composite_text = "\n".join(lines)
        assert "СОСТАВНОЙ JOIN" in composite_text, (
            "Должна быть рекомендация составного JOIN"
        )
        assert "tb_id" in composite_text
        assert "old_gosb_id" in composite_text or "gosb_id" in composite_text

    def test_format_join_analysis_includes_composite(self):
        """format_join_analysis выводит составной ON и не предлагает new_gosb_id как top-1."""
        df_dim = _make_dim_gosb_df()
        df_fact = _make_fact_outflow_df()
        output = format_join_analysis(
            "s", "uzp_dwh_fact_outflow", df_fact,
            "s", "uzp_dim_gosb", df_dim,
            pk_count1=1, pk_count2=2,
        )
        assert "СОСТАВНОЙ JOIN" in output, "Вывод должен содержать рекомендацию составного JOIN"
        assert "old_gosb_id" in output or "gosb_id" in output
        assert "tb_id" in output
