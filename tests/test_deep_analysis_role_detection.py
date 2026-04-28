"""Role-detection regression tests against the corp metadata.

We cannot run the profiler on live data, so we synthesize minimal DataFrames
matching the dtype and expected cardinality for each column, then assert the
role the profiler picks. The expectation table is hand-curated from
data_for_agent/attr_list.csv — any future heuristic change must keep these
classifications intact, otherwise the full pipeline will mis-interpret key
columns like `gosb_id`, `segment_name`, `_amt`, `_qty`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.deep_analysis.loader import LoadPlan
from core.deep_analysis.profiler import profile_dataframe
from core.deep_analysis.types import ColumnRole

# (column_name, dtype, approximate_cardinality, expected_role)
# Cardinality chosen to reflect the column's business meaning, not the exact
# corp value — what matters is the profiler's decision boundary.
CASES: tuple[tuple[str, str, int, ColumnRole], ...] = (
    # --- entity IDs (high cardinality) ---
    ("epk_id", "int64", 5000, ColumnRole.ID),
    ("saphr_id", "int64", 3000, ColumnRole.ID),
    ("inn", "int64", 4000, ColumnRole.ID),
    ("kpp", "int64", 4000, ColumnRole.ID),
    ("ogrn", "int64", 4000, ColumnRole.ID),
    ("oktmo", "object", 10000, ColumnRole.ID),
    ("acc_num", "object", 5000, ColumnRole.ID),
    ("agrmnt_num", "object", 4000, ColumnRole.ID),
    ("document_info_sha1", "int64", 6000, ColumnRole.ID),
    ("author_login", "object", 500, ColumnRole.ID),
    ("deal_code", "object", 2000, ColumnRole.ID),
    ("actual_client_tid", "int64", 4000, ColumnRole.ID),

    # --- low-cardinality *_id are categories, not entities ---
    ("status_id", "int16", 5, ColumnRole.CATEGORY),
    ("segment_id", "int16", 8, ColumnRole.CATEGORY),
    ("priority_id", "int16", 4, ColumnRole.CATEGORY),
    ("client_type_id", "int16", 6, ColumnRole.CATEGORY),
    ("industry_id", "int16", 30, ColumnRole.CATEGORY),

    # --- medium-cardinality branch-like ids (tb ~20, gosb ~90) ---
    # With cardinality ≥50, gosb_id classifies as ID (useful as entity_col).
    ("tb_id", "int16", 15, ColumnRole.CATEGORY),
    ("gosb_id", "int32", 90, ColumnRole.ID),

    # --- category names ---
    ("segment_name", "object", 8, ColumnRole.CATEGORY),
    ("status_name", "object", 5, ColumnRole.CATEGORY),
    ("industry_name", "object", 30, ColumnRole.CATEGORY),
    ("task_category", "object", 7, ColumnRole.CATEGORY),
    ("task_type", "object", 10, ColumnRole.CATEGORY),
    ("card_type", "object", 6, ColumnRole.CATEGORY),
    ("vertical_name", "object", 6, ColumnRole.CATEGORY),

    # --- money and percent ---
    ("amt", "float64", 5000, ColumnRole.MONEY),
    ("m_avg_salary_amt", "float64", 5000, ColumnRole.MONEY),
    ("next_m_avg_salary_amt", "float64", 5000, ColumnRole.MONEY),
    ("outflow_perc", "float64", 500, ColumnRole.PERCENT),
    ("other_inn_emp_perc", "float64", 500, ColumnRole.PERCENT),
    ("post_cnt_fill", "float64", 500, ColumnRole.NUMERIC),  # no perc/amt suffix — generic numeric

    # --- counts/quantities are numeric, never category ---
    ("staff_qty", "int32", 500, ColumnRole.NUMERIC),
    ("transaction_qty", "int16", 200, ColumnRole.NUMERIC),
    ("outflow_qty", "int32", 400, ColumnRole.NUMERIC),
    ("np_qty", "int32", 300, ColumnRole.NUMERIC),
    ("fact_qty", "int16", 100, ColumnRole.NUMERIC),
    ("prev_m_fl_val", "int32", 500, ColumnRole.NUMERIC),

    # --- flags (boolean) ---
    ("is_force", "bool", 2, ColumnRole.FLAG),
    ("is_key_client", "bool", 2, ColumnRole.FLAG),
    ("is_parent", "bool", 2, ColumnRole.FLAG),
    # int4 flags encoded as 0/1
    ("is_closed", "int32", 2, ColumnRole.FLAG),
    ("is_task_leader", "int32", 2, ColumnRole.FLAG),
    ("is_worked", "int32", 2, ColumnRole.FLAG),

    # --- dates / timestamps ---
    ("report_dt", "date", 365, ColumnRole.DATE),
    ("acc_open_dt", "date", 1000, ColumnRole.DATE),
    ("modified_dttm", "timestamp", 10000, ColumnRole.DATETIME),
    ("start_dttm", "timestamp", 10000, ColumnRole.DATETIME),
    ("task_created_dt", "timestamp", 10000, ColumnRole.DATETIME),

    # --- narrative/PII text is TEXT, not CATEGORY ---
    ("fio", "object", 2000, ColumnRole.TEXT_LONG),
    ("department_manager_saphr_fio", "object", 500, ColumnRole.TEXT_LONG),
    ("reason_comment", "object", 2000, ColumnRole.TEXT_LONG),
    ("client_communication_infopovod", "object", 2000, ColumnRole.TEXT_LONG),
    ("company_name", "object", 5000, ColumnRole.TEXT_LONG),

    # --- narrow varchar codes treated as categories when low-cardinality ---
    ("tb_code", "object", 15, ColumnRole.CATEGORY),
    ("gosb_code", "object", 90, ColumnRole.ID),           # enough cardinality to serve as entity
    # oktmo subject codes: 85 federal subjects cross the ID cardinality
    # threshold — equally reasonable as ID (cohort) or CATEGORY. We accept ID.
    ("oktmo_subject_code", "object", 90, ColumnRole.ID),
)


def _synthesize(name: str, dtype: str, cardinality: int, n_rows: int = 2000) -> pd.Series:
    """Build a synthetic series matching the described dtype + cardinality.

    We care only about the profiler's role decision, so values are arbitrary
    but chosen to exercise the right branches (length, uniqueness, nulls).
    """
    rng = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    if dtype == "bool":
        return pd.Series(rng.integers(0, 2, size=n_rows).astype(bool), name=name)
    if dtype.startswith("int"):
        if cardinality <= 2:
            arr = rng.integers(0, 2, size=n_rows)
        else:
            arr = rng.integers(0, max(cardinality, 3), size=n_rows)
        # For narrative-PII numeric cases (none expected) this is unused.
        return pd.Series(arr.astype(dtype), name=name)
    if dtype.startswith("float"):
        if "amt" in name or "salary" in name:
            return pd.Series(rng.normal(100000, 20000, size=n_rows).round(2), name=name)
        if "perc" in name:
            return pd.Series(rng.uniform(0, 1, size=n_rows), name=name)
        return pd.Series(rng.normal(0.5, 0.2, size=n_rows), name=name)
    if dtype == "date":
        start = pd.Timestamp("2023-01-01")
        days = rng.integers(0, max(cardinality, 2), size=n_rows)
        return pd.Series([start + pd.Timedelta(days=int(d)) for d in days], name=name)
    if dtype == "timestamp":
        start = pd.Timestamp("2023-01-01")
        secs = rng.integers(0, max(cardinality * 3600, 2), size=n_rows)
        return pd.Series([start + pd.Timedelta(seconds=int(s)) for s in secs], name=name)
    if dtype == "object":
        # Tune string length so narrative/PII values are > 50 chars (matching
        # the TEXT_LONG threshold), while coded categories stay compact.
        is_text_like = any(t in name.lower() for t in ("fio", "comment", "infopovod")) or (
            "name" in name.lower() and cardinality > 500
        )
        if is_text_like:
            pool = [
                f"Длинный narrative-текст позиция {i:05d}, тут описание-контент и ФИО-подобные данные"
                for i in range(cardinality)
            ]
        else:
            pool = [f"val_{i}" for i in range(max(cardinality, 2))]
        idx = rng.integers(0, len(pool), size=n_rows)
        return pd.Series([pool[i] for i in idx], name=name)
    raise ValueError(f"unknown dtype {dtype}")


def test_role_detection_matches_corp_metadata_expectations():
    cols = [_synthesize(name, dtype, card) for name, dtype, card, _ in CASES]
    df = pd.concat(cols, axis=1)
    plan = LoadPlan(
        schema="schema", table="corp_mix", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=100, est_full_bytes=100 * len(df),
    )
    _, profile = profile_dataframe(df, plan)

    failures = []
    for name, _dtype, _card, expected in CASES:
        actual = profile.columns[name].role
        if actual != expected:
            failures.append(f"  {name}: expected={expected.value}, got={actual.value}")
    assert not failures, (
        "Role detection drifted for corp-schema columns:\n"
        + "\n".join(failures)
    )
