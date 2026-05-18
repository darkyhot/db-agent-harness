import pandas as pd
import pytest

from core.exceptions import KerberosAuthError
from core.value_profiler import build_db_profile, fetch_table_profile_sample


class StubDB:
    def __init__(self, frame):
        self.frame = frame
        self.sql_calls = []

    def execute_query(self, sql: str, limit: int = 1000):
        self.sql_calls.append(sql)
        return self.frame


class KerberosStubDB:
    def execute_query(self, sql: str, limit: int = 1000):
        _ = (sql, limit)
        raise KerberosAuthError("kinit required")


def test_fetch_table_profile_sample_loads_table_once():
    db = StubDB(pd.DataFrame({"task_subtype": ["фактический отток", "отток"]}))

    sample = fetch_table_profile_sample(
        db,
        schema="dm",
        table="sale_funnel",
        columns=["task_subtype"],
    )

    assert list(sample["task_subtype"]) == ["фактический отток", "отток"]
    assert 'SELECT "task_subtype" FROM "dm"."sale_funnel" ORDER BY random() LIMIT 100000' == db.sql_calls[0]


def test_fetch_table_profile_sample_does_not_swallow_kerberos_error():
    with pytest.raises(KerberosAuthError):
        fetch_table_profile_sample(
            KerberosStubDB(),
            schema="dm",
            table="sale_funnel",
            columns=["task_subtype"],
        )


def test_build_db_profile_counts_values_from_sample_dataframe():
    sample = pd.DataFrame({"task_subtype": ["фактический отток", "отток", "фактический отток"]})

    profile = build_db_profile(
        sample,
        column="task_subtype",
        metadata_profile={"value_mode": "enum_like", "distinct_pct": 2.62},
    )

    assert profile["top_values"] == ["фактический отток", "отток"]
    assert profile["top_value_freq"] == [2, 1]
