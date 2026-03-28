"""Тесты SchemaLoader: проверка composite key логики."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from core.schema_loader import SchemaLoader


@pytest.fixture
def loader_with_data(tmp_path):
    """Create a SchemaLoader with test CSV data."""
    # tables_list.csv
    tables_df = pd.DataFrame({
        "schema_name": ["hr", "hr", "hr"],
        "table_name": ["emp", "dept", "emp_dept"],
        "table_description": ["Employees", "Departments", "Employee-Department link"],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)

    # attr_list.csv
    attrs_df = pd.DataFrame({
        "schema_name": ["hr", "hr", "hr", "hr", "hr", "hr", "hr"],
        "table_name": ["emp", "emp", "dept", "dept", "emp_dept", "emp_dept", "emp_dept"],
        "column_name": ["id", "dept_id", "id", "name", "emp_id", "dept_id", "role"],
        "data_type": ["int", "int", "int", "varchar", "int", "int", "varchar"],
        "is_primary_key": [True, False, True, False, True, True, False],
        "unique_perc": [100.0, 20.0, 100.0, 95.0, 80.0, 60.0, 30.0],
        "column_description": ["PK", "FK to dept", "PK", "Name", "FK", "FK", "Role"],
    })
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    return SchemaLoader(data_dir=tmp_path)


class TestCheckKeyUniqueness:
    """Тесты check_key_uniqueness с акцентом на composite key логику."""

    def test_single_pk_is_unique(self, loader_with_data):
        result = loader_with_data.check_key_uniqueness("hr", "emp", ["id"])
        assert result["is_unique"] is True

    def test_single_non_unique_column(self, loader_with_data):
        result = loader_with_data.check_key_uniqueness("hr", "emp", ["dept_id"])
        assert result["is_unique"] is False
        assert result["duplicate_pct"] == 80.0

    def test_single_column_100_perc_unique(self, loader_with_data):
        """Single column with unique_perc=100 should be unique even if not PK."""
        result = loader_with_data.check_key_uniqueness("hr", "dept", ["id"])
        assert result["is_unique"] is True

    def test_composite_all_pk_is_unique(self, loader_with_data):
        """Composite key where all columns are PKs → unique."""
        result = loader_with_data.check_key_uniqueness("hr", "emp_dept", ["emp_id", "dept_id"])
        assert result["is_unique"] is True
        assert result["all_pk"] is True

    def test_composite_high_unique_perc_not_all_pk(self, loader_with_data):
        """Composite: not all PK, but min_unique_perc >= 95 → unique."""
        # dept.id (PK, 100%) + dept.name (not PK, 95%) → min=95 → unique
        result = loader_with_data.check_key_uniqueness("hr", "dept", ["id", "name"])
        assert result["is_unique"] is True
        assert result["min_unique_perc"] == 95.0

    def test_composite_low_unique_perc_not_unique(self, loader_with_data):
        """Composite: not all PK, min_unique_perc < 95 → NOT unique."""
        # emp_dept: emp_id (PK, 80%) + role (not PK, 30%) → min=30 → not unique
        result = loader_with_data.check_key_uniqueness("hr", "emp_dept", ["emp_id", "role"])
        assert result["is_unique"] is False

    def test_composite_any_fully_unique_not_enough(self, loader_with_data):
        """For composite keys, one column with 100% is NOT enough (unlike single)."""
        # emp_dept: emp_id (PK, 80%) + dept_id (PK, 60%) → all_pk=True → unique
        # But emp_dept: emp_id (PK, 80%) + role (not PK, 30%) → min=30 < 95 → not unique
        # even though emp_id is PK
        result = loader_with_data.check_key_uniqueness("hr", "emp_dept", ["emp_id", "role"])
        assert result["is_unique"] is False

    def test_table_not_found(self, loader_with_data):
        result = loader_with_data.check_key_uniqueness("hr", "nonexistent", ["id"])
        assert result["is_unique"] is None
        assert "error" in result
