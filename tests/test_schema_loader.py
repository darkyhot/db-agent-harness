"""Тесты SchemaLoader: проверка composite key логики и literal search."""

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
        "description": ["Employees", "Departments", "Employee-Department link"],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)

    # attr_list.csv
    attrs_df = pd.DataFrame({
        "schema_name": ["hr", "hr", "hr", "hr", "hr", "hr", "hr"],
        "table_name": ["emp", "emp", "dept", "dept", "emp_dept", "emp_dept", "emp_dept"],
        "column_name": ["id", "dept_id", "id", "name", "emp_id", "dept_id", "role"],
        "dType": ["int", "int", "int", "varchar", "int", "int", "varchar"],
        "description": ["PK", "FK to dept", "PK", "Name", "FK", "FK", "Role"],
        "is_primary_key": [True, False, True, False, True, True, False],
        "unique_perc": [100.0, 20.0, 100.0, 95.0, 80.0, 60.0, 30.0],
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

    def test_single_composite_pk_member_not_unique(self, loader_with_data):
        """Single column that is PK but part of composite key with low unique_perc.

        emp_dept.emp_id: is_primary_key=True, unique_perc=80%, but table has 2 PKs.
        As a composite PK member with unique_perc < 90%, it is NOT unique alone.
        """
        result = loader_with_data.check_key_uniqueness("hr", "emp_dept", ["emp_id"])
        assert result["is_unique"] is False
        assert result["status"] == "risky"

    def test_single_sole_pk_is_unique(self, loader_with_data):
        """Single PK column in a table with only 1 PK → unique (not composite).

        emp.id: is_primary_key=True, unique_perc=100%, only PK in table.
        """
        result = loader_with_data.check_key_uniqueness("hr", "emp", ["id"])
        assert result["is_unique"] is True
        assert result["status"] == "safe"

    def test_status_field_present(self, loader_with_data):
        """check_key_uniqueness should always return 'status' field."""
        result = loader_with_data.check_key_uniqueness("hr", "emp", ["id"])
        assert result["status"] == "safe"

        result = loader_with_data.check_key_uniqueness("hr", "emp", ["dept_id"])
        assert result["status"] == "risky"

    def test_table_not_found(self, loader_with_data):
        result = loader_with_data.check_key_uniqueness("hr", "nonexistent", ["id"])
        assert result["is_unique"] is None
        assert "error" in result


class TestLiteralSearchRobustness:
    def test_search_tables_treats_regex_like_input_as_literal(self, loader_with_data):
        assert loader_with_data.search_tables("[").empty
        assert loader_with_data.search_tables("foo(bar").empty
        assert loader_with_data.search_tables("a+b").empty

    def test_search_tables_matches_literal_identifier(self, loader_with_data):
        result = loader_with_data.search_tables("emp")
        assert not result.empty
        assert "emp" in set(result["table_name"])

    def test_find_tables_with_column_handles_regex_like_input(self, loader_with_data):
        assert loader_with_data.find_tables_with_column("client_id").empty
        assert loader_with_data.find_tables_with_column("a+b").empty

    def test_search_by_description_handles_mixed_and_russian_input(self, tmp_path):
        tables_df = pd.DataFrame({
            "schema_name": ["dm", "support"],
            "table_name": ["orders", "tickets"],
            "description": ["Customer orders and invoices", "История обращений пользователей"],
        })
        tables_df.to_csv(tmp_path / "tables_list.csv", index=False)

        attrs_df = pd.DataFrame({
            "schema_name": ["dm", "support"],
            "table_name": ["orders", "tickets"],
            "column_name": ["client_id", "ticket_text"],
            "dType": ["int", "text"],
            "description": ["Client identifier", "Русское описание обращения"],
            "is_primary_key": [False, False],
            "unique_perc": [10.0, 0.0],
        })
        attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
        loader = SchemaLoader(data_dir=tmp_path)

        assert not loader.search_by_description("обращения").empty
        assert not loader.search_by_description("client_id").empty
        assert loader.search_by_description("[orders]+").empty


class TestGrainHelpers:
    def test_missing_grain_column_is_added(self, loader_with_data):
        assert "grain" in loader_with_data.tables_df.columns
        assert loader_with_data.get_table_grain("hr", "emp") == ""

    def test_infer_query_grain(self, loader_with_data):
        assert loader_with_data.infer_query_grain("Посчитай количество задач", ["задачи"]) == "task"
        assert loader_with_data.infer_query_grain("Сколько клиентов", ["клиенты"]) == "client"

    def test_value_profiles_generated(self, loader_with_data):
        loader_with_data.ensure_value_profiles()
        profile = loader_with_data.get_value_profile("hr", "dept", "name")
        assert profile == {}

    def test_column_semantics_generated(self, loader_with_data):
        loader_with_data.ensure_column_semantics()
        sem = loader_with_data.get_column_semantics("hr", "emp", "id")
        assert sem["semantic_class"] in {"identifier", "join_key"}

    def test_table_semantics_generated(self, loader_with_data):
        loader_with_data.ensure_table_semantics()
        sem = loader_with_data.get_table_semantics("hr", "dept")
        assert sem["table_role"] in {"dimension", "reference", "fact", "other"}
        assert "primary_subjects" in sem

    def test_filter_candidate_gets_value_profile(self, tmp_path):
        tables_df = pd.DataFrame({
            "schema_name": ["dm"],
            "table_name": ["tasks"],
            "description": ["Task registry"],
            "grain": ["task"],
        })
        tables_df.to_csv(tmp_path / "tables_list.csv", index=False)

        attrs_df = pd.DataFrame({
            "schema_name": ["dm", "dm"],
            "table_name": ["tasks", "tasks"],
            "column_name": ["task_subtype", "report_dt"],
            "dType": ["varchar", "date"],
            "description": ["Подтип задачи", "Отчетная дата"],
            "is_primary_key": [False, False],
            "unique_perc": [12.0, 1.0],
            "not_null_perc": [100.0, 100.0],
        })
        attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
        loader = SchemaLoader(data_dir=tmp_path)

        loader.ensure_column_semantics()
        loader.ensure_value_profiles()
        profile = loader.get_value_profile("dm", "tasks", "task_subtype")
        assert profile["value_mode"] == "enum_like"
        assert "=" in profile["allowed_operators"]
        assert "ILIKE" in profile["allowed_operators"]
