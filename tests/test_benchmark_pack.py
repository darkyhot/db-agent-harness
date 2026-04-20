from pathlib import Path

import pytest
import pandas as pd

from core.schema_loader import SchemaLoader
from core.semantic_frame import derive_semantic_frame

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


pytestmark = pytest.mark.skipif(not _YAML_AVAILABLE, reason="pyyaml required")


@pytest.fixture
def benchmark_loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["sale_funnel", "employees"],
        "description": ["Воронка продаж по задачам", "Справочник сотрудников"],
        "grain": ["task", "employee"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": ["sale_funnel", "sale_funnel", "sale_funnel", "employees", "employees", "employees"],
        "column_name": ["report_dt", "is_outflow", "task_subtype", "employee_id", "employee_name", "report_dt"],
        "dType": ["date", "int4", "text", "bigint", "text", "date"],
        "description": [
            "Отчетная дата",
            "Признак подтверждения оттока",
            "Подтип задачи",
            "ID сотрудника",
            "Имя сотрудника",
            "Отчетная дата",
        ],
        "is_primary_key": [False, False, False, False, False, False],
        "unique_perc": [1.0, 2.0, 10.0, 95.0, 80.0, 1.0],
        "not_null_perc": [99.0, 100.0, 100.0, 100.0, 99.0, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader.ensure_semantic_registry()
    return loader


def test_benchmark_pack_has_expected_shape():
    path = Path(__file__).parent / "benchmark_cases.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 5
    for case in data:
        assert "query" in case
        assert "expected" in case


def test_benchmark_pack_semantic_frame_expectations(benchmark_loader):
    path = Path(__file__).parent / "benchmark_cases.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for case in data:
        frame = derive_semantic_frame(case["query"], schema_loader=benchmark_loader)
        for key, value in case["expected"].items():
            assert frame.get(key) == value
