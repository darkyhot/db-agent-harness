"""Тесты core/metadata_health.check_catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.metadata_health import check_catalog
from core.schema_loader import SchemaLoader


def _write_catalog(tmp_path: Path, tables_csv: str, attrs_csv: str) -> SchemaLoader:
    (tmp_path / "tables_list.csv").write_text(tables_csv, encoding="utf-8")
    (tmp_path / "attr_list.csv").write_text(attrs_csv, encoding="utf-8")
    return SchemaLoader(data_dir=tmp_path)


class TestGrainCoverage:
    def test_empty_grain_reports_error(self, tmp_path):
        loader = _write_catalog(
            tmp_path,
            "schema_name,table_name,description,grain\ns,t,desc,\n",
            "schema_name,table_name,column_name,dType,is_not_null,description,is_primary_key,not_null_perc,unique_perc\n"
            "s,t,id,int8,True,,True,100.0,100.0\n",
        )
        report = check_catalog(loader)
        assert report.has_errors
        assert any(i.category == "grain" and i.subject == "s.t" for i in report.issues)

    def test_filled_grain_no_error(self, tmp_path):
        loader = _write_catalog(
            tmp_path,
            "schema_name,table_name,description,grain\ns,t,desc,event\n",
            "schema_name,table_name,column_name,dType,is_not_null,description,is_primary_key,not_null_perc,unique_perc\n"
            "s,t,id,int8,True,,True,100.0,100.0\n",
        )
        report = check_catalog(loader)
        assert not any(i.category == "grain" for i in report.issues)


class TestFKSuggestions:
    def test_suggest_unfilled_fk(self, tmp_path):
        loader = _write_catalog(
            tmp_path,
            "schema_name,table_name,description,grain\n"
            "s,dim,справочник,dictionary\n"
            "s,fact,факт,event\n",
            "schema_name,table_name,column_name,dType,is_not_null,description,is_primary_key,not_null_perc,unique_perc\n"
            "s,dim,inn,int8,True,,True,100.0,100.0\n"
            "s,fact,inn,int8,False,,False,100.0,0.5\n",
        )
        report = check_catalog(loader)
        fk_issues = [i for i in report.issues if i.category == "foreign_key"]
        assert len(fk_issues) == 1
        assert fk_issues[0].severity == "warning"
        assert fk_issues[0].subject == "s.fact.inn"

    def test_no_suggestion_when_fk_already_filled(self, tmp_path):
        loader = _write_catalog(
            tmp_path,
            "schema_name,table_name,description,grain\n"
            "s,dim,справочник,dictionary\n"
            "s,fact,факт,event\n",
            "schema_name,table_name,column_name,dType,is_not_null,description,is_primary_key,not_null_perc,unique_perc,foreign_key_target\n"
            "s,dim,inn,int8,True,,True,100.0,100.0,\n"
            "s,fact,inn,int8,False,,False,100.0,0.5,s.dim.inn\n",
        )
        report = check_catalog(loader)
        assert not any(i.category == "foreign_key" for i in report.issues)


class TestReportShape:
    def test_to_dict_serializable(self, tmp_path):
        loader = _write_catalog(
            tmp_path,
            "schema_name,table_name,description,grain\ns,t,desc,\n",
            "schema_name,table_name,column_name,dType,is_not_null,description,is_primary_key,not_null_perc,unique_perc\n"
            "s,t,id,int8,True,,True,100.0,100.0\n",
        )
        report = check_catalog(loader)
        d = report.to_dict()
        assert d["total_tables"] == 1
        assert isinstance(d["issues"], list)
        assert d["counts_by_severity"].get("error", 0) >= 1
