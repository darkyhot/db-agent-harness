from core.sql_static_checker import check_sql


class _FakeSchemaLoader:
    def get_table_columns(self, schema, table):
        import pandas as pd

        catalog = {
            ("dm", "sales"): pd.DataFrame([
                {"column_name": "task_code", "dType": "text"},
                {"column_name": "report_dt", "dType": "date"},
            ]),
        }
        return catalog.get((schema, table), pd.DataFrame(columns=["column_name", "dType"]))


def test_static_checker_accepts_columns_selected_from_cte_alias():
    sql = (
        "WITH sales_agg AS ("
        "SELECT report_dt, COUNT(*) AS cnt_udf "
        "FROM dm.sales GROUP BY report_dt"
        ") "
        "SELECT c1.report_dt, c1.cnt_udf "
        "FROM sales_agg c1 "
        "ORDER BY c1.report_dt"
    )

    result = check_sql(sql, schema_loader=_FakeSchemaLoader(), check_columns=True)

    assert not any("cnt_udf" in err for err in result.errors)
