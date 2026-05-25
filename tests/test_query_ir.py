import logging

import pandas as pd

from core.catalog_grounding import ground_query_spec
from core.column_binding import bind_columns, derive_entity_flag_filters
from core.join_analysis import detect_table_type
from core.query_ir import FilterSpec, QuerySpec, query_spec_json_schema
from core.schema_loader import SchemaLoader
from core.sql_planner_deterministic import build_blueprint
from core.where_resolver import resolve_where
from graph.nodes.query_ir import _strip_unstated_physical_hints
from graph.nodes.sql_pipeline import _apply_query_spec_blueprint_overrides


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["orders"],
        "description": ["Фактовая таблица заказов"],
        "grain": ["transaction"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "dm", "dm"],
        "table_name": ["orders", "orders", "orders"],
        "column_name": ["order_id", "order_dt", "amount"],
        "dType": ["bigint", "date", "numeric"],
        "description": ["ID заказа", "Дата заказа", "Сумма заказа"],
        "is_primary_key": [True, False, False],
        "unique_perc": [100.0, 2.0, 50.0],
        "not_null_perc": [100.0, 99.0, 95.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_query_spec_accepts_valid_payload():
    payload = {
        "task": "answer_data",
        "metrics": [
            {
                "operation": "sum",
                "target": "amount",
                "distinct_policy": "auto",
                "confidence": 0.9,
                "evidence": [{"source": "user", "text": "сумма заказов", "confidence": 0.9}],
            }
        ],
        "dimensions": [{"target": "order_dt", "confidence": 0.8}],
        "filters": [],
        "time_range": {"start": "2024-01-01", "end": "2024-02-01", "confidence": 0.8},
        "source_constraints": [{"schema": "dm", "table": "orders", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.88,
    }

    spec, errors = QuerySpec.from_dict(payload)

    assert errors == []
    assert spec is not None
    assert spec.to_legacy_intent()["aggregation_hint"] == "sum"
    assert spec.to_legacy_user_hints()["group_by_hints"] == ["order_dt"]


def test_query_interpreter_strips_unstated_physical_hints():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "sum", "target": "отток", "confidence": 0.9}],
        "entities": [{"name": "отток", "target_column_hint": "fact_payee_qty"}],
        "dimensions": [
            {
                "target": "ГОСБ",
                "label": "Название ГОСБ",
                "source_table": "dm.dim_gosb",
                "join_key": "gosb_id",
                "confidence": 0.9,
            }
        ],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    })
    assert spec is not None, errors

    _strip_unstated_physical_hints(spec, "Посчитай сумму оттока по дате и названию ГОСБ")

    assert spec.entities[0].target_column_hint is None
    assert spec.dimensions[0].source_table is None
    assert spec.dimensions[0].join_key is None


def test_query_interpreter_keeps_explicit_physical_hints():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "sum", "target": "outflow_qty", "confidence": 0.9}],
        "entities": [{"name": "outflow_qty", "target_column_hint": "outflow_qty"}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    })
    assert spec is not None, errors

    _strip_unstated_physical_hints(spec, "Покажи сумму outflow_qty")

    assert spec.entities[0].target_column_hint == "outflow_qty"


def test_query_spec_projects_multiple_count_metrics_without_loss():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [
            {"operation": "count", "target": "tb_id", "distinct_policy": "distinct", "confidence": 0.9},
            {"operation": "count", "target": "gosb_id", "distinct_policy": "distinct", "confidence": 0.9},
        ],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    })
    assert spec is not None, errors

    hints = spec.to_legacy_user_hints()

    assert hints["aggregation_preferences"] == {
        "function": "count",
        "column": "tb_id",
        "distinct": True,
    }
    assert hints["aggregation_preferences_list"] == [
        {"function": "count", "column": "tb_id", "distinct": True},
        {"function": "count", "column": "gosb_id", "distinct": True},
    ]
    assert spec.to_semantic_frame()["requires_single_entity_count"] is False


def test_query_spec_order_by_overrides_blueprint_direction():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "target": "inn", "distinct_policy": "all", "confidence": 0.9}],
        "dimensions": [],
        "filters": [],
        "order_by": {"target": "inn", "direction": "ASC", "confidence": 0.9},
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    })
    assert spec is not None, errors
    blueprint = {
        "aggregation": {"function": "COUNT", "column": "inn", "alias": "count_inn"},
        "aggregations": [{"function": "COUNT", "column": "inn", "alias": "count_inn"}],
        "order_by": "count_inn DESC",
    }

    _apply_query_spec_blueprint_overrides(blueprint, spec.to_dict())

    assert blueprint["order_by"] == "count_inn ASC"


def test_query_spec_rejects_invalid_metric_operation():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "median", "confidence": 0.5}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.5,
    })

    assert spec is None
    assert any("metrics[0].operation" in err for err in errors)


def test_query_spec_schema_exposes_required_contract():
    schema = query_spec_json_schema()

    assert schema["title"] == "QuerySpec"
    assert "metrics" in schema["required"]
    assert "source_constraints" in schema["properties"]
    assert schema["additionalProperties"] is False


def test_catalog_grounding_binds_explicit_source(tmp_path):
    loader = _loader(tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "target": "orders", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [{"schema": "dm", "table": "orders", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    result = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="сколько заказов")

    assert result.needs_clarification is False
    assert [s.full_name for s in result.sources] == ["dm.orders"]
    assert result.plan_ir is not None
    assert result.plan_ir.main_source.full_name == "dm.orders"


def test_catalog_grounding_binds_unique_table_only_required_source(tmp_path):
    loader = _loader(tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [{"table": "orders", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    result = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="сколько заказов")

    assert result.needs_clarification is False
    assert [source.full_name for source in result.sources] == ["dm.orders"]
    assert result.plan_ir is not None
    assert result.plan_ir.main_source.full_name == "dm.orders"


def test_catalog_grounding_keeps_required_table_when_other_table_covers_slots(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["requested_orders", "better_orders"],
        "description": ["Фактовая таблица заказов", "Заказы с суммой"],
        "grain": ["transaction", "transaction"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "dm", "dm"],
        "table_name": ["requested_orders", "better_orders", "better_orders"],
        "column_name": ["order_id", "order_id", "amount"],
        "dType": ["bigint", "bigint", "numeric"],
        "description": ["ID заказа", "ID заказа", "Сумма заказа"],
        "is_primary_key": [True, True, False],
        "unique_perc": [100.0, 100.0, 50.0],
        "not_null_perc": [100.0, 100.0, 95.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "sum", "target": "amount", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [{"table": "requested_orders", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    result = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="сумма заказов")

    assert result.needs_clarification is False
    assert result.sources[0].full_name == "dm.requested_orders"
    assert "dm.requested_orders" in [source.full_name for source in result.sources]


def test_catalog_grounding_stops_when_required_source_missing(tmp_path):
    loader = _loader(tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [{"table": "missing_orders", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    result = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="сколько заказов")

    assert result.needs_clarification is True
    assert result.clarification is not None
    assert result.clarification.reason == "catalog_grounding_missing_required_source"
    assert result.sources == []


def test_catalog_grounding_clarifies_ambiguous_table_only_required_source(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "ods"],
        "table_name": ["orders", "orders"],
        "description": ["Заказы DM", "Заказы ODS"],
        "grain": ["transaction", "transaction"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "ods"],
        "table_name": ["orders", "orders"],
        "column_name": ["order_id", "order_id"],
        "dType": ["bigint", "bigint"],
        "description": ["ID заказа", "ID заказа"],
        "is_primary_key": [True, True],
        "unique_perc": [100.0, 100.0],
        "not_null_perc": [100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [{"table": "orders", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    result = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="сколько заказов")

    assert result.needs_clarification is True
    assert result.clarification is not None
    assert set(result.clarification.options) == {"dm.orders", "ods.orders"}


def test_query_spec_promotes_calendar_literal_filter_to_time_range():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [{"target": "отчетный месяц", "operator": "=", "value": "февраль 2026", "confidence": 0.9}],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })

    assert spec is not None, errors
    assert spec.time_range is not None
    assert spec.time_range.start == "2026-02-01"
    assert spec.time_range.end == "2026-03-01"
    assert spec.filters == []


def test_required_sale_funnel_calendar_filter_binds_to_time_axis(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["uzp_data_split_mzp_sale_funnel", "uzp_dwh_fact_outflow"],
        "description": ["Воронка продаж по задачам", "Фактические события оттока"],
        "grain": ["task", "event"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": [
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
        ],
        "column_name": [
            "report_dt",
            "task_code",
            "task_subtype",
            "m_avg_salary_amt",
            "report_dt",
            "outflow_qty",
        ],
        "dType": ["date", "text", "text", "numeric", "date", "numeric"],
        "description": [
            "Отчетная дата",
            "Код задачи",
            "Подтип задачи",
            "Средняя зарплата в отчетный месяц",
            "Отчетная дата",
            "Количество оттока",
        ],
        "is_primary_key": [False, True, False, False, False, False],
        "unique_perc": [0.5, 100.0, 2.0, 10.0, 0.5, 5.0],
        "not_null_perc": [100.0, 100.0, 95.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "distinct_policy": "auto", "confidence": 1.0}],
        "entities": [{"name": "задача по фактическому оттоку", "confidence": 1.0}],
        "dimensions": [],
        "filters": [{"target": "отчетный месяц", "operator": "=", "value": "февраль 2026", "confidence": 1.0}],
        "source_constraints": [{"table": "uzp_data_split_mzp_sale_funnel", "required": True, "confidence": 1.0}],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 1.0,
    })
    assert spec is not None, errors
    assert spec.time_range is not None
    assert spec.filters == []

    grounded = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="")
    assert grounded.needs_clarification is False
    assert [source.full_name for source in grounded.sources] == ["dm.uzp_data_split_mzp_sale_funnel"]

    table_key = "dm.uzp_data_split_mzp_sale_funnel"
    table_structures = {table_key: loader.get_table_info("dm", "uzp_data_split_mzp_sale_funnel")}
    table_types = {
        table_key: detect_table_type(
            "uzp_data_split_mzp_sale_funnel",
            loader.get_table_columns("dm", "uzp_data_split_mzp_sale_funnel"),
        )
    }
    bound = bind_columns(
        query_spec=spec,
        table_structures=table_structures,
        table_types=table_types,
        schema_loader=loader,
    )
    assert bound is not None
    selected = bound["selected_columns"]
    assert selected[table_key]["filter"] == ["report_dt"]
    assert "m_avg_salary_amt" not in selected[table_key]["filter"]

    blueprint = build_blueprint(
        intent=spec.to_legacy_intent(),
        selected_columns=selected,
        join_spec=[],
        table_types=table_types,
        join_analysis_data={},
        user_hints=spec.to_legacy_user_hints(),
        schema_loader=loader,
        semantic_frame=spec.to_semantic_frame(),
        time_range=spec.to_dict().get("time_range") or {},
    )
    assert blueprint["main_table"] == table_key
    assert "report_dt >= '2026-02-01'::date" in blueprint["where_conditions"]
    assert "report_dt < '2026-03-01'::date" in blueprint["where_conditions"]
    assert not any("m_avg_salary_amt" in cond for cond in blueprint["where_conditions"])


def test_catalog_grounding_uses_metadata_filters_and_prunes_unrelated_helpers(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm", "dm"],
        "table_name": ["fact_outflow", "sale_funnel", "employee_assignment"],
        "description": [
            "Информация по фактическим оттокам",
            "Воронка продаж по задачам",
            "Данные по закреплению сотрудников за организациями",
        ],
        "grain": ["event", "task", "employee"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 7,
        "table_name": [
            "fact_outflow", "fact_outflow", "fact_outflow",
            "sale_funnel", "sale_funnel", "sale_funnel",
            "employee_assignment",
        ],
        "column_name": [
            "report_dt", "inn", "is_task",
            "report_dt", "task_subtype", "task_category",
            "end_dttm",
        ],
        "dType": ["date", "text", "boolean", "date", "text", "text", "timestamp"],
        "description": [
            "Отчетная дата", "ИНН", "Признак выставленной задачи",
            "Отчетная дата", "Подтип задачи", "Категория задачи",
            "Дата окончания закрепления",
        ],
        "is_primary_key": [False] * 7,
        "unique_perc": [1.0, 90.0, 2.0, 1.0, 10.0, 2.0, 5.0],
        "not_null_perc": [100.0] * 7,
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader._rule_registry = {
        "rules": [
            {
                "rule_id": "text:dm.sale_funnel.task_subtype",
                "column_key": "dm.sale_funnel.task_subtype",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["подтип задачи"],
                "value_candidates": ["фактический отток"],
            },
            {
                "rule_id": "text:dm.sale_funnel.task_category",
                "column_key": "dm.sale_funnel.task_category",
                "semantic_class": "enum_like",
                "match_kind": "text_search",
                "match_phrases": ["категория задачи"],
                "value_candidates": ["Задача"],
            },
        ]
    }

    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "target": "задачи", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [{"target": "фактический отток", "operator": "ILIKE", "value": "фактический отток", "confidence": 0.8}],
        "time_range": {"start": "2026-02-01", "end": "2026-03-01", "grain": "month", "confidence": 0.9},
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    result = ground_query_spec(
        query_spec=spec,
        schema_loader=loader,
        user_input=(
            "Сколько задач по фактическому оттоку поставили в феврале 26\n"
            "Вопрос уточнения: Уточните год?\n"
            "Уточнение пользователя: 2026"
        ),
        max_sources=3,
    )

    # H2: when two strong fact-table candidates remain after pruning and
    # neither is a user-pinned explicit source, the grounder asks instead
    # of silently picking one. Both candidates appear in the options
    # (rendered as "schema.table — description...").
    assert result.needs_clarification is True
    assert result.clarification is not None
    assert result.clarification.reason == "catalog_grounding_ambiguous_strong_sources"
    option_prefixes = {
        (str(opt).split(" — ", 1)[0]) for opt in (result.clarification.options or [])
    }
    assert option_prefixes == {"dm.sale_funnel", "dm.fact_outflow"}


def test_catalog_grounding_logs_table_scores(tmp_path, caplog):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["orders", "customers"],
        "description": ["Фактовая таблица заказов", "Справочник клиентов"],
        "grain": ["transaction", "customer"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "dm", "dm"],
        "table_name": ["orders", "orders", "customers"],
        "column_name": ["order_id", "amount", "customer_id"],
        "dType": ["bigint", "numeric", "bigint"],
        "description": ["ID заказа", "Сумма заказа", "ID клиента"],
        "is_primary_key": [True, False, True],
        "unique_perc": [100.0, 50.0, 100.0],
        "not_null_perc": [100.0, 95.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "sum", "target": "amount", "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    with caplog.at_level(logging.INFO, logger="core.catalog_grounding"):
        ground_query_spec(query_spec=spec, schema_loader=loader, user_input="сумма заказов")

    messages = [record.getMessage() for record in caplog.records]
    assert any("CatalogTableScore: scored 2 table(s)" in msg for msg in messages)
    assert any("table=dm.orders" in msg and "score=" in msg for msg in messages)
    assert any("table=dm.customers" in msg and "score=" in msg for msg in messages)


def _quality_loader(tmp_path, fact_segment_fill=1.5, include_join=True):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["fact_sales", "org_dict"],
        "description": ["Факты продаж по организациям", "Справочник организаций"],
        "grain": ["event", "organization"],
    })
    fact_cols = ["sale_dt", "org_id", "amount", "segment_name"]
    dict_cols = ["org_id" if include_join else "dict_org_id", "segment_name"]
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * (len(fact_cols) + len(dict_cols)),
        "table_name": ["fact_sales"] * len(fact_cols) + ["org_dict"] * len(dict_cols),
        "column_name": fact_cols + dict_cols,
        "dType": ["date", "bigint", "numeric", "text", "bigint", "text"],
        "description": [
            "Дата продажи",
            "Идентификатор организации",
            "Сумма продажи",
            "Наименование сегмента",
            "Идентификатор организации",
            "Наименование сегмента",
        ],
        "is_primary_key": [False, False, False, False, True, False],
        "unique_perc": [1.0, 90.0, 80.0, 0.1, 100.0, 0.1],
        "not_null_perc": [100.0, 100.0, 100.0, fact_segment_fill, 100.0, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def _sum_by_segment_spec():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "sum", "target": "amount", "distinct_policy": "auto", "confidence": 0.9}],
        "dimensions": [{"target": "segment", "confidence": 0.8}],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    })
    assert spec is not None, errors
    return spec


def test_catalog_grounding_adds_joinable_source_for_low_fill_dimension(tmp_path):
    loader = _quality_loader(tmp_path, fact_segment_fill=1.5, include_join=True)

    result = ground_query_spec(
        query_spec=_sum_by_segment_spec(),
        schema_loader=loader,
        user_input="сумма продаж по сегменту",
        max_sources=1,
    )

    assert result.needs_clarification is False
    assert [source.full_name for source in result.sources] == ["dm.fact_sales", "dm.org_dict"]
    assert result.sources[1].reason == "dimension_quality_enrichment"


def test_catalog_grounding_keeps_high_fill_fact_dimension_without_helper(tmp_path):
    loader = _quality_loader(tmp_path, fact_segment_fill=95.0, include_join=True)

    result = ground_query_spec(
        query_spec=_sum_by_segment_spec(),
        schema_loader=loader,
        user_input="сумма продаж по сегменту",
        max_sources=3,
    )

    assert result.needs_clarification is False
    assert [source.full_name for source in result.sources] == ["dm.fact_sales"]


def test_catalog_grounding_does_not_add_unjoinable_high_fill_source(tmp_path):
    loader = _quality_loader(tmp_path, fact_segment_fill=1.5, include_join=False)

    result = ground_query_spec(
        query_spec=_sum_by_segment_spec(),
        schema_loader=loader,
        user_input="сумма продаж по сегменту",
        max_sources=1,
    )

    assert result.needs_clarification is False
    assert [source.full_name for source in result.sources] == ["dm.fact_sales"]


def test_catalog_grounding_dictionary_cardinality_prefers_single_dict_source(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["uzp_dim_gosb", "uzp_data_epk_consolidation"],
        "description": ["Справочник ТБ и ГОСБ", "Консолидация ЕПК по ТБ и ГОСБ"],
        "grain": ["gosb", "epk"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": [
            "uzp_dim_gosb", "uzp_dim_gosb",
            "uzp_data_epk_consolidation", "uzp_data_epk_consolidation", "uzp_data_epk_consolidation",
        ],
        "column_name": ["tb_id", "old_gosb_id", "epk_id", "tb_id", "gosb_id"],
        "dType": ["int4", "int4", "int8", "int4", "int4"],
        "description": ["Идентификатор ТБ", "Идентификатор ГОСБ", "ЕПК", "Идентификатор ТБ", "Идентификатор ГОСБ"],
        "is_primary_key": [True, True, True, False, False],
        "unique_perc": [3.0, 95.0, 100.0, 3.0, 80.0],
        "not_null_perc": [100.0] * 5,
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [
            {"operation": "count", "target": "tb_id", "distinct_policy": "distinct", "confidence": 0.9},
            {"operation": "count", "target": "gosb_id", "distinct_policy": "distinct", "confidence": 0.9},
        ],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.9,
    })
    assert spec is not None, errors

    result = ground_query_spec(
        query_spec=spec,
        schema_loader=loader,
        user_input="Сколько всего есть тб и госб",
        max_sources=3,
    )

    assert result.needs_clarification is False
    assert [source.full_name for source in result.sources] == ["dm.uzp_dim_gosb"]


def test_catalog_grounding_clarifies_when_no_source(tmp_path):
    loader = _loader(tmp_path)
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "sum", "target": "unknown_metric", "distinct_policy": "auto", "confidence": 0.7}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.7,
    })
    assert spec is not None, errors

    result = ground_query_spec(query_spec=spec, schema_loader=loader, user_input="")

    assert result.needs_clarification is True
    assert result.clarification is not None
    assert result.clarification.field == "source_constraints"


def test_query_spec_rejects_extra_fields():
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [],
        "dimensions": [],
        "filters": [],
        "source_constraints": [],
        "join_constraints": [],
        "clarification_needed": False,
        "confidence": 0.5,
        "unexpected": True,
    })

    assert spec is None
    assert any("unexpected" in err for err in errors)


def test_where_resolver_accepts_filter_specs_directly(tmp_path):
    loader = _loader(tmp_path)
    loader._rule_registry = {
        "rules": [
            {
                "rule_id": "text:dm.orders.amount",
                "column_key": "dm.orders.amount",
                "semantic_class": "metric",
                "match_kind": "numeric_compare",
                "match_phrases": ["amount"],
                "value_candidates": [],
            }
        ]
    }

    result = resolve_where(
        user_input="",
        intent={"filter_conditions": []},
        selected_columns={"dm.orders": {"select": ["amount"], "filter": ["amount"], "aggregate": []}},
        selected_tables=["dm.orders"],
        schema_loader=loader,
        semantic_frame={},
        base_conditions=[],
        filter_specs=[
            FilterSpec(target="amount", operator=">=", value=100, confidence=0.9),
        ],
    )

    assert result["needs_clarification"] is False
    assert any("amount" in cond and ">=" in cond for cond in result["conditions"])


def test_where_resolver_skips_exact_date_filter_when_calendar_range_exists(tmp_path):
    loader = _loader(tmp_path)
    result = resolve_where(
        user_input="сколько заказов за февраль 2026",
        intent={"filter_conditions": []},
        selected_columns={"dm.orders": {"select": ["order_id"], "filter": ["order_dt"], "aggregate": ["order_id"]}},
        selected_tables=["dm.orders"],
        schema_loader=loader,
        semantic_frame={},
        base_conditions=["order_dt >= '2026-02-01'::date", "order_dt < '2026-03-01'::date"],
        filter_specs=[
            FilterSpec(target="order_dt", operator="=", value="2026-02-26", confidence=0.8),
        ],
        time_range={"start": "2026-02-01", "end": "2026-03-01", "grain": "month"},
    )

    assert result["conditions"] == ["order_dt >= '2026-02-01'::date", "order_dt < '2026-03-01'::date"]
    assert result["applied_rules"] == []


# ---------------------------------------------------------------------------
# H2: _detect_ambiguous_strong_sources — pick a clarification when 2+ tables
# tie within `min_gap_pct` of the top raw score; skip otherwise.
# ---------------------------------------------------------------------------


def _binding(
    table: str,
    score: float,
    *,
    reason: str = "query_spec_slot_score",
    confidence: float | None = None,
):
    from core.query_ir import SourceBinding
    # Confidence is clamped to [0.35, 0.95] in production; default it to a
    # plausible mid-range value when callers don't care, so the dataclass
    # stays valid.
    if confidence is None:
        confidence = max(0.35, min(0.95, score / 12.0))
    return SourceBinding(
        schema="dm",
        table=table,
        reason=reason,
        confidence=confidence,
        score=score,
        evidence=[],
    )


def test_h2_helper_returns_tied_sources_within_gap():
    from core.catalog_grounding import _detect_ambiguous_strong_sources
    # Raw scores 3.0 vs 2.0 → 33% gap < 40% threshold — both surface.
    tied = _detect_ambiguous_strong_sources([
        _binding("fact_outflow", 3.0),
        _binding("sale_funnel", 2.0),
        _binding("misc", 1.0),  # below score_floor
    ])
    assert {s.table for s in tied} == {"fact_outflow", "sale_funnel"}


def test_h2_helper_returns_empty_when_one_table_dominates():
    from core.catalog_grounding import _detect_ambiguous_strong_sources
    # 3.0 vs 1.0 → 67% gap > 55% — clear dominance, no ambiguity.
    tied = _detect_ambiguous_strong_sources([
        _binding("fact_a", 3.0),
        _binding("fact_b", 1.0),
    ])
    assert tied == []


def test_h2_helper_triggers_for_close_fact_tables():
    """Real-world regression (agent log 2026-05-25): scores 3.0 vs 1.7
    (gap≈43%) must surface H2 — these are the typical competing fact tables
    H2 was designed for. A 40% threshold was too narrow.
    """
    from core.catalog_grounding import _detect_ambiguous_strong_sources
    tied = _detect_ambiguous_strong_sources([
        _binding("sale_funnel_task", 3.0),
        _binding("fact_outflow", 1.7),
    ])
    assert {s.table for s in tied} == {"sale_funnel_task", "fact_outflow"}


def test_h2_helper_skips_dimension_quality_enrichment_sidekicks():
    """Dimension sidekicks added via JOIN enrichment are not alternatives —
    they coexist with the main fact, so H2 must not treat them as a tie.
    """
    from core.catalog_grounding import _detect_ambiguous_strong_sources
    tied = _detect_ambiguous_strong_sources([
        _binding("fact_sales", 3.0),
        _binding("org_dict", 4.0, reason="dimension_quality_enrichment"),
    ])
    assert tied == []


def test_h2_helper_skips_explicit_source_constraint():
    """User-pinned sources should never trigger ambiguity — the user already
    chose.
    """
    from core.catalog_grounding import _detect_ambiguous_strong_sources
    tied = _detect_ambiguous_strong_sources([
        _binding("fact_a", 4.0, reason="explicit_source_constraint"),
        _binding("fact_b", 3.5),
    ])
    assert tied == []


def test_h2_grounder_skips_clarification_when_join_constraints_present(tmp_path):
    """If QuerySpec carries join_constraints, multi-source is intentional —
    H2 must not emit a clarification.
    """
    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["fact_a", "fact_b"],
        "description": ["Факт A", "Факт B"],
        "grain": ["event", "event"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["fact_a", "fact_b"],
        "column_name": ["x", "y"],
        "dType": ["text", "text"],
        "description": ["X col", "Y col"],
        "is_primary_key": [False, False],
        "unique_perc": [50.0, 50.0],
        "not_null_perc": [100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    # Build a QuerySpec that pins both tables via join_constraints — the
    # ambiguity guard must bail.
    spec, errors = QuerySpec.from_dict({
        "task": "answer_data",
        "metrics": [{"operation": "count", "target": None, "distinct_policy": "auto", "confidence": 0.8}],
        "dimensions": [],
        "filters": [],
        "source_constraints": [
            {"table": "fact_a", "schema": "dm", "required": True, "confidence": 0.9},
            {"table": "fact_b", "schema": "dm", "required": True, "confidence": 0.9},
        ],
        "join_constraints": [
            {"left": "dm.fact_a", "right": "dm.fact_b", "key": "x", "confidence": 0.9},
        ],
        "clarification_needed": False,
        "confidence": 0.8,
    })
    assert spec is not None, errors

    result = ground_query_spec(
        query_spec=spec,
        schema_loader=loader,
        user_input="join fact_a and fact_b",
    )
    # Either we get both sources without clarification, or an unrelated
    # clarification — but never the H2 ambiguity reason.
    if result.clarification is not None:
        assert result.clarification.reason != "catalog_grounding_ambiguous_strong_sources"


# ---------------------------------------------------------------------------
# H6: derive_entity_flag_filters — entity «Задача» → is_task = TRUE
# ---------------------------------------------------------------------------


def test_derive_entity_flag_filters_emits_synthetic_boolean(tmp_path):
    """When the user's entity resolves to a boolean column on the main table
    (e.g. «Задача» → is_task), the helper emits a synthetic FilterSpec dict
    so the WHERE pipeline picks it up as direct_filter_specs and we don't
    lose the structural filter to a dtype mismatch on a parallel text intent.
    """
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["fact_outflow"],
        "description": ["Факты по фактическому оттоку, включая задачи"],
        "grain": ["row"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["fact_outflow"] * 3,
        "column_name": ["report_dt", "is_task", "amount"],
        "dType": ["date", "bool", "int8"],
        "description": ["Дата", "Признак: задача", "Сумма"],
        "is_primary_key": [False] * 3,
        "unique_perc": [0.5, 50.0, 80.0],
        "not_null_perc": [99.0, 99.0, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    spec, _errors = QuerySpec.from_dict({
        "task": "answer_data",
        "entities": [{"name": "Задача", "canonical": "Задача", "confidence": 1.0}],
        "metrics": [{"operation": "count", "target": "Задача", "confidence": 1.0}],
        "filters": [],
        "confidence": 0.9,
    })
    assert spec is not None

    selected_columns: dict[str, dict] = {
        "dm.fact_outflow": {"select": ["report_dt"], "aggregate": ["*"], "filter": ["report_dt"]}
    }
    synthetic = derive_entity_flag_filters(
        query_spec=spec,
        selected_columns=selected_columns,
        schema_loader=loader,
    )
    assert synthetic, "expected a synthetic flag filter for entity «Задача»"
    assert synthetic[0]["target"] == "is_task"
    assert synthetic[0]["value"] is True
    # The flag column is wired into the table's filter role so
    # _apply_exact_filter_specs can find it.
    assert "is_task" in selected_columns["dm.fact_outflow"]["filter"]


def test_derive_entity_flag_filters_skips_when_no_flag_column(tmp_path):
    """When the main table has no boolean column for the entity, the helper
    returns [] — no synthetic noise.
    """
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["orders"],
        "description": ["Заказы клиентов"],
        "grain": ["row"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["orders", "orders"],
        "column_name": ["order_id", "amount"],
        "dType": ["int8", "int8"],
        "description": ["ID заказа", "Сумма"],
        "is_primary_key": [True, False],
        "unique_perc": [100.0, 80.0],
        "not_null_perc": [100.0, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    spec, _errors = QuerySpec.from_dict({
        "task": "answer_data",
        "entities": [{"name": "Заказ", "canonical": "Заказ", "confidence": 1.0}],
        "metrics": [{"operation": "count", "target": "Заказ", "confidence": 1.0}],
        "filters": [],
        "confidence": 0.9,
    })
    assert spec is not None

    synthetic = derive_entity_flag_filters(
        query_spec=spec,
        selected_columns={"dm.orders": {"select": ["order_id"], "aggregate": ["*"]}},
        schema_loader=loader,
    )
    assert synthetic == []


def test_derive_entity_flag_filters_via_description_match(tmp_path):
    """Cross-lingual fallback (Step I): entity «Задача» (Russian) must match
    boolean column `is_task` (English name) via its Russian description
    «Признак выставленной задачи», even when the embedding-based
    entity_resolver can't bridge the language gap (e.g. no LLM available).
    """
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["fact_outflow"],
        "description": ["Факты"],
        "grain": ["row"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["fact_outflow"] * 3,
        "column_name": ["report_dt", "is_task", "amount"],
        "dType": ["date", "bool", "int8"],
        "description": [
            "Отчётная дата",
            "Признак выставленной задачи",  # Russian description bridges «Задача» → is_task
            "Сумма",
        ],
        "is_primary_key": [False] * 3,
        "unique_perc": [0.5, 50.0, 80.0],
        "not_null_perc": [99.0, 99.0, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    spec, _errors = QuerySpec.from_dict({
        "task": "answer_data",
        "entities": [{"name": "Задача", "canonical": "задача", "confidence": 1.0}],
        "metrics": [{"operation": "count", "target": "задача", "confidence": 1.0}],
        "filters": [],
        "confidence": 0.9,
    })
    assert spec is not None

    selected_columns: dict[str, dict] = {
        "dm.fact_outflow": {"select": ["report_dt"], "aggregate": ["*"], "filter": ["report_dt"]}
    }
    # llm_invoker=None → forces the entity_resolver to fall back to
    # embeddings (which may or may not bridge RU↔EN). Step I's description
    # match must catch this case regardless.
    synthetic = derive_entity_flag_filters(
        query_spec=spec,
        selected_columns=selected_columns,
        schema_loader=loader,
        llm_invoker=None,
    )
    assert synthetic, "expected synthetic filter via description match"
    assert synthetic[0]["target"] == "is_task"
    assert synthetic[0]["value"] is True
    assert "is_task" in selected_columns["dm.fact_outflow"]["filter"]
