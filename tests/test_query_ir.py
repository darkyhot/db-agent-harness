import pandas as pd

from core.catalog_grounding import ground_query_spec
from core.query_ir import FilterSpec, QuerySpec, query_spec_json_schema
from core.schema_loader import SchemaLoader
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

    assert result.needs_clarification is False
    assert [source.full_name for source in result.sources] == ["dm.sale_funnel", "dm.fact_outflow"]


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
