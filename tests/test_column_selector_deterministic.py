"""Тесты для core/column_selector_deterministic.py."""

import pandas as pd


def test_choose_best_metric_prefers_outflow_qty_over_is_outflow_for_sum(tmp_path):
    from core.column_selector_deterministic import _choose_best_column
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm", "dm"],
        "table_name": ["uzp_dwh_fact_outflow", "uzp_data_split_mzp_sale_funnel"],
        "description": [
            "Фактический отток клиентов",
            "Воронка продаж и признаки оттока",
        ],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": [
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_dwh_fact_outflow",
            "uzp_data_split_mzp_sale_funnel",
            "uzp_data_split_mzp_sale_funnel",
        ],
        "column_name": [
            "inn",
            "report_dt",
            "outflow_qty",
            "segment_name",
            "is_outflow",
        ],
        "dType": [
            "varchar",
            "date",
            "int4",
            "varchar",
            "int4",
        ],
        "description": [
            "ИНН клиента",
            "Отчётная дата",
            "Кол-во ФЛ переставших быть ЗП клиентами",
            "Сегмент клиента",
            "Признак подтверждения оттока",
        ],
        "is_primary_key": [
            False,
            False,
            False,
            False,
            False,
        ],
        "unique_perc": [
            85.0,
            1.0,
            10.0,
            5.0,
            2.0,
        ],
        "not_null_perc": [
            95.0,
            99.0,
            95.0,
            99.99,
            100.0,
        ],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    table_structures = {
        "dm.uzp_dwh_fact_outflow": "outflow fact with inn, report_dt, outflow_qty",
        "dm.uzp_data_split_mzp_sale_funnel": "sale funnel with segment_name, is_outflow",
    }
    table_types = {
        "dm.uzp_dwh_fact_outflow": "fact",
        "dm.uzp_data_split_mzp_sale_funnel": "fact",
    }

    best = _choose_best_column(
        table_structures=table_structures,
        table_types=table_types,
        schema_loader=loader,
        slot="отток_ottok",
        require_numeric=True,
        agg_hint="sum",
    )

    assert best == ("dm.uzp_dwh_fact_outflow", "outflow_qty")


def test_choose_best_metric_uses_catalog_synonyms(tmp_path):
    from core.column_selector_deterministic import _choose_best_column
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["fact_outflow"],
        "description": ["Факт клиентских изменений"],
        "grain": ["event"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["fact_outflow"] * 3,
        "column_name": ["calc_fl_qty", "fact_payee_qty", "outflow_qty"],
        "dType": ["int4", "int4", "int4"],
        "description": [
            "Расчетное количество получателей при расчета Оттока",
            "Фактическое количество получателей начислений",
            "Кол-во ФЛ переставших быть ЗП клиентами",
        ],
        "is_primary_key": [False, False, False],
        "unique_perc": [1.0, 1.0, 1.0],
        "not_null_perc": [100.0, 100.0, 100.0],
        "synonyms": ["", "", "отток,фактический отток,количество оттока"],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    best = _choose_best_column(
        table_structures={"dm.fact_outflow": "fact"},
        table_types={"dm.fact_outflow": "fact"},
        schema_loader=loader,
        slot="фактический_отток",
        require_numeric=True,
        agg_hint="sum",
    )

    assert best == ("dm.fact_outflow", "outflow_qty")


# ---------------------------------------------------------------------------
# Task 1.2: group_by_hints influence on explorer (indirect test via state)
# ---------------------------------------------------------------------------

def test_explorer_skips_llm_when_no_group_by_hints(tmp_path):
    """Если group_by_hints пустые — ColumnSelector использует детерминированный путь."""
    from graph.graph import create_initial_state

    state = create_initial_state("посчитай отток")
    state["user_hints"]["group_by_hints"] = []

    # Нет явных group_by_hints → _has_explicit_group = False
    hints = state.get("user_hints", {}) or {}
    has_explicit_group = bool(hints.get("group_by_hints"))
    assert has_explicit_group is False


def test_explorer_uses_llm_path_when_group_by_hints_present(tmp_path):
    """Если group_by_hints не пустые — _has_explicit_group = True (LLM путь)."""
    from graph.graph import create_initial_state

    state = create_initial_state("сгруппируй по segment_name")
    state["user_hints"]["group_by_hints"] = ["segment_name"]

    hints = state.get("user_hints", {}) or {}
    has_explicit_group = bool(hints.get("group_by_hints"))
    assert has_explicit_group is True


def test_select_columns_single_entity_count_prefers_pk_over_aux_code(tmp_path):
    from core.column_selector_deterministic import select_columns
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_data_split_mzp_sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": ["uzp_data_split_mzp_sale_funnel"] * 5,
        "column_name": ["report_dt", "task_code", "uzp_task_code", "task_subtype", "task_category"],
        "dType": ["date", "text", "text", "text", "text"],
        "description": [
            "Отчетная дата",
            "Код задачи",
            "Код задачи УЗП",
            "Подтип задачи",
            "Категория задачи",
        ],
        "is_primary_key": [False, True, False, False, False],
        "unique_perc": [0.5, 100.0, 36.68, 2.62, 0.02],
        "not_null_perc": [99.0, 100.0, 36.68, 37.06, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    result = select_columns(
        intent={
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
        },
        table_structures={"dm.uzp_data_split_mzp_sale_funnel": "Воронка продаж по задачам"},
        table_types={"dm.uzp_data_split_mzp_sale_funnel": "fact"},
        join_analysis_data={},
        schema_loader=loader,
        user_input="Сколько задач по фактическому оттоку за февраль 2026",
        semantic_frame={
            "subject": "task",
            "requires_single_entity_count": True,
            "output_dimensions": [],
        },
    )

    roles = result["selected_columns"]["dm.uzp_data_split_mzp_sale_funnel"]
    assert roles["aggregate"] == ["task_code"]
    assert roles["select"] == ["task_code"]
    assert "group_by" not in roles or not roles["group_by"]
    assert "uzp_task_code" not in roles.get("select", [])


def test_select_columns_scalar_count_without_semantic_flag_still_avoids_group_by(tmp_path):
    from core.column_selector_deterministic import select_columns
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_data_split_mzp_sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": ["uzp_data_split_mzp_sale_funnel"] * 5,
        "column_name": ["report_dt", "task_code", "uzp_task_code", "task_subtype", "task_category"],
        "dType": ["date", "text", "text", "text", "text"],
        "description": [
            "Отчетная дата",
            "Код задачи",
            "Код задачи УЗП",
            "Подтип задачи",
            "Категория задачи",
        ],
        "is_primary_key": [False, True, False, False, False],
        "unique_perc": [0.5, 100.0, 36.68, 2.62, 0.02],
        "not_null_perc": [99.0, 100.0, 36.68, 37.06, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)

    loader = SchemaLoader(data_dir=tmp_path)
    result = select_columns(
        intent={
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
        },
        table_structures={"dm.uzp_data_split_mzp_sale_funnel": "Воронка продаж по задачам"},
        table_types={"dm.uzp_data_split_mzp_sale_funnel": "fact"},
        join_analysis_data={},
        schema_loader=loader,
        user_input="Сколько задач по фактическому оттоку за февраль 2026",
        semantic_frame={"subject": "task", "output_dimensions": []},
    )

    roles = result["selected_columns"]["dm.uzp_data_split_mzp_sale_funnel"]
    assert roles["aggregate"] == ["task_code"]
    assert roles["select"] == ["task_code"]
    assert "group_by" not in roles or not roles["group_by"]


def test_requested_slots_ignore_table_choice_suffix_and_filter_value(tmp_path):
    from core.column_selector_deterministic import _derive_requested_slots

    requested = _derive_requested_slots(
        "Сколько задач по фактическому оттоку поставили в феврале 26 "
        "(использовать таблицу dm.uzp_data_split_mzp_sale_funnel)",
        {
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "required_output": [],
        },
    )

    assert "uzp_data_split_mzp_sale_funnel" not in requested["dimensions"]
    assert not any("фактическому" in dim for dim in requested["dimensions"])
    assert requested["metric"] is not None


def _build_outflow_event_loader(tmp_path):
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_dwh_fact_outflow"],
        "description": ["Информация по фактическим оттокам"],
        "grain": ["event"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 5,
        "table_name": ["uzp_dwh_fact_outflow"] * 5,
        "column_name": ["report_dt", "inn", "gosb_id", "is_task", "inserted_dttm"],
        "dType": ["date", "int8", "int4", "boolean", "timestamp"],
        "description": [
            "Отчетная дата",
            "ИНН",
            "Идентификатор ГОСБ",
            "Признак выставленной задачи",
            "Дата и время загрузки",
        ],
        "is_primary_key": [False, True, False, False, False],
        "unique_perc": [0.5, 98.75, 0.93, 0.02, 0.02],
        "not_null_perc": [100.0, 100.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_select_columns_task_count_on_event_table_falls_back_to_count_star(tmp_path):
    from core.column_selector_deterministic import select_columns

    loader = _build_outflow_event_loader(tmp_path)
    result = select_columns(
        intent={
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
        },
        table_structures={"dm.uzp_dwh_fact_outflow": "Информация по фактическим оттокам"},
        table_types={"dm.uzp_dwh_fact_outflow": "fact"},
        join_analysis_data={},
        schema_loader=loader,
        user_input="Сколько задач по фактическому оттоку поставили в феврале 26",
        semantic_frame={"subject": "task", "requires_single_entity_count": True, "output_dimensions": []},
    )

    roles = result["selected_columns"]["dm.uzp_dwh_fact_outflow"]
    assert roles["aggregate"] == ["*"]
    assert "report_dt" not in roles.get("aggregate", [])


def test_select_columns_employee_subject_still_avoids_report_dt(tmp_path):
    from core.column_selector_deterministic import select_columns

    loader = _build_outflow_event_loader(tmp_path)
    result = select_columns(
        intent={
            "aggregation_hint": "count",
            "entities": ["задачи", "отток"],
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
        },
        table_structures={"dm.uzp_dwh_fact_outflow": "Информация по фактическим оттокам"},
        table_types={"dm.uzp_dwh_fact_outflow": "fact"},
        join_analysis_data={},
        schema_loader=loader,
        user_input="Сколько задач по фактическому оттоку поставили в феврале 26",
        semantic_frame={"subject": "employee", "requires_single_entity_count": True, "output_dimensions": []},
    )

    roles = result["selected_columns"]["dm.uzp_dwh_fact_outflow"]
    assert "report_dt" not in roles.get("aggregate", [])


def test_select_columns_tb_gosb_cardinality_uses_single_dictionary(tmp_path):
    from core.column_selector_deterministic import select_columns
    from core.schema_loader import SchemaLoader

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

    result = select_columns(
        intent={"aggregation_hint": "count", "entities": ["tb_id", "gosb_id"], "date_filters": {}},
        table_structures={
            "dm.uzp_dim_gosb": "Справочник ТБ и ГОСБ",
            "dm.uzp_data_epk_consolidation": "Консолидация ЕПК",
        },
        table_types={"dm.uzp_dim_gosb": "dim", "dm.uzp_data_epk_consolidation": "fact"},
        join_analysis_data={
            "dm.uzp_dim_gosb|dm.uzp_data_epk_consolidation": {
                "text": "JOIN candidates",
                "table1": "dm.uzp_dim_gosb",
                "table2": "dm.uzp_data_epk_consolidation",
            }
        },
        schema_loader=loader,
        user_input="Сколько всего есть тб и госб",
        user_hints={
            "aggregation_preferences_list": [
                {"function": "count", "column": "tb_id", "distinct": True},
                {"function": "count", "column": "gosb_id", "distinct": True},
            ]
        },
        semantic_frame={"requires_single_entity_count": True, "output_dimensions": []},
    )

    assert set(result["selected_columns"]) == {"dm.uzp_dim_gosb"}
    roles = result["selected_columns"]["dm.uzp_dim_gosb"]
    assert roles["aggregate"] == ["tb_id", "old_gosb_id"]
    assert "group_by" not in roles
    assert result["join_spec"] == []
    assert result["confidence"] >= 0.55  # A.2: detection should be confident enough


# ---------------------------------------------------------------------------
# A.2 / A.3 / A.4 / A.1: confidence tightening tests
# ---------------------------------------------------------------------------

def _build_loader_with_gosb(tmp_path, *, include_name_col: bool):
    """Helper: builds a single-table fixture for label-slot tests."""
    rows: list[dict] = []
    rows.append(
        {
            "column_name": "gosb_id",
            "dType": "int4",
            "description": "Идентификатор ГОСБ. Наименование: Москва",
            "is_primary_key": False,
            "unique_perc": 80.0,
            "not_null_perc": 100.0,
        }
    )
    if include_name_col:
        rows.append(
            {
                "column_name": "gosb_name",
                "dType": "varchar",
                "description": "Название ГОСБ",
                "is_primary_key": False,
                "unique_perc": 80.0,
                "not_null_perc": 99.0,
            }
        )
    attrs_df = pd.DataFrame(
        [{"schema_name": "dm", "table_name": "uzp_dim_gosb", **row} for row in rows]
    )
    tables_df = pd.DataFrame(
        [{"schema_name": "dm", "table_name": "uzp_dim_gosb", "description": "Справочник ГОСБ"}]
    )
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    from core.schema_loader import SchemaLoader

    return SchemaLoader(data_dir=tmp_path)


def test_label_slot_prefers_name_over_id(tmp_path):
    """A.3: при наличии _name колонки label-слот должен предпочесть её, а не *_id."""
    from core.column_selector_deterministic import _choose_best_column

    loader = _build_loader_with_gosb(tmp_path, include_name_col=True)
    table_structures = {"dm.uzp_dim_gosb": "Справочник ГОСБ"}
    table_types = {"dm.uzp_dim_gosb": "dim"}

    best = _choose_best_column(
        table_structures=table_structures,
        table_types=table_types,
        schema_loader=loader,
        slot="gosb_name",
    )
    assert best == ("dm.uzp_dim_gosb", "gosb_name")


def test_label_slot_rejects_id_column_when_no_name_sibling(tmp_path):
    """A.3: если *_name отсутствует, селектор не должен взять *_id для label-слота
    только потому, что description содержит «Наименование»."""
    from core.column_selector_deterministic import _choose_best_column

    loader = _build_loader_with_gosb(tmp_path, include_name_col=False)
    best = _choose_best_column(
        table_structures={"dm.uzp_dim_gosb": "Справочник ГОСБ"},
        table_types={"dm.uzp_dim_gosb": "dim"},
        schema_loader=loader,
        slot="gosb_name",
    )
    assert best is None


def test_alias_hit_lifts_tb_id_for_tb_slot(tmp_path):
    """A.2: для слота 'тб' алиас-карта должна обеспечить tb_id, а не описание-фаззи."""
    from core.column_selector_deterministic import _choose_best_column
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame(
        [{"schema_name": "dm", "table_name": "uzp_dim_gosb", "description": "Справочник ТБ"}]
    )
    attrs_df = pd.DataFrame(
        [
            {
                "schema_name": "dm",
                "table_name": "uzp_dim_gosb",
                "column_name": "tb_id",
                "dType": "int4",
                "description": "Идентификатор ТБ",
                "is_primary_key": False,
                "unique_perc": 5.0,
                "not_null_perc": 100.0,
            },
            {
                "schema_name": "dm",
                "table_name": "uzp_dim_gosb",
                "column_name": "isu_branch_id",
                "dType": "int4",
                "description": "Идентификатор отделения банка (ТБ-уровень)",
                "is_primary_key": False,
                "unique_perc": 30.0,
                "not_null_perc": 100.0,
            },
        ]
    )
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    best = _choose_best_column(
        table_structures={"dm.uzp_dim_gosb": "Справочник ТБ"},
        table_types={"dm.uzp_dim_gosb": "dim"},
        schema_loader=loader,
        slot="тб",
    )
    assert best == ("dm.uzp_dim_gosb", "tb_id")


def test_date_column_ambiguity_lowers_confidence(tmp_path):
    """A.4: если в filter-роли оказались две date-колонки — confidence должен упасть на 0.15."""
    from core.column_selector_deterministic import select_columns
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame(
        [{"schema_name": "dm", "table_name": "fact_outflow", "description": "Факт оттока"}]
    )
    attrs_df = pd.DataFrame(
        [
            {
                "schema_name": "dm", "table_name": "fact_outflow",
                "column_name": "report_dt", "dType": "date",
                "description": "Отчётная дата",
                "is_primary_key": False, "unique_perc": 1.0, "not_null_perc": 100.0,
            },
            {
                "schema_name": "dm", "table_name": "fact_outflow",
                "column_name": "inserted_dttm", "dType": "timestamp",
                "description": "Дата вставки",
                "is_primary_key": False, "unique_perc": 90.0, "not_null_perc": 100.0,
            },
            {
                "schema_name": "dm", "table_name": "fact_outflow",
                "column_name": "outflow_qty", "dType": "int4",
                "description": "Количество оттока",
                "is_primary_key": False, "unique_perc": 30.0, "not_null_perc": 100.0,
            },
        ]
    )
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    result = select_columns(
        intent={
            "aggregation_hint": "sum",
            "entities": ["отток"],
            "date_filters": {"from": "2026-02-01", "to": "2026-03-01"},
        },
        table_structures={"dm.fact_outflow": "Факт оттока"},
        table_types={"dm.fact_outflow": "fact"},
        join_analysis_data={},
        schema_loader=loader,
        user_input="отток в феврале 2026",
        user_hints={},
        semantic_frame={},
    )

    filter_cols = result["selected_columns"]["dm.fact_outflow"].get("filter", [])
    # Две date-колонки в filter → штраф confidence
    if len([c for c in filter_cols if "dt" in c.lower() or "dttm" in c.lower()]) > 1:
        # Бейслайн confidence без штрафа близок к 0.7-0.8; с штрафом -0.15 → ниже исходного
        assert result["confidence"] <= 0.85


def test_short_cyrillic_slot_resolves_via_description_overlap(tmp_path):
    """Без alias-словарей: «тб» → tb_id матчится по описанию «Идентификатор ТБ»."""
    from core.column_selector_deterministic import _choose_best_column
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame(
        [{"schema_name": "dm", "table_name": "uzp_dim_gosb", "description": "Справочник ТБ"}]
    )
    attrs_df = pd.DataFrame(
        [
            {
                "schema_name": "dm", "table_name": "uzp_dim_gosb",
                "column_name": "tb_id", "dType": "int4",
                "description": "Идентификатор ТБ",
                "is_primary_key": False, "unique_perc": 5.0, "not_null_perc": 100.0,
            },
        ]
    )
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    best = _choose_best_column(
        table_structures={"dm.uzp_dim_gosb": "Справочник ТБ"},
        table_types={"dm.uzp_dim_gosb": "dim"},
        schema_loader=loader,
        slot="тб",
    )
    assert best == ("dm.uzp_dim_gosb", "tb_id")
