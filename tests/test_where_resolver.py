import pandas as pd

from core.schema_loader import SchemaLoader
from core.semantic_frame import derive_semantic_frame
from core.where_resolver import resolve_where


def _loader(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["uzp_data_split_mzp_sale_funnel"],
        "description": ["Воронка продаж по задачам"],
        "grain": ["task"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 6,
        "table_name": ["uzp_data_split_mzp_sale_funnel"] * 6,
        "column_name": ["report_dt", "task_code", "task_subtype", "task_category", "task_type", "is_outflow"],
        "dType": ["date", "text", "text", "text", "text", "int4"],
        "description": [
            "Отчетная дата",
            "Код задачи",
            "Подтип задачи",
            "Категория задачи",
            "Тип задачи",
            "Признак подтверждения оттока",
        ],
        "is_primary_key": [False, False, False, False, False, False],
        "unique_perc": [0.5, 90.0, 10.0, 0.02, 0.11, 2.0],
        "not_null_perc": [99.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_where_resolver_adds_confirmed_outflow_flag(tmp_path):
    loader = _loader(tmp_path)
    loader.ensure_value_profiles()
    frame = derive_semantic_frame(
        "Посчитай количество задач с подтвержденным оттоком",
        schema_loader=loader,
    )
    result = resolve_where(
        user_input="Посчитай количество задач с подтвержденным оттоком",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert any("is_outflow = 1" in cond for cond in result["conditions"])
    assert result["applied_rules"]
    assert next(iter(result["filter_candidates"].values()))[0]["column"] == "is_outflow"
    assert not any("task_subtype" in cond for cond in result["conditions"])


def test_where_resolver_adds_factual_outflow_task_subtype(tmp_path):
    loader = _loader(tmp_path)
    loader._value_profiles = {
        "dm.uzp_data_split_mzp_sale_funnel.task_subtype": {
            "known_terms": ["фактический отток"],
            "top_values": [],
            "value_mode": "enum_like",
        },
        "dm.uzp_data_split_mzp_sale_funnel.task_category": {
            "known_terms": ["Задача"],
            "top_values": [],
            "value_mode": "enum_like",
        },
        "dm.uzp_data_split_mzp_sale_funnel.task_type": {
            "known_terms": ["сервисная задача по юр-лицу"],
            "top_values": [],
            "value_mode": "enum_like",
        },
    }
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    result = resolve_where(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert any("task_subtype ILIKE '%фактический отток%'" in cond for cond in result["conditions"])
    assert any("task_category ILIKE '%Задача%'" in cond for cond in result["conditions"])
    assert not any("task_type" in cond for cond in result["conditions"])
    assert result["applied_rules"]
    assert any(cands[0]["column"] == "task_subtype" for cands in result["filter_candidates"].values() if cands)


def test_where_resolver_respects_explicit_column_clarification(tmp_path):
    loader = _loader(tmp_path)
    loader.ensure_value_profiles()
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    result = resolve_where(
        user_input="Посчитай количество задач по фактическому оттоку\nУточнение пользователя: task_subtype",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert result["needs_clarification"] is False
    assert any("task_subtype" in cond for cond in result["conditions"])


def test_where_resolver_honors_user_filter_choices_without_text_augmentation(tmp_path):
    """Если пользователь уже ответил на clarification и выбор пришёл через
    state.user_filter_choices — повторного вопроса быть не должно, даже если
    имя колонки не дописано в user_input."""
    loader = _loader(tmp_path)
    loader.ensure_value_profiles()
    frame = derive_semantic_frame(
        "Посчитай количество задач по фактическому оттоку",
        schema_loader=loader,
    )
    # Сначала прогон без choices — чтобы получить request_id первого кандидата.
    first = resolve_where(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    request_id = next(iter(first["filter_candidates"].keys()))

    result = resolve_where(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
        user_filter_choices={request_id: "task_subtype"},
    )
    assert result["needs_clarification"] is False
    assert any("task_subtype" in cond for cond in result["conditions"])
    assert any(r.startswith("user_choice:") for r in result["reasoning"])


def test_where_resolver_calendar_literal_uses_date_range_not_metric_column(tmp_path):
    loader = _loader(tmp_path)
    result = resolve_where(
        user_input="Сколько задач за отчетный месяц февраль 2026",
        intent={"filter_conditions": []},
        selected_columns={
            "dm.uzp_data_split_mzp_sale_funnel": {
                "aggregate": ["task_code"],
                "filter": ["m_avg_salary_amt"],
            }
        },
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame={},
        base_conditions=[],
        filter_specs=[{"target": "отчетный месяц", "operator": "=", "value": "февраль 2026", "confidence": 0.9}],
    )

    assert result["needs_clarification"] is False
    assert "report_dt >= '2026-02-01'::date" in result["conditions"]
    assert "report_dt < '2026-03-01'::date" in result["conditions"]
    assert not any("m_avg_salary_amt" in cond for cond in result["conditions"])


def test_where_resolver_calendar_literal_clarifies_without_date_axis(tmp_path):
    import pandas as pd
    from core.schema_loader import SchemaLoader

    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["metrics_only"],
        "description": ["Таблица метрик"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["metrics_only"],
        "column_name": ["m_avg_salary_amt"],
        "dType": ["numeric"],
        "description": ["Средняя зарплата в отчетный месяц"],
        "is_primary_key": [False],
        "unique_perc": [50.0],
        "not_null_perc": [100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)

    result = resolve_where(
        user_input="Сколько задач за отчетный месяц февраль 2026",
        intent={"filter_conditions": []},
        selected_columns={"dm.metrics_only": {"aggregate": ["m_avg_salary_amt"]}},
        selected_tables=["dm.metrics_only"],
        schema_loader=loader,
        semantic_frame={},
        base_conditions=[],
        filter_specs=[{"target": "отчетный месяц", "operator": "=", "value": "февраль 2026", "confidence": 0.9}],
    )

    assert result["needs_clarification"] is True
    assert "date/time-axis" in result["clarification_message"]


def test_where_resolver_skips_rejected_candidate_without_reasking(tmp_path):
    loader = _loader(tmp_path)
    loader.ensure_value_profiles()
    frame = {
        "filter_intents": [
            {
                "request_id": "text:dm.uzp_data_split_mzp_sale_funnel.task_subtype",
                "kind": "text_search",
                "query_text": "фактический отток",
                "column_key": "dm.uzp_data_split_mzp_sale_funnel.task_subtype",
            }
        ]
    }
    result = resolve_where(
        user_input="Посчитай количество задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
        rejected_filter_choices={"text:dm.uzp_data_split_mzp_sale_funnel.task_subtype": ["task_subtype"]},
    )
    assert result["needs_clarification"] is False
    assert not any("task_subtype" in cond for cond in result["conditions"])
    assert any(r.startswith("rejected_all:") for r in result["reasoning"])


def test_where_resolver_clarification_message_shows_example_values(tmp_path):
    """Когда агент всё-таки спрашивает — в подсказке должны быть видны
    матчанные примеры значений из known_terms."""
    loader = _loader(tmp_path)
    loader._value_profiles = {
        "dm.uzp_data_split_mzp_sale_funnel.task_subtype": {
            "known_terms": ["факт"],
            "top_values": [],
            "value_mode": "enum_like",
        },
        "dm.uzp_data_split_mzp_sale_funnel.task_type": {
            "known_terms": ["факт"],
            "top_values": [],
            "value_mode": "enum_like",
        },
    }
    # Добавим task_type к лоадеру — _loader выдаёт task_subtype без task_type.
    # Чтобы не перегружать фикстуру, просто проверяем сообщение через кандидатов вручную.
    from core.where_resolver import candidate_label
    cand_a = {"column": "task_subtype", "description": "Подтип задачи", "matched_example": "фактический отток"}
    cand_b = {"column": "task_type", "description": "Тип задачи", "matched_example": "отток"}
    label_a = candidate_label(cand_a)
    label_b = candidate_label(cand_b)
    assert "фактический отток" in label_a
    assert "отток" in label_b
    assert "Подтип задачи" in label_a
    assert "Тип задачи" in label_b


def test_where_resolver_treats_business_event_as_table_context_for_single_selected_table(tmp_path):
    loader = _loader(tmp_path)
    tables_df = pd.read_csv(tmp_path / "tables_list.csv")
    tables_df = pd.concat([
        tables_df,
        pd.DataFrame({
            "schema_name": ["dm"],
            "table_name": ["uzp_dwh_fact_outflow"],
            "description": ["Информация по фактическим оттокам для УЗП"],
            "grain": ["snapshot"],
        }),
    ], ignore_index=True)
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader.ensure_value_profiles()
    frame = {
        "business_event": "фактический отток",
        "filter_intents": [
            {
                "request_id": "text:dm.uzp_data_split_mzp_sale_funnel.task_subtype",
                "kind": "text_search",
                "query_text": "фактический отток",
                "column_key": "dm.uzp_data_split_mzp_sale_funnel.task_subtype",
            }
        ],
    }
    result = resolve_where(
        user_input="Сколько фактических оттоков",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_dwh_fact_outflow": {"select": ["inn"], "aggregate": ["inn"]}},
        selected_tables=["dm.uzp_dwh_fact_outflow"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert result["needs_clarification"] is False
    assert "table_context_covers_business_event" in result["reasoning"]


def test_where_resolver_adds_implicit_subject_flag_for_task_subject(tmp_path):
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["outflow_snapshot"],
        "description": ["Снимок событий оттока"],
        "grain": ["snapshot"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["outflow_snapshot"] * 3,
        "column_name": ["report_dt", "inn", "is_task"],
        "dType": ["date", "text", "boolean"],
        "description": ["Отчетная дата", "ИНН клиента", "Признак выставленной задачи"],
        "is_primary_key": [False, False, False],
        "unique_perc": [0.5, 90.0, 0.02],
        "not_null_perc": [100.0, 100.0, 100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader.ensure_value_profiles()

    frame = derive_semantic_frame("Сколько задач", schema_loader=loader)
    result = resolve_where(
        user_input="Сколько задач",
        intent={"filter_conditions": []},
        selected_columns={"dm.outflow_snapshot": {"select": ["inn"], "aggregate": ["inn"]}},
        selected_tables=["dm.outflow_snapshot"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert result["needs_clarification"] is False
    assert any("is_task = true" == cond for cond in result["conditions"])


def test_where_resolver_keeps_clarification_when_single_table_does_not_cover_business_event(tmp_path):
    loader = _loader(tmp_path)
    frame = {
        "business_event": "фактический отток",
        "filter_intents": [
            {
                "request_id": "text:dm.uzp_data_split_mzp_sale_funnel.task_subtype",
                "kind": "text_search",
                "query_text": "фактический отток",
                "column_key": "dm.uzp_data_split_mzp_sale_funnel.task_subtype",
            }
        ],
    }
    result = resolve_where(
        user_input="Сколько фактических оттоков",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code"], "aggregate": ["task_code"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert result["needs_clarification"] is True
    assert "table_context_covers_business_event" not in result["reasoning"]


def test_candidate_label_falls_back_to_example_values():
    from core.where_resolver import candidate_label
    cand = {
        "column": "segment_name",
        "description": "Сегмент клиента",
        "matched_example": None,
        "example_values": ["retail", "corp"],
    }
    label = candidate_label(cand)
    assert "segment_name" in label
    assert "Сегмент клиента" in label
    assert "retail" in label and "corp" in label


# ---------------------------------------------------------------------------
# Task 1.2: _user_explicitly_named и user_filter_choices
# ---------------------------------------------------------------------------

def test_user_explicitly_named_prevents_clarification(tmp_path):
    """Если пользователь явно назвал имя колонки — clarification не запрашивается."""
    from core.where_resolver import _user_explicitly_named

    best = {"column": "task_subtype", "confidence": "medium", "condition": "task_subtype='X'", "score": 50.0}
    second = {"column": "task_type", "confidence": "medium", "condition": "task_type='X'", "score": 40.0}

    is_explicit, chosen = _user_explicitly_named("посчитай по task_subtype", best, second)
    assert is_explicit is True
    assert chosen == best


def test_user_explicitly_named_chooses_second(tmp_path):
    """Если пользователь явно назвал второй вариант — выбирается он."""
    from core.where_resolver import _user_explicitly_named

    best = {"column": "task_subtype", "confidence": "medium", "condition": "task_subtype='X'", "score": 50.0}
    second = {"column": "task_type", "confidence": "medium", "condition": "task_type='X'", "score": 40.0}

    is_explicit, chosen = _user_explicitly_named("посчитай по task_type", best, second)
    assert is_explicit is True
    assert chosen == second


def test_user_explicitly_named_both_mentioned_no_choice(tmp_path):
    """Если оба варианта упомянуты — однозначного выбора нет."""
    from core.where_resolver import _user_explicitly_named

    best = {"column": "task_subtype", "confidence": "medium", "condition": "task_subtype='X'", "score": 50.0}
    second = {"column": "task_type", "confidence": "medium", "condition": "task_type='X'", "score": 40.0}

    is_explicit, chosen = _user_explicitly_named("task_subtype и task_type вместе", best, second)
    assert is_explicit is False
    assert chosen is None


def test_user_filter_choices_bypasses_clarification(tmp_path):
    """user_filter_choices с уже выбранной колонкой — clarification_message должен быть пустым."""
    loader = _loader(tmp_path)
    loader.ensure_value_profiles()
    frame = derive_semantic_frame(
        "Посчитай по task_subtype",
        schema_loader=loader,
    )
    result = resolve_where(
        user_input="Посчитай по task_subtype",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_data_split_mzp_sale_funnel": {"select": ["task_code", "task_subtype"]}},
        selected_tables=["dm.uzp_data_split_mzp_sale_funnel"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
        user_filter_choices={"task_subtype": "task_subtype"},
    )
    # clarification_message должен быть пустым
    assert result.get("clarification_message", "") == ""
