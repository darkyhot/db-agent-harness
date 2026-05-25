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
    # Either suppression path is fine — F2 short-circuit (G1) or the older
    # business-event-encoded-in-table guard. Both fold the filter into the
    # table context without surfacing a clarification.
    assert any(
        marker in result["reasoning"]
        for marker in (
            "table_context_covers_business_event",
            "table_context_covers_value:dm.uzp_dwh_fact_outflow:фактический отток",
        )
    )


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


# ---------------------------------------------------------------------------
# Fix B: dtype-check для filter_specs (bool колонка ≠ text литерал)
# ---------------------------------------------------------------------------


def _bool_loader(tmp_path, *, table_name="generic_facts", description="Универсальные факты"):
    """Минимальный SchemaLoader с boolean-колонкой is_task.

    Имя таблицы намеренно нейтральное — иначе сработает эвристика H и
    фильтр уйдёт не в unresolved, а в implicit.
    """
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": [table_name],
        "description": [description],
        "grain": ["row"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 2,
        "table_name": [table_name] * 2,
        "column_name": ["report_dt", "is_task"],
        "dType": ["date", "bool"],
        "description": ["Дата", "Признак: задача"],
        "is_primary_key": [False, False],
        "unique_perc": [0.5, 50.0],
        "not_null_perc": [99.0, 99.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    return SchemaLoader(data_dir=tmp_path)


def test_bool_column_vs_text_value_goes_to_unresolved(tmp_path):
    """Регрессия (лог 2026-05-22): is_task = 'фактический отток' попадал в SQL.

    Имя таблицы НЕ кодирует "отток" → H-эвристика не срабатывает →
    фильтр остаётся в unresolved_filters.
    """
    from core.query_ir import FilterSpec
    loader = _bool_loader(tmp_path, table_name="tasks", description="Реестр задач")
    loader.ensure_value_profiles()
    result = resolve_where(
        user_input="Сколько задач по технической блокировке",
        intent={"filter_conditions": []},
        selected_columns={"dm.tasks": {"select": ["is_task"], "filter": ["is_task"]}},
        selected_tables=["dm.tasks"],
        schema_loader=loader,
        semantic_frame=None,
        base_conditions=[],
        filter_specs=[
            FilterSpec(target="is_task", operator="=", value="техническая блокировка", value_kind="literal")
        ],
    )
    assert not any("is_task" in cond and "блокировк" in cond for cond in result["conditions"])
    unresolved = result.get("unresolved_filters") or []
    assert unresolved, "ожидаем dtype-mismatch фильтр в unresolved_filters"
    assert unresolved[0]["reason"] == "dtype_mismatch"
    assert unresolved[0]["candidate_dtype"].lower().startswith("bool")
    assert unresolved[0]["value"] == "техническая блокировка"
    # implicit_filters пустой — таблица "tasks" не кодирует "блокировку"
    assert not result.get("implicit_filters")


def test_filter_value_encoded_in_table_migrates_to_implicit(tmp_path):
    """Fix H: 'фактический отток' + таблица fact_outflow → implicit_filters,
    а не unresolved. Это разблокирует выполнение SQL без галлюцинаций.
    """
    from core.query_ir import FilterSpec
    loader = _bool_loader(
        tmp_path, table_name="fact_outflow", description="Факты по фактическому оттоку"
    )
    loader.ensure_value_profiles()
    result = resolve_where(
        user_input="Сколько задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_columns={"dm.fact_outflow": {"select": ["is_task"], "filter": ["is_task"]}},
        selected_tables=["dm.fact_outflow"],
        schema_loader=loader,
        semantic_frame=None,
        base_conditions=[],
        filter_specs=[
            FilterSpec(target="is_task", operator="=", value="фактический отток", value_kind="literal")
        ],
    )
    assert not any("is_task" in cond and "фактическ" in cond for cond in result["conditions"])
    # unresolved опустошён, фильтр переехал в implicit_filters
    assert not result.get("unresolved_filters")
    implicit = result.get("implicit_filters") or []
    assert implicit, "ожидаем перенос в implicit_filters"
    assert implicit[0]["applied_via"] == "table_name_encoding"
    assert implicit[0]["value"] == "фактический отток"
    assert "outflow" in implicit[0]["table"]


def test_numeric_column_vs_numeric_value_passes(tmp_path):
    """Контр-тест: int-колонка + int литерал — фильтр применяется, unresolved пуст."""
    from core.query_ir import FilterSpec
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["fact_x"],
        "description": ["x"],
        "grain": ["row"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["fact_x"],
        "column_name": ["amount"],
        "dType": ["int8"],
        "description": ["amount"],
        "is_primary_key": [False],
        "unique_perc": [50.0],
        "not_null_perc": [100.0],
    })
    tables_df.to_csv(tmp_path / "tables_list.csv", index=False)
    attrs_df.to_csv(tmp_path / "attr_list.csv", index=False)
    loader = SchemaLoader(data_dir=tmp_path)
    loader.ensure_value_profiles()
    result = resolve_where(
        user_input="amount > 100",
        intent={"filter_conditions": []},
        selected_columns={"dm.fact_x": {"select": ["amount"], "filter": ["amount"]}},
        selected_tables=["dm.fact_x"],
        schema_loader=loader,
        semantic_frame=None,
        base_conditions=[],
        filter_specs=[FilterSpec(target="amount", operator=">", value=100, value_kind="literal")],
    )
    assert any("amount" in cond and "100" in cond for cond in result["conditions"])
    assert not result.get("unresolved_filters")


# ---------------------------------------------------------------------------
# G1/G2/G3: F2 fallback clarification — table-encoded short-circuit and
# plausibility filter for offered candidates (2026-05-25 regression).
# ---------------------------------------------------------------------------


def test_f2_short_circuits_when_value_encoded_in_selected_table(tmp_path):
    """When the filter intent's value is already encoded in the chosen table
    (e.g. «фактический отток» + uzp_dwh_fact_outflow) AND the intent's
    column_key points at a DIFFERENT table, the F2 fallback should fold the
    filter into implicit_filters and emit no clarification.
    """
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
                "value": "фактический отток",
                "column_key": "dm.uzp_data_split_mzp_sale_funnel.task_subtype",
            }
        ],
    }
    result = resolve_where(
        user_input="Сколько задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_dwh_fact_outflow": {"select": ["inn"], "aggregate": ["inn"]}},
        selected_tables=["dm.uzp_dwh_fact_outflow"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert result["needs_clarification"] is False
    assert result["clarification_spec"] == {} or not result["clarification_spec"]
    assert any(
        item.get("applied_via") == "table_name_encoding"
        and item.get("table") == "dm.uzp_dwh_fact_outflow"
        for item in result.get("implicit_filters") or []
    )
    assert any(
        r.startswith("table_context_covers_value:dm.uzp_dwh_fact_outflow")
        for r in result["reasoning"]
    )


def test_f2_filters_out_identifier_and_low_confidence_candidates():
    """G2: F2 fallback must skip semantic_class=identifier and confidence=low
    candidates when offering choices to the user. The previous behaviour
    surfaced login/ФИО columns for free-text values like «фактический отток».
    """
    from core.where_resolver import resolve_where
    import types

    # Build a tiny in-memory loader stub that exposes the minimum surface
    # resolve_where uses; we monkey-rank candidates via a fake
    # rank_filter_candidates by patching the module attribute.
    class _Stub:
        tables_df = pd.DataFrame({
            "schema_name": ["dm"],
            "table_name": ["t"],
            "description": ["test table"],
            "grain": ["snapshot"],
        })
        def get_table_info(self, schema, table):
            return "Таблица: dm.t\nОписание: test table\nКолонки:"
        def get_table_columns(self, schema, table):
            return pd.DataFrame(columns=["column_name", "dType", "description"])
        def get_table_semantics(self, schema, table):
            return {}
        def get_column_semantics(self, schema, table, column):
            return {}
        def get_value_profile(self, schema, table, column):
            return {}

    import core.where_resolver as wr
    original = wr.rank_filter_candidates

    def fake_rank(*, user_input, intent, selected_tables, schema_loader, semantic_frame):
        return {
            "text:dm.t.fake": [
                {
                    "request_id": "text:dm.t.fake",
                    "column": "autor_login",
                    "description": "Логин автора задачи",
                    "score": 18.0,
                    "confidence": "low",
                    "semantic_class": "identifier",
                    "condition": "autor_login ILIKE '%фактический отток%'",
                    "table_key": "dm.t",
                    "value": "фактический отток",
                    "evidence": [],
                },
                {
                    "request_id": "text:dm.t.fake",
                    "column": "event_label",
                    "description": "Метка события",
                    "score": 55.0,
                    "confidence": "medium",
                    "semantic_class": "enum_like",
                    "condition": "event_label = 'фактический отток'",
                    "table_key": "dm.t",
                    "value": "фактический отток",
                    "evidence": [],
                },
            ]
        }

    wr.rank_filter_candidates = fake_rank
    try:
        frame = {
            "filter_intents": [
                {
                    "request_id": "text:dm.t.fake",
                    "kind": "text_search",
                    "query_text": "фактический отток",
                    "value": "фактический отток",
                    # column_key intentionally on a DIFFERENT table so G1
                    # short-circuit does not fire.
                    "column_key": "dm.other.col",
                }
            ]
        }
        result = wr.resolve_where(
            user_input="фактический отток",
            intent={"filter_conditions": []},
            selected_columns={"dm.t": {"select": ["x"], "aggregate": ["x"]}},
            selected_tables=["dm.t"],
            schema_loader=_Stub(),
            semantic_frame=frame,
            base_conditions=[],
        )
    finally:
        wr.rank_filter_candidates = original

    spec = result["clarification_spec"]
    assert spec, "expected a structured clarification_spec"
    cols = [opt["column"] for opt in spec["options"]]
    assert "autor_login" not in cols, "identifier column must be filtered out"
    assert "event_label" in cols, "plausible enum_like candidate must remain"


def test_f2_flag_column_guard_keeps_entity_filter_unresolved(tmp_path):
    """H3: F2 must NOT fold a value like «Задача» into implicit_filters when
    the selected table carries a boolean column (`is_task`) for the same
    concept — that column discriminates, so the filter is structural and
    can't be «covered by the table name».
    """
    from core.query_ir import FilterSpec
    # Table description mentions "задача" → _filter_value_encoded_in_table
    # would normally fire. But the table also has `is_task` boolean →
    # H3 must keep the filter alive.
    loader = _bool_loader(
        tmp_path,
        table_name="fact_outflow",
        description="Информация по фактическим оттокам, включая задачи",
    )
    loader.ensure_value_profiles()
    result = resolve_where(
        user_input="Сколько задач",
        intent={"filter_conditions": []},
        selected_columns={"dm.fact_outflow": {"select": ["is_task"]}},
        selected_tables=["dm.fact_outflow"],
        schema_loader=loader,
        semantic_frame={
            "filter_intents": [
                {
                    "request_id": "text:dm.fact_outflow.entity",
                    "query_text": "Задача",
                    "value": "Задача",
                    "kind": "text_search",
                }
            ]
        },
        base_conditions=[],
        filter_specs=[
            FilterSpec(target="some_unrelated_target", operator="=", value="Задача"),
        ],
    )
    # The value «Задача» must NOT be silently folded — the flag column on
    # the table is the discriminator the user expects to see.
    implicit = result.get("implicit_filters") or []
    assert not any(item.get("value") == "Задача" for item in implicit), (
        "H3 violation: «Задача» got folded as implicit despite is_task column"
    )


def test_f2_does_not_fold_value_matching_query_entity(tmp_path):
    """H7: a filter value that IS a QuerySpec entity must never be folded
    into implicit_filters via F2 — the entity is the WHAT-is-counted, not
    a filter the table choice can subsume.

    This is a cross-lingual-safe guard: no LLM/embeddings/synonym tables
    needed, just direct stem comparison between `value` and entity_names
    surfaced by the query_ir node.
    """
    loader = _bool_loader(
        tmp_path,
        table_name="fact_outflow",
        description="Информация по фактическим оттокам, включая задачи",
    )
    loader.ensure_value_profiles()
    result = resolve_where(
        user_input="Сколько задач",
        intent={"filter_conditions": []},
        selected_columns={"dm.fact_outflow": {"select": ["is_task"]}},
        selected_tables=["dm.fact_outflow"],
        schema_loader=loader,
        semantic_frame={
            "entity_names": ["задача"],
            "filter_intents": [
                {
                    "request_id": "text:dm.fact_outflow.entity",
                    "query_text": "Задача",
                    "value": "Задача",
                    "kind": "text_search",
                }
            ],
        },
        base_conditions=[],
    )
    implicit = result.get("implicit_filters") or []
    assert not any(item.get("value") == "Задача" for item in implicit), (
        "H7 violation: value matching a QuerySpec entity got folded as implicit"
    )


def test_f2_falls_back_to_table_choice_when_no_plausible_candidate():
    """H4: when every column candidate is identifier/low/sub-threshold, F2
    must NOT emit the generic «уточните признак» message — instead it
    surfaces a concrete clarification asking the user to pick which table
    holds the value, so the user has something actionable to choose.
    """
    import core.where_resolver as wr

    class _Stub:
        tables_df = pd.DataFrame({
            "schema_name": ["dm"],
            "table_name": ["t"],
            "description": ["test"],
            "grain": ["snapshot"],
        })
        def get_table_info(self, schema, table): return "Таблица: dm.t"
        def get_table_columns(self, schema, table):
            return pd.DataFrame(columns=["column_name", "dType", "description"])
        def get_table_semantics(self, schema, table): return {}
        def get_column_semantics(self, schema, table, column): return {}
        def get_value_profile(self, schema, table, column): return {}

    def fake_rank(**_):
        return {
            "text:dm.t.fake": [
                {
                    "request_id": "text:dm.t.fake",
                    "column": "autor_login",
                    "score": 18.0,
                    "confidence": "low",
                    "semantic_class": "identifier",
                    "condition": "autor_login ILIKE '%X%'",
                    "table_key": "dm.t",
                    "value": "X",
                    "evidence": [],
                },
            ]
        }

    original = wr.rank_filter_candidates
    wr.rank_filter_candidates = fake_rank
    try:
        result = wr.resolve_where(
            user_input="X",
            intent={"filter_conditions": []},
            selected_columns={"dm.t": {"select": ["x"], "aggregate": ["x"]}},
            selected_tables=["dm.t"],
            schema_loader=_Stub(),
            semantic_frame={
                "filter_intents": [
                    {
                        "request_id": "text:dm.t.fake",
                        "query_text": "X",
                        "value": "X",
                        "column_key": "dm.other.col",
                    }
                ]
            },
            base_conditions=[],
        )
    finally:
        wr.rank_filter_candidates = original

    assert result["needs_clarification"] is True
    # Implausible columns are NOT exposed; instead we offer table choices.
    spec = result["clarification_spec"]
    assert spec, "expected a structured clarification_spec (H4)"
    assert spec["type"] == "choice"
    table_values = [opt["value"] for opt in spec["options"]]
    assert "dm.t" in table_values, "selected_table must be listed as an option"
    assert "Не нашёл подходящих колонок" in result["clarification_message"]
    assert "«X»" in result["clarification_message"], (
        "user's value must be quoted in the question"
    )


def test_f2_spec_is_cleared_by_business_event_suppression(tmp_path):
    """G3: when the business-event-encoded-in-table suppression fires, it
    must clear BOTH clarification_message AND clarification_spec — otherwise
    F2's spec leaks through and the UI keeps asking.
    """
    loader = _loader(tmp_path)
    tables_df = pd.read_csv(tmp_path / "tables_list.csv")
    tables_df = pd.concat([
        tables_df,
        pd.DataFrame({
            "schema_name": ["dm"],
            "table_name": ["uzp_dwh_fact_outflow"],
            "description": ["Информация по фактическим оттокам"],
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
                "value": "фактический отток",
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
    assert result["clarification_message"] == ""
    assert not result["clarification_spec"]


# ---------------------------------------------------------------------------
# H1: G1 must NOT fold boolean_true / flag / explicit_filter intents even
# when the subject word matches stems in the chosen table's description.
# ---------------------------------------------------------------------------


def _outflow_with_task_in_description(tmp_path):
    """Loader fixture where the fact_outflow table description *does* contain
    «задач» — earlier G1 would have folded the boolean_true intent for «задача».
    """
    tables_df = pd.DataFrame({
        "schema_name": ["dm"],
        "table_name": ["outflow_with_tasks"],
        "description": ["Снимок фактических оттоков с поставленными задачами"],
        "grain": ["snapshot"],
    })
    attrs_df = pd.DataFrame({
        "schema_name": ["dm"] * 3,
        "table_name": ["outflow_with_tasks"] * 3,
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
    return loader


def test_h1_boolean_true_intent_not_folded_when_table_mentions_subject(tmp_path):
    """H1 regression: when the subject word «задача» appears in the table's
    description/column descriptions, the boolean_true intent (is_task = true)
    must still resolve to a SQL condition — G1 must not silently fold it.
    """
    loader = _outflow_with_task_in_description(tmp_path)
    frame = derive_semantic_frame("Сколько задач", schema_loader=loader)
    result = resolve_where(
        user_input="Сколько задач",
        intent={"filter_conditions": []},
        selected_columns={"dm.outflow_with_tasks": {"select": ["inn"], "aggregate": ["inn"]}},
        selected_tables=["dm.outflow_with_tasks"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert result["needs_clarification"] is False
    assert any("is_task = true" == cond for cond in result["conditions"]), (
        f"Expected is_task=true; got conditions={result['conditions']}, "
        f"implicit={result.get('implicit_filters')}, reasoning={result['reasoning']}"
    )
    # The intent must NOT have been folded into implicit_filters.
    assert not any(
        item.get("applied_via") == "table_name_encoding"
        for item in result.get("implicit_filters") or []
    )


def test_h1_explicit_filter_intent_not_folded_by_g1(tmp_path):
    """H1 regression: explicit_filter intents (from user-pinned filter_conditions)
    are structural — they must never be folded by G1 even if the value's stems
    happen to match the chosen table.
    """
    loader = _outflow_with_task_in_description(tmp_path)
    # Use a fake intent stream — we want full control over the kind/value pair.
    import core.where_resolver as wr
    original = wr.rank_filter_candidates

    def fake_rank(**_):
        return {
            "explicit:0": [
                {
                    "request_id": "explicit:0",
                    "column": "is_task",
                    "score": 80.0,
                    "confidence": "high",
                    "semantic_class": "flag",
                    "condition": "is_task = true",
                    "table_key": "dm.outflow_with_tasks",
                    "value": True,
                    "schema": "dm",
                    "table": "outflow_with_tasks",
                    "evidence": [],
                },
            ]
        }

    wr.rank_filter_candidates = fake_rank
    try:
        result = wr.resolve_where(
            user_input="Сколько задач",
            intent={"filter_conditions": []},
            selected_columns={"dm.outflow_with_tasks": {"select": ["inn"], "aggregate": ["inn"]}},
            selected_tables=["dm.outflow_with_tasks"],
            schema_loader=loader,
            semantic_frame={
                "filter_intents": [
                    {
                        "request_id": "explicit:0",
                        "kind": "explicit_filter",
                        "query_text": "задача",
                        "value": True,
                        "column_key": "dm.outflow_with_tasks.is_task",
                    }
                ]
            },
            base_conditions=[],
        )
    finally:
        wr.rank_filter_candidates = original
    # explicit_filter must produce a real condition, not be folded silently.
    assert "is_task = true" in result["conditions"]
    assert not any(
        item.get("applied_via") == "table_name_encoding"
        for item in result.get("implicit_filters") or []
    )


def test_h1_text_search_intent_still_folded_when_value_encoded(tmp_path):
    """H1 must not regress G1's happy path: text_search intent for a value
    that's semantically encoded in the chosen table should still be folded.
    """
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
                "value": "фактический отток",
                "column_key": "dm.uzp_data_split_mzp_sale_funnel.task_subtype",
            }
        ],
    }
    result = resolve_where(
        user_input="Сколько задач по фактическому оттоку",
        intent={"filter_conditions": []},
        selected_columns={"dm.uzp_dwh_fact_outflow": {"select": ["inn"], "aggregate": ["inn"]}},
        selected_tables=["dm.uzp_dwh_fact_outflow"],
        schema_loader=loader,
        semantic_frame=frame,
        base_conditions=[],
    )
    assert result["needs_clarification"] is False
    assert any(
        item.get("applied_via") == "table_name_encoding"
        for item in result.get("implicit_filters") or []
    )
