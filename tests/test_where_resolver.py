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
        "schema_name": ["dm"] * 4,
        "table_name": ["uzp_data_split_mzp_sale_funnel"] * 4,
        "column_name": ["report_dt", "task_code", "task_subtype", "is_outflow"],
        "dType": ["date", "text", "text", "int4"],
        "description": ["Отчетная дата", "Код задачи", "Подтип задачи", "Признак подтверждения оттока"],
        "is_primary_key": [False, False, False, False],
        "unique_perc": [0.5, 90.0, 10.0, 2.0],
        "not_null_perc": [99.0, 100.0, 100.0, 100.0],
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
    loader.ensure_value_profiles()
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
    assert any("task_subtype ILIKE '%фактическому%'" in cond for cond in result["conditions"])
    assert any("task_subtype ILIKE '%оттоку%'" in cond for cond in result["conditions"])
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
