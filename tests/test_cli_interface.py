from cli.interface import _build_augmented_clarification_input, _parse_deep_analysis_args
from cli.interface import CLIInterface
from core.deep_analysis import AnalysisMode


def test_parse_slash_command():
    command, args = CLIInterface._parse_command("/config params")
    assert command == "config"
    assert args == ["params"]


def test_parse_legacy_command():
    command, args = CLIInterface._parse_command("exit")
    assert command == "exit"
    assert args == []


def test_parse_plain_query_returns_none():
    command, args = CLIInterface._parse_command("покажи продажи по регионам")
    assert command is None
    assert args == []


def test_build_augmented_clarification_input_keeps_question_and_answer():
    result = _build_augmented_clarification_input(
        "Сколько задач по фактическому оттоку поставили в феврале",
        "Вы имели в виду февраль 2026 года?",
        "да",
    )
    assert result == (
        "Сколько задач по фактическому оттоку поставили в феврале\n"
        "Вопрос уточнения: Вы имели в виду февраль 2026 года?\n"
        "Уточнение пользователя: да"
    )


def test_build_augmented_clarification_input_skips_empty_question():
    result = _build_augmented_clarification_input(
        "Покажи задачи",
        "",
        "да",
    )
    assert result == "Покажи задачи\nУточнение пользователя: да"


def test_parse_command_supports_metadata_and_refresh():
    assert CLIInterface._parse_command("/metadata add dm.sales,dm.clients") == (
        "metadata",
        ["add", "dm.sales,dm.clients"],
    )
    assert CLIInterface._parse_command("/refresh metadata") == (
        "refresh",
        ["metadata"],
    )


def test_parse_deep_analysis_args_collects_where_equals_tokens():
    mode, table_ref, where_clause, hypothesis_text = _parse_deep_analysis_args([
        "dm.sales",
        "--mode=deep",
        "--where=report_dt",
        ">=",
        "'2026-01-01'",
    ])

    assert mode == AnalysisMode.DEEP
    assert table_ref == "dm.sales"
    assert where_clause == "report_dt >= '2026-01-01'"
    assert hypothesis_text == ""


def test_parse_deep_analysis_args_collects_where_space_tokens():
    mode, table_ref, where_clause, hypothesis_text = _parse_deep_analysis_args([
        "dm.sales",
        "--where",
        "inn",
        "IN",
        "('7707083893','7728168971')",
    ])

    assert mode == AnalysisMode.FAST
    assert table_ref == "dm.sales"
    assert where_clause == "inn IN ('7707083893','7728168971')"
    assert hypothesis_text == ""


def test_parse_deep_analysis_args_keeps_hypothesis_without_where():
    _, table_ref, where_clause, hypothesis_text = _parse_deep_analysis_args([
        "dm.sales",
        "проверь",
        "сезонность",
    ])

    assert table_ref == "dm.sales"
    assert where_clause is None
    assert hypothesis_text == "проверь сезонность"
