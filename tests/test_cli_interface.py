from cli.interface import _build_augmented_clarification_input
from cli.interface import CLIInterface


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
