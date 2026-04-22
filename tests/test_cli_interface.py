"""Тесты CLI helper-методов без тяжёлой инициализации агента."""

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
