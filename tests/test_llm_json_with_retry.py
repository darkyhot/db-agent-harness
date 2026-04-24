"""Тесты на BaseNodeMixin._llm_json_with_retry и _try_parse_json.

Проверяет: happy path, один retry при невалидном ответе, запись в memory
при повторной неудаче.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


def _ensure_mock_modules():
    for mod_name in (
        "langchain_gigachat",
        "langchain_gigachat.chat_models",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()


_ensure_mock_modules()

from graph.nodes.common import BaseNodeMixin  # noqa: E402


def _make_mixin(llm_returns: list[str]):
    """Собрать лёгкий объект BaseNodeMixin без реальных зависимостей."""
    responses = iter(llm_returns)

    llm = SimpleNamespace(
        invoke_with_system=lambda system, user, temperature=0.0: next(responses),
    )

    memory_state: dict[str, str] = {}

    def _set_memory(key: str, value: str) -> None:
        memory_state[key] = value

    def _get_memory(key: str) -> str | None:
        return memory_state.get(key)

    memory = SimpleNamespace(
        set_memory=_set_memory,
        get_memory=_get_memory,
    )

    obj = BaseNodeMixin.__new__(BaseNodeMixin)
    obj.llm = llm
    obj.memory = memory
    return obj, memory_state


class TestTryParseJson:
    def test_plain_json_object(self):
        result = BaseNodeMixin._try_parse_json('{"a": 1}')
        assert result == {"a": 1}

    def test_json_with_markdown_wrapper(self):
        result = BaseNodeMixin._try_parse_json('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_json_with_surrounding_text(self):
        result = BaseNodeMixin._try_parse_json('Вот ответ: {"a": 1} — готово')
        assert result == {"a": 1}

    def test_trailing_comma_repaired(self):
        result = BaseNodeMixin._try_parse_json('{"a": 1,}')
        assert result == {"a": 1}

    def test_array_expected(self):
        result = BaseNodeMixin._try_parse_json("[1, 2, 3]", expect="array")
        assert result == [1, 2, 3]

    def test_array_with_surrounding_text(self):
        result = BaseNodeMixin._try_parse_json("prefix [1,2] suffix", expect="array")
        assert result == [1, 2]

    def test_object_when_array_expected_returns_none(self):
        result = BaseNodeMixin._try_parse_json('{"a": 1}', expect="array")
        assert result is None

    def test_invalid_returns_none(self):
        result = BaseNodeMixin._try_parse_json("это просто текст без json")
        assert result is None


class TestLlmJsonWithRetry:
    def test_happy_path_no_retry(self):
        obj, mem = _make_mixin(['{"column": "inn"}'])
        result = obj._llm_json_with_retry("sys", "usr", failure_tag="count_id")
        assert result == {"column": "inn"}
        assert "json_parse_failures" not in mem

    def test_retry_recovers(self):
        obj, mem = _make_mixin(["not a json", '{"column": "inn"}'])
        result = obj._llm_json_with_retry("sys", "usr", failure_tag="count_id")
        assert result == {"column": "inn"}
        assert "json_parse_failures" not in mem

    def test_retry_fails_logs_to_memory(self):
        obj, mem = _make_mixin(["garbage 1", "garbage 2"])
        result = obj._llm_json_with_retry("sys", "usr", failure_tag="count_id")
        assert result is None
        assert "json_parse_failures" in mem
        log = json.loads(mem["json_parse_failures"])
        assert len(log) == 1
        assert log[0]["tag"] == "count_id"
        assert "garbage 1" in log[0]["primary"]
        assert "garbage 2" in log[0]["secondary"]

    def test_failures_append_to_existing_log(self):
        obj, mem = _make_mixin(["bad", "bad2"])
        mem["json_parse_failures"] = json.dumps([{"tag": "older"}])
        obj._llm_json_with_retry("sys", "usr", failure_tag="count_id")
        log = json.loads(mem["json_parse_failures"])
        assert len(log) == 2
        assert log[0]["tag"] == "older"
        assert log[1]["tag"] == "count_id"

    def test_array_expect(self):
        obj, _ = _make_mixin(["[1,2,3]"])
        result = obj._llm_json_with_retry(
            "sys", "usr", failure_tag="t", expect="array",
        )
        assert result == [1, 2, 3]
