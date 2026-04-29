"""Тесты кэша инстансов GigaChat и глобального rate-limit (Direction 6.4)."""

from __future__ import annotations

import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def _ensure_mock_modules():
    for mod_name in (
        "langchain_gigachat",
        "langchain_gigachat.chat_models",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
    if "langchain_core" not in sys.modules:
        core_module = ModuleType("langchain_core")
        messages_module = ModuleType("langchain_core.messages")
        messages_module.BaseMessage = object
        messages_module.HumanMessage = type("HumanMessage", (), {"__init__": lambda s, **k: None})
        messages_module.SystemMessage = type("SystemMessage", (), {"__init__": lambda s, **k: None})
        core_module.messages = messages_module
        sys.modules["langchain_core"] = core_module
        sys.modules["langchain_core.messages"] = messages_module


_ensure_mock_modules()


@pytest.fixture
def rllm():
    """RateLimitedLLM с мок-инстансом GigaChat, чтобы не дёргать сеть."""
    from core import llm as llm_module

    class _FakeGiga:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, messages):
            class _R:
                content = "ok"
            return _R()

    with patch.object(llm_module, "GigaChat", _FakeGiga):
        instance = llm_module.RateLimitedLLM()
        # сброс глобального счётчика между тестами
        llm_module.RateLimitedLLM._global_last_call_time = 0.0
        yield instance


class TestLLMCache:
    def test_no_cache_growth_beyond_limit(self, rllm):
        for t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            rllm._get_llm(temperature=t)
        assert len(rllm._llm_cache) <= rllm.MAX_CACHED_TEMPERATURES

    def test_lru_eviction_order(self, rllm):
        # заполняем ровно до лимита
        rllm._get_llm(temperature=0.1)
        rllm._get_llm(temperature=0.2)
        rllm._get_llm(temperature=0.3)
        assert set(rllm._llm_cache.keys()) == {0.1, 0.2, 0.3}
        # переиспользуем 0.1 — он становится most-recent
        rllm._get_llm(temperature=0.1)
        # добавляем новый → выселяется самый старый (0.2)
        rllm._get_llm(temperature=0.4)
        assert 0.2 not in rllm._llm_cache
        assert 0.1 in rllm._llm_cache
        assert 0.4 in rllm._llm_cache

    def test_none_temperature_uses_default(self, rllm):
        default = rllm._get_llm(None)
        assert default is rllm._llm
        assert len(rllm._llm_cache) == 0

    def test_invoke_logs_full_response(self, rllm, caplog):
        import logging

        caplog.set_level(logging.INFO, logger="core.llm")

        assert rllm.invoke("prompt") == "ok"

        messages = [rec.getMessage() for rec in caplog.records]
        assert "LLM полный ответ:\nok" in messages


class TestGlobalRateLimit:
    def test_global_last_call_shared(self):
        """Несколько инстансов разделяют _global_last_call_time."""
        from core import llm as llm_module

        class _FakeGiga:
            def __init__(self, **kwargs):
                pass

        with patch.object(llm_module, "GigaChat", _FakeGiga):
            llm_module.RateLimitedLLM._global_last_call_time = 0.0
            a = llm_module.RateLimitedLLM()
            b = llm_module.RateLimitedLLM()
            # пишем через класс — оба инстанса видят
            marker = time.time()
            llm_module.RateLimitedLLM._global_last_call_time = marker
            assert a._global_last_call_time == marker
            assert b._global_last_call_time == marker

    def test_wait_honours_min_interval(self):
        """_wait_for_rate_limit спит если прошло меньше MIN_INTERVAL."""
        from core import llm as llm_module

        class _FakeGiga:
            def __init__(self, **kwargs):
                pass

        with patch.object(llm_module, "GigaChat", _FakeGiga):
            rllm = llm_module.RateLimitedLLM()
            # эмулируем недавний вызов
            llm_module.RateLimitedLLM._global_last_call_time = time.time() - 0.1
            with patch("core.llm.time.sleep") as mock_sleep:
                rllm._wait_for_rate_limit()
                assert mock_sleep.called
                # должны ждать ~ MIN_INTERVAL - 0.1 сек
                wait_arg = mock_sleep.call_args.args[0]
                assert wait_arg > 0
                assert wait_arg <= rllm.MIN_INTERVAL

    def test_no_wait_when_interval_exceeded(self):
        from core import llm as llm_module

        class _FakeGiga:
            def __init__(self, **kwargs):
                pass

        with patch.object(llm_module, "GigaChat", _FakeGiga):
            rllm = llm_module.RateLimitedLLM()
            llm_module.RateLimitedLLM._global_last_call_time = (
                time.time() - rllm.MIN_INTERVAL - 5
            )
            with patch("core.llm.time.sleep") as mock_sleep:
                rllm._wait_for_rate_limit()
                assert not mock_sleep.called
