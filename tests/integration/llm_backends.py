"""Альтернативные LLM-бэкенды для тестов (drop-in замена RateLimitedLLM).

Прод-код в [core/llm.py] не меняется — фикстура `real_llm` в conftest
выбирает один из бэкендов по env-переменной `TEST_LLM_BACKEND`:

    deepseek    (дефолт)   — DeepSeek API, OpenAI-совместимый, дёшево
    gigachat               — реальный RateLimitedLLM (для финальной верификации
                              на той же модели, что и прод)

Любой класс из этого модуля имеет совместимый с `RateLimitedLLM` интерфейс:

    .invoke(prompt: str | list[BaseMessage], temperature: float | None) -> str
    .invoke_with_system(system: str, user: str, temperature: float | None) -> str
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Iterable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


def _msg_to_openai_dict(m: BaseMessage) -> dict[str, str]:
    """LangChain BaseMessage → OpenAI Chat-completions dict."""
    if isinstance(m, SystemMessage):
        role = "system"
    elif isinstance(m, AIMessage):
        role = "assistant"
    elif isinstance(m, HumanMessage):
        role = "user"
    else:
        role = "user"
    content = m.content if isinstance(m.content, str) else str(m.content)
    return {"role": role, "content": content}


class DeepseekLLM:
    """Совместимая с `RateLimitedLLM` обёртка над DeepSeek (OpenAI API).

    Те же методы и сигнатуры — узлы графа подменяются без правок.

    Env:
        DEEPSEEK_API_URL    base URL (default https://api.deepseek.com/v1)
        DEEPSEEK_API_KEY    Bearer-ключ
        DEEPSEEK_MODEL      имя модели (default deepseek-v4-flash)
    """

    DEFAULT_MODEL = "deepseek-v4-flash"
    # У DeepSeek нет такого жёсткого rate-limit как у GigaChat. Держим мягкое
    # окно (0.5с) на случай аномалий и из-за того, что параллельных вызовов из
    # одного графа на одну сессию мы и не делаем.
    MIN_INTERVAL: float = 0.0
    MAX_RETRIES: int = 4

    _global_last_call_time: float = 0.0
    _global_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY не задан — для DeepSeek-бэкенда нужен токен."
            )
        base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
        self._model = os.getenv("DEEPSEEK_MODEL", self.DEFAULT_MODEL)
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        logger.info("DeepseekLLM инициализирован (model=%s, base=%s)", self._model, base_url)

    # ---- совместимость с RateLimitedLLM ----

    def _wait_for_rate_limit(self) -> None:
        if self.MIN_INTERVAL <= 0:
            return
        with DeepseekLLM._global_lock:
            elapsed = time.time() - DeepseekLLM._global_last_call_time
            if elapsed < self.MIN_INTERVAL:
                time.sleep(self.MIN_INTERVAL - elapsed)

    def _build_messages(
        self, prompt: str | Iterable[BaseMessage]
    ) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        return [_msg_to_openai_dict(m) for m in prompt]

    def invoke(
        self,
        prompt: str | list[BaseMessage],
        temperature: float | None = None,
    ) -> str:
        """Совместимо с RateLimitedLLM.invoke."""
        messages = self._build_messages(prompt)
        # Для воспроизводимости интеграционных тестов: temperature=0 (если узел
        # графа не передал свою) и фиксированный seed. DeepSeek принимает оба
        # параметра через OpenAI-совместимый API. seed гарантирует одинаковый
        # вывод между запусками для тех узлов, где LLM детерминированно
        # форматирует QuerySpec/SQL.
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.0,
            "seed": 42,
        }

        last_err: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                DeepseekLLM._global_last_call_time = time.time()
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                logger.info("LLM ответ получен (попытка %d)", attempt)
                logger.info("LLM полный ответ:\n%s", content)
                return content
            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM ошибка (попытка %d/%d): %s",
                    attempt, self.MAX_RETRIES, e,
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(min(2 ** (attempt - 1), 8))

        err_msg = (
            f"Не удалось получить ответ от DeepSeek после {self.MAX_RETRIES} попыток. "
            f"Последняя ошибка: {last_err}"
        )
        logger.error(err_msg)
        return err_msg

    def invoke_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        """Совместимо с RateLimitedLLM.invoke_with_system."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return self.invoke(messages, temperature=temperature)


def build_llm() -> Any:
    """Выбрать LLM-бэкенд по env `TEST_LLM_BACKEND`.

    Returns:
        Объект с методами .invoke()/.invoke_with_system() — совместимый с
        тем, что ожидают узлы графа.

    Raises:
        RuntimeError: если выбран бэкенд, для которого нет токена.
    """
    backend = os.getenv("TEST_LLM_BACKEND", "deepseek").strip().lower()
    if backend in ("gigachat", "giga"):
        # Реальный RateLimitedLLM — для финального прогона на прод-модели.
        from core.llm import RateLimitedLLM
        return RateLimitedLLM()
    if backend in ("deepseek", "ds"):
        return DeepseekLLM()
    raise RuntimeError(
        f"Неизвестный TEST_LLM_BACKEND={backend!r}. Доступно: deepseek | gigachat"
    )
