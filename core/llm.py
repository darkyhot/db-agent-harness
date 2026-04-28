"""GigaChat клиент с rate-limit и retry логикой."""

import json
import os
import time
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "GigaChat-2-Max"
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def _resolve_model() -> str:
    """Определить модель GigaChat: env GIGACHAT_MODEL → config.json → default."""
    env_model = os.getenv("GIGACHAT_MODEL")
    if env_model:
        return env_model
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        model = cfg.get("llm_model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return DEFAULT_MODEL


class RateLimitedLLM:
    """Обёртка над GigaChat с обязательной паузой между запросами и retry при ошибках.

    Rate-limit держится ГЛОБАЛЬНЫМ (на уровне класса) — чтобы несколько
    инстансов RateLimitedLLM в одном процессе не могли параллельно
    превысить лимит GigaChat API. Кэш инстансов GigaChat ограничен
    MAX_CACHED_TEMPERATURES (LRU-эвикция).
    """

    MIN_INTERVAL: float = 5.0
    MAX_RETRIES: int = 5
    MAX_CACHED_TEMPERATURES: int = 3

    # Глобальные (на уровень класса) поля rate-limit — чтобы несколько
    # инстансов разделяли один таймер.
    _global_last_call_time: float = 0.0
    _global_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Инициализация GigaChat клиента из переменных окружения."""
        self._base_url = os.getenv("GIGACHAT_API_URL")
        self._access_token = os.getenv("JPY_API_TOKEN")
        self._model = _resolve_model()
        self._llm = GigaChat(
            base_url=self._base_url,
            access_token=self._access_token,
            model=self._model,
            timeout=120,
        )
        self._llm_cache: "OrderedDict[float, GigaChat]" = OrderedDict()
        logger.info("RateLimitedLLM инициализирован (model=%s)", self._model)

    def _get_llm(self, temperature: float | None = None) -> GigaChat:
        """Получить экземпляр GigaChat с указанной temperature.

        Кеширует до MAX_CACHED_TEMPERATURES инстансов по temperature,
        с LRU-эвикцией чтобы не раздувать память при свободном выборе
        температур из узлов графа.

        Args:
            temperature: Температура генерации (0.0-1.0). None — default модели.

        Returns:
            GigaChat клиент.
        """
        if temperature is None:
            return self._llm
        if temperature in self._llm_cache:
            self._llm_cache.move_to_end(temperature)
            return self._llm_cache[temperature]
        llm = GigaChat(
            base_url=self._base_url,
            access_token=self._access_token,
            model=self._model,
            timeout=120,
            temperature=temperature,
        )
        self._llm_cache[temperature] = llm
        while len(self._llm_cache) > self.MAX_CACHED_TEMPERATURES:
            evicted_temp, _ = self._llm_cache.popitem(last=False)
            logger.debug("LLM cache LRU-эвикция: temperature=%s", evicted_temp)
        return llm

    def _wait_for_rate_limit(self) -> None:
        """Ожидание до следующего разрешённого времени запроса (глобально)."""
        with RateLimitedLLM._global_lock:
            elapsed = time.time() - RateLimitedLLM._global_last_call_time
            if elapsed < self.MIN_INTERVAL:
                wait = self.MIN_INTERVAL - elapsed
                logger.debug("Rate-limit (global): ожидание %.1f сек", wait)
                time.sleep(wait)

    def invoke(
        self,
        prompt: str | list[BaseMessage],
        temperature: float | None = None,
    ) -> str:
        """Отправить запрос к LLM с retry логикой.

        Args:
            prompt: Текстовый промпт или список сообщений LangChain.
            temperature: Температура генерации (0.0-1.0). None — default модели.

        Returns:
            Текст ответа модели. При исчерпании попыток — сообщение об ошибке.
        """
        if isinstance(prompt, str):
            messages = [HumanMessage(content=prompt)]
        else:
            messages = prompt

        llm = self._get_llm(temperature)

        for attempt in range(1, self.MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                RateLimitedLLM._global_last_call_time = time.time()
                response = llm.invoke(messages)
                logger.info("LLM ответ получен (попытка %d)", attempt)
                return response.content
            except Exception as e:
                logger.warning(
                    "LLM ошибка (попытка %d/%d): %s", attempt, self.MAX_RETRIES, e
                )
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.MIN_INTERVAL)

        error_msg = (
            f"Не удалось получить ответ от LLM после {self.MAX_RETRIES} попыток. "
            "Проверьте подключение и переменные окружения."
        )
        logger.error(error_msg)
        return error_msg

    def invoke_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
    ) -> str:
        """Запрос с системным промптом.

        GigaChat лучше следует инструкциям из SystemMessage,
        поэтому это предпочтительный метод вызова.

        Args:
            system_prompt: Системное сообщение (роль и правила).
            user_prompt: Пользовательский запрос (данные и задача).
            temperature: Температура генерации (0.0-1.0). None — default модели.

        Returns:
            Текст ответа модели.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        return self.invoke(messages, temperature=temperature)
