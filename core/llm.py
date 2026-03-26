"""GigaChat клиент с rate-limit и retry логикой."""

import os
import time
import logging
from typing import Any

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class RateLimitedLLM:
    """Обёртка над GigaChat с обязательной паузой между запросами и retry при ошибках."""

    MIN_INTERVAL: float = 5.0
    MAX_RETRIES: int = 5

    def __init__(self) -> None:
        """Инициализация GigaChat клиента из переменных окружения."""
        self._base_url = os.getenv("GIGACHAT_API_URL")
        self._access_token = os.getenv("JPY_API_TOKEN")
        self._model = "GigaChat-2-Max"
        self._llm = GigaChat(
            base_url=self._base_url,
            access_token=self._access_token,
            model=self._model,
            timeout=120,
        )
        self._last_call_time: float = 0.0
        logger.info("RateLimitedLLM инициализирован (model=%s)", self._model)

    def _get_llm(self, temperature: float | None = None) -> GigaChat:
        """Получить экземпляр GigaChat с указанной temperature.

        Args:
            temperature: Температура генерации (0.0-1.0). None — default модели.

        Returns:
            GigaChat клиент.
        """
        if temperature is None:
            return self._llm
        return GigaChat(
            base_url=self._base_url,
            access_token=self._access_token,
            model=self._model,
            timeout=120,
            temperature=temperature,
        )

    def _wait_for_rate_limit(self) -> None:
        """Ожидание до следующего разрешённого времени запроса."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.MIN_INTERVAL:
            wait = self.MIN_INTERVAL - elapsed
            logger.debug("Rate-limit: ожидание %.1f сек", wait)
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
                self._last_call_time = time.time()
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
