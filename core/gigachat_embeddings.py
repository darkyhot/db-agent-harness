"""GigaChat Embeddings с rate-limit и батч-логикой.

Отдельный клиент — НЕ то же самое, что RateLimitedLLM (текстовый GigaChat).
Rate-limit для эмбеддингов: 6 секунд между батч-вызовами (не 5, как у текстового).
Батч: до MAX_BATCH_SIZE_PARTS текстов или до MAX_BATCH_SIZE_CHARS символов суммарно.
"""

import os
from time import perf_counter, sleep
from typing import List

from langchain_gigachat.embeddings import GigaChatEmbeddings

GIGA_EMBED_DELAY: float = 6.0
MAX_BATCH_SIZE_CHARS: int = 1_000_000
MAX_BATCH_SIZE_PARTS: int = 90

# Глобальный таймер последнего вызова (разделяется между инстансами модуля)
_GIGA_EMBED_LAST_INVOKE: float = 0.0


class GigaChatEmbeddingsDelayed(GigaChatEmbeddings):
    """GigaChatEmbeddings с автоматическим rate-limit и батчингом."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Эмбеддит тексты батчами с соблюдением 6-секундного rate-limit.

        Args:
            texts: Список строк для эмбеддинга.

        Returns:
            Список векторов (float) в том же порядке.
        """
        global _GIGA_EMBED_LAST_INVOKE

        result: List[List[float]] = []
        local_texts: List[str] = []
        size: int = 0
        embed_kwargs: dict = {}
        if self.model is not None:
            embed_kwargs["model"] = self.model

        def _flush() -> None:
            nonlocal local_texts, size
            global _GIGA_EMBED_LAST_INVOKE

            if not local_texts:
                return

            elapsed = perf_counter() - _GIGA_EMBED_LAST_INVOKE
            if elapsed < GIGA_EMBED_DELAY:
                sleep(GIGA_EMBED_DELAY - elapsed)
            _GIGA_EMBED_LAST_INVOKE = perf_counter()

            response = self._client.embeddings(texts=local_texts, **embed_kwargs)
            for item in response.data:
                result.append(item.embedding)

            local_texts.clear()
            size = 0

        for text in texts:
            local_texts.append(text)
            size += len(text)
            if size > MAX_BATCH_SIZE_CHARS or len(local_texts) >= MAX_BATCH_SIZE_PARTS:
                _flush()
        _flush()

        return result


def build_embedder(
    base_url: str | None = None,
    access_token: str | None = None,
    *,
    verify_ssl: bool = False,
) -> GigaChatEmbeddingsDelayed:
    """Фабрика эмбеддера с конфигурацией из параметров или env-переменных.

    Args:
        base_url: URL GigaChat API. По умолчанию — из GIGACHAT_API_URL.
        access_token: Токен доступа. По умолчанию — из JPY_API_TOKEN.
        verify_ssl: Проверять ли SSL-сертификат сервера.

    Returns:
        Готовый экземпляр GigaChatEmbeddingsDelayed.
    """
    return GigaChatEmbeddingsDelayed(
        base_url=base_url or os.getenv("GIGACHAT_API_URL"),
        access_token=access_token or os.getenv("JPY_API_TOKEN"),
        verify_ssl_certs=verify_ssl,
    )
