"""Семантический индекс колонок на базе GigaChat Embeddings.

Строится один раз при старте (или загружается из кэша data_for_agent/embeddings.npz).
Предоставляет косинусное сходство для семантического поиска колонок и таблиц.

Правила:
- embeddings.npz НЕ коммитится в git (добавлен в .gitignore).
- Если эмбеддер недоступен — все методы поиска возвращают пустые результаты.
- GigaChat Embeddings rate-limit: 6 сек между батч-вызовами (отдельный клиент).
"""

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from core.gigachat_embeddings import GigaChatEmbeddingsDelayed

logger = logging.getLogger(__name__)

_EMBEDDINGS_FILE = "embeddings.npz"
_LRU_CAPACITY = 256


class _LRUCache:
    """Простой LRU-кэш фиксированного размера."""

    def __init__(self, capacity: int = _LRU_CAPACITY) -> None:
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._capacity = capacity

    def get(self, key: str) -> list[float] | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: list[float]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._capacity:
                self._cache.popitem(last=False)
        self._cache[key] = value


class SemanticIndex:
    """Индекс эмбеддингов колонок БД с кэшем на диске и LRU для query.

    Структура npz-кэша:
        keys   — dtype=object, shape=(N,): строки "schema.table.column"
        vectors — dtype=float32, shape=(N, dim): векторы эмбеддингов
    """

    def __init__(
        self,
        embedder: "GigaChatEmbeddingsDelayed",
        data_dir: Path,
        *,
        attrs_df: Any = None,
    ) -> None:
        """Инициализация индекса.

        Args:
            embedder: Клиент GigaChat Embeddings с rate-limit.
            data_dir: Директория с CSV-файлами и кэшем embeddings.npz.
            attrs_df: pandas DataFrame с атрибутами (schema_name, table_name,
                      column_name, description). Если не передан — индекс
                      не строится (только загрузка из npz если есть).
        """
        self._embedder = embedder
        self._data_dir = data_dir
        self._npz_path = data_dir / _EMBEDDINGS_FILE
        self._query_cache = _LRUCache(_LRU_CAPACITY)

        # Индекс в памяти
        self._keys: list[str] = []
        self._vectors: np.ndarray | None = None  # shape (N, dim)

        self._load_or_build(attrs_df)

    # ------------------------------------------------------------------
    # Построение / загрузка
    # ------------------------------------------------------------------

    def _is_cache_fresh(self) -> bool:
        """Вернуть True если npz существует и новее CSV-источников."""
        if not self._npz_path.exists():
            return False
        npz_mtime = self._npz_path.stat().st_mtime
        for name in ("attr_list.csv", "attrs_list.csv", "tables_list.csv"):
            csv_path = self._data_dir / name
            if csv_path.exists() and csv_path.stat().st_mtime > npz_mtime:
                return False
        return True

    def _load_from_cache(self) -> bool:
        """Загрузить индекс из npz. Вернуть True при успехе."""
        try:
            data = np.load(str(self._npz_path), allow_pickle=True)
            keys = data["keys"].tolist()
            vectors = data["vectors"].astype(np.float32)
            if len(keys) != vectors.shape[0]:
                logger.warning("semantic_index: npz повреждён (размеры не совпадают)")
                return False
            self._keys = keys
            self._vectors = vectors
            logger.info("semantic_index: загружен кэш, N=%d колонок", len(self._keys))
            return True
        except Exception as e:
            logger.warning("semantic_index: ошибка загрузки npz: %s", e)
            return False

    def _build_from_df(self, attrs_df: Any) -> None:
        """Построить индекс из DataFrame атрибутов, сохранить в npz."""
        try:
            rows = attrs_df[
                ["schema_name", "table_name", "column_name", "description"]
            ].dropna(subset=["column_name"])
        except Exception as e:
            logger.warning("semantic_index: ошибка чтения attrs_df: %s", e)
            return

        texts: list[str] = []
        keys: list[str] = []
        for _, row in rows.iterrows():
            schema = str(row.get("schema_name") or "").strip()
            table = str(row.get("table_name") or "").strip()
            col = str(row.get("column_name") or "").strip()
            desc = str(row.get("description") or "").strip()
            if not (schema and table and col):
                continue
            key = f"{schema}.{table}.{col}".lower()
            text = f"{schema}.{table}.{col} {desc}".strip()
            keys.append(key)
            texts.append(text)

        if not texts:
            logger.warning("semantic_index: нет данных для построения индекса")
            return

        logger.info("semantic_index: строим индекс для %d колонок...", len(texts))
        t0 = time.monotonic()
        try:
            vectors = self._embedder.embed_documents(texts)
        except Exception as e:
            logger.error("semantic_index: ошибка эмбеддингов: %s", e)
            return
        elapsed = time.monotonic() - t0
        logger.info(
            "semantic_index: построен индекс, N=%d колонок, time=%.1fs",
            len(texts), elapsed,
        )

        self._keys = keys
        self._vectors = np.array(vectors, dtype=np.float32)

        try:
            np.savez_compressed(
                str(self._npz_path),
                keys=np.array(keys, dtype=object),
                vectors=self._vectors,
            )
            logger.info("semantic_index: кэш сохранён → %s", self._npz_path)
        except Exception as e:
            logger.warning("semantic_index: ошибка сохранения npz: %s", e)

    def _load_or_build(self, attrs_df: Any) -> None:
        """Загрузить из кэша или построить индекс."""
        if self._is_cache_fresh():
            if self._load_from_cache():
                return

        if attrs_df is not None:
            self._build_from_df(attrs_df)
        else:
            logger.info(
                "semantic_index: кэш устарел или отсутствует, "
                "attrs_df не передан — индекс пуст"
            )

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Индекс загружен и готов к поиску."""
        return self._vectors is not None and len(self._keys) > 0

    def _embed_query(self, query: str) -> list[float] | None:
        """Эмбеддит query с LRU-кэшем."""
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        try:
            vecs = self._embedder.embed_documents([query])
            if vecs:
                self._query_cache.put(query, vecs[0])
                return vecs[0]
        except Exception as e:
            logger.warning("semantic_index: ошибка эмбеддинга query: %s", e)
        return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Косинусное сходство вектора a с матрицей b."""
        norm_a = np.linalg.norm(a)
        if norm_a == 0:
            return np.zeros(b.shape[0], dtype=np.float32)
        norm_b = np.linalg.norm(b, axis=1)
        norm_b = np.where(norm_b == 0, 1e-9, norm_b)
        return (b @ a) / (norm_b * norm_a)

    def similarity(
        self, query: str, candidate_keys: list[str]
    ) -> dict[str, float]:
        """Косинусное сходство query с заданным списком ключей.

        Args:
            query: Текстовый запрос пользователя.
            candidate_keys: Список ключей формата "schema.table.column".

        Returns:
            Словарь {key: score}. Недостающие/неизвестные ключи имеют score=0.0.
        """
        result = {k: 0.0 for k in candidate_keys}
        if not self.is_ready or not candidate_keys:
            return result

        q_vec = self._embed_query(query)
        if q_vec is None:
            return result

        key_set = {k.lower() for k in candidate_keys}
        indices = [i for i, k in enumerate(self._keys) if k in key_set]
        if not indices:
            return result

        sub_vectors = self._vectors[indices]
        q_arr = np.array(q_vec, dtype=np.float32)
        scores = self._cosine(q_arr, sub_vectors)
        for idx, score in zip(indices, scores.tolist()):
            result[self._keys[idx]] = float(score)
        return result

    def semantic_search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Семантический поиск: топ-K ключей по косинусному сходству.

        Args:
            query: Текстовый запрос.
            top_k: Максимальное количество результатов.

        Returns:
            Список (key, score) отсортированный по убыванию score.
        """
        if not self.is_ready:
            return []

        q_vec = self._embed_query(query)
        if q_vec is None:
            return []

        q_arr = np.array(q_vec, dtype=np.float32)
        scores = self._cosine(q_arr, self._vectors)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._keys[i], float(scores[i]))
            for i in top_indices
        ]
