"""Тесты для SemanticIndex, GigaChatEmbeddingsDelayed и semantic bonus в filter_ranking.

Все обращения к реальному GigaChat Embeddings API замокированы.
"""

import math
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile

import numpy as np
import pytest

from core.semantic_index import SemanticIndex, _LRUCache


# ─────────────────────────────────────────────────────────────────
# Вспомогательные утилиты
# ─────────────────────────────────────────────────────────────────

def _make_embedder(vectors: list[list[float]]) -> MagicMock:
    """Фабрика мок-эмбеддера, возвращающего заданные векторы по очереди."""
    embedder = MagicMock()
    embedder.model = None
    embedder.embed_documents.return_value = vectors
    return embedder


def _make_attrs_df(rows: list[tuple[str, str, str, str]]):
    """Создать минимальный DataFrame атрибутов."""
    import pandas as pd
    return pd.DataFrame(
        rows,
        columns=["schema_name", "table_name", "column_name", "description"],
    )


def _unit_vec(*components: float) -> list[float]:
    """Вернуть нормализованный вектор."""
    mag = math.sqrt(sum(c ** 2 for c in components))
    return [c / mag for c in components]


# ─────────────────────────────────────────────────────────────────
# Тест _LRUCache
# ─────────────────────────────────────────────────────────────────

def test_lru_cache_put_get():
    cache = _LRUCache(capacity=2)
    cache.put("a", [1.0, 2.0])
    assert cache.get("a") == [1.0, 2.0]
    assert cache.get("missing") is None


def test_lru_cache_evicts_oldest():
    cache = _LRUCache(capacity=2)
    cache.put("a", [1.0])
    cache.put("b", [2.0])
    cache.put("c", [3.0])  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == [2.0]
    assert cache.get("c") == [3.0]


def test_lru_cache_access_refreshes_order():
    cache = _LRUCache(capacity=2)
    cache.put("a", [1.0])
    cache.put("b", [2.0])
    cache.get("a")           # refresh "a"
    cache.put("c", [3.0])    # should evict "b", not "a"
    assert cache.get("a") == [1.0]
    assert cache.get("b") is None


# ─────────────────────────────────────────────────────────────────
# Тест SemanticIndex — построение из DataFrame
# ─────────────────────────────────────────────────────────────────

def test_build_index_from_df(tmp_path):
    attrs = _make_attrs_df([
        ("dm", "fact_churn", "reason_code", "код причины оттока"),
        ("dm", "dim_reason", "reason_name", "название причины"),
    ])
    vecs = [_unit_vec(1, 0, 0), _unit_vec(0, 1, 0)]
    embedder = _make_embedder(vecs)

    idx = SemanticIndex(embedder, tmp_path, attrs_df=attrs)

    assert idx.is_ready
    assert "dm.fact_churn.reason_code" in idx._keys
    assert "dm.dim_reason.reason_name" in idx._keys


def test_index_saved_to_npz(tmp_path):
    attrs = _make_attrs_df([
        ("dm", "fact_churn", "task_id", "id задачи"),
    ])
    embedder = _make_embedder([_unit_vec(1, 0)])
    idx = SemanticIndex(embedder, tmp_path, attrs_df=attrs)

    assert (tmp_path / "embeddings.npz").exists()


# ─────────────────────────────────────────────────────────────────
# Тест SemanticIndex — загрузка из кэша
# ─────────────────────────────────────────────────────────────────

def test_load_from_cache_skips_embedder(tmp_path):
    # Первый запуск — строим индекс
    attrs = _make_attrs_df([("dm", "t", "col", "desc")])
    embedder1 = _make_embedder([_unit_vec(1, 0)])
    idx1 = SemanticIndex(embedder1, tmp_path, attrs_df=attrs)

    # Второй запуск с тем же data_dir — должен загрузить из кэша
    embedder2 = _make_embedder([])
    idx2 = SemanticIndex(embedder2, tmp_path, attrs_df=attrs)

    embedder2.embed_documents.assert_not_called()
    assert idx2.is_ready


# ─────────────────────────────────────────────────────────────────
# Тест SemanticIndex.similarity — косинусное сходство
# ─────────────────────────────────────────────────────────────────

def test_similarity_returns_correct_scores(tmp_path):
    attrs = _make_attrs_df([
        ("dm", "t", "col_a", "описание A"),
        ("dm", "t", "col_b", "описание B"),
    ])
    vecs_build = [_unit_vec(1, 0), _unit_vec(0, 1)]
    embedder = _make_embedder(vecs_build)

    idx = SemanticIndex(embedder, tmp_path, attrs_df=attrs)

    # Запрос совпадает с col_a
    embedder.embed_documents.return_value = [_unit_vec(1, 0)]
    scores = idx.similarity("запрос к col_a", ["dm.t.col_a", "dm.t.col_b"])

    assert scores["dm.t.col_a"] > scores["dm.t.col_b"]
    assert abs(scores["dm.t.col_a"] - 1.0) < 0.01


def test_similarity_unknown_key_returns_zero(tmp_path):
    attrs = _make_attrs_df([("dm", "t", "col", "desc")])
    embedder = _make_embedder([_unit_vec(1, 0)])
    idx = SemanticIndex(embedder, tmp_path, attrs_df=attrs)

    embedder.embed_documents.return_value = [_unit_vec(1, 0)]
    scores = idx.similarity("query", ["nonexistent.key"])
    assert scores["nonexistent.key"] == 0.0


# ─────────────────────────────────────────────────────────────────
# Тест SemanticIndex.semantic_search — топ-K результаты
# ─────────────────────────────────────────────────────────────────

def test_semantic_search_top_k(tmp_path):
    attrs = _make_attrs_df([
        ("dm", "t", "col_a", ""),
        ("dm", "t", "col_b", ""),
        ("dm", "t", "col_c", ""),
    ])
    # col_a ближе к запросу, col_c дальше
    vecs_build = [_unit_vec(1, 0), _unit_vec(0.6, 0.8), _unit_vec(0, 1)]
    embedder = _make_embedder(vecs_build)
    idx = SemanticIndex(embedder, tmp_path, attrs_df=attrs)

    embedder.embed_documents.return_value = [_unit_vec(1, 0)]
    results = idx.semantic_search("query", top_k=2)

    assert len(results) == 2
    keys = [k for k, _ in results]
    assert keys[0] == "dm.t.col_a"


# ─────────────────────────────────────────────────────────────────
# Тест: query LRU-кэш — эмбеддинг не вызывается повторно
# ─────────────────────────────────────────────────────────────────

def test_query_lru_cache(tmp_path):
    attrs = _make_attrs_df([("dm", "t", "col", "desc")])
    embedder = _make_embedder([_unit_vec(1, 0)])
    idx = SemanticIndex(embedder, tmp_path, attrs_df=attrs)

    q_vec = [_unit_vec(1, 0)]
    embedder.embed_documents.return_value = q_vec

    idx.semantic_search("same_query", top_k=1)
    idx.semantic_search("same_query", top_k=1)  # Повторный вызов

    # embed_documents вызывался для построения индекса (1 батч) + 1 раз для query
    calls = embedder.embed_documents.call_count
    # Первый вызов — построение индекса (1 текст), второй — query embedding
    # Третий вызов НЕ должен был произойти (LRU кэш)
    assert calls <= 2


# ─────────────────────────────────────────────────────────────────
# Тест GigaChatEmbeddingsDelayed — batch flush и rate-limit
# ─────────────────────────────────────────────────────────────────

def test_batch_flush_at_max_parts():
    """Батч сбрасывается при достижении MAX_BATCH_SIZE_PARTS."""
    from core.gigachat_embeddings import GigaChatEmbeddingsDelayed, MAX_BATCH_SIZE_PARTS

    embedder = MagicMock(spec=GigaChatEmbeddingsDelayed)
    embedder.model = None

    call_batches: list[list[str]] = []

    def fake_client_embeddings(texts, **kwargs):
        call_batches.append(list(texts))
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1] * 3) for _ in texts]
        return mock_resp

    embedder._client = MagicMock()
    embedder._client.embeddings.side_effect = fake_client_embeddings

    texts = [f"text_{i}" for i in range(MAX_BATCH_SIZE_PARTS + 5)]

    with patch("core.gigachat_embeddings.sleep"), \
         patch("core.gigachat_embeddings.perf_counter", return_value=100.0):
        GigaChatEmbeddingsDelayed.embed_documents(embedder, texts)

    # Должен быть хотя бы 2 батча (1 полный + 1 остаток)
    assert len(call_batches) >= 2
    assert len(call_batches[0]) == MAX_BATCH_SIZE_PARTS


def test_rate_limit_sleep_is_called():
    """sleep вызывается если не прошло GIGA_EMBED_DELAY секунд."""
    from core.gigachat_embeddings import GigaChatEmbeddingsDelayed
    import core.gigachat_embeddings as ge_module

    embedder = MagicMock(spec=GigaChatEmbeddingsDelayed)
    embedder.model = None
    embedder._client = MagicMock()
    embedder._client.embeddings.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1])]
    )

    ge_module._GIGA_EMBED_LAST_INVOKE = 1_000_000.0  # в будущем

    with patch("core.gigachat_embeddings.sleep") as mock_sleep, \
         patch("core.gigachat_embeddings.perf_counter", return_value=1_000_000.5):
        GigaChatEmbeddingsDelayed.embed_documents(embedder, ["text"])

    mock_sleep.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# Тест: semantic bonus в _column_reference_score
# ─────────────────────────────────────────────────────────────────

def test_column_reference_score_semantic_bonus_high():
    """sem_score >= 0.8 добавляет +20 к базовому score."""
    from core.filter_ranking import _column_reference_score

    sem_idx = MagicMock()
    sem_idx.similarity.return_value = {"dm.t.col": 0.9}

    score = _column_reference_score(
        "фактический отток", "col", "код причины",
        semantic_index=sem_idx,
        column_key="dm.t.col",
    )
    # base=0 (col не в запросе, desc score < порога) + bonus=20
    assert score == pytest.approx(20.0, abs=1.0)


def test_column_reference_score_semantic_bonus_medium():
    """sem_score 0.6–0.8 добавляет +10."""
    from core.filter_ranking import _column_reference_score

    sem_idx = MagicMock()
    sem_idx.similarity.return_value = {"dm.t.col": 0.7}

    score = _column_reference_score(
        "фактический отток", "col", "другое",
        semantic_index=sem_idx,
        column_key="dm.t.col",
    )
    assert score == pytest.approx(10.0, abs=1.0)


def test_column_reference_score_no_semantic_index():
    """Без semantic_index — обычная логика без бонуса."""
    from core.filter_ranking import _column_reference_score

    score = _column_reference_score("reason_code", "reason_code", "код причины")
    assert score == pytest.approx(120.0)


# ─────────────────────────────────────────────────────────────────
# Тест: semantic_search_tables через SchemaLoader
# ─────────────────────────────────────────────────────────────────

def test_semantic_search_tables_no_index():
    """Без semantic_index возвращает пустой список."""
    from unittest.mock import MagicMock
    from core.schema_loader import SchemaLoader

    schema = MagicMock(spec=SchemaLoader)
    schema.semantic_index = None
    schema.semantic_search_tables = SchemaLoader.semantic_search_tables.__get__(schema)
    result = schema.semantic_search_tables("запрос")
    assert result == []


def test_semantic_search_tables_with_index():
    """С semantic_index возвращает дедуплицированные таблицы."""
    from core.schema_loader import SchemaLoader

    sem_idx = MagicMock()
    sem_idx.semantic_search.return_value = [
        ("dm.fact_churn.reason_code", 0.9),
        ("dm.fact_churn.task_id", 0.85),  # та же таблица — дедупликация
        ("dm.dim_reason.reason_name", 0.7),
    ]

    schema = MagicMock(spec=SchemaLoader)
    schema.semantic_index = sem_idx
    schema._data_dir = MagicMock()
    result = SchemaLoader.semantic_search_tables(schema, "фактический отток", top_k=10)

    # Две уникальные таблицы (дедупликация по dm.fact_churn)
    assert len(result) == 2
    schemas = [r[0] for r in result]
    tables = [r[1] for r in result]
    assert "fact_churn" in tables
    assert "dim_reason" in tables
