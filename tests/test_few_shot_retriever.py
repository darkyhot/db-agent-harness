"""Тесты FewShotRetriever: загрузка из JSON-memory, scoring, invalidation."""

from __future__ import annotations

import pytest

from core.few_shot_retriever import FewShotRetriever, _jaccard_similarity, _tokenize
from core.memory import MemoryManager


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(memory_dir=tmp_path)


def _seed(memory: MemoryManager, rows: list[dict]) -> None:
    memory.start_session("test-user")
    for r in rows:
        memory.log_sql_execution(
            user_input=r["user_input"],
            sql=r["sql"],
            row_count=r.get("row_count", 5),
            status=r.get("status", "success"),
            duration_ms=r.get("duration_ms", 123),
            retry_count=r.get("retry_count", 0),
            error_type=r.get("error_type", ""),
        )


class TestTokenize:
    def test_stopwords_removed(self):
        toks = _tokenize("покажи все заказы по регионам")
        assert "заказы" in toks
        assert "регионам" in toks
        assert "покажи" not in toks
        assert "все" not in toks

    def test_short_tokens_dropped(self):
        assert _tokenize("а я по") == set()


class TestJaccard:
    def test_identical(self):
        a = {"x", "y"}
        assert _jaccard_similarity(a, a) == 1.0

    def test_disjoint(self):
        assert _jaccard_similarity({"x"}, {"y"}) == 0.0

    def test_empty_pair(self):
        assert _jaccard_similarity(set(), set()) == 0.0


class TestLoadFromJsonMemory:
    def test_loads_only_successful_first_try(self, memory):
        _seed(memory, [
            {"user_input": "посчитай клиентов", "sql": "SELECT COUNT(*) FROM dm.clients", "status": "success", "retry_count": 0},
            {"user_input": "ошибка", "sql": "SELECT bad", "status": "error", "retry_count": 0},
            {"user_input": "пересчитано со 2-го раза", "sql": "SELECT * FROM dm.sales", "status": "success", "retry_count": 2},
            {"user_input": "короткий", "sql": "SELECT 1", "status": "success", "retry_count": 0},  # короче 20 символов
        ])
        retriever = FewShotRetriever(memory)
        queries = retriever._load_successful_queries()
        assert len(queries) == 1
        assert queries[0]["user_input"] == "посчитай клиентов"

    def test_cache_invalidate(self, memory):
        _seed(memory, [
            {"user_input": "запрос один", "sql": "SELECT * FROM dm.orders WHERE id=1"},
        ])
        retriever = FewShotRetriever(memory)
        first = retriever._load_successful_queries()
        assert len(first) == 1
        # Добавили новую запись — без invalidate не должно повлиять
        _seed(memory, [
            {"user_input": "запрос два", "sql": "SELECT * FROM dm.orders WHERE id=2"},
        ])
        cached = retriever._load_successful_queries()
        assert len(cached) == 1  # cache hit
        retriever.invalidate_cache()
        fresh = retriever._load_successful_queries()
        assert len(fresh) == 2


class TestGetSimilar:
    def test_returns_top_n_by_similarity(self, memory):
        _seed(memory, [
            {"user_input": "количество клиентов по регионам", "sql": "SELECT region, COUNT(*) FROM dm.clients GROUP BY region"},
            {"user_input": "сумма продаж по товарам", "sql": "SELECT product, SUM(amount) FROM dm.sales GROUP BY product"},
            {"user_input": "клиенты по сегментам", "sql": "SELECT segment, COUNT(*) FROM dm.clients GROUP BY segment"},
        ])
        retriever = FewShotRetriever(memory)
        results = retriever.get_similar("сколько клиентов по регионам", n=2, min_similarity=0.1)
        assert len(results) >= 1
        assert "клиентов" in results[0]["user_input"]

    def test_min_similarity_filter(self, memory):
        _seed(memory, [
            {"user_input": "сумма продаж", "sql": "SELECT SUM(amount) FROM dm.sales"},
        ])
        retriever = FewShotRetriever(memory)
        results = retriever.get_similar("запрос про товары", n=2, min_similarity=0.99)
        assert results == []

    def test_semantic_frame_metric_bonus(self, memory):
        _seed(memory, [
            {"user_input": "записей в таблице", "sql": "SELECT COUNT(*) FROM dm.clients"},
            {"user_input": "записей в таблице", "sql": "SELECT * FROM dm.clients"},
        ])
        retriever = FewShotRetriever(memory)
        frame = {"metric_intent": "count", "subject": "client"}
        results = retriever.get_similar(
            "сколько записей в таблице", n=2, min_similarity=0.1, semantic_frame=frame
        )
        # Оба прошли Jaccard — COUNT(*) должен быть выше за счёт bonus по метрике
        assert results[0]["sql"].upper().startswith("SELECT COUNT")

    def test_last_similarities_recorded(self, memory):
        _seed(memory, [
            {"user_input": "продажи по менеджерам", "sql": "SELECT manager, SUM(amount) FROM dm.sales GROUP BY manager"},
        ])
        retriever = FewShotRetriever(memory)
        _ = retriever.get_similar("продажи менеджеров", n=1, min_similarity=0.1)
        assert len(retriever.last_similarities) >= 1
        assert retriever.last_similarities[0] > 0.0

    def test_empty_history_returns_empty(self, memory):
        memory.start_session("empty")
        retriever = FewShotRetriever(memory)
        assert retriever.get_similar("любой запрос") == []

    def test_format_for_prompt(self, memory):
        examples = [
            {"user_input": "пример", "sql": "SELECT 1 FROM dm.t"},
        ]
        retriever = FewShotRetriever(memory)
        out = retriever.format_for_prompt(examples)
        assert "ПОХОЖИЕ ЗАПРОСЫ" in out
        assert "SELECT 1" in out


class TestIterSqlAudit:
    def test_filters_by_status_and_retry(self, memory):
        _seed(memory, [
            {"user_input": "a", "sql": "SELECT * FROM t WHERE a=1", "status": "success", "retry_count": 0},
            {"user_input": "b", "sql": "SELECT * FROM t WHERE a=2", "status": "success", "retry_count": 1},
            {"user_input": "c", "sql": "SELECT * FROM t WHERE a=3", "status": "error", "retry_count": 0},
        ])
        rows = memory.iter_sql_audit(status="success", max_retry_count=0)
        assert len(rows) == 1
        assert rows[0]["user_input"] == "a"

    def test_min_sql_length(self, memory):
        _seed(memory, [
            {"user_input": "short", "sql": "SEL 1", "status": "success", "retry_count": 0},
            {"user_input": "long", "sql": "SELECT * FROM long_table WHERE id=1", "status": "success", "retry_count": 0},
        ])
        rows = memory.iter_sql_audit(min_sql_length=20)
        assert len(rows) == 1

    def test_row_count_bounds(self, memory):
        _seed(memory, [
            {"user_input": "zero", "sql": "SELECT * FROM t WHERE a=1", "row_count": 0},
            {"user_input": "many", "sql": "SELECT * FROM t WHERE a=2", "row_count": 10},
            {"user_input": "huge", "sql": "SELECT * FROM t WHERE a=3", "row_count": 1_000_000},
        ])
        rows = memory.iter_sql_audit(min_row_count=1, max_row_count=100)
        labels = {r["user_input"] for r in rows}
        assert labels == {"many"}

    def test_limit(self, memory):
        _seed(memory, [
            {"user_input": f"q{i}", "sql": f"SELECT * FROM t WHERE id={i}"}
            for i in range(5)
        ])
        rows = memory.iter_sql_audit(limit=2)
        assert len(rows) == 2
