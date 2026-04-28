"""Тесты для feedback loop (задача 3.2):
- MemoryManager.log_user_feedback / load_feedback
- FewShotRetriever.retrieve_examples с include_negatives
"""

import json
import tempfile
from pathlib import Path

import pytest

from core.memory import MemoryManager
from core.few_shot_retriever import FewShotRetriever


# ─────────────────────────────────────────────────────────────────
# Вспомогательные утилиты
# ─────────────────────────────────────────────────────────────────

def _make_memory(tmp_path: Path) -> MemoryManager:
    return MemoryManager(memory_dir=tmp_path)


# ─────────────────────────────────────────────────────────────────
# Тест MemoryManager.log_user_feedback
# ─────────────────────────────────────────────────────────────────

def test_log_feedback_creates_file(tmp_path):
    mem = _make_memory(tmp_path)
    mem.log_user_feedback("запрос", "SELECT 1", verdict="up")
    assert (tmp_path / "feedback.jsonl").exists()


def test_log_feedback_appends_entries(tmp_path):
    mem = _make_memory(tmp_path)
    mem.log_user_feedback("запрос 1", "SELECT 1", verdict="up")
    mem.log_user_feedback("запрос 2", "SELECT 2", verdict="down", corrected_sql="SELECT 3")

    lines = (tmp_path / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["verdict"] == "up"
    assert entry1["query"] == "запрос 1"

    entry2 = json.loads(lines[1])
    assert entry2["verdict"] == "down"
    assert entry2["corrected_sql"] == "SELECT 3"


def test_log_feedback_entry_has_timestamp(tmp_path):
    mem = _make_memory(tmp_path)
    mem.log_user_feedback("q", "sql", verdict="up")
    entries = mem.load_feedback()
    assert entries[0].get("timestamp")


# ─────────────────────────────────────────────────────────────────
# Тест MemoryManager.load_feedback с фильтрацией по verdict
# ─────────────────────────────────────────────────────────────────

def test_load_feedback_filter_down(tmp_path):
    mem = _make_memory(tmp_path)
    mem.log_user_feedback("q1", "sql1", verdict="up")
    mem.log_user_feedback("q2", "sql2", verdict="down", corrected_sql="fix")

    entries = mem.load_feedback(verdict="down")
    assert len(entries) == 1
    assert entries[0]["query"] == "q2"


def test_load_feedback_no_filter(tmp_path):
    mem = _make_memory(tmp_path)
    mem.log_user_feedback("q1", "sql1", verdict="up")
    mem.log_user_feedback("q2", "sql2", verdict="down")

    entries = mem.load_feedback()
    assert len(entries) == 2


def test_load_feedback_empty_file(tmp_path):
    mem = _make_memory(tmp_path)
    entries = mem.load_feedback()
    assert entries == []


# ─────────────────────────────────────────────────────────────────
# Тест FewShotRetriever.retrieve_examples с include_negatives
# ─────────────────────────────────────────────────────────────────

def _make_retriever_with_feedback(
    tmp_path: Path,
    negatives: list[dict],
) -> FewShotRetriever:
    """Создать FewShotRetriever с предзаполненными негативными примерами."""
    mem = _make_memory(tmp_path)
    for neg in negatives:
        mem.log_user_feedback(
            query=neg["query"],
            sql=neg["sql"],
            verdict="down",
            corrected_sql=neg.get("corrected_sql"),
        )

    retriever = FewShotRetriever(mem)
    # Нет реальной истории успешных запросов — кэш пуст
    retriever._cache = []
    return retriever


def test_retrieve_examples_include_negatives(tmp_path):
    """include_negatives=True → prompt содержит секцию антипримеров."""
    negatives = [
        {
            "query": "покажи клиентов",
            "sql": "SELECT * FROM clients",
            "corrected_sql": "SELECT id, name FROM dm.clients WHERE active=true",
        }
    ]
    retriever = _make_retriever_with_feedback(tmp_path, negatives)
    result = retriever.retrieve_examples("покажи клиентов", include_negatives=True)

    assert "НЕ ДЕЛАЙ ТАК" in result
    assert "SELECT * FROM clients" in result
    assert "SELECT id, name" in result


def test_retrieve_examples_no_negatives_by_default(tmp_path):
    """По умолчанию (include_negatives=False) негативные примеры НЕ включаются."""
    negatives = [
        {"query": "q", "sql": "bad sql", "corrected_sql": "good sql"},
    ]
    retriever = _make_retriever_with_feedback(tmp_path, negatives)
    result = retriever.retrieve_examples("q", include_negatives=False)

    assert "НЕ ДЕЛАЙ ТАК" not in result


def test_retrieve_examples_only_with_corrected_sql(tmp_path):
    """Антипримеры без corrected_sql не попадают в промпт."""
    mem = _make_memory(tmp_path)
    mem.log_user_feedback("q", "bad sql", verdict="down")  # нет corrected_sql

    retriever = FewShotRetriever(mem)
    retriever._cache = []
    result = retriever.retrieve_examples("q", include_negatives=True)

    assert "НЕ ДЕЛАЙ ТАК" not in result


def test_retrieve_examples_empty_when_no_history(tmp_path):
    """Без истории и без негативов — пустая строка."""
    mem = _make_memory(tmp_path)
    retriever = FewShotRetriever(mem)
    retriever._cache = []
    result = retriever.retrieve_examples("q", include_negatives=False)
    assert result == ""


# ─────────────────────────────────────────────────────────────────
# Тест: feedback roadmap acceptance scenario
# ─────────────────────────────────────────────────────────────────

def test_negative_example_appears_in_future_retrieval(tmp_path):
    """
    Acceptance: запрос → 👎 → corrected_sql → повтор →
    retrieval показывает negative example.
    """
    mem = _make_memory(tmp_path)
    bad_sql = "SELECT * FROM dm.orders"
    good_sql = "SELECT order_id, amount FROM dm.orders WHERE status='active'"

    # Пользователь поставил 👎 и предоставил правильный SQL
    mem.log_user_feedback(
        "покажи активные заказы", bad_sql, verdict="down",
        corrected_sql=good_sql,
    )

    retriever = FewShotRetriever(mem)
    retriever._cache = []

    # При повторном запросе negative example должен присутствовать
    prompt = retriever.retrieve_examples(
        "покажи активные заказы", include_negatives=True
    )

    assert bad_sql in prompt
    assert good_sql in prompt
    assert "НЕ ДЕЛАЙ ТАК" in prompt
