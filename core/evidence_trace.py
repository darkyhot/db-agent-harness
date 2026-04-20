"""Стандартизированная схема записи evidence_trace узлов графа (Direction 4.1).

Эта функция вынесена в отдельный модуль (а не в graph.nodes.common), чтобы
её можно было тестировать без тяжёлых зависимостей (sqlalchemy, langchain).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def record_evidence(
    evidence_trace: dict[str, Any] | None,
    node: str,
    decision: str,
    evidence: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Записать запись в evidence_trace по стандартной схеме.

    Имя ноды — ключ словаря, значение —
    {decision, evidence, warnings, finished_at}.
    Если нода уже встречалась в trace (retry-сценарий), поле `history`
    содержит предыдущие записи (последние 4) для отладки.

    Returns:
        Новый словарь evidence_trace (безопасно присваивать в state).
    """
    trace = dict(evidence_trace or {})
    now = datetime.now(timezone.utc).isoformat()
    entry: dict[str, Any] = {
        "decision": decision,
        "evidence": dict(evidence or {}),
        "warnings": list(warnings or []),
        "finished_at": now,
    }
    if node in trace and isinstance(trace[node], dict):
        prev = trace[node]
        history = list(prev.get("history", []))
        history.append({k: v for k, v in prev.items() if k != "history"})
        entry["history"] = history[-4:]
    trace[node] = entry
    return trace
