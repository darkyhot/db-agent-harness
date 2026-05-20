"""Grounded Q&A over a finished deep-analysis report.

After ``/deep_table_analysis`` the CLI enters a Q&A mode: the business
``report.md`` plus an index of the exported CSV files become the *only*
context the LLM may use. The system prompt forbids inventing anything that
is not in those materials — this is the same anti-hallucination stance that
the analysis stages now take, applied to follow-up questions.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from core.deep_analysis.logging_setup import get_logger
from core.llm import RateLimitedLLM

_MAX_CSV_FILES = 40
_MAX_CSV_SAMPLE_ROWS = 15
_MAX_CSV_CHARS = 3000  # per-file cap on the inlined sample

_SYSTEM_PROMPT = """Ты — бизнес-аналитик. Пользователь уже запустил глубокий анализ таблицы. Тебе дают готовый отчёт (report.md) и список выгруженных CSV-файлов с примерами строк. Других данных у тебя нет.

Правила:
- Отвечай ТОЛЬКО на основе отчёта и приложенных CSV. Никаких внешних знаний, домыслов и общих рассуждений.
- Если ответа в материалах нет — честно скажи «В отчёте нет данных по этому вопросу» и подскажи, какой раздел отчёта или CSV-файл мог бы помочь (или что нужен отдельный анализ).
- НЕ выдумывай числа, даты, имена, проценты, сегменты — бери их строго из отчёта/CSV дословно.
- Отвечай кратко и деловым языком, без статистического жаргона (никаких p-value, z-score, Cramér's V и т.п.).
- Если вопрос про конкретных нарушителей/объекты — назови соответствующий CSV-файл по имени.
"""


def build_csv_index(output_dir: Path) -> str:
    """Compact, bounded index of exported CSVs: name + header + a few rows.

    Keeps the prompt small — we inline only the header and the first handful
    of rows per file so the LLM can reference files and cite concrete values
    without us dumping potentially huge violator lists.
    """
    try:
        csv_files = sorted(p for p in output_dir.glob("*.csv") if p.is_file())
    except OSError:
        return "(CSV-файлы недоступны)"
    if not csv_files:
        return "(CSV-файлы не выгружались)"

    blocks: list[str] = []
    for path in csv_files[:_MAX_CSV_FILES]:
        try:
            manifest = _csv_manifest(path)
            with path.open("r", encoding="utf-8") as f:
                sample_lines: list[str] = []
                for i, line in enumerate(f):
                    if i > _MAX_CSV_SAMPLE_ROWS:
                        break
                    sample_lines.append(line.rstrip("\n"))
            sample = "\n".join(sample_lines)
            if len(sample) > _MAX_CSV_CHARS:
                sample = sample[:_MAX_CSV_CHARS] + " …(обрезано)"
            blocks.append(f"### {path.name}\n{manifest}\n```\n{sample}\n```")
        except OSError as exc:
            blocks.append(f"### {path.name}\n(не удалось прочитать: {exc})")

    extra = ""
    if len(csv_files) > _MAX_CSV_FILES:
        extra = f"\n\n(ещё {len(csv_files) - _MAX_CSV_FILES} CSV-файлов не показаны)"
    return "\n\n".join(blocks) + extra


def _csv_manifest(path: Path) -> str:
    """Small semantic summary for Q&A grounding."""
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            row_count = 0
            counters: dict[str, Counter[str]] = {
                c: Counter() for c in ("entity", "entity_col", "metric", "reason", "peer_group")
                if c in columns
            }
            for row in reader:
                row_count += 1
                if row_count <= 5000:
                    for col, counter in counters.items():
                        val = str(row.get(col) or "").strip()
                        if val:
                            counter[val] += 1
    except Exception as exc:  # noqa: BLE001
        return f"(summary недоступен: {exc})"

    lines = [
        f"- rows: {row_count}",
        f"- columns: {', '.join(columns) or '—'}",
    ]
    for col, counter in counters.items():
        if not counter:
            continue
        top = ", ".join(f"{v} ({n})" for v, n in counter.most_common(5))
        lines.append(f"- top {col}: {top}")
    return "\n".join(lines)


def answer_from_report(
    llm: RateLimitedLLM,
    question: str,
    report_md_text: str,
    csv_index: str,
) -> str:
    """Answer a follow-up question strictly from the report + CSV index.

    Returns a plain-text answer. On LLM failure returns a short message so
    the CLI can show something instead of crashing the session.
    """
    log = get_logger()
    user = (
        "=== ОТЧЁТ (report.md) ===\n"
        f"{report_md_text}\n\n"
        "=== ВЫГРУЖЕННЫЕ CSV-ФАЙЛЫ (имя + первые строки) ===\n"
        f"{csv_index}\n\n"
        "=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===\n"
        f"{question}"
    )
    try:
        return llm.invoke_with_system(_SYSTEM_PROMPT, user, temperature=0.1).strip()
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the REPL
        log.warning("report_qa LLM call failed: %s", exc)
        return (
            "Не удалось получить ответ от модели по отчёту. "
            "Повторите вопрос или выполните /reset, чтобы выйти из режима."
        )
