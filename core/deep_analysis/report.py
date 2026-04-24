"""Report writer: markdown summary + structured findings.jsonl + diagnostics.

The markdown report is the primary deliverable AND the debugging channel —
because the tool runs in a closed environment, the user cannot share a
database, only logs and reports. So the report doubles as an execution
trace: profile digest, per-hypothesis status (ok / skip / error / budget),
timing, and skip reasons, so an off-site engineer can see exactly what
happened from the file alone.

findings.jsonl exists for programmatic consumers.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.profiler import profile_to_brief
from core.deep_analysis.types import (
    AnalysisMode,
    Finding,
    HypothesisSpec,
    TableProfile,
)

if TYPE_CHECKING:
    from core.deep_analysis.orchestrator import HypothesisRunRecord

_SEVERITY_ORDER = {"critical": 0, "strong": 1, "notable": 2, "info": 3}
_SEVERITY_ICON = {"critical": "🔥", "strong": "⚠️", "notable": "ℹ️", "info": "•"}
_STATUS_ICON = {"ok": "✅", "skip": "⏭", "error": "❌", "budget": "⏰", "pending": "…"}


def write_report(
    findings: list[Finding],
    profile: TableProfile,
    hypotheses: list[HypothesisSpec],
    mode: AnalysisMode,
    output_dir: Path,
    *,
    run_records: "list[HypothesisRunRecord] | None" = None,
    wall_seconds: float = 0.0,
) -> Path:
    log = get_logger()
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "findings.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for fnd in findings:
            f.write(json.dumps(dataclasses.asdict(fnd), ensure_ascii=False, default=str))
            f.write("\n")

    if run_records:
        diag_path = output_dir / "diagnostics.jsonl"
        with diag_path.open("w", encoding="utf-8") as f:
            for rec in run_records:
                f.write(json.dumps(dataclasses.asdict(rec), ensure_ascii=False, default=str))
                f.write("\n")

    md_path = output_dir / "report.md"
    md_path.write_text(
        _render_markdown(
            findings, profile, hypotheses, mode, output_dir,
            run_records=run_records or [], wall_seconds=wall_seconds,
        ),
        encoding="utf-8",
    )
    log.info("Report written: %s", md_path)
    return md_path


def _render_markdown(
    findings: list[Finding],
    profile: TableProfile,
    hypotheses: list[HypothesisSpec],
    mode: AnalysisMode,
    output_dir: Path,
    *,
    run_records: "list[HypothesisRunRecord]",
    wall_seconds: float,
) -> str:
    lines: list[str] = []
    lines.append(f"# Глубокий анализ {profile.schema}.{profile.table}")
    lines.append("")
    lines.append(f"- Режим: **{mode.value}**")
    lines.append(f"- Всего строк в таблице: {profile.n_rows}")
    lines.append(f"- Колонок проанализировано: {profile.n_cols}")
    lines.append(f"- Стратегия загрузки: `{profile.load_strategy}`")
    lines.append(f"- Wall time: {wall_seconds:.1f}s")
    if profile.dropped_wide_text:
        lines.append(f"- Откинутые широкие текстовые колонки: {', '.join(profile.dropped_wide_text)}")
    if profile.table_description:
        lines.append(f"- Описание: {profile.table_description}")
    lines.append("")

    if not findings:
        lines.append("## Итог")
        lines.append("")
        lines.append("Проверены все гипотезы, значимых отклонений не найдено.")
        lines.append("")
        lines.append(f"Всего проверено гипотез: {len(hypotheses)}. "
                     "Подробности — в разделе диагностики ниже.")
        lines.append("")
    else:
        sorted_findings = sorted(
            findings,
            key=lambda f: (_SEVERITY_ORDER.get(f.severity, 99), -f.metrics.get("n_violators", 0)),
        )
        lines.append(f"## Найдено значимых находок: {len(findings)}")
        lines.append("")
        for fnd in sorted_findings:
            icon = _SEVERITY_ICON.get(fnd.severity, "•")
            lines.append(f"### {icon} [{fnd.severity}] {fnd.title}")
            lines.append("")
            lines.append(fnd.summary)
            lines.append("")
            if fnd.metrics:
                lines.append("Метрики:")
                for k, v in fnd.metrics.items():
                    if isinstance(v, float):
                        v = f"{v:.4g}"
                    lines.append(f"- `{k}`: {v}")
                lines.append("")
            if fnd.entity_csv:
                lines.append(f"Полный список: `{fnd.entity_csv}`")
                lines.append("")

    # Diagnostics section — execution trace for remote debugging.
    lines.append("---")
    lines.append("")
    lines.append("## Диагностика выполнения")
    lines.append("")
    if run_records:
        status_counts: dict[str, int] = {}
        for rec in run_records:
            status_counts[rec.status] = status_counts.get(rec.status, 0) + 1
        counts_line = ", ".join(
            f"{_STATUS_ICON.get(k, '?')} {k}={v}" for k, v in sorted(status_counts.items())
        )
        lines.append(f"Счётчик статусов: {counts_line}")
        lines.append("")
        total_runner_sec = sum(r.seconds for r in run_records)
        lines.append(f"Суммарное время runner-ов: {total_runner_sec:.1f}s")
        lines.append("")
        lines.append("| Status | Source | Runner | Prio | Sec | Findings | Hypothesis | Error |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for rec in run_records:
            icon = _STATUS_ICON.get(rec.status, rec.status)
            lines.append(
                f"| {icon} {rec.status} | {rec.source} | {rec.runner} | "
                f"{rec.priority:.2f} | {rec.seconds:.1f} | {rec.n_findings} | "
                f"{rec.title} | {rec.error_summary or '—'} |"
            )
    else:
        lines.append("(диагностика пустая — список run_records не передан)")
    lines.append("")

    lines.append("## Все проверенные гипотезы")
    lines.append("")
    lines.append("| Приоритет | Источник | Runner | Гипотеза |")
    lines.append("| --- | --- | --- | --- |")
    for h in sorted(hypotheses, key=lambda x: -x.priority):
        lines.append(f"| {h.priority:.2f} | {h.source} | {h.runner} | {h.title} |")
    lines.append("")

    # Profile digest — makes the report self-contained for remote debugging.
    lines.append("## Профайл колонок (дайджест)")
    lines.append("")
    lines.append("```")
    lines.append(profile_to_brief(profile))
    lines.append("```")
    lines.append("")
    lines.append(f"Артефакты в каталоге: `{output_dir}`")
    return "\n".join(lines)
