"""Report writer: markdown summary + structured findings.jsonl + diagnostics.

The markdown report is the primary deliverable AND the debugging channel —
because the tool runs in a closed environment, the user cannot share a
database, only logs and reports. So the report doubles as an execution
trace: profile digest, per-hypothesis status (ok / skip / error / budget),
timing, and skip reasons, so an off-site engineer can see exactly what
happened from the file alone.

Findings are grouped by *business theme* (calendar / anomalous entities /
regime shifts / dependencies / outliers) so the analyst sees structure, not
a flat dump. A short TL;DR block at the top spotlights the top-N most
business-relevant findings — see _build_tldr below.

findings.jsonl exists for programmatic consumers.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.profiler import profile_to_brief
from core.deep_analysis.types import (
    AnalysisMode,
    BusinessInsight,
    Finding,
    HypothesisSpec,
    TableProfile,
)

if TYPE_CHECKING:
    from core.deep_analysis.orchestrator import HypothesisRunRecord

_SEVERITY_ORDER = {"critical": 0, "strong": 1, "notable": 2, "info": 3}
_SEVERITY_ICON = {"critical": "🔥", "strong": "⚠️", "notable": "ℹ️", "info": "•"}
_STATUS_ICON = {"ok": "✅", "skip": "⏭", "error": "❌", "budget": "⏰", "pending": "…"}

# Runner → human-readable theme. Theme order in the rendered report follows
# this dict's insertion order (Python 3.7+ guarantees insertion order).
_THEME_BY_RUNNER: dict[str, str] = {
    "group_anomalies": "Аномальные сущности и группы",
    "regime_shifts": "Структурные сдвиги во времени",
    "seasonality": "Календарные паттерны",
    "outliers": "Выбросы в значениях",
    "dependencies": "Связи между колонками",
}
_THEME_ORDER = list(_THEME_BY_RUNNER.values()) + ["Прочее"]

_TLDR_MAX_ITEMS = 5

_INSIGHT_PRIORITY_ORDER = {"top": 0, "high": 1, "medium": 2}
_INSIGHT_PRIORITY_LABEL = {"top": "🔝 TOP", "high": "HIGH", "medium": "MEDIUM"}
_INSIGHT_CONFIDENCE_LABEL = {"high": "высокое", "medium": "среднее", "low": "низкое"}


def write_report(
    findings: list[Finding],
    profile: TableProfile,
    hypotheses: list[HypothesisSpec],
    mode: AnalysisMode,
    output_dir: Path,
    *,
    run_records: "list[HypothesisRunRecord] | None" = None,
    wall_seconds: float = 0.0,
    business_insights: list[BusinessInsight] | None = None,
) -> Path:
    log = get_logger()
    output_dir.mkdir(parents=True, exist_ok=True)
    insights = business_insights or []

    jsonl_path = output_dir / "findings.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for fnd in findings:
            f.write(json.dumps(dataclasses.asdict(fnd), ensure_ascii=False, default=str))
            f.write("\n")

    if insights:
        ins_path = output_dir / "business_insights.jsonl"
        with ins_path.open("w", encoding="utf-8") as f:
            for ins in insights:
                f.write(json.dumps(dataclasses.asdict(ins), ensure_ascii=False, default=str))
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
            business_insights=insights,
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
    business_insights: list[BusinessInsight] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Глубокий анализ {profile.schema}.{profile.table}")
    lines.append("")
    lines.append(f"- Режим: **{mode.value}**")
    lines.append(f"- Всего строк в таблице: {profile.n_rows}")
    lines.append(f"- Колонок проанализировано: {profile.n_cols}")
    lines.append(f"- Стратегия загрузки: `{profile.load_strategy}`")
    lines.append(f"- Wall time: {wall_seconds:.1f}s")
    if profile.where_clause:
        lines.append(f"- Фильтр (WHERE): `{profile.where_clause}`")
    if profile.dropped_wide_text:
        lines.append(f"- Откинутые широкие текстовые колонки: {', '.join(profile.dropped_wide_text)}")
    if profile.table_description:
        lines.append(f"- Описание: {profile.table_description}")
    lines.append("")

    sorted_findings = sorted(
        findings,
        key=lambda f: (
            _SEVERITY_ORDER.get(f.severity, 99),
            -f.metrics.get("n_violators", 0),
        ),
    )

    # Business insights (LLM-curated) — sits above the mechanical TL;DR. If
    # the stage didn't run / produced nothing, this is silently skipped.
    if business_insights:
        lines.extend(_render_business_insights(business_insights))

    # TL;DR — top-N business-relevant findings, surfaced before the dump.
    lines.extend(_render_tldr(sorted_findings))

    if not findings:
        lines.append("## Итог")
        lines.append("")
        lines.append("Проверены все гипотезы, значимых отклонений не найдено.")
        lines.append("")
        lines.append(f"Всего проверено гипотез: {len(hypotheses)}. "
                     "Подробности — в разделе диагностики ниже.")
        lines.append("")
    else:
        lines.append(f"## Найдено значимых находок: {len(findings)}")
        lines.append("")
        # Grouped by theme. Within a theme we keep severity-sorted order.
        grouped = _group_findings_by_theme(sorted_findings)
        for theme in _THEME_ORDER:
            theme_findings = grouped.get(theme)
            if not theme_findings:
                continue
            severity_count = _severity_breakdown(theme_findings)
            lines.append(f"### Раздел: {theme} ({severity_count})")
            lines.append("")
            for fnd in theme_findings:
                _render_finding(lines, fnd, profile)
            lines.append("")

    # Equivalence summary — show analysts which columns we folded so they
    # don't think findings are missing for `post_name` etc.
    eq_block = _render_equivalence_block(profile)
    if eq_block:
        lines.extend(eq_block)

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


def _group_findings_by_theme(findings: list[Finding]) -> dict[str, list[Finding]]:
    out: dict[str, list[Finding]] = {}
    for fnd in findings:
        theme = _THEME_BY_RUNNER.get(fnd.runner, "Прочее")
        out.setdefault(theme, []).append(fnd)
    return out


def _severity_breakdown(findings: list[Finding]) -> str:
    """Render `🔥 3 · ⚠️ 7 · ℹ️ 2` for a finding list — gives a quick header eye-bite."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    parts = []
    for sev in ("critical", "strong", "notable", "info"):
        if counts.get(sev):
            parts.append(f"{_SEVERITY_ICON[sev]} {counts[sev]}")
    return " · ".join(parts) or f"{len(findings)} шт."


def _render_finding(lines: list[str], fnd: Finding, profile: TableProfile) -> None:
    icon = _SEVERITY_ICON.get(fnd.severity, "•")
    # Anchor lets business-insight links drill down to the underlying finding.
    lines.append(f'<a id="{fnd.hypothesis_id}"></a>')
    lines.append(f"#### {icon} [{fnd.severity}] {fnd.title}")
    lines.append("")
    lines.append(fnd.summary)
    lines.append("")
    # Show equivalence note so analysts see the finding covers a whole class.
    eq_note = _equivalence_note_for_finding(fnd, profile)
    if eq_note:
        lines.append(eq_note)
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


def _equivalence_note_for_finding(fnd: Finding, profile: TableProfile) -> str:
    """If the finding's column has equivalent siblings, list them inline.

    We scan the finding's metrics/details/title for column names appearing in
    multi-member equivalence classes. The note tells the analyst "this finding
    is equally true for these other columns" — explains why we don't repeat
    the same finding three times.
    """
    if not profile.equivalence_groups:
        return ""
    multi_classes = {
        rep: members for rep, members in profile.equivalence_groups.items()
        if len(members) > 1
    }
    if not multi_classes:
        return ""
    text_blob = " ".join([
        fnd.title or "",
        fnd.summary or "",
        " ".join(str(v) for v in (fnd.metrics or {}).values()),
        " ".join(str(v) for v in (fnd.details or {}).values()),
    ])
    matched: list[tuple[str, list[str]]] = []
    for rep, members in multi_classes.items():
        if any(_word_appears(m, text_blob) for m in members):
            siblings = [m for m in members if m != rep]
            if siblings:
                matched.append((rep, siblings))
    if not matched:
        return ""
    parts = [
        f"`{rep}` ≡ {', '.join(f'`{s}`' for s in sibs)}"
        for rep, sibs in matched
    ]
    return "_Эквивалентно для:_ " + "; ".join(parts)


def _word_appears(name: str, blob: str) -> bool:
    """Whole-word match for column names. Avoids false positives when a short
    name like `id` is a substring of `saphr_id` etc."""
    if not name:
        return False
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, blob) is not None


def _render_equivalence_block(profile: TableProfile) -> list[str]:
    multi_classes = {
        rep: members for rep, members in profile.equivalence_groups.items()
        if len(members) > 1
    }
    if not multi_classes:
        return []
    lines = ["## Эквивалентные колонки (свернуты в анализе)", ""]
    lines.append(
        "Колонки ниже определяются друг через друга 1:1 (Cramér's V ≥ 0.99). "
        "Анализ вёлся только по представителю — все находки распространяются "
        "на остальных членов класса автоматически."
    )
    lines.append("")
    lines.append("| Представитель | Эквивалентные колонки |")
    lines.append("| --- | --- |")
    for rep, members in sorted(multi_classes.items()):
        siblings = [m for m in members if m != rep]
        lines.append(f"| `{rep}` | {', '.join(f'`{m}`' for m in siblings)} |")
    lines.append("")
    return lines


def _render_tldr(sorted_findings: list[Finding]) -> list[str]:
    """Top-N executive summary. Surfaces the highest-severity, biggest-impact
    findings so a reader who scrolls past the first screen still gets the
    point.

    Ranking inside severity buckets prefers findings with explicit "violator"
    counts (`n_violators` / `n_outliers`) — they are the most actionable.
    """
    lines = ["## TL;DR", ""]
    if not sorted_findings:
        lines.append("Существенных аномалий в таблице не найдено.")
        lines.append("")
        return lines

    # Pick the worst-severity tier that has any items, then sort by impact.
    tiers = {"critical": [], "strong": [], "notable": [], "info": []}
    for f in sorted_findings:
        if f.severity in tiers:
            tiers[f.severity].append(f)

    chosen: list[Finding] = []
    for tier in ("critical", "strong", "notable", "info"):
        if not tiers[tier]:
            continue
        ranked = sorted(
            tiers[tier],
            key=lambda f: -_impact_score(f),
        )
        for f in ranked:
            if f not in chosen:
                chosen.append(f)
            if len(chosen) >= _TLDR_MAX_ITEMS:
                break
        if len(chosen) >= _TLDR_MAX_ITEMS:
            break

    for f in chosen:
        sev_icon = _SEVERITY_ICON.get(f.severity, "•")
        first_sentence = _first_sentence(f.summary)
        lines.append(f"- {sev_icon} **{f.title}** — {first_sentence}")
    lines.append("")
    severity_summary = _severity_breakdown(sorted_findings)
    lines.append(f"_Всего находок: {len(sorted_findings)} ({severity_summary}). "
                 "Полный разбор — в разделах ниже._")
    lines.append("")
    return lines


def _impact_score(f: Finding) -> float:
    """Heuristic: prefer findings with concrete entity counts, then high
    deviation magnitudes, then statistical strength."""
    m = f.metrics or {}
    score = 0.0
    for k in ("n_violators", "n_outliers", "n_changepoints"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            score += float(v) * 100
    for k in ("max_abs_z", "max_rel_deviation_pct", "max_abs_rel_shift_pct"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            score += abs(float(v))
    for k in ("cramer_v", "spearman_rho", "eta_sq"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            score += abs(float(v)) * 10
    return score


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _render_business_insights(insights: list[BusinessInsight]) -> list[str]:
    """LLM-curated «куда смотреть → на что влияет → что сделать» block.

    Sits above the mechanical TL;DR. Drill-down links use the per-finding
    anchors emitted by _render_finding.
    """
    ordered = sorted(
        insights,
        key=lambda i: _INSIGHT_PRIORITY_ORDER.get(i.priority, 99),
    )
    lines = ["## 🎯 Главное для бизнеса", ""]
    lines.append(
        f"_LLM выделил {len(ordered)} "
        f"{'инсайт' if len(ordered) == 1 else 'инсайтов' if len(ordered) >= 5 else 'инсайта'} "
        "из всех находок ниже. Если нужна полная картина — смотрите TL;DR и тематические разделы._"
    )
    lines.append("")
    for ins in ordered:
        prio = _INSIGHT_PRIORITY_LABEL.get(ins.priority, ins.priority.upper())
        lines.append(f"### [{prio}] {ins.title}")
        lines.append("")
        lines.append(f"- **Куда смотреть:** {ins.where_to_look}")
        lines.append(f"- **На что влияет:** {ins.business_impact}")
        lines.append(f"- **Что сделать:** {ins.recommended_action}")
        if ins.related_finding_ids:
            refs = ", ".join(f"[{fid}](#{fid})" for fid in ins.related_finding_ids)
            confidence = _INSIGHT_CONFIDENCE_LABEL.get(ins.confidence, ins.confidence)
            lines.append(f"- _Доверие: {confidence} · подробнее: {refs}_")
        else:
            confidence = _INSIGHT_CONFIDENCE_LABEL.get(ins.confidence, ins.confidence)
            lines.append(f"- _Доверие: {confidence}_")
        lines.append("")
    return lines


def _first_sentence(text: str, limit: int = 220) -> str:
    if not text:
        return ""
    parts = _SENTENCE_END.split(text.strip(), maxsplit=1)
    head = parts[0]
    if len(head) > limit:
        head = head[: limit - 1].rstrip() + "…"
    return head
