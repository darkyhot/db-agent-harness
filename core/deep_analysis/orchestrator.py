"""Orchestrator — runs profile → hypothesis generation → runners → report.

Budget:
- FAST mode: hard deadline of FAST_BUDGET_SEC (default 40 min).
- DEEP mode: hard deadline of DEEP_BUDGET_SEC (default 3 hours).

Hypotheses are sorted by priority and executed until the deadline. Each runner
invocation gets its own try/except so one crash doesn't kill the whole run.

Loader and the LLM call are injectable so integration tests can run the full
pipeline without a live database or a real LLM endpoint.
"""

from __future__ import annotations

import time
import traceback
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.database import DatabaseManager
from core.deep_analysis.business_insights import extract_business_insights
from core.deep_analysis.equivalence import compute_equivalence_groups
from core.deep_analysis.hypothesis_catalog import generate_catalog_hypotheses
from core.deep_analysis.hypothesis_llm import enrich_hypotheses
from core.deep_analysis.loader import LoadPlan, SafeLoader
from core.deep_analysis.logging_setup import attach_run_log, detach_run_log, get_logger
from core.deep_analysis.profiler import profile_dataframe, profile_to_brief
from core.deep_analysis.progress import ProgressCallback, ProgressReporter
from core.deep_analysis.report import write_report
from core.deep_analysis.runners import get_runner
from core.deep_analysis.types import (
    AnalysisContext,
    AnalysisMode,
    BusinessInsight,
    Finding,
    HypothesisSpec,
    TableProfile,
)
from core.deep_analysis.user_hypothesis import UserHypothesisPlan
from core.llm import RateLimitedLLM
from core.schema_loader import SchemaLoader

FAST_BUDGET_SEC = 40 * 60     # 40 min
DEEP_BUDGET_SEC = 3 * 60 * 60  # 3 hours

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent / "workspace" / "deep_analysis"

# Signature: loader_fn(db, schema, table, progress_cb, where=None)
#   -> (DataFrame, LoadPlan).
# Default implementation wraps SafeLoader; tests inject a fake. ``where`` is
# keyword-only with a default of None so existing test loaders that don't
# accept the kwarg still work via positional arity, but production callers
# always thread the user-supplied filter through.
LoaderFn = Callable[..., tuple[pd.DataFrame, LoadPlan]]
EnrichFn = Callable[[Any, TableProfile, list[HypothesisSpec], str], list[HypothesisSpec]]
# Signature: insights_fn(llm, findings, profile, user_hypothesis_text, table_semantics)
#   -> list[BusinessInsight]. Default wraps extract_business_insights; tests
# inject a fake to bypass the LLM and return a canned list.
InsightsFn = Callable[..., list[BusinessInsight]]


def _default_loader(
    db: DatabaseManager, schema: str, table: str, progress_cb, *, where: str | None = None
):
    return SafeLoader(db).plan_and_load(
        schema, table, progress_cb=progress_cb, where=where,
    )


@dataclass
class HypothesisRunRecord:
    """Per-hypothesis diagnostics, surfaced into the report.

    Captures everything we need to debug a run from logs alone — the tool has
    no remote access to the target environment, so the report itself is the
    primary feedback channel.
    """

    hypothesis_id: str
    runner: str
    title: str
    priority: float
    source: str
    status: str                        # "ok" | "skip" | "error" | "budget"
    n_findings: int = 0
    seconds: float = 0.0
    error_summary: str = ""


@dataclass
class AnalysisResult:
    output_dir: Path
    findings: list[Finding]
    profile: TableProfile
    hypotheses: list[HypothesisSpec]
    mode: AnalysisMode
    wall_seconds: float
    report_path: Path
    run_records: list[HypothesisRunRecord] = field(default_factory=list)
    business_insights: list[BusinessInsight] = field(default_factory=list)


def run_deep_analysis(
    schema: str,
    table: str,
    *,
    mode: AnalysisMode,
    db: DatabaseManager | None,
    llm: RateLimitedLLM | None,
    schema_loader: SchemaLoader | None,
    user_hypothesis: UserHypothesisPlan | None = None,
    progress_cb: ProgressCallback | None = None,
    loader_fn: LoaderFn | None = None,
    enrich_fn: EnrichFn | None = None,
    insights_fn: InsightsFn | None = None,
    output_root: Path | None = None,
    where: str | None = None,
) -> AnalysisResult:
    started_at = time.time()
    budget_sec = FAST_BUDGET_SEC if mode == AnalysisMode.FAST else DEEP_BUDGET_SEC
    deadline = started_at + budget_sec

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = output_root or WORKSPACE_ROOT
    output_dir = base_root / f"{schema}.{table}" / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handler = attach_run_log(output_dir)
    log = get_logger()
    log.info("Deep analysis start: %s.%s mode=%s budget=%ds out=%s",
             schema, table, mode.value, budget_sec, output_dir)

    progress = ProgressReporter(progress_cb)
    progress.set_stages(total=5)
    ctx = AnalysisContext(
        schema=schema, table=table, mode=mode,
        deadline_ts=deadline, output_dir=str(output_dir), progress=progress,
    )
    run_records: list[HypothesisRunRecord] = []

    try:
        # Stage 1: load
        progress.stage("Загружаю таблицу")
        loader = loader_fn or _default_loader
        if _loader_accepts_where(loader):
            df, plan = loader(db, schema, table, progress.sub, where=where)
        else:
            # Backward-compat: legacy/test loaders without the `where` kwarg.
            if where:
                log.warning(
                    "Loader %r does not support `where`; ignoring --where=%r",
                    loader, where,
                )
            df, plan = loader(db, schema, table, progress.sub)

        # Stage 2: profile
        progress.stage("Профилирую колонки")
        table_desc = _get_table_description(schema_loader, schema, table)
        df, profile = profile_dataframe(df, plan, table_description=table_desc)
        # Equivalence (post_id ↔ post_name ↔ pos_name etc.) — catalogue/runners
        # use representatives only so the report doesn't triple-count findings.
        profile.equivalence_groups = compute_equivalence_groups(df, profile)
        log.info("Profile brief:\n%s", profile_to_brief(profile))

        # Stage 3: hypothesis generation
        progress.stage("Формирую гипотезы")
        catalog = generate_catalog_hypotheses(profile)
        progress.sub(f"Каталог: {len(catalog)} шт.")
        if enrich_fn is not None:
            enriched = enrich_fn(llm, profile, catalog, table_desc)
        elif llm is not None:
            try:
                enriched = enrich_hypotheses(llm, profile, catalog, table_semantics=table_desc)
            except Exception as exc:
                log.warning("LLM enrichment failed: %s", exc)
                enriched = catalog
        else:
            enriched = catalog
        hypotheses = list(enriched)
        if user_hypothesis is not None:
            hypotheses.insert(0, user_hypothesis.hypothesis)
        hypotheses.sort(key=lambda h: h.priority, reverse=True)
        progress.sub(f"Всего гипотез: {len(hypotheses)} (с учётом LLM)")

        # Stage 4: execute runners
        progress.stage("Проверяю гипотезы")
        findings = _execute_hypotheses(df, profile, hypotheses, ctx, run_records)

        # Stage 4b: business insights — distill findings into 0..5 actionable
        # takeaways. Non-fatal: if the LLM fails or no findings qualify, the
        # report still renders without the "Главное для бизнеса" section.
        business_insights: list[BusinessInsight] = []
        user_focus = (
            user_hypothesis.hypothesis.rationale
            if user_hypothesis is not None else None
        )
        if insights_fn is not None:
            try:
                business_insights = insights_fn(
                    llm, findings, profile, user_focus, table_desc,
                )
            except Exception as exc:
                log.warning("Injected insights_fn failed: %s", exc)
        elif llm is not None and findings:
            progress.sub("Формирую бизнес-инсайты")
            try:
                business_insights = extract_business_insights(
                    llm, findings, profile,
                    user_hypothesis_text=user_focus,
                    table_semantics=table_desc,
                )
            except Exception as exc:
                log.warning("Business insights stage failed: %s", exc)

        # Stage 5: report
        progress.stage("Формирую отчёт")
        report_path = write_report(
            findings, profile, hypotheses, mode, output_dir,
            run_records=run_records,
            wall_seconds=time.time() - started_at,
            business_insights=business_insights,
        )
    finally:
        detach_run_log(log_handler)

    wall = time.time() - started_at
    log.info("Deep analysis done in %.1fs, findings=%d", wall, len(findings) if 'findings' in locals() else -1)
    return AnalysisResult(
        output_dir=output_dir,
        findings=findings,
        profile=profile,
        hypotheses=hypotheses,
        mode=mode,
        wall_seconds=wall,
        report_path=report_path,
        business_insights=business_insights if 'business_insights' in locals() else [],
        run_records=run_records,
    )


def _loader_accepts_where(loader: LoaderFn) -> bool:
    try:
        sig = inspect.signature(loader)
    except (TypeError, ValueError):
        return True
    return (
        "where" in sig.parameters
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    )


def _execute_hypotheses(
    df,
    profile: TableProfile,
    hypotheses: list[HypothesisSpec],
    ctx: AnalysisContext,
    run_records: list[HypothesisRunRecord],
) -> list[Finding]:
    log = get_logger()
    findings: list[Finding] = []
    total = len(hypotheses)
    for i, spec in enumerate(hypotheses, start=1):
        rec = HypothesisRunRecord(
            hypothesis_id=spec.hypothesis_id, runner=spec.runner,
            title=spec.title, priority=spec.priority, source=spec.source,
            status="pending",
        )
        if ctx.seconds_left() <= 10:
            rec.status = "budget"
            run_records.append(rec)
            log.warning("Budget exhausted, stopping at %d/%d hypotheses", i - 1, total)
            ctx.progress.sub(f"Бюджет исчерпан на {i - 1}/{total} гипотезах")
            # Record the rest as skipped so the report reflects what was dropped.
            for later in hypotheses[i:]:
                run_records.append(HypothesisRunRecord(
                    hypothesis_id=later.hypothesis_id, runner=later.runner,
                    title=later.title, priority=later.priority, source=later.source,
                    status="budget",
                ))
            break
        runner = get_runner(spec.runner)
        if runner is None:
            rec.status = "skip"
            rec.error_summary = f"unknown runner {spec.runner!r}"
            run_records.append(rec)
            log.warning("Unknown runner: %s (skip)", spec.runner)
            continue
        ctx.progress.hypothesis(i, total, spec.title)
        started = time.time()
        try:
            produced = runner(df, profile, spec, ctx) or []
            rec.status = "ok"
            rec.n_findings = len(produced)
        except Exception as exc:
            log.error("Runner %s failed for %s: %s\n%s",
                      spec.runner, spec.hypothesis_id, exc, traceback.format_exc())
            produced = []
            rec.status = "error"
            rec.error_summary = f"{type(exc).__name__}: {exc}"
        rec.seconds = time.time() - started
        run_records.append(rec)
        findings.extend(produced)
        log.info("Hypothesis %s took %.1fs, produced %d findings",
                 spec.hypothesis_id, rec.seconds, len(produced))
    return findings


def _get_table_description(schema_loader: SchemaLoader | None, schema: str, table: str) -> str:
    if schema_loader is None:
        return ""
    try:
        tdf = schema_loader.tables_df
        row = tdf[(tdf["schema_name"] == schema) & (tdf["table_name"] == table)]
        if not row.empty:
            return str(row.iloc[0].get("description") or "").strip()
    except Exception:
        pass
    return ""
