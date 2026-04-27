"""Tests for the post-v1 refinements:

- seasonality: effect-size-aware severity (small absolute deviation must not
  reach `critical`, even if p-value is astronomical due to sample size).
- equivalence: 1:1-related columns are folded; representatives drive runners.
- dependencies: reference-table siblings (post_id ↔ post_name) are filtered.
- report: thematic grouping renders the right section headers; TL;DR block
  appears at the top with up to 5 bullets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.deep_analysis.equivalence import compute_equivalence_groups
from core.deep_analysis.loader import LoadPlan
from core.deep_analysis.orchestrator import run_deep_analysis
from core.deep_analysis.profiler import profile_dataframe
from core.deep_analysis.progress import ProgressReporter
from core.deep_analysis.runners.dependencies import (
    _is_reference_pair,
    run_dependencies,
)
from core.deep_analysis.runners.seasonality import run_seasonality
from core.deep_analysis.types import (
    AnalysisContext,
    AnalysisMode,
    Finding,
    HypothesisSpec,
)


# -----------------------------------------------------------------------------
# #1 — seasonality severity by effect size
# -----------------------------------------------------------------------------
def test_seasonality_micro_deviation_is_not_critical():
    """At N=200k a 0.5% deviation will be p<<1e-10 but is business noise.

    The new rule requires |dev| ≥ 0.20 AND p<1e-10 for `critical`. A tiny
    deviation injected on top of a uniform calendar must therefore land in
    `notable` or `info`, not `critical`.
    """
    rng = np.random.default_rng(0)
    n = 200_000
    dates = pd.to_datetime(
        rng.choice(pd.date_range("2024-01-01", "2025-12-31", freq="D"), size=n)
    )
    # Create the smallest possible bias: 0.6% extra rows on day-of-week==0.
    boost = (dates.dayofweek == 0) & (rng.random(n) < 0.006)
    extra_dates = dates[boost]
    df = pd.DataFrame({"event_dt": list(dates) + list(extra_dates)})

    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=8, est_full_bytes=8 * len(df),
    )
    df, profile = profile_dataframe(df, plan)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="t", table="x", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        spec = HypothesisSpec(
            hypothesis_id="weak_seasonality", runner="seasonality",
            title="rows by date", rationale="",
            params={"date_col": "event_dt", "value_col": None, "agg": "count"},
            priority=1.0,
        )
        findings = run_seasonality(df, profile, spec, ctx)
        # Any finding for a near-uniform calendar with 0.6% bias must NOT be
        # critical — that severity is reserved for ≥20% effects.
        criticals = [f for f in findings if f.severity == "critical"]
        assert not criticals, (
            f"Expected no `critical` findings on 0.6%-bias seasonality; "
            f"got: {[(f.title, f.metrics.get('max_rel_deviation_pct')) for f in criticals]}"
        )


def test_seasonality_strong_deviation_is_critical():
    """Inverse direction: a +50% bias on weekends MUST land as `critical`."""
    rng = np.random.default_rng(0)
    base = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    rows: list[pd.Timestamp] = []
    for d in base:
        # Weekday: 100 events. Weekend: 200 events.
        n_events = 200 if d.dayofweek >= 5 else 100
        rows.extend([d] * n_events)
    df = pd.DataFrame({"event_dt": rows})
    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=8, est_full_bytes=8 * len(df),
    )
    df, profile = profile_dataframe(df, plan)
    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="t", table="x", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        spec = HypothesisSpec(
            hypothesis_id="strong_seasonality", runner="seasonality",
            title="rows by date", rationale="",
            params={"date_col": "event_dt", "value_col": None, "agg": "count"},
            priority=1.0,
        )
        findings = run_seasonality(df, profile, spec, ctx)
        dow_findings = [
            f for f in findings if f.metrics.get("horizon") == "день недели"
        ]
        assert dow_findings, "Expected a day-of-week seasonality finding"
        assert any(f.severity == "critical" for f in dow_findings), (
            f"Expected day-of-week finding with severity=critical; got "
            f"{[(f.severity, f.metrics.get('max_rel_deviation_pct')) for f in dow_findings]}"
        )


# -----------------------------------------------------------------------------
# #2 — equivalence detection + dedup in catalog/runners
# -----------------------------------------------------------------------------
def test_equivalence_groups_fold_1to1_columns():
    """post_id ↔ post_name ↔ pos_name with V=1.0 must collapse to one rep."""
    rng = np.random.default_rng(0)
    n = 5000
    post_id = rng.choice([1, 2, 3, 4, 5], size=n)
    name_map = {1: "manager", 2: "specialist", 3: "director", 4: "intern", 5: "lead"}
    post_name = np.array([name_map[i] for i in post_id])
    pos_name = post_name  # identical column under another name
    other = rng.choice(["a", "b", "c"], size=n)  # independent
    df = pd.DataFrame({
        "post_id": post_id, "post_name": post_name,
        "pos_name": pos_name, "other_cat": other,
    })
    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=20, est_full_bytes=20 * len(df),
    )
    df, profile = profile_dataframe(df, plan)
    groups = compute_equivalence_groups(df, profile)

    # Exactly one multi-member class containing all three trio members.
    multi = [members for members in groups.values() if len(members) > 1]
    assert len(multi) == 1, f"Expected one equivalence class, got: {multi}"
    assert set(multi[0]) == {"post_id", "post_name", "pos_name"}, multi[0]
    # Independent column stays singleton.
    assert ["other_cat"] in groups.values()


def test_catalog_does_not_duplicate_hypotheses_across_equivalents():
    """When 3 columns are equivalent, the seasonality_cat catalog should emit
    one hypothesis covering the representative — not three siblings."""
    from core.deep_analysis.hypothesis_catalog import generate_catalog_hypotheses

    rng = np.random.default_rng(0)
    n = 4000
    pid = rng.choice([10, 20, 30], size=n)
    pname = np.array(["a", "b", "c"])[(pid // 10) - 1]
    posname = pname
    df = pd.DataFrame({
        "report_dt": pd.to_datetime(
            rng.choice(pd.date_range("2024-01-01", "2024-12-31"), size=n)
        ),
        "post_id": pid, "post_name": pname, "pos_name": posname,
    })
    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=20, est_full_bytes=20 * len(df),
    )
    df, profile = profile_dataframe(df, plan)
    profile.equivalence_groups = compute_equivalence_groups(df, profile)

    catalog = generate_catalog_hypotheses(profile)
    cat_seasonality = [
        h for h in catalog
        if h.runner == "seasonality" and h.params.get("group_col") in
        {"post_id", "post_name", "pos_name"}
    ]
    assert len(cat_seasonality) == 1, (
        f"Expected exactly one seasonality_cat hypothesis across the "
        f"equivalence class; got {len(cat_seasonality)}: "
        f"{[h.params.get('group_col') for h in cat_seasonality]}"
    )


# -----------------------------------------------------------------------------
# #3 — reference-pair filter in dependencies
# -----------------------------------------------------------------------------
def test_is_reference_pair_recognises_id_name_siblings():
    assert _is_reference_pair("post_id", "post_name")
    assert _is_reference_pair("tb_code", "tb_id")
    assert _is_reference_pair("manager_lvl_1_post_id", "manager_lvl_1_name")
    assert _is_reference_pair("post", "post_id")
    # Non-pairs: different stems, or the same suffix on both sides.
    assert not _is_reference_pair("post_id", "client_id")
    assert not _is_reference_pair("region", "segment")


def test_dependencies_skips_reference_pair():
    """A perfectly-1:1 id↔name pair must NOT show up as a dependency finding."""
    rng = np.random.default_rng(0)
    n = 5000
    post_id = rng.choice([1, 2, 3, 4, 5], size=n)
    post_name = np.array(["m", "s", "d", "i", "l"])[post_id - 1]
    region = rng.choice(["moscow", "spb", "other"], size=n)
    df = pd.DataFrame({"post_id": post_id, "post_name": post_name, "region": region})
    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=20, est_full_bytes=20 * len(df),
    )
    df, profile = profile_dataframe(df, plan)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="t", table="x", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        spec = HypothesisSpec(
            hypothesis_id="dep", runner="dependencies",
            title="pairs", rationale="",
            params={"columns": ["post_id", "post_name", "region"], "max_pairs": 10},
            priority=1.0,
        )
        findings = run_dependencies(df, profile, spec, ctx)
        for f in findings:
            assert not (
                "post_id" in f.title and "post_name" in f.title
            ), f"Reference pair leaked through filter: {f.title}"


# -----------------------------------------------------------------------------
# Report — TL;DR + thematic grouping
# -----------------------------------------------------------------------------
def test_first_sentence_truncates_long_text():
    from core.deep_analysis.report import _first_sentence as _fs
    assert _fs("Hello. World.") == "Hello."
    assert _fs("Single short line") == "Single short line"
    long_text = "x" * 400
    out = _fs(long_text, limit=50)
    assert len(out) <= 50
    assert out.endswith("…")


def _make_finding(
    *, runner: str, severity: str, title: str, summary: str = "S.", **metrics
) -> Finding:
    return Finding(
        hypothesis_id=f"h_{title}", runner=runner, title=title,
        severity=severity, summary=summary, metrics=metrics,
    )


def test_report_renders_tldr_and_themes():
    from core.deep_analysis.report import _render_markdown
    from core.deep_analysis.types import TableProfile

    findings = [
        _make_finding(
            runner="group_anomalies", severity="critical",
            title="A critical group", summary="Severe anomaly in cohort.",
            n_violators=42, max_abs_z=10.0,
        ),
        _make_finding(
            runner="seasonality", severity="strong",
            title="Strong seasonality", summary="Clear weekly pattern.",
            max_rel_deviation_pct=25.0,
        ),
        _make_finding(
            runner="dependencies", severity="notable",
            title="Notable link", summary="V=0.4 between X and Y.",
            cramer_v=0.4,
        ),
    ]
    profile = TableProfile(
        schema="s", table="t", n_rows=1000, n_cols=3, columns={},
        equivalence_groups={},
    )
    md = _render_markdown(
        findings, profile, hypotheses=[], mode=AnalysisMode.FAST,
        output_dir=Path("/tmp"), run_records=[], wall_seconds=1.0,
    )
    # TL;DR appears, lists the critical finding first.
    assert "## TL;DR" in md
    assert md.index("## TL;DR") < md.index("## Найдено значимых")
    assert "A critical group" in md
    # Thematic sections present with their localized names.
    assert "Аномальные сущности и группы" in md
    assert "Календарные паттерны" in md
    assert "Связи между колонками" in md
    # Equivalence block absent when no multi-member groups.
    assert "## Эквивалентные колонки" not in md


def test_report_renders_equivalence_block():
    from core.deep_analysis.report import _render_markdown
    from core.deep_analysis.types import TableProfile

    profile = TableProfile(
        schema="s", table="t", n_rows=1000, n_cols=4,
        columns={},
        equivalence_groups={
            "post_id": ["pos_name", "post_id", "post_name"],
            "other": ["other"],
        },
    )
    md = _render_markdown(
        findings=[], profile=profile, hypotheses=[],
        mode=AnalysisMode.FAST, output_dir=Path("/tmp"),
        run_records=[], wall_seconds=0.0,
    )
    assert "## Эквивалентные колонки" in md
    assert "`post_id`" in md
    assert "`post_name`" in md
    assert "`pos_name`" in md


# -----------------------------------------------------------------------------
# Integration smoke — orchestrator must populate equivalence_groups field.
# -----------------------------------------------------------------------------
def test_orchestrator_populates_equivalence_field():
    """End-to-end: 3 1:1 columns in the input → orchestrator detects them."""
    rng = np.random.default_rng(0)
    n = 3000
    pid = rng.choice([10, 20, 30, 40], size=n)
    name_map = {10: "a", 20: "b", 30: "c", 40: "d"}
    df = pd.DataFrame({
        "report_dt": pd.to_datetime(
            rng.choice(pd.date_range("2024-01-01", "2024-12-31"), size=n)
        ),
        "post_id": pid,
        "post_name": np.array([name_map[i] for i in pid]),
        "pos_name": np.array([name_map[i] for i in pid]),
        "amount": rng.normal(100, 10, size=n),
    })

    def fake_loader(db, schema, table, progress_cb):
        plan = LoadPlan(
            schema=schema, table=table, total_rows=len(df),
            kept_columns=list(df.columns), dropped_wide_text=[],
            strategy="full", sample_rows=None,
            est_bytes_per_row=30, est_full_bytes=30 * len(df),
        )
        return df, plan

    def fake_enrich(llm, profile, catalog, semantics):
        return catalog

    with tempfile.TemporaryDirectory() as tmp:
        result = run_deep_analysis(
            "s", "t",
            mode=AnalysisMode.FAST,
            db=None, llm=None, schema_loader=None,
            loader_fn=fake_loader, enrich_fn=fake_enrich,
            output_root=Path(tmp),
        )
        groups = result.profile.equivalence_groups
        multi = [m for m in groups.values() if len(m) > 1]
        assert multi, f"Expected at least one equivalence class; got {groups}"
        assert set(multi[0]) == {"post_id", "post_name", "pos_name"}
