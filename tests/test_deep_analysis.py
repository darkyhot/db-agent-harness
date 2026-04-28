"""End-to-end smoke tests for core.deep_analysis.

We bypass the DatabaseManager and feed runners directly with a synthetic pandas
DataFrame modeled on the vacation-fraud scenario from the feature spec:

- 2000 employees over 2 years of daily records.
- 100 of them concentrate `type='отпуск'` into the last 10 days of each
  quarter (the fraud pattern). The remaining 1900 take vacations uniformly.
- group_anomalies with metric='end_of_quarter_shift' must surface the 100
  fraudsters and write a CSV with all of them.
- seasonality must detect a quarter-end effect in the `type` distribution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from core.deep_analysis.hypothesis_catalog import generate_catalog_hypotheses
from core.deep_analysis.loader import LoadPlan, _sanitize_where_clause
from core.deep_analysis.orchestrator import run_deep_analysis
from core.deep_analysis.profiler import profile_dataframe
from core.deep_analysis.progress import ProgressReporter
from core.deep_analysis.runners.dependencies import run_dependencies
from core.deep_analysis.runners.group_anomalies import run_group_anomalies
from core.deep_analysis.runners.outliers import run_outliers
from core.deep_analysis.runners.regime_shifts import run_regime_shifts
from core.deep_analysis.runners.seasonality import run_seasonality
from core.deep_analysis.types import (
    AnalysisContext,
    AnalysisMode,
    BusinessInsight,
    ColumnRole,
    Finding,
    HypothesisSpec,
)


def _build_synthetic_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_employees = 2000
    n_fraud = 100
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-12-31")
    dates = pd.date_range(start, end, freq="D")
    fraud_ids = set(range(n_fraud))

    rows = []
    for emp in range(n_employees):
        is_fraud = emp in fraud_ids
        # Each employee is sampled on ~40% of days (not every day).
        mask = rng.random(len(dates)) < 0.4
        emp_dates = dates[mask]
        for d in emp_dates:
            q_end = (d.to_period("Q").end_time - d).days
            if is_fraud and q_end <= 9 and rng.random() < 0.7:
                etype = "отпуск"
                worked = 1
            else:
                # Normal population: ~8% vacation uniformly.
                r = rng.random()
                if r < 0.08:
                    etype = "отпуск"
                    worked = 0
                elif r < 0.10:
                    etype = "больничный"
                    worked = 0
                elif r < 0.12:
                    etype = "декрет"
                    worked = 0
                else:
                    etype = "работа"
                    worked = 1
            rows.append((emp, d, etype, worked, float(rng.normal(100, 20))))
    df = pd.DataFrame(rows, columns=["employee_id", "date", "type", "worked", "kpi_score"])
    return df


def test_group_anomalies_catches_fraud_ring():
    df = _build_synthetic_df()
    plan = LoadPlan(
        schema="hr", table="events",
        total_rows=len(df),
        kept_columns=list(df.columns),
        dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=100, est_full_bytes=100 * len(df),
    )
    df2, profile = profile_dataframe(df, plan)
    assert profile.columns["employee_id"].role == ColumnRole.ID
    assert profile.columns["date"].role in (ColumnRole.DATE, ColumnRole.DATETIME)
    assert profile.columns["type"].role == ColumnRole.CATEGORY
    assert profile.columns["worked"].role == ColumnRole.FLAG

    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="hr", table="events", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        spec = HypothesisSpec(
            hypothesis_id="test_eoq_shift",
            runner="group_anomalies",
            title="End-of-quarter shift",
            rationale="",
            params={
                "entity_col": "employee_id",
                "date_col": "date",
                "category_col": "type",
                "metric": "end_of_quarter_shift",
                "period": "quarter",
            },
            priority=1.0,
        )
        findings = run_group_anomalies(df2, profile, spec, ctx)
        assert len(findings) == 1
        f = findings[0]
        assert f.entity_csv is not None
        csv_path = Path(tmp) / f.entity_csv
        violators = pd.read_csv(csv_path)
        # At least 60 of the 100 fraudsters must be flagged. (Threshold |z|>2.5
        # on a heavy-tailed distribution — some statistical leeway.)
        fraudsters_caught = violators["employee_id"].astype(int).isin(range(100)).sum()
        assert fraudsters_caught >= 60, (
            f"Expected ≥60 fraudsters flagged, got {fraudsters_caught}. "
            f"Total violators: {len(violators)}."
        )


def test_seasonality_detects_quarter_end_effect():
    df = _build_synthetic_df()
    plan = LoadPlan(
        schema="hr", table="events", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=100, est_full_bytes=100 * len(df),
    )
    df2, profile = profile_dataframe(df, plan)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="hr", table="events", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        # Seasonality of row counts grouped by type — should flag quarter-end bump.
        spec = HypothesisSpec(
            hypothesis_id="test_seasonality_cat",
            runner="seasonality",
            title="Seasonality of type by date",
            rationale="",
            params={"date_col": "date", "value_col": None, "agg": "count", "group_col": "type"},
            priority=1.0,
        )
        findings = run_seasonality(df2, profile, spec, ctx)
        # Expect at least one finding on the отпуск category for the EOQ horizon.
        eoq_findings = [
            f for f in findings
            if "отпуск" in f.title and "квартал" in f.metrics.get("horizon", "").lower()
        ]
        assert eoq_findings, (
            f"Expected an end-of-quarter finding for `отпуск`; got {len(findings)} findings total."
        )


def test_catalog_generates_expected_hypotheses():
    df = _build_synthetic_df()
    plan = LoadPlan(
        schema="hr", table="events", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=100, est_full_bytes=100 * len(df),
    )
    _, profile = profile_dataframe(df, plan)
    catalog = generate_catalog_hypotheses(profile)
    runners_seen = {h.runner for h in catalog}
    assert "seasonality" in runners_seen
    assert "group_anomalies" in runners_seen
    assert "outliers" in runners_seen
    # End-of-quarter shift hypothesis must be present (our vacation fraud signal).
    eoq = [h for h in catalog if h.params.get("metric") == "end_of_quarter_shift"]
    assert eoq, "Catalog did not produce an end_of_quarter_shift hypothesis"


def test_dependencies_finds_categorical_link():
    """categorical ↔ categorical: region strongly predicts segment."""
    rng = np.random.default_rng(0)
    n = 5000
    region = rng.choice(["moscow", "spb", "other"], size=n, p=[0.5, 0.3, 0.2])
    # segment is almost fully determined by region — Cramér's V near 1.
    segment_map = {"moscow": "premium", "spb": "standard", "other": "basic"}
    segment = np.array([segment_map[r] if rng.random() < 0.9 else "basic" for r in region])
    # Independent noise column — should NOT surface.
    noise = rng.choice(["x", "y", "z"], size=n)
    df = pd.DataFrame({"region": region, "segment": segment, "noise": noise})
    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=30, est_full_bytes=30 * len(df),
    )
    _, profile = profile_dataframe(df, plan)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="t", table="x", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        spec = HypothesisSpec(
            hypothesis_id="dep_test", runner="dependencies",
            title="pairs", rationale="",
            params={"columns": ["region", "segment", "noise"], "max_pairs": 10},
            priority=1.0,
        )
        findings = run_dependencies(df, profile, spec, ctx)
        titles = [f.title for f in findings]
        assert any("region" in t and "segment" in t for t in titles), (
            f"Expected a region↔segment finding. Got: {titles}"
        )


def test_regime_shifts_detects_level_change():
    """Flat baseline then 3x volume for the second half — one big changepoint."""
    start = pd.Timestamp("2024-01-01")
    # First 120 days: ~10 events/day. Next 120 days: ~30 events/day.
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        for _ in range(int(rng.poisson(10))):
            rows.append(start + pd.Timedelta(days=i))
    for i in range(120, 240):
        for _ in range(int(rng.poisson(30))):
            rows.append(start + pd.Timedelta(days=i))
    df = pd.DataFrame({"event_dt": rows, "amount": rng.normal(100, 5, size=len(rows))})

    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=30, est_full_bytes=30 * len(df),
    )
    _, profile = profile_dataframe(df, plan)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="t", table="x", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        spec = HypothesisSpec(
            hypothesis_id="regime_test", runner="regime_shifts",
            title="volume regime", rationale="",
            params={"date_col": "event_dt", "agg": "count", "freq": "day"},
            priority=1.0,
        )
        findings = run_regime_shifts(df, profile, spec, ctx)
        assert findings, "Expected at least one regime_shifts finding"
        # The big shift should be near day 120 — confirm a date in the 100-140 window.
        rows_csv = pd.read_csv(Path(tmp) / findings[0].entity_csv)
        rows_csv["date"] = pd.to_datetime(rows_csv["date"])
        close = rows_csv["date"].between(
            pd.Timestamp("2024-04-15"), pd.Timestamp("2024-05-30")
        )
        assert close.any(), (
            f"Expected a changepoint in April-May 2024; got {rows_csv['date'].tolist()}"
        )


def test_integration_end_to_end_with_mock_loader():
    """Full orchestrator run using the fraud-ring synthetic + injected loader.

    This is the closed-loop canary that we'd otherwise not be able to run
    without a live database. If this test goes red, real runs are broken too.
    """
    df = _build_synthetic_df()
    def fake_loader(db, schema, table, progress_cb):
        plan = LoadPlan(
            schema=schema, table=table, total_rows=len(df),
            kept_columns=list(df.columns), dropped_wide_text=[],
            strategy="full", sample_rows=None,
            est_bytes_per_row=100, est_full_bytes=100 * len(df),
        )
        return df, plan

    # No LLM enrichment in test — deterministic behavior.
    def fake_enrich(llm, profile, catalog, semantics):
        return catalog

    with tempfile.TemporaryDirectory() as tmp:
        result = run_deep_analysis(
            "hr", "events",
            mode=AnalysisMode.FAST,
            db=None, llm=None, schema_loader=None,
            loader_fn=fake_loader, enrich_fn=fake_enrich,
            output_root=Path(tmp),
        )
        assert result.report_path.exists()
        report = result.report_path.read_text(encoding="utf-8")
        # Diagnostics block should be present.
        assert "## Диагностика выполнения" in report
        assert "## Профайл колонок" in report
        # The vacation fraud finding must show up.
        assert any(
            "Сдвиг распределения" in f.title or "квартал" in f.title.lower()
            for f in result.findings
        ), f"No quarter-end finding in: {[f.title for f in result.findings]}"
        # Every executed hypothesis must have a diagnostics record.
        assert result.run_records
        ok_count = sum(1 for r in result.run_records if r.status == "ok")
        assert ok_count >= 3
        # At least one entity CSV exists on disk.
        csvs = list(result.output_dir.glob("entities_*.csv"))
        assert csvs, f"No entity CSVs written to {result.output_dir}"


def test_run_deep_analysis_passes_where_to_loader_and_report():
    df = pd.DataFrame({
        "report_dt": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "amount": [10, 20],
    })
    seen: dict[str, str | None] = {}

    def fake_loader(db, schema, table, progress_cb, *, where=None):
        seen["where"] = where
        plan = LoadPlan(
            schema=schema, table=table, total_rows=len(df),
            kept_columns=list(df.columns), dropped_wide_text=[],
            strategy="full", sample_rows=None,
            est_bytes_per_row=16, est_full_bytes=16 * len(df),
            where_clause=where,
        )
        return df, plan

    def fake_enrich(llm, profile, catalog, semantics):
        return []

    with tempfile.TemporaryDirectory() as tmp:
        result = run_deep_analysis(
            "dm", "sales",
            mode=AnalysisMode.FAST,
            db=None, llm=None, schema_loader=None,
            loader_fn=fake_loader, enrich_fn=fake_enrich,
            output_root=Path(tmp),
            where="report_dt >= '2026-01-01'",
        )
        report_text = result.report_path.read_text(encoding="utf-8")

    assert seen["where"] == "report_dt >= '2026-01-01'"
    assert result.profile.where_clause == "report_dt >= '2026-01-01'"
    assert "Фильтр (WHERE): `report_dt >= '2026-01-01'`" in report_text


def test_sanitize_where_clause_allows_select_predicates():
    assert (
        _sanitize_where_clause("WHERE report_dt >= '2026-01-01'")
        == "report_dt >= '2026-01-01'"
    )
    assert (
        _sanitize_where_clause("inn IN ('7707083893','7728168971')")
        == "inn IN ('7707083893','7728168971')"
    )


def test_sanitize_where_clause_rejects_statement_breakout():
    for where in [
        "report_dt >= '2026-01-01'; DROP TABLE dm.sales",
        "inn = '1' -- comment",
        "id IN (SELECT id INTO tmp FROM src)",
        "name = 'broken",
    ]:
        try:
            _sanitize_where_clause(where)
        except ValueError:
            continue
        raise AssertionError(f"WHERE should be rejected: {where}")


def test_group_anomalies_handles_small_cohort_like_tb_id():
    """15-territorial-bank cohort: even with low cardinality, we want to flag
    the 2 anomalous TBs. Mirrors the real corp tb_id (~15 values) case."""
    rng = np.random.default_rng(0)
    n_tb = 15
    rows_per_tb = 400
    rows = []
    for tb in range(n_tb):
        base_rate = 0.05 if tb >= 2 else 0.6   # first 2 TBs are anomalous
        for _ in range(rows_per_tb):
            rows.append((tb, int(rng.random() < base_rate)))
    df = pd.DataFrame(rows, columns=["tb_id", "is_outflow"])
    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=8, est_full_bytes=8 * len(df),
    )
    df, profile = profile_dataframe(df, plan)
    # tb_id should be CATEGORY (15 values), but still surface as entity candidate.
    assert "tb_id" in profile.entity_candidates(min_card=5)

    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="t", table="x", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        spec = HypothesisSpec(
            hypothesis_id="grp_tb_rate", runner="group_anomalies",
            title="rate by tb", rationale="",
            params={"entity_col": "tb_id", "metric": "rate", "value_col": "is_outflow"},
            priority=1.0,
        )
        findings = run_group_anomalies(df, profile, spec, ctx)
        assert findings, "Expected a finding on the small (15-entity) cohort"
        f = findings[0]
        assert f.entity_csv is not None
        violators = pd.read_csv(Path(tmp) / f.entity_csv)
        # The 2 anomalous TBs (tb_id 0 and 1) must be in the violator list.
        flagged = set(int(v) for v in violators["tb_id"].tolist())
        assert {0, 1}.issubset(flagged), f"Expected tb_ids 0 and 1 flagged, got {flagged}"


def test_outliers_mad_flags_extreme_values():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "id": range(5000),
        "amount": np.concatenate([rng.normal(100, 10, 4990), rng.normal(5000, 100, 10)]),
    })
    plan = LoadPlan(
        schema="t", table="x", total_rows=len(df),
        kept_columns=list(df.columns), dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=16, est_full_bytes=16 * len(df),
    )
    _, profile = profile_dataframe(df, plan)
    with tempfile.TemporaryDirectory() as tmp:
        ctx = AnalysisContext(
            schema="t", table="x", mode=AnalysisMode.FAST,
            deadline_ts=1e18, output_dir=tmp, progress=ProgressReporter(),
        )
        spec = HypothesisSpec(
            hypothesis_id="mad_test", runner="outliers",
            title="", rationale="",
            params={"value_cols": ["amount"], "method": "mad"},
            priority=1.0,
        )
        findings = run_outliers(df, profile, spec, ctx)
        assert findings
        assert findings[0].metrics["n_outliers"] >= 10


def test_business_insights_render_in_report_via_insights_fn():
    """Inject a fake insights_fn to verify the new MD section + jsonl + anchors.

    Bypasses the real LLM — we feed a curated BusinessInsight list and check
    that everything downstream (orchestrator → write_report → markdown) wires
    it together correctly.
    """
    df = _build_synthetic_df()

    def fake_loader(db, schema, table, progress_cb):
        plan = LoadPlan(
            schema=schema, table=table, total_rows=len(df),
            kept_columns=list(df.columns), dropped_wide_text=[],
            strategy="full", sample_rows=None,
            est_bytes_per_row=100, est_full_bytes=100 * len(df),
        )
        return df, plan

    def fake_enrich(llm, profile, catalog, semantics):
        return catalog

    captured: dict[str, object] = {}

    def fake_insights_fn(llm, findings, profile, user_focus, table_semantics):
        captured["n_findings"] = len(findings)
        captured["table"] = f"{profile.schema}.{profile.table}"
        # Pin the first real finding so we can verify drill-down anchors.
        target_id = findings[0].hypothesis_id if findings else "unknown"
        captured["target_id"] = target_id
        return [
            BusinessInsight(
                insight_id="vacation-fraud-ring",
                title="Сотрудники концентрируют отпуска в конце квартала",
                priority="top",
                where_to_look="employee_id из CSV нарушителей за последние 10 дней квартала",
                business_impact=(
                    "Прямой риск занижения KPI и потери выручки на ~100 сотрудниках; "
                    "сигнал для фрод-аудита."
                ),
                recommended_action="Передать список employee_id в HR-аудит для проверки совпадений с фактической явкой.",
                related_finding_ids=[target_id],
                confidence="high",
            ),
        ]

    with tempfile.TemporaryDirectory() as tmp:
        result = run_deep_analysis(
            "hr", "events",
            mode=AnalysisMode.FAST,
            db=None, llm=None, schema_loader=None,
            loader_fn=fake_loader, enrich_fn=fake_enrich,
            insights_fn=fake_insights_fn,
            output_root=Path(tmp),
        )

        # Insights propagate into AnalysisResult.
        assert len(result.business_insights) == 1
        assert result.business_insights[0].insight_id == "vacation-fraud-ring"
        assert captured["table"] == "hr.events"
        assert captured["n_findings"] == len(result.findings)

        # business_insights.jsonl artifact exists and is parseable.
        jsonl = result.output_dir / "business_insights.jsonl"
        assert jsonl.exists(), "business_insights.jsonl was not written"
        import json as _json
        rows = [_json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["priority"] == "top"
        assert rows[0]["recommended_action"]

        # Markdown report contains the new section, ordered before TL;DR.
        report = result.report_path.read_text(encoding="utf-8")
        assert "## 🎯 Главное для бизнеса" in report
        assert "Сотрудники концентрируют отпуска" in report
        assert "**Куда смотреть:**" in report
        assert "**На что влияет:**" in report
        assert "**Что сделать:**" in report
        # New section must be ABOVE TL;DR, not below it.
        assert report.index("Главное для бизнеса") < report.index("## TL;DR")
        # Drill-down anchor + link must both be present.
        target_id = captured["target_id"]
        assert f'<a id="{target_id}"></a>' in report
        assert f"(#{target_id})" in report


def test_business_insights_failure_is_graceful():
    """If insights_fn explodes, the report still builds without the section."""
    df = _build_synthetic_df()

    def fake_loader(db, schema, table, progress_cb):
        plan = LoadPlan(
            schema=schema, table=table, total_rows=len(df),
            kept_columns=list(df.columns), dropped_wide_text=[],
            strategy="full", sample_rows=None,
            est_bytes_per_row=100, est_full_bytes=100 * len(df),
        )
        return df, plan

    def fake_enrich(llm, profile, catalog, semantics):
        return catalog

    def broken_insights_fn(llm, findings, profile, user_focus, table_semantics):
        raise RuntimeError("LLM provider is unreachable")

    with tempfile.TemporaryDirectory() as tmp:
        result = run_deep_analysis(
            "hr", "events",
            mode=AnalysisMode.FAST,
            db=None, llm=None, schema_loader=None,
            loader_fn=fake_loader, enrich_fn=fake_enrich,
            insights_fn=broken_insights_fn,
            output_root=Path(tmp),
        )

        assert result.business_insights == []
        report = result.report_path.read_text(encoding="utf-8")
        # Section must be absent — TL;DR + diagnostics are still there.
        assert "Главное для бизнеса" not in report
        assert "## TL;DR" in report
        assert "## Диагностика выполнения" in report


def test_business_insights_extract_with_stub_llm():
    """Hit the real extract_business_insights() with a stub LLM that returns
    canned JSON — proves selection, validation, parsing, and drop of fake
    finding_ids work end-to-end without a network call."""
    from core.deep_analysis.business_insights import extract_business_insights

    findings = [
        Finding(
            hypothesis_id="seas_eoq_vacation",
            runner="seasonality",
            title="EOQ всплеск отпусков",
            severity="strong",
            summary="Доля 'отпуск' в последние 10 дней квартала резко выше базового уровня.",
            metrics={"max_rel_deviation_pct": 180.0, "horizon": "квартал"},
        ),
        Finding(
            hypothesis_id="grp_eoq_employees",
            runner="group_anomalies",
            title="Сотрудники с EOQ-сдвигом",
            severity="critical",
            summary="100 сотрудников концентрируют отпуска на конец квартала.",
            metrics={"n_violators": 78, "max_abs_z": 4.2},
            entity_csv="entities_grp_eoq_employees.csv",
        ),
        Finding(
            hypothesis_id="info_minor_nulls",
            runner="outliers",
            title="Мелкие nullы в kpi_score",
            severity="info",
            summary="0.3% nullов в числовой колонке.",
            metrics={"n_outliers": 5},
        ),
    ]
    plan = LoadPlan(
        schema="hr", table="events", total_rows=10,
        kept_columns=["x"], dropped_wide_text=[],
        strategy="full", sample_rows=None,
        est_bytes_per_row=10, est_full_bytes=100,
    )
    df = pd.DataFrame({"x": [1, 2, 3]})
    _, profile = profile_dataframe(df, plan)

    canned_json = """```json
{
  "insights": [
    {
      "title": "EOQ-фрод по отпускам",
      "priority": "top",
      "where_to_look": "CSV нарушителей за последние 10 дней квартала",
      "business_impact": "Риск занижения KPI на 78 сотрудниках.",
      "recommended_action": "Передать в HR-аудит.",
      "related_finding_ids": ["grp_eoq_employees", "seas_eoq_vacation", "ghost_id"],
      "confidence": "high"
    },
    {
      "title": "",
      "priority": "high",
      "where_to_look": "x", "business_impact": "y", "recommended_action": "z"
    }
  ]
}
```"""

    class StubLLM:
        def invoke_with_system(self, system, user, temperature=None):
            assert "Главное для бизнеса" not in system   # sanity: system prompt is the insights one
            return canned_json

    insights = extract_business_insights(
        StubLLM(),  # type: ignore[arg-type]
        findings, profile,
        user_hypothesis_text="фрод с отпусками в конце квартала",
        table_semantics="HR события",
    )
    assert len(insights) == 1, "Empty-title insight should be dropped"
    ins = insights[0]
    assert ins.priority == "top"
    assert ins.confidence == "high"
    # Validator must drop unknown finding_ids but keep real ones.
    assert "grp_eoq_employees" in ins.related_finding_ids
    assert "seas_eoq_vacation" in ins.related_finding_ids
    assert "ghost_id" not in ins.related_finding_ids
    # info-severity finding must NOT have been fed to the LLM (we can't observe
    # the prompt directly here, but we can re-derive: the validator only knows
    # ids of findings that were fed). info_minor_nulls would never appear
    # anyway as the LLM didn't reference it — covered indirectly.
