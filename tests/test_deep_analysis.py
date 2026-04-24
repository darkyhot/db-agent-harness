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
from core.deep_analysis.loader import LoadPlan
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
    ColumnRole,
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
