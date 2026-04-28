"""Regime shifts runner — detect dates when a metric's behavior changed.

Uses ruptures' Pelt algorithm with an L2 cost on a daily/weekly aggregate of
the metric. Each detected changepoint is emitted as a finding with the date,
the before/after metric, and the relative shift magnitude. Small noisy shifts
are filtered by a minimum-magnitude threshold to keep the output actionable.

The algorithm is global (whole history): it doesn't know what's "expected"
business-wise. Findings are framed as "поведение поменялось с X", for the
analyst to interpret. This is usually enough to surface the obvious candidates
(marketing campaign starts, tariff changes, regulation dates, incidents).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.runners._common import severity_from_score, write_entities_csv
from core.deep_analysis.types import (
    AnalysisContext,
    Finding,
    HypothesisSpec,
    TableProfile,
)

_MIN_POINTS = 30                  # need at least a month of points for PELT
_MIN_REL_SHIFT = 0.2              # ignore regime shifts < 20% of baseline
_MAX_CHANGEPOINTS = 8
_FREQ_MAP = {"day": "D", "week": "W-MON", "month": "MS"}


def run_regime_shifts(
    df: pd.DataFrame,
    profile: TableProfile,
    spec: HypothesisSpec,
    ctx: AnalysisContext,
) -> list[Finding]:
    log = get_logger()
    params = spec.params
    date_col = params.get("date_col")
    value_col = params.get("value_col")
    agg = params.get("agg", "count")
    freq = params.get("freq", "day")

    if date_col not in df.columns:
        return []
    pelt_freq = _FREQ_MAP.get(freq, "D")

    dt = pd.to_datetime(df[date_col], errors="coerce")
    mask = dt.notna()
    work = df.loc[mask].assign(_date=dt[mask])
    if work.empty:
        return []

    # Aggregate into a time series.
    if agg == "count" or not value_col:
        series = work.groupby(pd.Grouper(key="_date", freq=pelt_freq)).size()
    else:
        if value_col not in work.columns:
            return []
        v = pd.to_numeric(work[value_col], errors="coerce")
        series = (
            work.assign(_v=v)
            .groupby(pd.Grouper(key="_date", freq=pelt_freq))["_v"]
            .agg(agg)
        )
    series = series.dropna()
    if len(series) < _MIN_POINTS:
        log.info("regime_shifts: too few points (%d)", len(series))
        return []

    try:
        import ruptures as rpt
    except ImportError:
        log.warning("ruptures not installed — skipping regime_shifts runner")
        return []

    arr = series.values.astype(float)
    # Scale by series std so PELT's penalty has the same meaning regardless of
    # metric magnitude. Penalty=3 ≈ roughly "3σ move required" — conservative.
    std = float(np.std(arr))
    if std == 0:
        return []
    try:
        algo = rpt.Pelt(model="l2").fit(arr.reshape(-1, 1))
        bkps = algo.predict(pen=3 * std * std)
    except Exception as exc:
        log.warning("ruptures failed: %s", exc)
        return []

    # bkps includes the series length as last element — drop it.
    changepoints = [b for b in bkps if 0 < b < len(arr)]
    if not changepoints:
        return []

    output_dir = Path(ctx.output_dir)
    rows = []
    for cp in changepoints[:_MAX_CHANGEPOINTS]:
        before = arr[max(0, cp - 30): cp]
        after = arr[cp: cp + 30]
        if len(before) < 5 or len(after) < 5:
            continue
        mean_before = float(np.mean(before))
        mean_after = float(np.mean(after))
        baseline = max(abs(mean_before), 1e-9)
        rel = (mean_after - mean_before) / baseline
        if abs(rel) < _MIN_REL_SHIFT:
            continue
        rows.append({
            "date": str(series.index[cp].date()),
            "mean_before_30": round(mean_before, 4),
            "mean_after_30": round(mean_after, 4),
            "rel_shift_pct": round(rel * 100, 2),
        })
    if not rows:
        return []

    rows.sort(key=lambda r: abs(r["rel_shift_pct"]), reverse=True)
    csv = write_entities_csv(pd.DataFrame(rows), output_dir, spec.hypothesis_id)
    top = rows[0]
    max_abs = max(abs(r["rel_shift_pct"]) for r in rows)
    severity = severity_from_score(max_abs / 20, thresholds=(1.0, 2.0, 5.0))
    metric_label = "количество событий" if agg == "count" else f"{agg}({value_col})"
    summary = (
        f"Найдено {len(rows)} точек смены режима в ряду {metric_label} "
        f"(частота {freq}). Сильнейшая — {top['date']}: "
        f"{top['mean_before_30']:.2f} → {top['mean_after_30']:.2f} "
        f"({top['rel_shift_pct']:+.1f}%). Полный список — в `{csv}`."
    )
    return [Finding(
        hypothesis_id=spec.hypothesis_id,
        runner=spec.runner,
        title=spec.title,
        severity=severity,
        summary=summary,
        metrics={
            "n_changepoints": len(rows),
            "max_abs_rel_shift_pct": max_abs,
            "freq": freq,
            "agg": agg,
        },
        entity_csv=csv,
    )]
