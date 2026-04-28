"""Runners: each one consumes a HypothesisSpec and emits Findings."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from core.deep_analysis.runners.dependencies import run_dependencies
from core.deep_analysis.runners.group_anomalies import run_group_anomalies
from core.deep_analysis.runners.outliers import run_outliers
from core.deep_analysis.runners.regime_shifts import run_regime_shifts
from core.deep_analysis.runners.seasonality import run_seasonality
from core.deep_analysis.types import (
    AnalysisContext,
    Finding,
    HypothesisSpec,
    TableProfile,
)

RunnerFn = Callable[
    [pd.DataFrame, TableProfile, HypothesisSpec, AnalysisContext],
    list[Finding],
]

RUNNERS: dict[str, RunnerFn] = {
    "seasonality": run_seasonality,
    "group_anomalies": run_group_anomalies,
    "outliers": run_outliers,
    "dependencies": run_dependencies,
    "regime_shifts": run_regime_shifts,
}


def get_runner(name: str) -> RunnerFn | None:
    return RUNNERS.get(name)
