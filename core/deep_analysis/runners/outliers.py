"""Outliers runner.

Two paths:
- method="mad": per-column robust z-score (median absolute deviation). Cheap,
  exact, interpretable.
- method="isolation_forest": multivariate IsolationForest on the numeric
  subspace. Catches joint outliers that are not extreme in any single column.

Both modes export the outlier rows as a CSV, sorted by severity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.runners._common import robust_z, severity_from_score, write_entities_csv
from core.deep_analysis.types import (
    AnalysisContext,
    Finding,
    HypothesisSpec,
    TableProfile,
)

_MAD_THRESHOLD = 4.0
_IFOREST_CONTAMINATION = 0.01
_IFOREST_MAX_ROWS = 500_000   # above this we subsample for the fit


def run_outliers(
    df: pd.DataFrame,
    profile: TableProfile,
    spec: HypothesisSpec,
    ctx: AnalysisContext,
) -> list[Finding]:
    log = get_logger()
    value_cols = [c for c in spec.params.get("value_cols", []) if c in df.columns]
    method = spec.params.get("method", "mad")
    if not value_cols:
        return []

    output_dir = Path(ctx.output_dir)

    if method == "mad":
        return _run_mad(df, value_cols, spec, output_dir, log)
    if method == "isolation_forest":
        return _run_isolation_forest(df, value_cols, spec, output_dir, log)
    return []


def _run_mad(
    df: pd.DataFrame,
    value_cols: list[str],
    spec: HypothesisSpec,
    output_dir: Path,
    log,
) -> list[Finding]:
    findings: list[Finding] = []
    for col in value_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 50:
            continue
        z = robust_z(s)
        mask = z.abs() >= _MAD_THRESHOLD
        n_out = int(mask.sum())
        if n_out == 0:
            continue
        top = df.loc[mask].assign(_col=col, _value=s[mask], _robust_z=z[mask])
        top = top.sort_values("_robust_z", key=lambda x: x.abs(), ascending=False)
        csv_path = write_entities_csv(top, output_dir, spec.hypothesis_id + f"_{_slug(col)}")
        max_z = float(z[mask].abs().max())
        findings.append(Finding(
            hypothesis_id=spec.hypothesis_id + f"_{_slug(col)}",
            runner=spec.runner,
            title=f"Выбросы в колонке {col}",
            severity=severity_from_score(max_z, thresholds=(2.5, 4.0, 6.0)),
            summary=(
                f"В колонке `{col}` найдено {n_out} строк с |robust-z|≥{_MAD_THRESHOLD} "
                f"(макс z={max_z:.2f}). Полный список — в `{csv_path}`."
            ),
            metrics={
                "column": col,
                "n_outliers": n_out,
                "max_abs_z": max_z,
                "threshold": _MAD_THRESHOLD,
            },
            entity_csv=csv_path,
        ))
    return findings


def _run_isolation_forest(
    df: pd.DataFrame,
    value_cols: list[str],
    spec: HypothesisSpec,
    output_dir: Path,
    log,
) -> list[Finding]:
    from sklearn.ensemble import IsolationForest

    num = df[value_cols].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    # Drop rows with all-NaN across the chosen features.
    mask_any = num.notna().any(axis=1)
    num = num.loc[mask_any]
    # Impute remaining NaNs with column medians — IsolationForest dislikes NaN.
    num = num.fillna(num.median(numeric_only=True))
    if len(num) < 200 or num.shape[1] == 0:
        return []

    fit_sample = num.sample(n=min(_IFOREST_MAX_ROWS, len(num)), random_state=42)
    model = IsolationForest(
        n_estimators=200,
        contamination=_IFOREST_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(fit_sample)
    score = -model.decision_function(num)    # higher = more anomalous
    pred = model.predict(num)                 # -1 = outlier
    mask = pred == -1
    n_out = int(mask.sum())
    if n_out == 0:
        return []

    result = df.loc[num.index[mask]].assign(_iforest_score=score[mask])
    result = result.sort_values("_iforest_score", ascending=False)
    csv_path = write_entities_csv(result, output_dir, spec.hypothesis_id)

    max_score = float(score[mask].max())
    # Map score into our severity buckets; scores are ~O(0.1..0.5).
    severity = severity_from_score(max_score * 10, thresholds=(1.5, 3.0, 5.0))
    return [Finding(
        hypothesis_id=spec.hypothesis_id,
        runner=spec.runner,
        title=spec.title,
        severity=severity,
        summary=(
            f"IsolationForest на {num.shape[1]} числовых колонках "
            f"выявил {n_out} многомерных аномалий из {len(num)} строк. Полный список — `{csv_path}`."
        ),
        metrics={
            "n_outliers": n_out,
            "max_score": max_score,
            "features": value_cols,
            "contamination": _IFOREST_CONTAMINATION,
        },
        entity_csv=csv_path,
    )]


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text.lower())[:40]
