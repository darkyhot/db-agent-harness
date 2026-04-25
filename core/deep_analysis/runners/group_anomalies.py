"""Group anomalies runner — entity-level mass-deviation detection.

The typical ask: "out of 2000 employees, which 100 are fraudulent?". We
aggregate a metric per (entity, period), compute a robust z-score across the
cohort, and export every entity above the threshold to a CSV so the analyst
can review the full list.

Supported metrics:
- row_count: number of records per entity.
- mean: mean of a numeric column per entity.
- rate: probability a flag/binary column is 1 for the entity.
- end_of_quarter_shift: how much more the entity concentrates a given
  category value into the last 10 days of the quarter vs. peers — exactly the
  vacation-fraud pattern described in the feature spec.

Output: entity-level CSV with all entities above |z| > threshold, sorted by
severity descending. No truncation — this is the deliverable.
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

_THRESHOLD = 2.5          # |robust-z| threshold for inclusion in violators list
_MIN_ENTITY_SAMPLE = 10   # minimum observations per entity to consider it
_MIN_COHORT_SIZE = 8      # minimum #entities to form a meaningful cohort
                          # (allows small entity sets like ~15 territorial banks)


def run_group_anomalies(
    df: pd.DataFrame,
    profile: TableProfile,
    spec: HypothesisSpec,
    ctx: AnalysisContext,
) -> list[Finding]:
    log = get_logger()
    params = spec.params
    entity_col = params.get("entity_col")
    metric = params.get("metric")
    if entity_col not in df.columns:
        return []

    date_col = params.get("date_col")
    period = params.get("period")  # "quarter" | "month" | None
    value_col = params.get("value_col")
    category_col = params.get("category_col")

    work = df.copy()
    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
    if work.empty:
        return []

    # Compute per-entity (per-period if requested) metric.
    try:
        if metric == "row_count":
            agg_df = _agg_row_count(work, entity_col, date_col, period)
        elif metric == "mean":
            if not value_col or value_col not in work.columns:
                return []
            agg_df = _agg_mean(work, entity_col, value_col, date_col, period)
        elif metric == "rate":
            if not value_col or value_col not in work.columns:
                return []
            agg_df = _agg_rate(work, entity_col, value_col, date_col, period)
        elif metric == "end_of_quarter_shift":
            if not category_col or category_col not in work.columns or not date_col:
                return []
            agg_df = _agg_end_of_quarter_shift(work, entity_col, date_col, category_col)
        else:
            return []
    except Exception as exc:
        log.warning("group_anomalies agg failed for %s: %s", spec.hypothesis_id, exc)
        return []

    if agg_df is None or agg_df.empty or "metric_value" not in agg_df.columns:
        return []

    # Drop entities with too little data to avoid spurious outliers.
    if "n_obs" in agg_df.columns:
        agg_df = agg_df[agg_df["n_obs"] >= _MIN_ENTITY_SAMPLE]
    if len(agg_df) < _MIN_COHORT_SIZE:
        # Not enough peers for cohort comparison.
        return []

    agg_df["robust_z"] = robust_z(agg_df["metric_value"])
    violators = agg_df[agg_df["robust_z"].abs() >= _THRESHOLD].copy()
    violators = violators.sort_values("robust_z", key=lambda s: s.abs(), ascending=False)

    if violators.empty:
        return [Finding(
            hypothesis_id=spec.hypothesis_id,
            runner=spec.runner,
            title=spec.title,
            severity="info",
            summary=(
                f"Проверено {len(agg_df)} сущностей ({entity_col}) — "
                f"массовых отклонений (|z|>{_THRESHOLD}) не обнаружено."
            ),
            metrics={"n_entities": int(len(agg_df)), "n_violators": 0},
        )]

    output_dir = Path(ctx.output_dir)
    csv_path = write_entities_csv(violators, output_dir, spec.hypothesis_id)

    max_z = float(violators["robust_z"].abs().max())
    severity = severity_from_score(max_z)
    # Describe top violators in-line so the report is useful without opening the CSV.
    top_rows = violators.head(5)
    top_lines = ", ".join(
        f"{r[entity_col]}={r['metric_value']:.2f} (z={r['robust_z']:+.2f})"
        for _, r in top_rows.iterrows()
        if pd.notna(r.get(entity_col))
    )
    summary = (
        f"Из {len(agg_df)} сущностей ({entity_col}) {len(violators)} "
        f"имеют аномальную метрику (|z|>{_THRESHOLD}). "
        f"Топ: {top_lines}. Полный список в `{csv_path}`."
    )
    return [Finding(
        hypothesis_id=spec.hypothesis_id,
        runner=spec.runner,
        title=spec.title,
        severity=severity,
        summary=summary,
        metrics={
            "n_entities": int(len(agg_df)),
            "n_violators": int(len(violators)),
            "max_abs_z": max_z,
            "metric": metric,
            "period": period,
        },
        entity_csv=csv_path,
        details={
            "entity_col": entity_col,
            "value_col": value_col,
            "category_col": category_col,
        },
    )]


# ---------- metric aggregations ----------


def _period_index(dt: pd.Series, period: str | None) -> pd.Series | None:
    if period is None:
        return None
    if period == "quarter":
        return dt.dt.to_period("Q").astype(str)
    if period == "month":
        return dt.dt.to_period("M").astype(str)
    if period == "year":
        return dt.dt.to_period("Y").astype(str)
    return None


def _agg_row_count(
    df: pd.DataFrame, entity_col: str, date_col: str | None, period: str | None
) -> pd.DataFrame:
    if date_col and period:
        p_idx = _period_index(df[date_col], period)
        # Per-entity mean count across periods: smooths out entities that exist
        # only briefly, and compares apples to apples.
        per_ep = df.groupby([entity_col, p_idx]).size().reset_index(name="cnt")
        out = per_ep.groupby(entity_col).agg(
            metric_value=("cnt", "mean"),
            n_obs=("cnt", "count"),
        ).reset_index()
    else:
        out = df.groupby(entity_col).size().reset_index(name="metric_value")
        out["n_obs"] = out["metric_value"]
    return out


def _agg_mean(
    df: pd.DataFrame,
    entity_col: str,
    value_col: str,
    date_col: str | None,
    period: str | None,
) -> pd.DataFrame:
    s = pd.to_numeric(df[value_col], errors="coerce")
    work = df.assign(_v=s).dropna(subset=["_v"])
    if date_col and period:
        p_idx = _period_index(work[date_col], period)
        per_ep = work.groupby([entity_col, p_idx]).agg(
            _m=("_v", "mean"), _n=("_v", "count")
        ).reset_index()
        out = per_ep.groupby(entity_col).agg(
            metric_value=("_m", "mean"),
            n_obs=("_n", "sum"),
        ).reset_index()
    else:
        out = work.groupby(entity_col).agg(
            metric_value=("_v", "mean"), n_obs=("_v", "count")
        ).reset_index()
    return out


def _agg_rate(
    df: pd.DataFrame,
    entity_col: str,
    value_col: str,
    date_col: str | None,
    period: str | None,
) -> pd.DataFrame:
    s = df[value_col]
    if pd.api.types.is_bool_dtype(s):
        s = s.astype(float)
    elif not pd.api.types.is_numeric_dtype(s):
        # Map truthy-ish strings to 1.
        s = s.astype(str).str.lower().isin({"1", "true", "yes", "y", "да", "t"}).astype(float)
    work = df.assign(_v=s)
    if date_col and period:
        p_idx = _period_index(work[date_col], period)
        per_ep = work.groupby([entity_col, p_idx]).agg(
            _r=("_v", "mean"), _n=("_v", "count")
        ).reset_index()
        out = per_ep.groupby(entity_col).agg(
            metric_value=("_r", "mean"), n_obs=("_n", "sum")
        ).reset_index()
    else:
        out = work.groupby(entity_col).agg(
            metric_value=("_v", "mean"), n_obs=("_v", "count")
        ).reset_index()
    return out


def _agg_end_of_quarter_shift(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    category_col: str,
) -> pd.DataFrame:
    """For each entity, compute the lift of any category value in the last 10
    days of the quarter vs. the rest of the quarter.

    This captures the vacation-fraud pattern: an employee whose "отпуск" share
    spikes in the last 10 days of every quarter, while peers don't.
    """
    dt = df[date_col]
    q_end = dt.dt.to_period("Q").dt.end_time
    is_eoq = ((q_end - dt).dt.days <= 9)
    work = df.assign(_eoq=is_eoq.astype(int))

    # Global share for each category value during EOQ vs non-EOQ — used as
    # the reference the entity is compared against.
    global_eoq = work.loc[work["_eoq"] == 1, category_col].value_counts(normalize=True)
    global_non = work.loc[work["_eoq"] == 0, category_col].value_counts(normalize=True)
    if global_eoq.empty:
        return pd.DataFrame()

    # For each entity, compute max(entity_eoq_share - peer_eoq_share) across
    # category values — big positive lift = suspicious concentration.
    rows = []
    for entity, grp in work.groupby(entity_col):
        n_total = len(grp)
        if n_total < _MIN_ENTITY_SAMPLE:
            continue
        eoq_rows = grp[grp["_eoq"] == 1]
        if eoq_rows.empty:
            continue
        entity_share = eoq_rows[category_col].value_counts(normalize=True)
        # Compute lift for each category value, pick the max.
        best_val = None
        best_lift = 0.0
        for val, share in entity_share.items():
            ref = float(global_eoq.get(val, 0.0))
            lift = share - ref
            if lift > best_lift:
                best_lift = lift
                best_val = val
        rows.append({
            entity_col: entity,
            "metric_value": best_lift,
            "top_category": best_val,
            "n_eoq_rows": len(eoq_rows),
            "n_obs": n_total,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
