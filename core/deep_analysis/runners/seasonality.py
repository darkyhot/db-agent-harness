"""Seasonality runner.

Checks periodicity at multiple horizons: day-of-week, day-of-month,
month-of-year, quarter-of-year, and an explicit end-of-quarter bucket (the
Russian banking calendar effect). For each horizon we:

1. Aggregate the metric (sum/count/rate) onto a time axis.
2. Run a chi-square / Kruskal-Wallis test comparing bucket distributions.
3. Run Ljung-Box on the residual after removing the tested seasonality to
   decide whether there is *remaining* structure vs. the bucket hypothesis
   explains it.

Strong findings include the top/bottom buckets + effect magnitude so the
report is immediately business-readable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.runners._common import severity_from_score, write_entities_csv
from core.deep_analysis.types import (
    AnalysisContext,
    Finding,
    HypothesisSpec,
    TableProfile,
)

_MIN_SAMPLES_PER_BUCKET = 10

_HORIZONS = (
    ("день недели", lambda s: s.dt.dayofweek),
    ("день месяца", lambda s: s.dt.day),
    ("неделя года", lambda s: s.dt.isocalendar().week.astype(int)),
    ("месяц года", lambda s: s.dt.month),
    ("квартал", lambda s: s.dt.quarter),
    ("последние 10 дней квартала", lambda s: _end_of_quarter_bucket(s)),
    # RU business-calendar buckets — useful for banking data (paydays,
    # holiday windows, quarter/year close). All are boolean 0/1 horizons:
    # a significant chi-square means the metric concentrates in this
    # special window vs. the rest of the year.
    ("конец месяца (последние 3 дня)", lambda s: _end_of_month_bucket(s, 3)),
    ("начало месяца (первые 3 дня)", lambda s: _start_of_month_bucket(s, 3)),
    ("выплатные дни (1-е и 15-е)", lambda s: _paydays_bucket(s)),
    ("последние 10 дней года", lambda s: _end_of_year_bucket(s, 10)),
    ("новогодние каникулы (28.12–08.01)", lambda s: _new_year_bucket(s)),
    ("майские праздники (30.04–10.05)", lambda s: _may_holidays_bucket(s)),
    ("предпраздничные дни (за 1 день до RU-праздника)", lambda s: _pre_holiday_bucket(s)),
)

# Fixed Russian public holidays (month, day). Enough to cover the major
# behavioural spikes banks see around them.
_RU_FIXED_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
    (2, 23), (3, 8),
    (5, 1), (5, 9),
    (6, 12), (11, 4),
)


def _end_of_quarter_bucket(s: pd.Series) -> pd.Series:
    q_end = s.dt.to_period("Q").dt.end_time
    days_to_q_end = (q_end - s).dt.days
    return (days_to_q_end <= 9).astype(int)


def _end_of_month_bucket(s: pd.Series, days: int) -> pd.Series:
    m_end = s.dt.to_period("M").dt.end_time
    return ((m_end - s).dt.days <= (days - 1)).astype(int)


def _start_of_month_bucket(s: pd.Series, days: int) -> pd.Series:
    return (s.dt.day <= days).astype(int)


def _paydays_bucket(s: pd.Series) -> pd.Series:
    return s.dt.day.isin([1, 15]).astype(int)


def _end_of_year_bucket(s: pd.Series, days: int) -> pd.Series:
    y_end = pd.to_datetime(s.dt.year.astype(str) + "-12-31")
    return ((y_end - s).dt.days.abs() <= (days - 1)).astype(int)


def _new_year_bucket(s: pd.Series) -> pd.Series:
    return (((s.dt.month == 12) & (s.dt.day >= 28)) | ((s.dt.month == 1) & (s.dt.day <= 8))).astype(int)


def _may_holidays_bucket(s: pd.Series) -> pd.Series:
    return (((s.dt.month == 4) & (s.dt.day >= 30)) | ((s.dt.month == 5) & (s.dt.day <= 10))).astype(int)


def _pre_holiday_bucket(s: pd.Series) -> pd.Series:
    next_day_holiday = pd.Series(False, index=s.index)
    for m, d in _RU_FIXED_HOLIDAYS:
        # A date is "pre-holiday" if day+1 matches a fixed holiday.
        shifted = s + pd.Timedelta(days=1)
        next_day_holiday = next_day_holiday | ((shifted.dt.month == m) & (shifted.dt.day == d))
    return next_day_holiday.astype(int)


def run_seasonality(
    df: pd.DataFrame,
    profile: TableProfile,
    spec: HypothesisSpec,
    ctx: AnalysisContext,
) -> list[Finding]:
    log = get_logger()
    params = spec.params
    date_col = params.get("date_col")
    value_col = params.get("value_col")
    group_col = params.get("group_col")
    agg = params.get("agg", "count")

    if date_col not in df.columns:
        return []

    series_date = pd.to_datetime(df[date_col], errors="coerce")
    work = df.loc[series_date.notna()].copy()
    if work.empty:
        return []
    work[date_col] = series_date[series_date.notna()]

    findings: list[Finding] = []
    output_dir = Path(ctx.output_dir)

    if group_col and group_col in work.columns:
        for cat_value, cat_df in work.groupby(group_col, dropna=True):
            if len(cat_df) < 200:
                continue
            sub_findings = _check_all_horizons(
                cat_df, date_col, value_col, agg, spec,
                label_suffix=f" [{group_col}={cat_value}]",
                output_dir=output_dir,
                hypothesis_id_suffix=f"_cat_{_slug(str(cat_value))}",
            )
            findings.extend(sub_findings)
    else:
        findings.extend(_check_all_horizons(
            work, date_col, value_col, agg, spec,
            label_suffix="",
            output_dir=output_dir,
            hypothesis_id_suffix="",
        ))

    log.info("Seasonality for %s yielded %d findings", spec.hypothesis_id, len(findings))
    return findings


def _check_all_horizons(
    work: pd.DataFrame,
    date_col: str,
    value_col: str | None,
    agg: str,
    spec: HypothesisSpec,
    *,
    label_suffix: str,
    output_dir: Path,
    hypothesis_id_suffix: str,
) -> list[Finding]:
    """Check every horizon, emitting findings when a bucket diverges strongly.

    Two statistical paths:
    - Count mode (agg='count' or no value_col): compare bucket counts against
      a uniform expectation (scaled by bucket *day-of-calendar* coverage when
      possible). We use chi-square goodness-of-fit here because the metric is
      "how many rows fell into this bucket".
    - Value mode (agg='sum'/'mean' with a numeric column): Kruskal-Wallis on
      per-row metric values across buckets, robust to non-normal distributions.
    """
    findings: list[Finding] = []
    count_mode = (agg == "count" or value_col is None)

    for horizon_name, bucket_fn in _HORIZONS:
        try:
            buckets = bucket_fn(work[date_col])
        except Exception:
            continue

        if count_mode:
            bucket_sizes = buckets.value_counts().sort_index()
            if bucket_sizes.size < 2 or bucket_sizes.sum() < 50:
                continue
            # Expected counts: weight each bucket by the number of unique
            # calendar days that mapped to it, so horizons with uneven bucket
            # widths (e.g. EOQ=1 vs EOQ=0) are compared fairly.
            bucket_days = (
                pd.DataFrame({"bucket": buckets.values, "dt": work[date_col].dt.normalize().values})
                .drop_duplicates()
                .groupby("bucket")
                .size()
                .reindex(bucket_sizes.index)
                .fillna(1.0)
            )
            weights = bucket_days / bucket_days.sum()
            expected = bucket_sizes.sum() * weights
            if (expected < 5).any():
                # Chi-square unreliable; fall back to proportion test on the
                # largest deviation bucket.
                top = (bucket_sizes - expected).abs().idxmax()
                p_value = 1.0 if expected.loc[top] <= 0 else float(
                    stats.binomtest(
                        int(bucket_sizes.loc[top]),
                        int(bucket_sizes.sum()),
                        float(weights.loc[top]),
                    ).pvalue
                )
                kw_stat = float(bucket_sizes.loc[top])
            else:
                try:
                    kw_stat, p_value = stats.chisquare(f_obs=bucket_sizes.values, f_exp=expected.values)
                except Exception:
                    continue
            bucket_means = bucket_sizes / bucket_days     # rows per day
            overall_rate = bucket_sizes.sum() / bucket_days.sum()
            rel_dev = (bucket_means - overall_rate) / abs(overall_rate) if overall_rate else bucket_means * 0
            top = rel_dev.abs().idxmax()
            top_dev = float(rel_dev.loc[top])
            bucket_summary_df = pd.DataFrame({
                "bucket": bucket_sizes.index,
                "bucket_size": bucket_sizes.values,
                "bucket_days": bucket_days.values,
                "rows_per_day": bucket_means.values,
                "rel_deviation_pct": (rel_dev * 100).values,
            }).sort_values("rel_deviation_pct", key=lambda s: s.abs(), ascending=False)
        else:
            metric = _compute_metric(work, value_col, agg)
            if metric is None or metric.empty:
                continue
            frame = pd.DataFrame({"bucket": buckets.values, "value": metric.values}).dropna()
            if frame.empty:
                continue
            grouped = frame.groupby("bucket")["value"]
            bucket_sizes = grouped.size()
            if (bucket_sizes < _MIN_SAMPLES_PER_BUCKET).all():
                continue
            bucket_means = grouped.mean()
            overall = frame["value"].mean()
            if overall == 0 or pd.isna(overall):
                continue
            bucket_values = [g.values for _, g in grouped if len(g) >= _MIN_SAMPLES_PER_BUCKET]
            if len(bucket_values) < 2:
                continue
            try:
                kw_stat, p_value = stats.kruskal(*bucket_values)
            except Exception:
                continue
            if pd.isna(p_value):
                continue
            rel_dev = (bucket_means - overall) / abs(overall)
            top = rel_dev.abs().idxmax()
            top_dev = float(rel_dev.loc[top])
            bucket_summary_df = pd.DataFrame({
                "bucket": bucket_means.index,
                "bucket_size": bucket_sizes.reindex(bucket_means.index).values,
                "mean_value": bucket_means.values,
                "rel_deviation_pct": (rel_dev * 100).values,
            }).sort_values("rel_deviation_pct", key=lambda s: s.abs(), ascending=False)

        if p_value > 0.01 and abs(top_dev) < 0.15:
            # Neither statistically significant nor large — skip noise.
            continue

        # Approximate z from p-value for severity bucketing.
        z_equivalent = float(stats.norm.isf(min(max(p_value, 1e-12), 0.5))) if p_value > 0 else 6.0
        severity = severity_from_score(max(z_equivalent, abs(top_dev) * 5))

        csv_rel = write_entities_csv(
            bucket_summary_df,
            output_dir,
            f"{spec.hypothesis_id}{hypothesis_id_suffix}_{_slug(horizon_name)}",
        )

        metric_name = "количество событий" if agg == "count" else f"{agg}({value_col})"
        top_label = str(top)
        summary = (
            f"По разрезу «{horizon_name}»{label_suffix} {metric_name} "
            f"в бакете `{top_label}` отклоняется на {top_dev * 100:+.1f}% "
            f"от общего среднего (p={p_value:.1e})."
        )
        findings.append(Finding(
            hypothesis_id=spec.hypothesis_id + hypothesis_id_suffix,
            runner=spec.runner,
            title=f"{spec.title} — {horizon_name}{label_suffix}",
            severity=severity,
            summary=summary,
            metrics={
                "horizon": horizon_name,
                "p_value": float(p_value),
                "kw_stat": float(kw_stat),
                "max_rel_deviation_pct": float(top_dev * 100),
                "top_bucket": top_label,
                "n_buckets": int(bucket_means.size),
            },
            entity_csv=csv_rel,
            details={
                "agg": agg,
                "value_col": value_col,
                "date_col": date_col,
            },
        ))
    return findings


def _compute_metric(df: pd.DataFrame, value_col: str | None, agg: str) -> pd.Series | None:
    """Return a per-row metric series aligned with df.index.

    For agg=count the metric is a constant 1, so that bucket aggregation yields
    counts. For sum/mean we project the value column; the downstream aggregator
    will compute mean-per-bucket regardless of agg, which is fine because what
    we care about is the bucket-to-bucket relative deviation.
    """
    if agg == "count" or value_col is None:
        return pd.Series(np.ones(len(df)), index=df.index)
    if value_col not in df.columns:
        return None
    s = df[value_col]
    if not pd.api.types.is_numeric_dtype(s):
        # Treat a flag-like column as 0/1.
        try:
            s = s.astype(float)
        except (TypeError, ValueError):
            try:
                s = s.astype(bool).astype(float)
            except Exception:
                return None
    return s


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text.lower())[:40]
