"""Shared helpers for runners."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_entities_csv(
    df: pd.DataFrame,
    output_dir: Path,
    hypothesis_id: str,
) -> str:
    """Persist violator/entity list CSV and return its path relative to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = f"entities_{hypothesis_id}.csv"
    fpath = output_dir / fname
    df.to_csv(fpath, index=False, encoding="utf-8", sep=",")
    return fname


def severity_from_score(score: float, thresholds=(1.5, 2.5, 4.0)) -> str:
    """Map an absolute deviation score to severity buckets.

    Defaults correspond to |z| > 1.5/2.5/4 for robust-z (roughly the 87th/99th/
    ~1-in-100k percentiles of a normal distribution).
    """
    a = abs(score)
    if a >= thresholds[2]:
        return "critical"
    if a >= thresholds[1]:
        return "strong"
    if a >= thresholds[0]:
        return "notable"
    return "info"


def robust_z(series: pd.Series) -> pd.Series:
    """Median/MAD-based z-score, stable under heavy tails."""
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0:
        # Fall back to std so we don't emit NaN for degenerate MAD.
        std = series.std(ddof=0)
        if std == 0 or pd.isna(std):
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - series.mean()) / std
    return 0.6745 * (series - med) / mad
