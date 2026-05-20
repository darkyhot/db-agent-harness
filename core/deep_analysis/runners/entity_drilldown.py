"""Entity drilldown runner: pinpoint anomalies for individual entities.

Unlike ``group_anomalies`` this runner is optimized for "one INN out of 1000"
cases. It compares each entity to peers inside the same period and optional
business segment, then exports explanation cards rather than raw rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.deep_analysis.runners._common import robust_z, severity_from_score, write_entities_csv
from core.deep_analysis.types import (
    AnalysisContext,
    ColumnRole,
    Finding,
    HypothesisSpec,
    TableProfile,
)

_MIN_PEER_ENTITIES = 8
_MIN_ABS_Z = 3.5
_MAX_METRICS = 6
_MAX_PEERS = 4
_MAX_DIMENSIONS = 4


def run_entity_drilldown(
    df: pd.DataFrame,
    profile: TableProfile,
    spec: HypothesisSpec,
    ctx: AnalysisContext,
) -> list[Finding]:
    params = spec.params
    entity_col = params.get("entity_col")
    if entity_col not in df.columns:
        return []

    date_col = params.get("date_col")
    metric_cols = [
        c for c in (params.get("metric_cols") or [])
        if c in df.columns and c != entity_col
    ][:_MAX_METRICS]
    peer_cols = [
        c for c in (params.get("peer_cols") or [])
        if c in df.columns and c != entity_col
    ][:_MAX_PEERS]
    dimension_cols = [
        c for c in (params.get("dimension_cols") or [])
        if c in df.columns and c not in {entity_col, date_col}
    ][:_MAX_DIMENSIONS]
    time_grains = [
        str(g) for g in (params.get("time_grains") or ["month"])
        if str(g) in {"month", "quarter", "year"}
    ]
    top_k = int(params.get("top_k") or (30 if ctx.mode.value == "fast" else 100))

    work = df.copy()
    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col])
    if work.empty:
        return []

    cards: list[dict[str, Any]] = []
    # Row volume is always useful, even when no numeric metric exists.
    cards.extend(_scan_metric(
        work, profile, entity_col, None, date_col, time_grains, peer_cols,
        metric_label="row_count",
    ))
    for metric_col in metric_cols:
        cards.extend(_scan_metric(
            work, profile, entity_col, metric_col, date_col, time_grains, peer_cols,
            metric_label=metric_col,
        ))

    for dim_col in dimension_cols:
        cards.extend(_scan_category_concentration(
            work, entity_col, dim_col, date_col, time_grains, peer_cols
        ))

    if not cards:
        return [Finding(
            hypothesis_id=spec.hypothesis_id,
            runner=spec.runner,
            title=spec.title,
            severity="info",
            summary=(
                f"Точечная проверка {entity_col}: сущностей с заметным "
                "отклонением от peer-группы не найдено."
            ),
            metrics={"entity_col": entity_col, "n_entity_cards": 0},
            details={"entity_col": entity_col},
        )]

    out = pd.DataFrame(cards)
    out = out.sort_values(["abs_robust_z", "abs_deviation_pct"], ascending=False)
    out = out.drop_duplicates(
        subset=["entity", "metric", "period", "peer_group", "reason"],
        keep="first",
    )
    out = out.head(top_k)
    csv_path = write_entities_csv(out, Path(ctx.output_dir), spec.hypothesis_id)

    max_z = float(out["abs_robust_z"].max())
    severity = severity_from_score(max_z, thresholds=(2.5, 3.5, 5.0))
    examples = _format_examples(out.head(5))
    period_note = " по периодам" if date_col else ""
    summary = (
        f"Найдены {len(out)} адресные карточки отклонений для {entity_col}"
        f"{period_note}. Первые для проверки: {examples}. "
        f"Полный список — в `{csv_path}`."
    )
    return [Finding(
        hypothesis_id=spec.hypothesis_id,
        runner=spec.runner,
        title=spec.title,
        severity=severity,
        summary=summary,
        metrics={
            "entity_col": entity_col,
            "n_entity_cards": int(len(out)),
            "max_abs_z": max_z,
            "top_entity": str(out.iloc[0]["entity"]) if len(out) else "",
            "csv_columns": list(out.columns),
        },
        entity_csv=csv_path,
        details={
            "entity_col": entity_col,
            "date_col": date_col,
            "metric_cols": metric_cols,
            "peer_cols": peer_cols,
            "dimension_cols": dimension_cols,
            "examples": out.head(5).to_dict(orient="records"),
        },
    )]


def _scan_metric(
    work: pd.DataFrame,
    profile: TableProfile,
    entity_col: str,
    metric_col: str | None,
    date_col: str | None,
    time_grains: list[str],
    peer_cols: list[str],
    *,
    metric_label: str,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    period_specs: list[tuple[str, pd.Series | None]] = [("all", None)]
    if date_col and date_col in work.columns:
        for grain in time_grains:
            period_specs.append((grain, _period_index(work[date_col], grain)))

    peer_specs: list[tuple[str | None, str]] = [(None, "all")]
    peer_specs.extend((p, p) for p in peer_cols)

    for grain, period_idx in period_specs:
        for peer_col, peer_label in peer_specs:
            group_cols = [entity_col]
            peer_series = None
            if peer_col:
                group_cols.insert(0, peer_col)
                peer_series = work[peer_col].astype(str)
            if period_idx is not None:
                group_cols.insert(0, "_period")
            frame = work.copy(deep=False)
            if period_idx is not None:
                frame = frame.assign(_period=period_idx.values)

            if metric_col is None:
                agg = frame.groupby(group_cols, dropna=True).size().reset_index(name="actual")
                agg["n_obs"] = agg["actual"]
                business_columns = [entity_col]
            else:
                values = pd.to_numeric(frame[metric_col], errors="coerce")
                frame = frame.assign(_metric_value=values).dropna(subset=["_metric_value"])
                if frame.empty:
                    continue
                agg_func = _metric_agg(profile, metric_col)
                grouped = frame.groupby(group_cols, dropna=True)["_metric_value"]
                agg = grouped.agg(actual=agg_func, n_obs="count").reset_index()
                business_columns = [entity_col, metric_col]
            if agg.empty:
                continue

            peer_keys = []
            if period_idx is not None:
                peer_keys.append("_period")
            if peer_col:
                peer_keys.append(peer_col)

            if peer_keys:
                for _, sub in agg.groupby(peer_keys, dropna=True):
                    cards.extend(_cards_from_peer_frame(
                        sub, entity_col, metric_label, grain, peer_col,
                        business_columns, peer_series=peer_series,
                    ))
            else:
                cards.extend(_cards_from_peer_frame(
                    agg, entity_col, metric_label, grain, peer_col,
                    business_columns, peer_series=peer_series,
                ))
    return cards


def _cards_from_peer_frame(
    sub: pd.DataFrame,
    entity_col: str,
    metric: str,
    grain: str,
    peer_col: str | None,
    business_columns: list[str],
    *,
    peer_series: pd.Series | None,
) -> list[dict[str, Any]]:
    if len(sub) < _MIN_PEER_ENTITIES:
        return []
    vals = pd.to_numeric(sub["actual"], errors="coerce")
    if vals.notna().sum() < _MIN_PEER_ENTITIES:
        return []
    z = robust_z(vals)
    mask = z.abs() >= _MIN_ABS_Z
    if not bool(mask.any()):
        return []

    baseline = float(vals.median())
    cards: list[dict[str, Any]] = []
    for idx in sub.index[mask]:
        actual = float(vals.loc[idx])
        rz = float(z.loc[idx])
        deviation_pct = _safe_deviation_pct(actual, baseline)
        row = sub.loc[idx]
        period = str(row.get("_period") or "all")
        peer_value = str(row.get(peer_col)) if peer_col else "all"
        peer_group = f"{peer_col}={peer_value}" if peer_col else "all"
        direction = "выше" if actual >= baseline else "ниже"
        reason = (
            f"{metric}: {actual:.4g} {direction} peer-baseline {baseline:.4g} "
            f"({deviation_pct:+.1f}%)"
        )
        cards.append({
            "entity": row[entity_col],
            "entity_col": entity_col,
            "peer_group": peer_group,
            "metric": metric,
            "baseline": baseline,
            "actual": actual,
            "deviation_pct": deviation_pct,
            "abs_deviation_pct": abs(deviation_pct),
            "period": period,
            "time_grain": grain,
            "reason": reason,
            "business_columns": ", ".join(business_columns),
            "n_peer_entities": int(len(sub)),
            "n_obs": int(row.get("n_obs") or 0),
            "robust_z": rz,
            "abs_robust_z": abs(rz),
        })
    return cards


def _scan_category_concentration(
    work: pd.DataFrame,
    entity_col: str,
    dim_col: str,
    date_col: str | None,
    time_grains: list[str],
    peer_cols: list[str],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    period_specs: list[tuple[str, pd.Series | None]] = [("all", None)]
    if date_col and date_col in work.columns:
        for grain in time_grains:
            period_specs.append((grain, _period_index(work[date_col], grain)))

    for grain, period_idx in period_specs:
        frame = work[[entity_col, dim_col]].copy()
        if period_idx is not None:
            frame["_period"] = period_idx.values
        group_cols = [entity_col]
        if "_period" in frame.columns:
            group_cols.insert(0, "_period")
        counts = frame.groupby(group_cols + [dim_col], dropna=True).size().reset_index(name="cnt")
        totals = counts.groupby(group_cols, dropna=True)["cnt"].sum().reset_index(name="total")
        shares = counts.merge(totals, on=group_cols)
        shares["actual"] = shares["cnt"] / shares["total"]
        idx = shares.groupby(group_cols, dropna=True)["actual"].idxmax()
        top = shares.loc[idx].copy()
        for _, sub in top.groupby(["_period"] if "_period" in top.columns else lambda _: 0):
            if len(sub) < _MIN_PEER_ENTITIES:
                continue
            vals = pd.to_numeric(sub["actual"], errors="coerce")
            z = robust_z(vals)
            mask = z.abs() >= _MIN_ABS_Z
            baseline = float(vals.median())
            for ridx in sub.index[mask]:
                row = sub.loc[ridx]
                actual = float(row["actual"])
                rz = float(z.loc[ridx])
                deviation_pct = _safe_deviation_pct(actual, baseline)
                dim_value = row[dim_col]
                reason = (
                    f"{dim_col}={dim_value}: доля {actual * 100:.1f}% против "
                    f"peer-baseline {baseline * 100:.1f}%"
                )
                cards.append({
                    "entity": row[entity_col],
                    "entity_col": entity_col,
                    "peer_group": "all",
                    "metric": f"share({dim_col}={dim_value})",
                    "baseline": baseline,
                    "actual": actual,
                    "deviation_pct": deviation_pct,
                    "abs_deviation_pct": abs(deviation_pct),
                    "period": str(row.get("_period") or "all"),
                    "time_grain": grain,
                    "reason": reason,
                    "business_columns": f"{entity_col}, {dim_col}",
                    "n_peer_entities": int(len(sub)),
                    "n_obs": int(row.get("total") or 0),
                    "robust_z": rz,
                    "abs_robust_z": abs(rz),
                })
    return cards


def _metric_agg(profile: TableProfile, metric_col: str):
    col = profile.columns.get(metric_col)
    if col and col.role in {ColumnRole.MONEY, ColumnRole.NUMERIC}:
        return "sum"
    return "mean"


def _period_index(dt: pd.Series, grain: str) -> pd.Series:
    if grain == "quarter":
        return dt.dt.to_period("Q").astype(str)
    if grain == "year":
        return dt.dt.to_period("Y").astype(str)
    return dt.dt.to_period("M").astype(str)


def _safe_deviation_pct(actual: float, baseline: float) -> float:
    if baseline == 0 or pd.isna(baseline):
        return float(np.sign(actual) * 999.0) if actual else 0.0
    return (actual - baseline) / abs(baseline) * 100.0


def _format_examples(rows: pd.DataFrame) -> str:
    parts: list[str] = []
    for _, r in rows.iterrows():
        parts.append(
            f"{r['entity']} — {r['metric']} {r['period']} "
            f"({r['deviation_pct']:+.1f}%)"
        )
    return "; ".join(parts) if parts else "—"
