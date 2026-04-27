"""Dependencies runner — surface hidden relationships between column pairs.

Banking data often has non-obvious column associations that matter for
analysis: "segment is predicted by region" or "churn flag follows product_type".
We scan column pairs and emit findings for statistically strong + large-effect
associations, biased toward pairs that are NOT trivially related by name
(to avoid noise like "client_id <-> client_hash").

Three sub-methods, chosen by the type pair:
- numeric–numeric: Spearman rank correlation (robust to non-linear monotone).
- categorical–categorical: Cramér's V from a chi-square contingency table.
- numeric–categorical: eta² from one-way ANOVA on the numeric-by-category
  grouping (same intuition as MI but cheaper and interpretable).

The output per finding is a pair-level CSV: for categorical-categorical it is
the contingency table with row/column percentages; for numeric pairs it is the
top residual rows; for mixed it is per-category mean + count. Analysts can
open the CSV to see *why* the relationship is strong.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.runners._common import write_entities_csv
from core.deep_analysis.types import (
    AnalysisContext,
    ColumnRole,
    Finding,
    HypothesisSpec,
    TableProfile,
)

_MAX_ROWS_FOR_TESTS = 200_000       # subsample above this to keep scans fast
_MIN_NON_NULL = 200                 # skip pairs with too little data
_CRAMER_V_STRONG = 0.25             # effect thresholds for inclusion
_SPEARMAN_STRONG = 0.4
_ETA_SQ_STRONG = 0.1

# Suffixes that mark a column as one face of a reference-table pair
# (post_id / post_name / post_code etc.). Pairs that share a stem and differ
# only by such a suffix are skipped — by definition V≈1 and not a discovery.
_REFERENCE_SUFFIXES = (
    "_id", "_name", "_nm", "_code", "_cd", "_key", "_desc", "_descr", "_text",
)


def _strip_reference_suffix(name: str) -> str | None:
    low = name.lower()
    for s in _REFERENCE_SUFFIXES:
        if low.endswith(s) and len(low) > len(s):
            return low[: -len(s)]
    return None


def _is_reference_pair(a: str, b: str) -> bool:
    """True when a/b look like two faces of the same dictionary entry.

    Two columns are a reference pair when:

    1. Both share an exact stem after stripping a known role suffix
       (`post_id` ↔ `post_name`, `tb_code` ↔ `tb_id`).
    2. One stem is a single-token extension of the other (`manager_lvl_1`
       vs `manager_lvl_1_post`) — covers nested `*_post_id` ↔ `*_name`
       patterns we see in flattened SAP/Сбер extracts.
    3. One side is bare and the other is `bare + role_suffix`
       (`post` ↔ `post_id`).
    """
    al, bl = a.lower(), b.lower()
    if al == bl:
        return False
    sa = _strip_reference_suffix(al)
    sb = _strip_reference_suffix(bl)

    if sa is None and sb is None:
        return False

    if sa is None:
        # `post` ↔ `post_id`
        return sb == al
    if sb is None:
        return sa == bl

    if sa == sb:
        return True

    short, long_ = (sa, sb) if len(sa) < len(sb) else (sb, sa)
    if long_.startswith(short + "_"):
        remainder = long_[len(short) + 1:]
        # Single-token tail only (e.g. `_post`); refuse multi-token to avoid
        # folding `region` with `region_capital_market_share`.
        if remainder and "_" not in remainder:
            return True
    return False


def run_dependencies(
    df: pd.DataFrame,
    profile: TableProfile,
    spec: HypothesisSpec,
    ctx: AnalysisContext,
) -> list[Finding]:
    log = get_logger()
    cols = [c for c in spec.params.get("columns", []) if c in df.columns]
    if len(cols) < 2:
        return []

    if len(df) > _MAX_ROWS_FOR_TESTS:
        df = df.sample(n=_MAX_ROWS_FOR_TESTS, random_state=42)

    output_dir = Path(ctx.output_dir)
    findings: list[Finding] = []
    max_pairs = int(spec.params.get("max_pairs", 80))

    pairs = list(combinations(cols, 2))
    # Drop reference-table siblings (post_id ↔ post_name and so on) — by
    # definition V≈1 and no analytical value. Catches the ~14 trivial
    # findings we used to emit per run on Сбер-style flat extracts.
    pairs = [(a, b) for a, b in pairs if not _is_reference_pair(a, b)]
    # Bias toward pairs whose names don't share a common prefix — those are
    # the "non-obvious" ones worth surfacing. Obvious pairs (client_id,
    # client_name) still get checked but ranked last.
    pairs.sort(key=lambda ab: _name_overlap(*ab))
    pairs = pairs[:max_pairs]

    for a, b in pairs:
        if ctx.seconds_left() <= 2:
            log.info("dependencies runner: budget low, stopping pair scan")
            break
        role_a = profile.columns[a].role if a in profile.columns else ColumnRole.UNKNOWN
        role_b = profile.columns[b].role if b in profile.columns else ColumnRole.UNKNOWN
        try:
            finding = _check_pair(df, a, b, role_a, role_b, spec, output_dir)
        except Exception as exc:
            log.warning("dependencies pair %s × %s failed: %s", a, b, exc)
            continue
        if finding is not None:
            findings.append(finding)
    log.info("dependencies: produced %d findings over %d pairs", len(findings), len(pairs))
    return findings


def _check_pair(
    df: pd.DataFrame,
    a: str,
    b: str,
    role_a: ColumnRole,
    role_b: ColumnRole,
    spec: HypothesisSpec,
    output_dir: Path,
) -> Finding | None:
    numeric_roles = {ColumnRole.NUMERIC, ColumnRole.MONEY, ColumnRole.PERCENT}
    categorical_roles = {ColumnRole.CATEGORY, ColumnRole.FLAG}

    sub = df[[a, b]].dropna()
    if len(sub) < _MIN_NON_NULL:
        return None

    if role_a in numeric_roles and role_b in numeric_roles:
        return _finding_num_num(sub, a, b, spec, output_dir)
    if role_a in categorical_roles and role_b in categorical_roles:
        return _finding_cat_cat(sub, a, b, spec, output_dir)
    if role_a in numeric_roles and role_b in categorical_roles:
        return _finding_num_cat(sub, a, b, spec, output_dir)
    if role_b in numeric_roles and role_a in categorical_roles:
        return _finding_num_cat(sub.rename(columns={a: b, b: a})[[b, a]].rename(columns={b: a, a: b}), b, a, spec, output_dir)
    return None


def _finding_num_num(
    sub: pd.DataFrame, a: str, b: str, spec: HypothesisSpec, output_dir: Path
) -> Finding | None:
    x = pd.to_numeric(sub[a], errors="coerce")
    y = pd.to_numeric(sub[b], errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < _MIN_NON_NULL:
        return None
    rho, p = stats.spearmanr(x[mask], y[mask])
    if pd.isna(rho) or abs(rho) < _SPEARMAN_STRONG or p > 0.001:
        return None
    severity = "critical" if abs(rho) >= 0.7 else "strong" if abs(rho) >= 0.55 else "notable"
    # Export bin-level means so the analyst can eyeball the relationship shape.
    bins = pd.qcut(x[mask], q=10, duplicates="drop")
    grouped = y[mask].groupby(bins, observed=True).agg(["mean", "count"]).reset_index()
    grouped.columns = [a, f"mean_{b}", "n"]
    grouped[a] = grouped[a].astype(str)
    csv = write_entities_csv(
        grouped, output_dir, f"{spec.hypothesis_id}_{_slug(a)}__{_slug(b)}"
    )
    return Finding(
        hypothesis_id=f"{spec.hypothesis_id}_{_slug(a)}__{_slug(b)}",
        runner=spec.runner,
        title=f"Корреляция {a} ↔ {b}",
        severity=severity,
        summary=(
            f"Spearman ρ={rho:+.2f} (p={p:.1e}) между `{a}` и `{b}`. "
            f"Разрез по децилям `{a}` с средним `{b}` — в `{csv}`."
        ),
        metrics={"spearman_rho": float(rho), "p_value": float(p), "n": int(mask.sum())},
        entity_csv=csv,
    )


def _finding_cat_cat(
    sub: pd.DataFrame, a: str, b: str, spec: HypothesisSpec, output_dir: Path
) -> Finding | None:
    cont = pd.crosstab(sub[a], sub[b])
    # Keep table compact so chi-square is stable and CSV is readable.
    if cont.shape[0] < 2 or cont.shape[1] < 2:
        return None
    if cont.shape[0] > 30 or cont.shape[1] > 30:
        top_rows = cont.sum(axis=1).sort_values(ascending=False).head(30).index
        top_cols = cont.sum(axis=0).sort_values(ascending=False).head(30).index
        cont = cont.loc[top_rows, top_cols]
    try:
        chi2, p, dof, _expected = stats.chi2_contingency(cont.values)
    except Exception:
        return None
    n = cont.values.sum()
    k = min(cont.shape) - 1
    if n == 0 or k == 0:
        return None
    cramer_v = float(np.sqrt(chi2 / (n * k)))
    if cramer_v < _CRAMER_V_STRONG or p > 0.001:
        return None
    severity = "critical" if cramer_v >= 0.5 else "strong" if cramer_v >= 0.35 else "notable"
    # Row percentages — analyst instantly sees the conditional distribution.
    pct = (cont.div(cont.sum(axis=1), axis=0) * 100).round(2).reset_index()
    csv = write_entities_csv(
        pct, output_dir, f"{spec.hypothesis_id}_{_slug(a)}__{_slug(b)}"
    )
    return Finding(
        hypothesis_id=f"{spec.hypothesis_id}_{_slug(a)}__{_slug(b)}",
        runner=spec.runner,
        title=f"Зависимость {a} ↔ {b}",
        severity=severity,
        summary=(
            f"Cramér's V={cramer_v:.2f} (χ²={chi2:.0f}, p={p:.1e}) — "
            f"значения `{a}` и `{b}` сильно связаны. Таблица долей — в `{csv}`."
        ),
        metrics={"cramer_v": cramer_v, "chi2": float(chi2), "p_value": float(p), "n": int(n)},
        entity_csv=csv,
    )


def _finding_num_cat(
    sub: pd.DataFrame, a: str, b: str, spec: HypothesisSpec, output_dir: Path
) -> Finding | None:
    # a is numeric, b is categorical.
    num = pd.to_numeric(sub[a], errors="coerce")
    cat = sub[b].astype("category")
    mask = num.notna()
    if mask.sum() < _MIN_NON_NULL:
        return None
    groups = [num[mask][cat[mask] == g].values for g in cat[mask].cat.categories]
    groups = [g for g in groups if len(g) >= 20]
    if len(groups) < 2:
        return None
    # One-way ANOVA F-test; eta² from ss_between / ss_total.
    try:
        f_stat, p = stats.f_oneway(*groups)
    except Exception:
        return None
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_total = ((all_vals - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
    if eta_sq < _ETA_SQ_STRONG or p > 0.001:
        return None
    severity = "critical" if eta_sq >= 0.3 else "strong" if eta_sq >= 0.2 else "notable"
    summary_df = (
        pd.DataFrame({a: num[mask], b: cat[mask]})
        .groupby(b, observed=True)[a]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    csv = write_entities_csv(
        summary_df, output_dir, f"{spec.hypothesis_id}_{_slug(a)}__{_slug(b)}"
    )
    return Finding(
        hypothesis_id=f"{spec.hypothesis_id}_{_slug(a)}__{_slug(b)}",
        runner=spec.runner,
        title=f"`{a}` зависит от `{b}`",
        severity=severity,
        summary=(
            f"η²={eta_sq:.2f} (F={f_stat:.1f}, p={p:.1e}) — "
            f"среднее `{a}` значимо различается между категориями `{b}`. "
            f"Сводка по категориям — в `{csv}`."
        ),
        metrics={"eta_sq": eta_sq, "f_stat": float(f_stat), "p_value": float(p), "n_groups": len(groups)},
        entity_csv=csv,
    )


def _name_overlap(a: str, b: str) -> int:
    """Negative score of shared longest prefix — pairs sharing a prefix (like
    client_id / client_name) rank last in the pair queue."""
    la, lb = a.lower(), b.lower()
    n = 0
    for x, y in zip(la, lb):
        if x != y:
            break
        n += 1
    return -n


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text.lower())[:40]
