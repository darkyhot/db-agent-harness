"""Column-equivalence detection.

Many corporate tables expose the same business attribute under several names —
e.g. `post_id`, `post_name`, `pos_name` are 1:1 because they come from the same
dictionary. Without deduplication, every runner produces three near-identical
findings per such trio, blowing up the report.

We compute equivalence classes once after profiling and let the rest of the
pipeline operate on representatives only. Two columns are equivalent when:

- Both are categorical/flag/id with cardinality ≤ 200, AND
- Cramér's V on a subsample is ≥ 0.99 (one fully determines the other).

The rule is conservative on purpose — we don't want to fold semantically
distinct columns just because they correlate. V≥0.99 is the empirical noise
floor for a true 1:1 mapping on real samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from core.deep_analysis.logging_setup import get_logger
from core.deep_analysis.types import ColumnRole, TableProfile

_EQUIV_CRAMER_V = 0.99
_EQUIV_MAX_CARDINALITY = 200
_EQUIV_SAMPLE_ROWS = 20_000


def compute_equivalence_groups(
    df: pd.DataFrame, profile: TableProfile
) -> dict[str, list[str]]:
    """Return {representative_column: sorted_member_list}.

    Every column from the profile is included — singletons map to themselves.
    Within a multi-member class, the representative is the column with the
    shortest, alphabetically-earliest name (tends to be the canonical id).
    """
    log = get_logger()
    candidates = [
        name for name, c in profile.columns.items()
        if c.role in (ColumnRole.CATEGORY, ColumnRole.FLAG, ColumnRole.ID)
        and 2 <= c.n_unique <= _EQUIV_MAX_CARDINALITY
    ]
    parent: dict[str, str] = {c: c for c in candidates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _better_rep(a: str, b: str) -> str:
        # Shorter wins; ties broken alphabetically — stable, canonical pick.
        if (len(a), a) <= (len(b), b):
            return a
        return b

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        winner = _better_rep(rx, ry)
        loser = ry if winner == rx else rx
        parent[loser] = winner

    if len(candidates) >= 2 and len(df) > 0:
        sample = (
            df.sample(_EQUIV_SAMPLE_ROWS, random_state=42)
            if len(df) > _EQUIV_SAMPLE_ROWS else df
        )
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                if find(a) == find(b):
                    continue
                try:
                    cont = pd.crosstab(sample[a], sample[b])
                except Exception:
                    continue
                if cont.shape[0] < 2 or cont.shape[1] < 2:
                    continue
                try:
                    chi2, _p, _dof, _exp = stats.chi2_contingency(cont.values)
                except Exception:
                    continue
                n = int(cont.values.sum())
                k = min(cont.shape) - 1
                if n == 0 or k == 0:
                    continue
                v = float(np.sqrt(chi2 / (n * k)))
                if v >= _EQUIV_CRAMER_V:
                    union(a, b)

    groups: dict[str, list[str]] = {}
    for c in candidates:
        rep = find(c)
        groups.setdefault(rep, []).append(c)
    for c in profile.columns:
        if c not in parent:
            groups.setdefault(c, []).append(c)
    for rep, members in groups.items():
        members.sort()

    multi = {rep: ms for rep, ms in groups.items() if len(ms) > 1}
    if multi:
        log.info(
            "Equivalence: %d multi-member groups (%d redundant cols folded)",
            len(multi),
            sum(len(ms) - 1 for ms in multi.values()),
        )
    return groups
