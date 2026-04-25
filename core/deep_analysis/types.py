"""Type definitions for deep table analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnalysisMode(str, Enum):
    FAST = "fast"
    DEEP = "deep"


class ColumnRole(str, Enum):
    """Semantic role of a column, inferred by the profiler.

    Drives which hypotheses are instantiable on a column. Mutually non-exclusive
    in theory — a column may be both id-like and category-like — but the
    profiler picks the strongest single role.
    """

    ID = "id"
    DATE = "date"
    DATETIME = "datetime"
    MONEY = "money"
    PERCENT = "percent"
    NUMERIC = "numeric"
    CATEGORY = "category"
    FLAG = "flag"
    TEXT_SHORT = "text_short"
    TEXT_LONG = "text_long"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Profiling data for a single column."""

    name: str
    dtype: str
    role: ColumnRole
    n_rows: int
    n_null: int
    n_unique: int
    top_values: list[tuple[Any, int]] = field(default_factory=list)
    numeric_stats: dict[str, float] | None = None
    date_range: tuple[str, str] | None = None
    max_str_len: int | None = None

    @property
    def null_pct(self) -> float:
        return (self.n_null / self.n_rows * 100) if self.n_rows else 0.0

    @property
    def unique_pct(self) -> float:
        non_null = self.n_rows - self.n_null
        return (self.n_unique / non_null * 100) if non_null else 0.0


@dataclass
class TableProfile:
    """Profile of an entire table after safe load."""

    schema: str
    table: str
    n_rows: int
    n_cols: int
    columns: dict[str, ColumnProfile]
    dropped_wide_text: list[str] = field(default_factory=list)
    load_strategy: str = "full"
    table_description: str = ""

    def cols_by_role(self, role: ColumnRole) -> list[str]:
        return [name for name, c in self.columns.items() if c.role == role]

    def date_columns(self) -> list[str]:
        return [
            name for name, c in self.columns.items()
            if c.role in (ColumnRole.DATE, ColumnRole.DATETIME)
        ]

    def numeric_columns(self) -> list[str]:
        return [
            name for name, c in self.columns.items()
            if c.role in (ColumnRole.NUMERIC, ColumnRole.MONEY, ColumnRole.PERCENT)
        ]

    def category_columns(self, max_cardinality: int = 200) -> list[str]:
        return [
            name for name, c in self.columns.items()
            if c.role == ColumnRole.CATEGORY and c.n_unique <= max_cardinality
        ]

    def flag_columns(self) -> list[str]:
        return [name for name, c in self.columns.items() if c.role == ColumnRole.FLAG]

    def id_columns(self) -> list[str]:
        return [name for name, c in self.columns.items() if c.role == ColumnRole.ID]

    def entity_candidates(self, min_card: int = 5, max_card: int = 500) -> list[str]:
        """Columns that can serve as cohort/entity keys for group_anomalies.

        Includes both true entity IDs (high cardinality) and low-cardinality
        categorical IDs like `tb_id` (15 territorial banks) — those are real
        cohorts even when their count is small. Result is sorted by cardinality
        descending so the catalog can pick multiple granularity levels.
        """
        out: list[tuple[str, int]] = []
        for name, c in self.columns.items():
            if c.role == ColumnRole.ID:
                out.append((name, c.n_unique))
            elif c.role == ColumnRole.CATEGORY and min_card <= c.n_unique <= max_card:
                out.append((name, c.n_unique))
        out.sort(key=lambda x: -x[1])
        return [n for n, _ in out]


@dataclass
class HypothesisSpec:
    """A single hypothesis to check.

    Produced either by the deterministic catalog, the LLM layer, or translated
    from a user's free-form hypothesis. Feeds into exactly one runner.
    """

    hypothesis_id: str
    runner: str                              # "seasonality" | "outliers" | "group_anomalies" | ...
    title: str                                # short human-readable name
    rationale: str                            # why this is worth checking
    params: dict[str, Any]                    # runner-specific arguments
    priority: float = 0.5                     # 0..1, bigger = run earlier
    source: str = "catalog"                   # "catalog" | "llm" | "user"
    est_cost_seconds: float = 10.0


@dataclass
class Finding:
    """A single result produced by a runner."""

    hypothesis_id: str
    runner: str
    title: str
    severity: str                             # "info" | "notable" | "strong" | "critical"
    summary: str                              # 1-3 sentences, business-readable
    metrics: dict[str, Any] = field(default_factory=dict)
    entity_csv: str | None = None             # relative path to CSV with violators
    chart_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisContext:
    """Runtime context threaded through the pipeline."""

    schema: str
    table: str
    mode: AnalysisMode
    deadline_ts: float                        # unix timestamp, hard stop
    output_dir: str                           # workspace/deep_analysis/<table>/<ts>/
    progress: "ProgressReporter"              # forward ref

    def seconds_left(self) -> float:
        import time
        return max(0.0, self.deadline_ts - time.time())
