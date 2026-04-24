"""Deep table analysis: comprehensive pattern discovery pipeline.

Separate from the main SQL graph. Accessed through /deep_table_analysis CLI
command. Provides two modes: fast (~30-40 min) and deep (time-unbounded, until
all hypotheses exhausted).

Public entry points:
- run_deep_analysis(schema, table, ...) — full auto-scan orchestration.
- build_user_hypothesis_plan(text, profile, semantics) — translate free-form
  user hypothesis into a formal check plan for plan-preview approval.
"""

from core.deep_analysis.orchestrator import run_deep_analysis
from core.deep_analysis.types import AnalysisMode, Finding, HypothesisSpec
from core.deep_analysis.user_hypothesis import build_user_hypothesis_plan

__all__ = [
    "AnalysisMode",
    "Finding",
    "HypothesisSpec",
    "build_user_hypothesis_plan",
    "run_deep_analysis",
]
