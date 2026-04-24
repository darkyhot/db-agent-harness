"""Фасад GraphNodes: объединяет все mixin-узлы в один класс для графа."""

from graph.nodes.common import BaseNodeMixin
from graph.nodes.intent import IntentNodes
from graph.nodes.hint_extractor import HintExtractorNodes
from graph.nodes.hint_extractor_llm import HintExtractorLLMNodes
from graph.nodes.explicit_mode_dispatcher import ExplicitModeDispatcherNodes
from graph.nodes.explorer import ExplorerNodes
from graph.nodes.sql_pipeline import SqlPipelineNodes
from graph.nodes.plan_preview import PlanPreviewNodes
from graph.nodes.plan_edit import PlanEditNodes
from graph.nodes.correction import CorrectionNodes
from graph.nodes.summarizer import SummarizerNodes
from graph.nodes.dispatcher import DispatcherNodes


class GraphNodes(
    IntentNodes,
    HintExtractorLLMNodes,
    HintExtractorNodes,
    ExplicitModeDispatcherNodes,
    ExplorerNodes,
    SqlPipelineNodes,
    PlanPreviewNodes,
    PlanEditNodes,
    CorrectionNodes,
    SummarizerNodes,
    DispatcherNodes,
    BaseNodeMixin,
):
    """Узлы графа агента — фасад, объединяющий все mixin-классы.

    Каждый mixin содержит 1-3 узла графа:
    - IntentNodes: intent_classifier, table_resolver
    - HintExtractorLLMNodes: hint_extractor_llm (LLM-экстрактор подсказок)
    - HintExtractorNodes: hint_extractor (regex + merge с LLM-результатом)
    - ExplicitModeDispatcherNodes: explicit_mode_dispatcher (детерминированный)
    - ExplorerNodes: table_explorer, column_selector
    - SqlPipelineNodes: sql_planner, sql_writer, sql_static_checker, sql_validator_node
    - PlanPreviewNodes: plan_preview (детерминированный)
    - PlanEditNodes: plan_edit_router, plan_patcher, source_rebinder, intent_rewriter,
      plan_edit_validator, plan_diff_renderer
    - CorrectionNodes: error_diagnoser, sql_fixer
    - SummarizerNodes: summarizer
    - DispatcherNodes: tool_dispatcher
    - BaseNodeMixin: общие методы (парсинг, вызов инструментов, контекст)
    """
    pass
