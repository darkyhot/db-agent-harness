"""Фасад GraphNodes: объединяет все mixin-узлы в один класс для графа."""

from graph.nodes.common import BaseNodeMixin
from graph.nodes.intent import IntentNodes
from graph.nodes.explorer import ExplorerNodes
from graph.nodes.sql_pipeline import SqlPipelineNodes
from graph.nodes.correction import CorrectionNodes
from graph.nodes.summarizer import SummarizerNodes
from graph.nodes.dispatcher import DispatcherNodes


class GraphNodes(
    IntentNodes,
    ExplorerNodes,
    SqlPipelineNodes,
    CorrectionNodes,
    SummarizerNodes,
    DispatcherNodes,
    BaseNodeMixin,
):
    """Узлы графа агента — фасад, объединяющий все mixin-классы.

    Каждый mixin содержит 1-3 узла графа:
    - IntentNodes: intent_classifier, table_resolver
    - ExplorerNodes: table_explorer, column_selector
    - SqlPipelineNodes: sql_planner, sql_writer, sql_validator_node
    - CorrectionNodes: error_diagnoser, sql_fixer
    - SummarizerNodes: summarizer
    - DispatcherNodes: tool_dispatcher
    - BaseNodeMixin: общие методы (парсинг, вызов инструментов, контекст)
    """
    pass
