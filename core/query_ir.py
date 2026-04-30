"""Semantic intermediate representation for analytical requests.

QuerySpec is the primary semantic contract of the agent.  Pydantic is used as
an in-process validation library only; no network access is needed at runtime.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


QueryTask = Literal["answer_data", "inspect_schema", "edit_plan", "clarify"]
QueryStrategy = Literal["count_attributes", "aggregate", "list", "inspect_schema", "edit_plan", "clarify"]
MetricOperation = Literal["count", "sum", "avg", "min", "max", "list"]
DistinctPolicy = Literal["auto", "distinct", "all"]
TimeGrain = Literal["day", "week", "month", "quarter", "year"]
OrderDirection = Literal["ASC", "DESC"]


class StrictModel(BaseModel):
    """Base model for IR objects: reject unknown fields."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(self.model_dump(mode="json", by_alias=True))


class Evidence(StrictModel):
    """Small proof object explaining why a semantic field exists."""

    source: str = "user"
    text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Alternative(StrictModel):
    """Alternative interpretation for low-confidence semantic fields."""

    value: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class MetricSpec(StrictModel):
    """Metric requested by the user, before binding to a physical column."""

    operation: MetricOperation
    target: str | None = None
    distinct_policy: DistinctPolicy = "auto"
    label: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternatives: list[Alternative] = Field(default_factory=list)

    @field_validator("operation", "distinct_policy", mode="before")
    @classmethod
    def _lower_enum(cls, value: Any) -> Any:
        return str(value).strip().lower() if value is not None else value


class DimensionSpec(StrictModel):
    """Output axis requested by the user."""

    target: str
    role: str = "group_by"
    source_table: str | None = None
    join_key: str | None = None
    label: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternatives: list[Alternative] = Field(default_factory=list)


class EntitySpec(StrictModel):
    """Semantic business entity mentioned by the user before physical binding."""

    name: str
    canonical: str | None = None
    target_column_hint: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FilterSpec(StrictModel):
    """Business filter before resolving it into a SQL predicate."""

    target: str
    operator: str = "="
    value: Any = None
    value_kind: str = "literal"
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    alternatives: list[Alternative] = Field(default_factory=list)

    @field_validator("operator", mode="before")
    @classmethod
    def _normalize_operator(cls, value: Any) -> str:
        return str(value or "=").strip().upper()


class TimeRangeSpec(StrictModel):
    """Calendar range or granularity extracted from the request."""

    start: str | None = None
    end: str | None = None
    grain: TimeGrain | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("grain", mode="before")
    @classmethod
    def _normalize_grain(cls, value: Any) -> Any:
        return str(value).strip().lower() if value not in (None, "") else None


class SourceConstraint(StrictModel):
    """User or model constraint for table/source selection."""

    table: str | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    semantic: str | None = None
    required: bool = False
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def schema(self) -> str | None:
        return self.schema_name


class JoinConstraint(StrictModel):
    """Requested join relation before join-governor validation."""

    left: str | None = None
    right: str | None = None
    key: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ClarificationSpec(StrictModel):
    """Typed clarification contract shown to the user instead of guessing."""

    question: str
    reason: str = ""
    field: str = ""
    options: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class OrderBySpec(StrictModel):
    """Requested result ordering before binding to a SQL expression."""

    target: str
    direction: OrderDirection = "DESC"
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, value: Any) -> str:
        return str(value or "DESC").strip().upper()


class QuerySpec(StrictModel):
    """Single semantic source of truth for a user request."""

    task: QueryTask = "answer_data"
    strategy: QueryStrategy = "aggregate"
    entities: list[EntitySpec] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    dimensions: list[DimensionSpec] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    time_range: TimeRangeSpec | None = None
    having: list[FilterSpec] = Field(default_factory=list)
    order_by: OrderBySpec | None = None
    limit: int | None = None
    source_constraints: list[SourceConstraint] = Field(default_factory=list)
    excluded_source_constraints: list[SourceConstraint] = Field(default_factory=list)
    join_constraints: list[JoinConstraint] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification: ClarificationSpec | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)

    @field_validator("task", mode="before")
    @classmethod
    def _normalize_task(cls, value: Any) -> Any:
        return str(value or "answer_data").strip().lower()

    @field_validator("strategy", mode="before")
    @classmethod
    def _normalize_strategy(cls, value: Any) -> Any:
        return str(value or "aggregate").strip().lower()

    @model_validator(mode="after")
    def _validate_strategy_contract(self) -> "QuerySpec":
        if self.task == "inspect_schema":
            self.strategy = "inspect_schema"
        elif self.task == "edit_plan":
            self.strategy = "edit_plan"
        elif self.task == "clarify":
            self.strategy = "clarify"

        if self.strategy == "count_attributes":
            count_targets = [
                metric for metric in self.metrics
                if metric.operation == "count" and str(metric.target or "").strip()
            ]
            if len(self.entities) < 2 and len(count_targets) < 2:
                raise ValueError("count_attributes requires at least two entities or count metric targets")
        return self

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> tuple["QuerySpec | None", list[str]]:
        """Backward-compatible validation wrapper."""
        try:
            spec = cls.model_validate(payload)
        except ValidationError as exc:
            return None, _validation_errors(exc)
        if spec.task == "clarify" and spec.clarification is None:
            return None, ["clarification: required when task='clarify'"]
        _normalize_calendar_literal_filters(spec)
        return spec, []

    def to_legacy_intent(self) -> dict[str, Any]:
        """Compatibility projection for existing pipeline nodes."""
        first_metric = self.metrics[0] if self.metrics else None
        entities: list[str] = []
        for entity in self.entities:
            entities.append(entity.canonical or entity.name)
        for metric in self.metrics:
            if metric.target:
                entities.append(metric.target)
        entities.extend(dim.target for dim in self.dimensions)
        entities.extend(f.target for f in self.filters)
        entities = list(dict.fromkeys([e for e in entities if e]))

        intent_name = {
            "answer_data": "analytics",
            "inspect_schema": "schema_question",
            "edit_plan": "followup",
            "clarify": "clarification",
        }.get(self.task, "analytics")

        date_filters = {"from": None, "to": None}
        if self.time_range:
            date_filters = {"from": self.time_range.start, "to": self.time_range.end}

        return {
            "intent": intent_name,
            "strategy": self.strategy,
            "entities": entities,
            "date_filters": date_filters,
            "aggregation_hint": first_metric.operation if first_metric else ("count" if self.strategy == "count_attributes" else None),
            "needs_search": False,
            "complexity": "join" if self.join_constraints or len(self.source_constraints) > 1 else "single_table",
            "clarification_question": self.clarification.question if self.clarification else "",
            "filter_conditions": [
                {
                    "column_hint": f.target,
                    "operator": f.operator,
                    "value": f.value,
                }
                for f in self.filters
                if f.value not in (None, "")
            ],
            "explicit_join": [
                {"table_hint": item.right or item.left, "column_hint": item.key}
                for item in self.join_constraints
                if item.key
            ],
            "required_output": [dim.target for dim in self.dimensions],
            "month_without_year": False,
        }

    def to_legacy_user_hints(self) -> dict[str, Any]:
        """Compatibility projection for existing hint-aware deterministic code."""
        agg_hints = [
            {
                "function": metric.operation,
                "column": metric.target,
                "distinct": metric.distinct_policy == "distinct",
            }
            for metric in self.metrics
        ]
        if self.strategy == "count_attributes" and not agg_hints:
            for entity in self.entities:
                target = entity.target_column_hint or entity.canonical or entity.name
                if target:
                    agg_hints.append({
                        "function": "count",
                        "column": target,
                        "distinct": True,
                    })
        preferences_list = [dict(item) for item in agg_hints if item.get("function")]
        for idx, preferences in enumerate(preferences_list):
            if preferences.get("column") == "*":
                preferences["force_count_star"] = True
            if (
                idx < len(self.metrics)
                and preferences.get("distinct") is False
                and self.metrics[idx].distinct_policy == "all"
            ):
                preferences["distinct"] = False
        preferences = dict(preferences_list[0]) if preferences_list else {}

        must_keep: list[tuple[str, str]] = []
        for source in self.source_constraints:
            schema = source.schema
            table = source.table
            if table and "." in table and not schema:
                schema, table = table.split(".", 1)
            if schema and table:
                must_keep.append((schema, table))

        excluded_tables: list[str] = []
        for source in self.excluded_source_constraints:
            schema = source.schema
            table = source.table
            if table and "." in table and not schema:
                schema, table = table.split(".", 1)
            if schema and table:
                excluded_tables.append(f"{schema}.{table}")

        dim_sources: dict[str, dict[str, str]] = {}
        for dim in self.dimensions:
            if dim.source_table:
                dim_sources[dim.target] = {"table": dim.source_table}
                if dim.join_key:
                    dim_sources[dim.target]["join_col"] = dim.join_key

        return {
            "must_keep_tables": must_keep,
            "excluded_tables": excluded_tables,
            "join_fields": [j.key for j in self.join_constraints if j.key],
            "dim_sources": dim_sources,
            "having_hints": [
                {
                    "op": item.operator,
                    "value": item.value,
                    "unit_hint": item.target,
                }
                for item in self.having
                if item.value not in (None, "")
            ],
            "group_by_hints": [dim.target for dim in self.dimensions],
            "aggregate_hints": agg_hints,
            "aggregation_preferences": preferences,
            "aggregation_preferences_list": preferences_list,
            "time_granularity": (
                self.time_range.grain
                if self.time_range and any(_target_looks_calendar(dim.target) for dim in self.dimensions)
                else None
            ),
            "negative_filters": [
                str(f.value)
                for f in self.filters
                if str(f.operator or "").upper() in {"!=", "<>", "NOT IN"}
                and f.value not in (None, "")
            ],
        }

    def to_semantic_frame(self) -> dict[str, Any]:
        first_metric = self.metrics[0] if self.metrics else None
        count_metrics = [
            metric for metric in self.metrics
            if metric.operation == "count" and metric.target
        ]
        single_count_metric = len(count_metrics) == 1 and not self.dimensions
        filter_intents = []
        for idx, item in enumerate(self.filters):
            filter_intents.append({
                "request_id": f"query_spec:{idx}",
                "kind": "query_spec_filter",
                "query_text": str(item.value if item.value is not None else item.target),
                "column_hint": item.target,
                "operator": item.operator,
                "value": item.value,
                "match_score": item.confidence,
                "match_source": "query_spec",
            })
        return {
            "subject": count_metrics[0].target if single_count_metric else None,
            "metric_intent": first_metric.operation if first_metric else None,
            "business_event": self.filters[0].target if self.filters else (first_metric.target if first_metric else None),
            "qualifier": None,
            "output_dimensions": [dim.target for dim in self.dimensions],
            "requires_listing": first_metric.operation == "list" if first_metric else False,
            "requires_single_entity_count": single_count_metric,
            "requested_grain": None,
            "period_kind": "calendar" if self.time_range else None,
            "ambiguities": ["clarification"] if self.clarification_needed else [],
            "filter_intents": filter_intents,
        }


class SourceBinding(StrictModel):
    """Physical source selected for a semantic source constraint."""

    schema_name: str = Field(alias="schema")
    table: str
    reason: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def schema(self) -> str:
        return self.schema_name

    @property
    def full_name(self) -> str:
        return f"{self.schema_name}.{self.table}"

    def to_tuple(self) -> tuple[str, str]:
        return (self.schema_name, self.table)


class PlanIR(StrictModel):
    """Structured SQL plan before compilation to SQL text."""

    main_source: SourceBinding | None = None
    sources: list[SourceBinding] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    dimensions: list[DimensionSpec] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    joins: list[JoinConstraint] = Field(default_factory=list)
    time_range: TimeRangeSpec | None = None
    having: list[FilterSpec] = Field(default_factory=list)
    order_by: OrderBySpec | None = None
    limit: int | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class EditSpec(StrictModel):
    """Structured user edit against an existing QuerySpec/PlanIR."""

    action: str
    metrics: list[MetricSpec] = Field(default_factory=list)
    dimensions: list[DimensionSpec] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


def query_spec_json_schema() -> dict[str, Any]:
    """Return the JSON Schema used in LLM prompts."""
    schema = QuerySpec.model_json_schema()
    schema["required"] = [
        "task",
        "strategy",
        "entities",
        "metrics",
        "dimensions",
        "filters",
        "source_constraints",
        "join_constraints",
        "clarification_needed",
        "confidence",
    ]
    return schema


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            clean = _drop_empty(item)
            if clean not in (None, [], {}, ""):
                result[key] = clean
        return result
    if isinstance(value, list):
        return [_drop_empty(item) for item in value if _drop_empty(item) not in (None, [], {}, "")]
    return value


_RU_MONTHS = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "май": 5,
    "мая": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}
_CALENDAR_TARGET_TOKENS = (
    "дат", "date", "dt", "dttm", "timestamp", "месяц", "month",
    "период", "period", "квартал", "quarter", "год", "year", "отчет",
    "отчёт",
)


def _target_looks_calendar(target: str) -> bool:
    text = str(target or "").strip().lower()
    return bool(text) and any(token in text for token in _CALENDAR_TARGET_TOKENS)


def _parse_calendar_period(value: Any) -> tuple[str, str, TimeGrain] | None:
    text = str(value or "").strip().lower().replace("ё", "е")
    if not text:
        return None

    quarter_match = re.search(r"(?:^|\b)(?:q|кв(?:артал)?\.?)\s*([1-4])(?:\D+|$).*?\b(20\d{2})\b", text)
    if not quarter_match:
        quarter_match = re.search(r"\b([1-4])\s*(?:кв(?:артал)?\.?|quarter|q)\D+?\b(20\d{2})\b", text)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        month = (quarter - 1) * 3 + 1
        end_month = month + 3
        end_year = year + 1 if end_month > 12 else year
        end_month = 1 if end_month > 12 else end_month
        return f"{year:04d}-{month:02d}-01", f"{end_year:04d}-{end_month:02d}-01", "quarter"

    month = None
    for stem, number in _RU_MONTHS.items():
        if stem in text:
            month = number
            break
    year_match = re.search(r"\b(20\d{2})\b", text)
    if month and year_match:
        year = int(year_match.group(1))
        end_year = year + 1 if month == 12 else year
        end_month = 1 if month == 12 else month + 1
        return f"{year:04d}-{month:02d}-01", f"{end_year:04d}-{end_month:02d}-01", "month"

    if re.fullmatch(r"20\d{2}", text):
        year = int(text)
        return f"{year:04d}-01-01", f"{year + 1:04d}-01-01", "year"

    return None


def _normalize_calendar_literal_filters(spec: QuerySpec) -> None:
    """Promote literal calendar filters into TimeRangeSpec and drop duplicates."""
    kept: list[FilterSpec] = []
    promoted: tuple[str, str, TimeGrain, FilterSpec] | None = None
    has_time_range = bool(spec.time_range and (spec.time_range.start or spec.time_range.end))
    for item in spec.filters:
        parsed = _parse_calendar_period(item.value)
        if parsed and _target_looks_calendar(item.target):
            if not has_time_range:
                promoted = (parsed[0], parsed[1], parsed[2], item)
            continue
        kept.append(item)

    if promoted is None:
        spec.filters = kept
        return

    start, end, grain, source_filter = promoted
    spec.time_range = TimeRangeSpec(
        start=start,
        end=end,
        grain=grain,
        evidence=source_filter.evidence or [
            Evidence(
                source="query_spec_calendar_filter",
                text=f"{source_filter.target} {source_filter.operator} {source_filter.value}",
                confidence=source_filter.confidence or 0.8,
            )
        ],
        confidence=max(source_filter.confidence, 0.8),
    )
    spec.filters = kept


def _validation_errors(exc: ValidationError) -> list[str]:
    errors: list[str] = []
    for item in exc.errors():
        loc = _format_loc(item.get("loc", ()))
        msg = str(item.get("msg") or "invalid")
        errors.append(f"{loc}: {msg}" if loc else msg)
    return errors


def _format_loc(loc: Any) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += ("." if out else "") + str(part)
    return out
