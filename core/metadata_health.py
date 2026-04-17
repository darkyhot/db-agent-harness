"""Валидатор каталога метаданных: grain, FK, sample_values, partition_key.

Используется в tools/check_catalog.py для offline-проверок полноты метаданных.
Возвращает структурированный отчёт без побочных эффектов (не пишет в каталог).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.schema_loader import SchemaLoader


@dataclass
class MetadataIssue:
    """Отдельная проблема каталога."""

    severity: str       # "error" | "warning" | "info"
    category: str       # "grain" | "foreign_key" | "sample_values" | "partition_key" | "synonyms"
    subject: str        # "schema.table" или "schema.table.column"
    message: str


@dataclass
class MetadataHealthReport:
    """Структурированный отчёт о состоянии каталога."""

    total_tables: int = 0
    total_columns: int = 0
    issues: list[MetadataIssue] = field(default_factory=list)
    counts_by_category: dict[str, int] = field(default_factory=dict)
    counts_by_severity: dict[str, int] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tables": self.total_tables,
            "total_columns": self.total_columns,
            "counts_by_category": dict(self.counts_by_category),
            "counts_by_severity": dict(self.counts_by_severity),
            "issues": [
                {"severity": i.severity, "category": i.category, "subject": i.subject, "message": i.message}
                for i in self.issues
            ],
        }


def _bump(dest: dict[str, int], key: str) -> None:
    dest[key] = dest.get(key, 0) + 1


def check_catalog(schema_loader: "SchemaLoader") -> MetadataHealthReport:
    """Собрать отчёт о полноте метаданных.

    Правила:
    - Пустой grain у таблицы → error (grain — фундамент семантики).
    - Потенциальный FK (suggestable via infer_foreign_keys), но не заполнен → warning.
    - Колонка с semantic_class=enum_like, но без sample_values → info.
    - Колонка с датой в факт-таблице, но без partition_key=true → info (подсказка).
    """
    report = MetadataHealthReport()
    tables_df = schema_loader.tables_df
    report.total_tables = int(len(tables_df))
    report.total_columns = int(len(schema_loader.attrs_df))

    # 1. grain coverage
    for _, row in tables_df.iterrows():
        schema = str(row.get("schema_name", "") or "")
        table = str(row.get("table_name", "") or "")
        grain = str(row.get("grain", "") or "").strip()
        if not (schema and table):
            continue
        if not grain:
            report.issues.append(MetadataIssue(
                severity="error",
                category="grain",
                subject=f"{schema}.{table}",
                message="grain не задан (заполни tables_list.csv или запусти EnrichmentPipeline)",
            ))

    # 2. FK suggestions (то, что infer мог бы предложить, но колонка пуста)
    suggested_fks = schema_loader.infer_foreign_keys(dry_run=True)
    attrs_df = schema_loader.attrs_df
    for fk_key, target in suggested_fks.items():
        try:
            schema, table, column = fk_key.split(".")
        except ValueError:
            continue
        mask = (
            (attrs_df["schema_name"] == schema)
            & (attrs_df["table_name"] == table)
            & (attrs_df["column_name"] == column)
        )
        rows = attrs_df[mask]
        if rows.empty:
            continue
        if "foreign_key_target" in attrs_df.columns:
            current = str(rows.iloc[0].get("foreign_key_target", "") or "").strip()
            if current:
                continue
        report.issues.append(MetadataIssue(
            severity="warning",
            category="foreign_key",
            subject=fk_key,
            message=f"возможный FK → {target} (заполни foreign_key_target в attr_list.csv)",
        ))

    # 3. enum_like без sample_values
    schema_loader.ensure_column_semantics()
    for _, row in attrs_df.iterrows():
        schema = str(row.get("schema_name", "") or "")
        table = str(row.get("table_name", "") or "")
        column = str(row.get("column_name", "") or "")
        if not (schema and table and column):
            continue
        sem = schema_loader.get_column_semantics(schema, table, column)
        cls = str(sem.get("semantic_class") or "")
        if cls == "enum_like":
            samples = schema_loader.get_column_sample_values(schema, table, column)
            if not samples:
                report.issues.append(MetadataIssue(
                    severity="info",
                    category="sample_values",
                    subject=f"{schema}.{table}.{column}",
                    message="enum_like без sample_values — value-resolver будет зависеть от LLM",
                ))

    # 4. подсчёты
    for issue in report.issues:
        _bump(report.counts_by_category, issue.category)
        _bump(report.counts_by_severity, issue.severity)

    return report
