"""CLI для проверки полноты каталога метаданных.

Запуск:
    python -m tools.check_catalog
    python -m tools.check_catalog --json
    python -m tools.check_catalog --fail-on-warning
"""

from __future__ import annotations

import argparse
import json
import sys

from core.metadata_health import check_catalog
from core.schema_loader import SchemaLoader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверка полноты каталога метаданных")
    parser.add_argument("--json", action="store_true", help="Вывод в JSON")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Ненулевой exit-code, если есть warnings (по умолчанию — только errors)",
    )
    args = parser.parse_args(argv)

    loader = SchemaLoader()
    report = check_catalog(loader)

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print("=== Catalog Health Report ===")
        print(f"Таблиц: {report.total_tables}")
        print(f"Колонок: {report.total_columns}")
        print(f"Issues: {len(report.issues)}")
        print(f"  по severity: {report.counts_by_severity}")
        print(f"  по category: {report.counts_by_category}")
        print("")
        if report.issues:
            # Сначала errors, потом warnings, потом info
            order = {"error": 0, "warning": 1, "info": 2}
            sorted_issues = sorted(report.issues, key=lambda i: (order.get(i.severity, 3), i.subject))
            for issue in sorted_issues:
                tag = {
                    "error": "[ERROR]  ",
                    "warning": "[WARN]   ",
                    "info": "[INFO]   ",
                }.get(issue.severity, "[?]      ")
                print(f"{tag}{issue.category:15} {issue.subject}")
                print(f"          {issue.message}")
        else:
            print("Каталог полностью заполнен — проблем не найдено.")

    if report.has_errors:
        return 1
    if args.fail_on_warning and report.counts_by_severity.get("warning", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
