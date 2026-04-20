"""CLI для отчёта о качестве генерации SQL за период.

Запуск:
    python -m tools.quality_report
    python -m tools.quality_report --days 7
    python -m tools.quality_report --json
    python -m tools.quality_report --top 20
"""

from __future__ import annotations

import argparse
import json
import sys

from core.memory import MemoryManager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Отчёт о качестве SQL-генерации")
    parser.add_argument("--days", type=int, default=30, help="Период в днях")
    parser.add_argument("--top", type=int, default=10, help="Сколько проблемных запросов показать")
    parser.add_argument("--json", action="store_true", help="Вывод в JSON")
    args = parser.parse_args(argv)

    mgr = MemoryManager()
    metrics = mgr.get_sql_quality_metrics(days=args.days)
    top_problems = mgr.get_top_problem_queries(days=args.days, limit=args.top)
    error_breakdown = mgr.get_error_type_breakdown(days=args.days)

    payload = {
        "period_days": args.days,
        "metrics": metrics,
        "top_problem_queries": top_problems,
        "error_type_breakdown": error_breakdown,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    total = metrics.get("total_queries", 0)
    print(f"=== SQL Quality Report (last {args.days} days) ===")
    print(f"Всего запросов: {total}")
    if total == 0:
        print("Нет данных за указанный период.")
        return 0
    print(f"Success rate:         {metrics.get('success_rate', 0)}%")
    print(f"First-try success:    {metrics.get('first_try_success_rate', 0)}%")
    print(f"Avg retries:          {metrics.get('avg_retries', 0)}")
    print(f"Max retries:          {metrics.get('max_retries', 0)}")
    print(f"Avg duration (ms):    {metrics.get('avg_duration_ms', 0)}")
    print("")

    print("Status distribution:")
    for k, v in sorted(metrics.get("status_distribution", {}).items()):
        print(f"  {k:25} {v}")
    print("")

    if error_breakdown:
        print("Error type breakdown:")
        rows = sorted(error_breakdown.items(), key=lambda x: -x[1]["count"])
        for et, info in rows:
            print(f"  {et:25} count={info['count']:4} median_retries={info['median_retries']}")
            sample = info.get("sample_user_input") or ""
            if sample:
                print(f"    пример: {sample[:120]}")
        print("")

    if top_problems:
        print(f"Top-{len(top_problems)} проблемных запросов (по total_retries):")
        for i, p in enumerate(top_problems, 1):
            print(
                f"  {i:2}. [{p['last_status']:8} / {p['last_error_type']:15}] "
                f"attempts={p['attempts']} retries={p['total_retries']}"
            )
            print(f"      {p['user_input'][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
