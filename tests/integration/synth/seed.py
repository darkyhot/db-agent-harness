"""CLI для пересоздания тестовой БД с синтетикой.

Использование:

    # Пересоздать схему и залить 200 строк/таблицу (дефолт)
    python -m tests.integration.synth.seed --drop

    # Тонкая настройка
    python -m tests.integration.synth.seed --drop --rows 500 --seed 7

    # Указать путь к тестовому конфигу (по умолчанию tests/integration/test_config.json)
    python -m tests.integration.synth.seed --drop --config tests/integration/test_config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from core.database import DatabaseManager
from tests.integration.synth.data_generator import generate_all
from tests.integration.synth.ddl_generator import load_metadata
from tests.integration.synth.loader import drop_and_create_schema, load_all

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST_CONFIG = ROOT / "tests" / "integration" / "test_config.json"


def _ensure_config_path(arg_path: str | None) -> Path:
    """Найти или собрать на лету тестовый config.json из env-переменных."""
    if arg_path:
        path = Path(arg_path).resolve()
        if not path.exists():
            sys.exit(f"--config указывает на несуществующий файл: {path}")
        return path

    if DEFAULT_TEST_CONFIG.exists():
        return DEFAULT_TEST_CONFIG

    # Собираем временный конфиг из env-переменных.
    cfg = {
        "user_id": os.getenv("TEST_PG_USER", "test"),
        "host": os.getenv("TEST_PG_HOST", "localhost"),
        "port": int(os.getenv("TEST_PG_PORT", "55432")),
        "database": os.getenv("TEST_PG_DB", "agent_test"),
        "llm_model": "GigaChat-2-Max",
        "debug_prompt": False,
        "show_plan": False,
        "llm_verifier_enabled": False,
    }
    tmp = Path(tempfile.gettempdir()) / "db_agent_test_config.json"
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return tmp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Залить синтетику в тестовый Postgres")
    parser.add_argument("--rows", type=int, default=int(os.getenv("SYNTH_ROWS", "200")),
                        help="строк на таблицу (по умолчанию 200, или env SYNTH_ROWS)")
    parser.add_argument("--seed", type=int, default=42, help="seed RNG (по умолчанию 42)")
    parser.add_argument("--drop", action="store_true",
                        help="DROP SCHEMA CASCADE перед созданием")
    parser.add_argument("--config", default=None,
                        help="путь к тестовому config.json (по умолчанию tests/integration/test_config.json)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = _ensure_config_path(args.config)
    db = DatabaseManager(config_path=config_path)
    if not db.is_configured:
        sys.exit(f"DatabaseManager: тестовый конфиг неполон: {config_path}")

    print(f"[seed] using config: {config_path}", flush=True)
    print(f"[seed] target: {db.config_summary}", flush=True)

    tables = load_metadata()
    print(f"[seed] metadata loaded: {len(tables)} tables", flush=True)

    if args.drop:
        print("[seed] DROP + CREATE schema/tables/FKs...", flush=True)
        drop_and_create_schema(db, tables)

    print(f"[seed] generating rows ({args.rows}/table, seed={args.seed})...", flush=True)
    order, data = generate_all(rows_per_table=args.rows, seed=args.seed)

    print("[seed] inserting...", flush=True)
    counts = load_all(db, order, data)

    total = sum(counts.values())
    print(f"[seed] done: {len(counts)} tables, {total} rows total", flush=True)
    for full, n in counts.items():
        print(f"  {full}: {n}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
