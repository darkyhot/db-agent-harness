import logging
from unittest.mock import MagicMock

from core.log_safety import summarize_dict_keys, summarize_sql, summarize_text


def test_summarize_sql_hides_literals_and_keeps_table_names():
    sql = "SELECT * FROM dm.clients WHERE full_name = 'Иван Иванов' AND inn = '1234567890'"
    summary = summarize_sql(sql)

    assert "Иван Иванов" not in summary
    assert "1234567890" not in summary
    assert "dm.clients" in summary
    assert "sha=" in summary


def test_summarize_text_hides_raw_content():
    text = "паспорт 1234 567890"
    summary = summarize_text(text, label="error")

    assert "1234 567890" not in summary
    assert summary.startswith("error[")
    assert "sha=" in summary


def test_summarize_dict_keys_only_logs_keys():
    payload = {"value": "secret", "items": [1, 2, 3], "nested": {"name": "Ivan"}}
    summary = summarize_dict_keys(payload, label="semantic_frame")

    assert "secret" not in summary
    assert "Ivan" not in summary
    assert "value" in summary
    assert "items" in summary
    assert "nested" in summary


def test_sql_validator_does_not_log_raw_sql(caplog):
    from core.sql_validator import SQLValidator

    caplog.set_level(logging.INFO)
    db = MagicMock()
    db.explain_query.return_value = "Seq Scan"
    db.check_key_uniqueness.return_value = {"is_unique": True, "duplicate_pct": 0.0}

    validator = SQLValidator(db_manager=db, schema_loader=None)
    validator.validate("SELECT * FROM dm.clients WHERE full_name = 'Иван Иванов'")

    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Иван Иванов" not in joined
    assert "SELECT * FROM dm.clients" not in joined
    assert "dm.clients" in joined
