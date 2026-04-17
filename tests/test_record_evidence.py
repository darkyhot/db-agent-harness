"""Тесты record_evidence (Direction 4.1)."""

from __future__ import annotations

from core.evidence_trace import record_evidence


class TestRecordEvidence:
    def test_new_trace_from_none(self):
        trace = record_evidence(
            None, "table_resolver", decision="ok",
            evidence={"n": 3}, warnings=["w1"],
        )
        assert "table_resolver" in trace
        entry = trace["table_resolver"]
        assert entry["decision"] == "ok"
        assert entry["evidence"] == {"n": 3}
        assert entry["warnings"] == ["w1"]
        assert "finished_at" in entry
        assert "history" not in entry

    def test_second_write_pushes_history(self):
        trace = record_evidence(
            None, "column_selector", decision="fail", evidence={"v": 1},
        )
        trace = record_evidence(
            trace, "column_selector", decision="ok", evidence={"v": 2},
        )
        entry = trace["column_selector"]
        assert entry["decision"] == "ok"
        assert entry["evidence"] == {"v": 2}
        assert "history" in entry
        assert len(entry["history"]) == 1
        assert entry["history"][0]["decision"] == "fail"

    def test_history_capped_at_4(self):
        trace = None
        for i in range(10):
            trace = record_evidence(
                trace, "node_x", decision=f"d{i}", evidence={"i": i},
            )
        entry = trace["node_x"]
        assert entry["decision"] == "d9"
        assert len(entry["history"]) == 4
        assert entry["history"][-1]["decision"] == "d8"

    def test_multiple_nodes_isolated(self):
        trace = record_evidence(None, "a", decision="a1")
        trace = record_evidence(trace, "b", decision="b1")
        assert trace["a"]["decision"] == "a1"
        assert trace["b"]["decision"] == "b1"
        assert "history" not in trace["a"]
        assert "history" not in trace["b"]

    def test_defaults_empty_containers(self):
        trace = record_evidence(None, "n", decision="x")
        entry = trace["n"]
        assert entry["evidence"] == {}
        assert entry["warnings"] == []

    def test_returns_new_dict(self):
        orig = {"a": {"decision": "a1"}}
        trace = record_evidence(orig, "b", decision="b1")
        assert "b" not in orig
        assert "b" in trace
