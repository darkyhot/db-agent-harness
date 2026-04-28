"""Контракт plan-level верификатора: PlanVerdict / PlanEdit и empty_verdict."""

import pytest
from pydantic import ValidationError

from core.plan_verifier_models import PlanEdit, PlanVerdict, empty_verdict


def test_empty_verdict_default_shape():
    v = empty_verdict()
    assert v == {"verdict": "approved", "reasons": [], "edits": []}


def test_plan_verdict_approved_minimal():
    parsed = PlanVerdict.model_validate({"verdict": "approved"})
    assert parsed.verdict == "approved"
    assert parsed.reasons == []
    assert parsed.edits == []


def test_plan_verdict_rejected_with_replace_column():
    parsed = PlanVerdict.model_validate({
        "verdict": "rejected",
        "reasons": ["id вместо name"],
        "edits": [
            {
                "op": "replace_column",
                "target_role": "group_by",
                "from_ref": "s.t.gosb_id",
                "to_ref": "s.dim.gosb_name",
                "reason": "label slot needs name",
            }
        ],
    })
    assert parsed.verdict == "rejected"
    assert len(parsed.edits) == 1
    assert parsed.edits[0].op == "replace_column"
    assert parsed.edits[0].target_role == "group_by"


def test_plan_edit_rejects_unknown_op():
    with pytest.raises(ValidationError):
        PlanEdit.model_validate({"op": "rewrite_everything", "to_ref": "x"})


def test_plan_edit_rejects_extra_field():
    with pytest.raises(ValidationError):
        PlanEdit.model_validate({
            "op": "replace_column",
            "from_ref": "a.b.c",
            "to_ref": "a.b.d",
            "evil_field": "should_not_pass",
        })


def test_plan_verdict_rejects_extra_field_at_top():
    with pytest.raises(ValidationError):
        PlanVerdict.model_validate({"verdict": "approved", "extra": 1})
