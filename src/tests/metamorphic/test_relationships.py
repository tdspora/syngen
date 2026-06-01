"""Relationship & dependency preservation (Part-A sections 4.3, 5.2, 11.1, 11.2).

These check that the generator preserves relationships *between* columns, not just
each column on its own. Run at slightly higher epochs so the joint structure has a
chance to be learned; thresholds stay lenient (anti-catastrophe).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import mr_asserts as mr        # noqa: E402

EPOCHS = 8


@pytest.mark.metamorphic
@pytest.mark.slow
def test_correlation_preserved(mrun):
    """5.2 / 11.2: age<->score correlation and the income=f(age) dependency survive."""
    res = mrun("numeric_explicit", "csv", epochs=EPOCHS)
    assert not mr.correlation_preserved(
        res, "numeric_explicit", [("age", "score"), ("age", "income")])


@pytest.mark.metamorphic
@pytest.mark.slow
def test_conditional_amount_by_status(mrun):
    """11.1: the per-status mean of `amount` keeps its ordering across statuses."""
    res = mrun("mixed_wide", "csv", epochs=EPOCHS)
    assert not mr.conditional_preserved(res, "mixed_wide", "status", "amount")


@pytest.mark.metamorphic
@pytest.mark.slow
def test_conditional_tier_by_country(mrun):
    """11.1: tier (Low<Mid<High) stays conditional on country."""
    res = mrun("categorical_explicit", "csv", epochs=EPOCHS)
    assert not mr.conditional_preserved(
        res, "categorical_explicit", "country", "tier",
        ordinal_map={"Low": 0, "Mid": 1, "High": 2})


@pytest.mark.metamorphic
@pytest.mark.slow
def test_relationship_cardinality(mrun):
    """4.3: the average number of child rows per parent (1:N fan-out) is preserved."""
    res = mrun("keys_relational_clean", "csv")
    assert not mr.cardinality_preserved(res)
