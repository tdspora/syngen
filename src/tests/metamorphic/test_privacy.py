"""Privacy / leakage (Part-A 10.1 memorization, 10.2/10.3 DCR & membership-inference proxy).

The synthetic data should be *similar to* but not a *copy of* the training data, and
should be no closer to the data it was trained on than to unseen real data.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import mr_asserts as mr, specs        # noqa: E402

DS = "numeric_explicit"
EPOCHS = 12


@pytest.mark.metamorphic
@pytest.mark.slow
def test_no_memorization(mrun):
    """10.1: very few synthetic rows are (near-)exact copies of a training row."""
    res = mrun(DS, "csv", epochs=EPOCHS)
    frac = mr.memorization_fraction(res, DS)
    assert frac < 0.02, f"{frac:.1%} of synthetic rows are near-exact copies (memorisation)"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_dcr_no_privacy_leak(mrun):
    """10.2/10.3: synthetic data is no systematically closer to the training split than
    to an unseen holdout split (a DCR ratio near 1 ⇒ low membership-inference risk)."""
    spec = specs.all_datasets()[DS]
    base = mrun(DS, "csv")
    full = base.originals[DS]
    train = full.sample(frac=0.7, random_state=6)
    holdout = full.drop(train.index)
    res = mrun(DS, "csv", epochs=EPOCHS, tables_override={DS: train.reset_index(drop=True)})
    ratio = mr.dcr_ratio(spec, DS, res.generated[DS], train, holdout)
    assert 0.5 <= ratio <= 2.0, \
        f"DCR ratio {ratio:.2f} (synthetic systematically closer to train ⇒ leakage)"
