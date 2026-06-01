"""ML utility & realism (Part-A 12.1 TSTR, 11.3 discriminator AUC).

These are **lenient anti-catastrophe** gates, not production quality bars: without a
captured baseline and at test-scale epochs, the backend-agnostic signal is "the
synthetic data is usable / not trivially fake", not "it matches a tuned target".
Run at higher epochs so the signal is real.
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
def test_tstr_not_catastrophic(mrun):
    """12.1: a model trained on synthetic and tested on real is not much worse than
    one trained on real (Train-Synthetic-Test-Real vs Train-Real-Test-Real)."""
    spec = specs.all_datasets()[DS]
    base = mrun(DS, "csv")
    full = base.originals[DS]
    train = full.sample(frac=0.8, random_state=5)
    test = full.drop(train.index)
    res = mrun(DS, "csv", epochs=EPOCHS, tables_override={DS: train.reset_index(drop=True)})
    # Target `income` (a strong power-law FUNCTION of age) rather than `score` (rho~0.6 with
    # heavy noise): it gives a real, stable downstream signal so the gate measures whether the
    # age->income relationship survives, not the noise floor of a weak target.
    tstr, trtr = mr.tstr_scores(spec, DS, "income", res.generated[DS], train, test)
    assert tstr > -0.5, f"synthetic data has no predictive utility (TSTR R^2 ={tstr:.2f})"
    assert tstr >= trtr - 0.6, f"TSTR R^2 {tstr:.2f} far below TRTR {trtr:.2f}"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_discriminator_not_trivially_separable(mrun):
    """11.3: a classifier should not separate real from synthetic almost perfectly
    (which would mean the synthetic data lives in a disjoint region)."""
    res = mrun(DS, "csv", epochs=EPOCHS)
    auc = mr.discriminator_auc(res, DS)
    assert auc <= 0.99, f"real vs synthetic almost perfectly separable (AUC={auc:.3f})"
