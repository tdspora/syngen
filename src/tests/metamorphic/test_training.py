"""Training-process behaviour (Part-A 13.1 loss convergence, 13.2 more-epochs-no-worse,
13.4 batch-size sensitivity).

13.3 (KL collapse) is intentionally **not** asserted as a balance ratio here: the
open-source CVAE trains with KL weight 0 and the enterprise model uses an MMD penalty,
so there is no reconstruction/KL balance to monitor. The latent/mode-collapse risk it
targets is instead guarded by ``generation_diverse`` (see test_diversity).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import mr_asserts as mr        # noqa: E402

DS = "numeric_explicit"


@pytest.mark.metamorphic
@pytest.mark.slow
def test_loss_converges(mrun):
    """13.1: training loss at the last epoch is below the first epoch."""
    res = mrun(DS, "csv", epochs=8)
    if res.losses is None:
        pytest.skip("no losses.csv captured in this environment")
    problems = mr.loss_decreased(res)
    assert not problems, "; ".join(problems)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_more_epochs_no_collapse(mrun):
    """13.2: training longer must not make the output worse (here: must not collapse)."""
    res = mrun(DS, "csv", epochs=15)
    problems = mr.preservation(res, DS)
    assert not problems, "; ".join(problems)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_batch_size_stability(mrun):
    """13.4: a different batch size yields a similarly-shaped (non-collapsed) output."""
    base = mrun(DS, "csv")                       # default batch size
    big = mrun(DS, "csv", batch_size=64)
    problems = mr.distribution_equivalent(base, big, tol=mr.preservation_tol())
    assert not problems, "; ".join(problems)
