"""MR-IO2 / MR-IO3 — backend + pipeline consistency across file formats.

The flagship transfer-consistency relations: the *same* dataset trained/inferred
with the *same* seed must produce equivalent output regardless of the source file
format (CSV/Avro/Excel/Pickle/Parquet) or whether the source was Fernet-encrypted.
"""
import os
import sys

import numpy as np
import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import specs, mr_asserts as mr        # noqa: E402
from _lib.generate_datasets import SEED          # noqa: E402

# Trainable source formats with distinct on-disk representations. pkl is excluded
# (syngen's BinaryLoader returns no schema, so a .pkl source is not trainable); tsv/psv
# are excluded as pure delimiter variants of csv (identical content).
_TRAINABLE = {"csv", "avro", "xlsx", "parquet"}
CROSS = {n: [f for f in s.formats if f in _TRAINABLE]
         for n, s in specs.all_datasets().items()}
CROSS = {n: fs for n, fs in CROSS.items() if len(fs) >= 2}


@pytest.mark.metamorphic
@pytest.mark.slow
@pytest.mark.parametrize("dataset", list(CROSS))
def test_cross_format_equivalence(dataset, mrun):
    """MR-IO2: output is equivalent across source formats (same seed)."""
    fmts = CROSS[dataset]
    base = mrun(dataset, fmts[0])
    problems = []
    for fmt in fmts[1:]:
        problems += mr.cross_format(base, mrun(dataset, fmt))
    assert not problems, "; ".join(problems)


@pytest.mark.metamorphic
def test_encrypted_dat_roundtrip():
    """MR-IO3: Fernet ``.dat`` encryption I/O is lossless.

    Syngen's encryption is for its own *at-rest persisted samples* (an encrypted
    ``.dat`` is not a trainable source — metadata validation reads source columns
    with a keyless loader), so we exercise the supported surface: round-trip a
    DataFrame through syngen's own ``DataEncryptor`` and assert exact fidelity
    (pickle preserves dtypes, so NaN vs "" is preserved too).
    """
    import tempfile
    from syngen.ml.data_loaders import DataEncryptor

    spec = specs.all_datasets()["mixed_wide"]            # has NaN + empty strings + many dtypes
    df = spec.builder(np.random.default_rng(SEED))["mixed_wide"]
    key = Fernet.generate_key().decode()                 # ephemeral; never committed
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "encrypted.dat")
        enc = DataEncryptor(path, key)
        enc.save_data(df)
        back, _ = enc.load_data()
    problems = mr.frame_equivalent(df, back, text_null_equiv=False)
    assert not problems, "; ".join(problems)
