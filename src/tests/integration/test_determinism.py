"""End-to-end determinism: same table, same config, same seed => identical output.

This is the property whose absence let EPMCTDM-7630's reproducibility bugs ship. The
round-2 unit tests assert the `row_limit` draw at the `_preprocess_data` level, which is the
right unit test but cannot catch a *different* unseeded decision further down the fitting
path - and there were two more (scaler selection, tied-mode imputation).

Deliberately end-to-end and deliberately cheap: a few hundred rows, one epoch, a small
`size`. It is a determinism assertion, not a quality one.

Each run happens in a **subprocess**. That is the faithful reproduction of what is being
asserted - two separate processes, each with its own hash seed and RNG state. It was also
required for isolation in round 3, when ``Validator.errors`` was still a bare class
attribute shared by every ``Validator`` in a process; that is fixed now (EPMCTDM-7630,
``syngen.ml.config.validation``), but subprocesses remain the more faithful reproduction
of the scenario, so the design is unchanged. `setup_log_process` validating the log level
before writing to `os.environ` (EPMCTDM-7630) means an earlier test's `LOGURU_LEVEL` can no
longer be an invalid value, so no `env.pop("LOGURU_LEVEL", ...)` is needed here either. See
``tmp/os-3rd-report.md`` §6 and ``tmp/os-4rd-report.md`` §3.

The fixture table exercises all three seeded fitting decisions in one run:

* ``row_limit`` below the row count            -> ``processors.ROW_SUBSET_SEED``
* ``borderline`` (normal + one huge outlier)   -> ``features.NORMALITY_SAMPLE_SEED``;
  a 500-of-N Shapiro subsample contains the outlier about half the time, which flips the
  selected scaler
* ``tied_cat`` (three tied values, with nulls) -> ``dataset.FITTING_SAMPLE_SEED``;
  ``mode()`` returns all three, so the fill value is a real choice
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from tests.conftest import SUCCESSFUL_MESSAGE

N_ROWS = 300
ROW_LIMIT = 200          # below N_ROWS, so the sampling path is exercised
EPOCHS = 1
INFER_SIZE = 100
TABLE = "determinism_probe"

_RUNNER = textwrap.dedent(
    """
    import sys
    from syngen.train import launch_train
    from syngen.infer import launch_infer

    source, table = sys.argv[1], sys.argv[2]
    epochs, row_limit, size = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    launch_train(
        source=source, table_name=table, epochs=epochs, row_limit=row_limit,
        reports="none", log_level="ERROR",
    )
    launch_infer(
        table_name=table, size=size, random_seed=1, reports="none", log_level="ERROR",
    )
    """
)


def _src_root() -> str:
    """Absolute path of the ``src`` dir, so the subprocess imports this working tree."""
    here = os.path.dirname(os.path.abspath(__file__))          # src/tests/integration
    return os.path.abspath(os.path.join(here, "..", ".."))     # src


def _build_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    numeric = rng.normal(size=N_ROWS)
    numeric[0] = 1e6                                  # borderline-normality driver
    tied = (["a"] * 90 + ["b"] * 90 + ["c"] * 90 + [None] * 30)[:N_ROWS]
    return pd.DataFrame(
        {
            "borderline": numeric,
            "plain_num": rng.integers(0, 100, size=N_ROWS),
            "tied_cat": tied,
            "steady_cat": rng.choice(["x", "y"], size=N_ROWS),
        }
    )


def _run_once(workdir, source_csv, hash_seed: str) -> pd.DataFrame:
    """One full train+infer in its own process, returning the generated frame."""
    runner = workdir / "_runner.py"
    runner.write_text(_RUNNER)

    env = dict(os.environ)
    # a different hash seed per run, so anything depending on set/dict iteration order
    # differs between the two - the runs must still agree despite that
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = _src_root()
    env["OMP_NUM_THREADS"] = "4"

    result = subprocess.run(
        [
            sys.executable, str(runner), str(source_csv), TABLE,
            str(EPOCHS), str(ROW_LIMIT), str(INFER_SIZE),
        ],
        cwd=str(workdir), env=env, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, (
        f"train+infer subprocess failed (rc={result.returncode})\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}"
    )

    generated = sorted(
        path
        for path in (workdir / "model_artifacts" / "tmp_store").rglob("merged_infer_*.csv")
        if not path.stem.endswith("_preview")
    )
    assert generated, f"infer produced no output file in {workdir}"
    return pd.read_csv(generated[0])


@pytest.mark.determinism
def test_same_seed_runs_produce_identical_output(tmp_path, rp_logger):
    """Two independent train+infer runs of the same table at the same seed must produce
    identical generated data.

    The two subprocesses get different ``PYTHONHASHSEED`` values and independent RNG state,
    so they differ exactly the way two real runs do - which is what the unseeded fitting
    draws used to pick up.
    """
    rp_logger.info("Test two same-seed end-to-end runs produce identical output")
    source = tmp_path / "source.csv"
    _build_table().to_csv(source, index=False)

    frames = []
    for index, hash_seed in enumerate(("0", "12345"), start=1):
        run_dir = tmp_path / f"run{index}"
        run_dir.mkdir()
        frames.append(_run_once(run_dir, source, hash_seed))

    first, second = frames
    assert list(first.columns) == list(second.columns), "column set/order must match"
    assert len(first) == len(second) == INFER_SIZE
    pd.testing.assert_frame_equal(
        first, second, obj="generated output of two same-seed runs"
    )
    rp_logger.info(SUCCESSFUL_MESSAGE)
