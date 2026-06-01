"""Pytest fixtures and markers for the metamorphic test suite.

The suite is opt-in:

    pytest src/tests/metamorphic -m metamorphic            # everything
    pytest src/tests/metamorphic -m "metamorphic and not slow"   # fast subset (loader IO)

``mrun`` runs a real ``launch_train`` -> ``launch_infer`` round-trip and memoises
the result for the session (the generated data is held in memory, so the temp
working dir is deleted immediately to keep disk usage flat).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import specs as specs_mod      # noqa: E402
from _lib import runner as runner_mod    # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "metamorphic: synthetic-data metamorphic relations via real train->infer round-trips",
    )
    config.addinivalue_line("markers", "slow: long-running tests (opt-in)")


@pytest.fixture(scope="session", autouse=True)
def ensure_data():
    """Generate the committed test-data on first use if it is missing."""
    manifest = os.path.join(os.path.dirname(__file__), "test-data", "manifest.json")
    if not os.path.exists(manifest):
        from _lib import generate_datasets
        generate_datasets.main()


@pytest.fixture(scope="session")
def datasets():
    return specs_mod.all_datasets()


@pytest.fixture(scope="session")
def mrun():
    """Memoised train->infer runner: ``mrun(dataset, fmt=..., seed=..., epochs=...)``."""
    cache = {}

    def _run(dataset, fmt="csv", *, seed=10, epochs=2, size_override=None,
             tables_override=None, generator=None, batch_size=None):
        # `dataset` may be a registry name or an ad-hoc DatasetSpec (e.g. single-table)
        spec = dataset if isinstance(dataset, specs_mod.DatasetSpec) \
            else specs_mod.all_datasets()[dataset]
        key = (spec.name, fmt, seed, epochs, size_override, generator, batch_size)
        if tables_override is None and key in cache:
            return cache[key]
        res = runner_mod.run(spec, fmt, seed=seed, epochs=epochs,
                             size_override=size_override, tables_override=tables_override,
                             generator=generator, batch_size=batch_size)
        runner_mod.cleanup(res)            # data is in memory; free the temp dir now
        if tables_override is None:
            cache[key] = res
        return res

    return _run
