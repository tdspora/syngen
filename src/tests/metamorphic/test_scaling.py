"""Data-volume & composition relations (Part-A 3.3 subset, 16.2 / 9.2 single-vs-related).

(3.1 duplication and 3.2 volume scaling live in test_transformations; 16.1 additivity
is covered as 'both sub-populations present' by the bimodal mode-coverage check in
test_diversity.)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lib import mr_asserts as mr, stats, runner, specs        # noqa: E402


@pytest.mark.metamorphic
@pytest.mark.slow
def test_subset_consistency(mrun):
    """3.3: training on a 50% random subsample yields a similar (non-collapsed) output."""
    base = mrun("numeric_explicit", "csv")
    sub = base.originals["numeric_explicit"].sample(frac=0.5, random_state=2).reset_index(drop=True)
    sub["id"] = range(1, len(sub) + 1)
    res = mrun("numeric_explicit", "csv", tables_override={"numeric_explicit": sub})
    problems = [d for d in mr.distribution_equivalent(base, res, tol=mr.preservation_tol())
                if "row_count" not in d]
    assert not problems, "; ".join(problems)


@pytest.mark.metamorphic
@pytest.mark.slow
def test_single_vs_related(mrun):
    """16.2 / 9.2: a child table's own column distributions are preserved whether it is
    generated standalone or as part of its parent/child relationship."""
    spec = specs.all_datasets()["keys_relational_clean"]
    full = mrun("keys_relational_clean", "csv")                 # child within the relation
    child_df = full.originals["kr_child"]
    single = runner.single_table_spec(spec, "kr_child")         # FK dropped -> standalone
    res = mrun(single, "csv", tables_override={"kr_child": child_df})

    kinds = spec.tables["kr_child"].column_kinds
    p_full = stats.profile_table(full.generated["kr_child"], kinds)
    p_single = stats.profile_table(res.generated["kr_child"], kinds)
    disc = [d for d in stats.compare_profiles(p_full, p_single, mr.preservation_tol())
            if "row_count" not in d]
    assert not disc, "; ".join(disc)
