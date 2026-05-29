"""Shared harness for running the real syngen train -> infer path on a fixture.

Both ``capture_baseline.py`` and ``test_parity.py`` use this so the baseline and
the candidate are produced through the *identical*, public code path
(``launch_train`` / ``launch_infer``), which also validates the CLI/orchestration
contract.

Each run happens in an isolated temporary working directory: ``model_artifacts/``
is created there and removed afterwards, and source/destination paths in the
metadata are rewritten to absolute paths so nothing leaks into the repo.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")
BASELINES_DIR = os.path.join(HERE, "baselines")


@dataclass
class FixtureSpec:
    """Describes one fixture: its metadata file, tables, and per-column kinds."""
    name: str                                   # baseline file stem
    metadata_file: str                          # yaml under fixtures/
    # table -> {column -> kind}; kinds: numeric/categorical/binary/text/email/datetime
    column_kinds: Dict[str, Dict[str, str]]
    email_domains: Dict[str, str] = field(default_factory=dict)
    pk_columns: Dict[str, List[str]] = field(default_factory=dict)   # table -> pk/uq cols
    uuid_columns: Dict[str, List[str]] = field(default_factory=dict)  # table -> uuid cols
    # each: {"child","parent","fk_col","parent_pk"}
    fks: List[Dict] = field(default_factory=list)
    # FK/key-focused fixtures use small tables to test linkage/uniqueness, not
    # per-column distribution parity -> catastrophic-collapse-only tolerances
    # (see Tolerances.catastrophic_collapse_only).
    key_focused: bool = False


# Central registry. Column kinds intentionally exclude opaque key columns that are
# regenerated (PK/UQ/FK) since those are validated via pk_columns / fk instead.
FIXTURES: Dict[str, FixtureSpec] = {
    "numeric_wide": FixtureSpec(
        name="numeric_wide",
        metadata_file="numeric_wide.yaml",
        column_kinds={
            "numeric_wide": {
                "age": "numeric",
                "income": "numeric",
                "score": "numeric",
                "near_constant": "categorical",
                "zero_heavy": "numeric",
                "nullable_num": "numeric",
            }
        },
        pk_columns={"numeric_wide": ["id"]},
    ),
    "categorical": FixtureSpec(
        name="categorical",
        metadata_file="categorical.yaml",
        column_kinds={
            "categorical": {
                "gender": "binary",
                "country": "categorical",
                "city": "categorical",
                "plan": "categorical",
            }
        },
        pk_columns={"categorical": ["id"]},
    ),
    "text_email": FixtureSpec(
        name="text_email",
        metadata_file="text_email.yaml",
        column_kinds={
            "text_email": {
                "email": "email",
                "short_code": "text",
                "bio": "text",
            }
        },
        # EmailFeature regenerates name@<its default domain> (dataset.py:1344 passes
        # no domain), so the *generated* domain is the feature default, not the
        # source's example.com.
        email_domains={"email": "tdspora.ai"},
        pk_columns={"text_email": ["id"]},
    ),
    "datetime": FixtureSpec(
        name="datetime",
        metadata_file="datetime.yaml",
        column_kinds={
            "datetime": {
                "signup_date": "datetime",
                "event_ts": "datetime",
                "tz_ts": "datetime",
            }
        },
        pk_columns={"datetime": ["id"]},
    ),
    "keys": FixtureSpec(
        name="keys",
        metadata_file="keys.yaml",
        column_kinds={
            "keys_parent": {"region": "categorical", "median_income": "numeric"},
            "keys_child": {"age": "numeric"},
        },
        pk_columns={"keys_parent": ["household_id"],
                    "keys_child": ["member_id", "ssn"]},
        fks=[{"child": "keys_child", "parent": "keys_parent",
              "fk_col": "household_id", "parent_pk": "household_id"}],
        key_focused=True,
    ),
    "mixed_complex": FixtureSpec(
        name="mixed_complex",
        metadata_file="mixed_complex.yaml",
        column_kinds={
            "mixed_complex": {
                "uniform_int": "numeric",      # uniform distribution
                "bimodal": "numeric",          # two-mode mixture (collapse probe)
                "exponential": "numeric",      # exponential distribution
                "signed": "numeric",           # negative values (is_positive=False)
                "poisson_count": "numeric",    # integer counts
                "heavy_tail": "numeric",       # pareto-ish -> QuantileTransformer
                "zero_inflated": "numeric",    # zero-heavy
                "nullable_float": "numeric",   # null-heavy
                "is_active": "binary",         # boolean
                "status": "categorical",       # imbalanced categories
                "segment": "categorical",      # high-cardinality (~50)
                "notes": "text",               # short free text
                "contact_email": "email",
                "created_date": "datetime",    # %d/%m/%Y
                "updated_at": "datetime",      # ISO 8601 with T
            }
        },
        email_domains={"contact_email": "tdspora.ai"},  # generated domain (feature default)
        pk_columns={"mixed_complex": ["record_id"]},
        uuid_columns={"mixed_complex": ["external_uuid"]},
    ),
    "relations_chain": FixtureSpec(
        name="relations_chain",
        metadata_file="relations_chain.yaml",
        column_kinds={
            "products": {"category": "categorical", "unit_price": "numeric"},
            "regions": {"region_name": "categorical"},
            "stores": {"store_type": "categorical", "sqft": "numeric"},
            "sales": {"amount": "numeric", "qty": "numeric", "sale_date": "datetime"},
        },
        pk_columns={
            "products": ["product_id"],
            "regions": ["region_id"],
            "stores": ["store_id"],
            "sales": ["sale_id"],
        },
        # region <- store <- sale  (chain) and sale -> product (second FK on sales)
        fks=[
            {"child": "stores", "parent": "regions",
             "fk_col": "region_id", "parent_pk": "region_id"},
            {"child": "sales", "parent": "stores",
             "fk_col": "store_id", "parent_pk": "store_id"},
            {"child": "sales", "parent": "products",
             "fk_col": "product_id", "parent_pk": "product_id"},
        ],
        key_focused=True,
    ),
}


def _rewrite_metadata(metadata: Dict, out_dir: str) -> Dict:
    """Make source paths absolute and pin destinations into ``out_dir``."""
    for table, cfg in metadata.items():
        train = cfg.setdefault("train_settings", {})
        src = train.get("source")
        if src:
            train["source"] = os.path.join(FIXTURES_DIR, os.path.basename(src))
        infer = cfg.setdefault("infer_settings", {})
        infer["destination"] = os.path.join(out_dir, f"{table}_generated.csv")
    return metadata


@dataclass
class RunResult:
    spec: FixtureSpec
    originals: Dict[str, pd.DataFrame]    # table -> source dataframe
    generated: Dict[str, pd.DataFrame]   # table -> generated dataframe


def run_fixture(spec: FixtureSpec, size: Optional[int] = None,
                random_seed: int = 10, epochs: Optional[int] = None) -> RunResult:
    """Train then infer the fixture through the public API in an isolated dir."""
    # Imported lazily so this module (and stats.py) stay importable without the
    # heavy syngen/TF/PyTorch backend installed.
    from syngen.train import launch_train
    from syngen.infer import launch_infer

    with open(os.path.join(FIXTURES_DIR, spec.metadata_file)) as fp:
        metadata = yaml.safe_load(fp)

    work_dir = tempfile.mkdtemp(prefix=f"parity_{spec.name}_")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    metadata = _rewrite_metadata(metadata, out_dir)
    if epochs is not None:
        for cfg in metadata.values():
            cfg["train_settings"]["epochs"] = epochs

    meta_path = os.path.join(work_dir, "metadata.yaml")
    with open(meta_path, "w") as fp:
        yaml.safe_dump(metadata, fp, sort_keys=False)

    originals = {
        table: pd.read_csv(cfg["train_settings"]["source"])
        for table, cfg in metadata.items()
    }

    prev_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        launch_train(metadata_path=meta_path, log_level="ERROR")
        infer_size = size if size is not None else None
        launch_infer(metadata_path=meta_path, size=infer_size,
                     random_seed=random_seed, log_level="ERROR")
    finally:
        os.chdir(prev_cwd)

    generated = {
        table: pd.read_csv(os.path.join(out_dir, f"{table}_generated.csv"))
        for table in metadata
    }
    result = RunResult(spec=spec, originals=originals, generated=generated)
    shutil.rmtree(work_dir, ignore_errors=True)
    return result
