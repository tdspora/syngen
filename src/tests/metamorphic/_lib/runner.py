"""Format-parameterized train -> infer harness for the metamorphic suite.

Extends the idea of ``src/tests/integration/parity/runner.py`` but:

* drives the run from a :class:`~_lib.specs.DatasetSpec` (not a committed YAML),
* lets the **source/destination file format** vary (csv/tsv/psv/avro/xlsx/pkl/parquet),
* optionally substitutes **in-memory transformed tables** for the on-disk data
  (so transformation MRs — permutation, duplication, affine scaling — reuse the
  exact same code path).

Each run happens in an isolated temp working directory; ``model_artifacts/`` is
created there and removed afterwards. Everything goes through the public
``launch_train`` / ``launch_infer`` API, so the CLI/orchestration contract is
exercised too.

The ``syngen`` import is lazy so this module (and ``stats``/``specs``) stay
importable without the heavy backend installed.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import pandas as pd
import yaml

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import specs as specs_mod  # noqa: E402
from _lib.generate_datasets import (  # noqa: E402
    FORMAT_EXT, CSV_SEP, write_table, TEST_DATA_DIR,
)


# --------------------------------------------------------------------------- #
# Format-aware readers (mirror what syngen reads with)
# --------------------------------------------------------------------------- #
def read_table(path: str, fmt: str) -> pd.DataFrame:
    if fmt in CSV_SEP:
        return pd.read_csv(path, sep=CSV_SEP[fmt])
    if fmt == "avro":
        import pandavro as pdx
        return pdx.from_avro(path)
    if fmt == "xlsx":
        return pd.read_excel(path, engine="openpyxl")
    if fmt == "pkl":
        return pd.read_pickle(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format {fmt!r}")


def source_path(dataset: str, table: str, fmt: str, data_dir: str = TEST_DATA_DIR) -> str:
    return os.path.join(data_dir, dataset, f"{table}{FORMAT_EXT[fmt]}")


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    spec: "specs_mod.DatasetSpec"
    fmt: str
    originals: Dict[str, pd.DataFrame]
    generated: Dict[str, pd.DataFrame]
    losses: Optional[pd.DataFrame] = None        # captured training losses.csv (if any)
    work_dir: str = field(default="", repr=False)


# --------------------------------------------------------------------------- #
# Metadata assembly
# --------------------------------------------------------------------------- #
def build_metadata(spec: "specs_mod.DatasetSpec", fmt: str, *, src_dir: str,
                   out_dir: str, seed: int, epochs: Optional[int],
                   size_override: Optional[int], generator: Optional[str],
                   batch_size: Optional[int] = None) -> Dict:
    meta: Dict = {}
    for tname, t in spec.tables.items():
        train = {
            "source": os.path.join(src_dir, f"{tname}{FORMAT_EXT[fmt]}"),
            "epochs": epochs if epochs is not None else t.epochs,
            "drop_null": False,
        }
        if batch_size is not None:
            train["batch_size"] = batch_size
        if t.passthrough:
            train["column_types"] = {"passthrough": list(t.passthrough)}
        if generator is not None:                      # enterprise: vae/auto
            train["generator"] = generator
        infer = {
            "size": size_override if size_override is not None else t.size,
            "random_seed": seed,
            "run_parallel": False,
            "destination": os.path.join(out_dir, f"{tname}_generated{FORMAT_EXT[fmt]}"),
        }
        cfg: Dict = {"train_settings": train, "infer_settings": infer}
        keys = t.keys_metadata()
        if keys:
            cfg["keys"] = keys
        meta[tname] = cfg
    return meta


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run(spec: "specs_mod.DatasetSpec", fmt: str, *, seed: int = 10,
        epochs: Optional[int] = None, size_override: Optional[int] = None,
        tables_override: Optional[Dict[str, pd.DataFrame]] = None,
        generator: Optional[str] = None, batch_size: Optional[int] = None,
        launch_train: Optional[Callable] = None,
        launch_infer: Optional[Callable] = None) -> RunResult:
    """Train then infer ``spec`` in format ``fmt`` through the public API.

    ``tables_override`` (table -> DataFrame) writes those frames as the source
    instead of the committed test-data, enabling transformation MRs.
    ``launch_train``/``launch_infer`` default to the open-source ``syngen`` entry
    points; the EE copy passes ``syngen_ee`` ones.
    """
    if launch_train is None or launch_infer is None:
        from syngen.train import launch_train as _lt
        from syngen.infer import launch_infer as _li
        launch_train = launch_train or _lt
        launch_infer = launch_infer or _li

    work_dir = tempfile.mkdtemp(prefix=f"metamorph_{spec.name}_{fmt}_")
    src_dir = os.path.join(work_dir, "src")
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Materialise sources: either committed test-data or in-memory overrides.
    originals: Dict[str, pd.DataFrame] = {}
    for tname in spec.tables:
        if tables_override is not None and tname in tables_override:
            df = tables_override[tname]
        else:
            df = read_table(source_path(spec.name, tname, fmt), fmt)
        originals[tname] = df
        write_table(df, os.path.join(src_dir, f"{tname}{FORMAT_EXT[fmt]}"), fmt)

    meta = build_metadata(spec, fmt, src_dir=src_dir, out_dir=out_dir, seed=seed,
                          epochs=epochs, size_override=size_override, generator=generator,
                          batch_size=batch_size)
    meta_path = os.path.join(work_dir, "metadata.yaml")
    with open(meta_path, "w") as fp:
        yaml.safe_dump(meta, fp, sort_keys=False)

    prev_cwd = os.getcwd()
    try:
        os.chdir(work_dir)
        launch_train(metadata_path=meta_path, log_level="ERROR")
        launch_infer(metadata_path=meta_path, size=None, log_level="ERROR")
    finally:
        os.chdir(prev_cwd)

    generated = {
        tname: read_table(os.path.join(out_dir, f"{tname}_generated{FORMAT_EXT[fmt]}"), fmt)
        for tname in spec.tables
    }
    losses = _capture_losses(work_dir)
    return RunResult(spec=spec, fmt=fmt, originals=originals,
                     generated=generated, losses=losses, work_dir=work_dir)


def _capture_losses(work_dir: str) -> Optional[pd.DataFrame]:
    """Read any per-epoch ``*losses*.csv`` syngen wrote during training (for the
    training-process MRs). Returns a combined DataFrame or None."""
    import glob
    frames = []
    for path in glob.glob(os.path.join(work_dir, "**", "*losses*.csv"), recursive=True):
        try:
            frames.append(pd.read_csv(path))
        except Exception:                               # noqa: BLE001 - best effort
            pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def cleanup(result: RunResult) -> None:
    if result.work_dir and os.path.isdir(result.work_dir):
        shutil.rmtree(result.work_dir, ignore_errors=True)


def single_table_spec(spec: "specs_mod.DatasetSpec", table: str) -> "specs_mod.DatasetSpec":
    """A one-table DatasetSpec from one table of a multi-table spec, with FK keys
    dropped so it can train standalone (for single-vs-related MRs). Use it with
    ``tables_override`` so no committed file is needed."""
    t = spec.tables[table]
    keys = [k for k in t.keys if k.type not in ("FK",)]
    new_t = specs_mod.TableSpec(
        name=t.name, column_kinds=dict(t.column_kinds), keys=keys, size=t.size,
        epochs=t.epochs, uuid_columns=list(t.uuid_columns), pk_columns=list(t.pk_columns),
        passthrough=list(t.passthrough), email_domains=dict(t.email_domains))
    return specs_mod.DatasetSpec(
        name=f"{spec.name}__{table}_only", title=f"{table} (standalone)",
        builder=lambda rng, _t=table: {table: spec.builder(rng)[_t]},
        tables={table: new_t}, formats=["csv"], explicit_patterns="",
        implicit_patterns="", null_policy="", key_focused=spec.key_focused,
        backend=spec.backend)
