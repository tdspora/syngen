"""Deterministically generate the metamorphic test datasets in every supported
file format, plus a manifest and one markdown "card" per dataset.

Run from the repo root (or anywhere):

    python -m tests.metamorphic._lib.generate_datasets
    # or:  python src/tests/metamorphic/_lib/generate_datasets.py

Outputs (committed, regenerable byte-for-byte):

    src/tests/metamorphic/test-data/<dataset>/<table>.<ext>   # data, one per (table, format)
    src/tests/metamorphic/test-data/manifest.json             # what was produced
    src/tests/metamorphic/dataset-cards/<dataset>.md          # human-readable card

Notes
-----
* Files are written with the *same* libraries syngen reads with (pandas / pandavro
  / openpyxl / pickle) so every generated file is guaranteed loadable by the
  ``DataLoader`` round-trip.
* The Fernet ``.dat`` format is **not** materialised here (to avoid committing an
  encryption key); the encryption metamorphic relation encrypts a committed CSV at
  test time with an ephemeral key — see ``mr_asserts.encrypt_dat``.
* Each format writer is guarded: a format that cannot represent a particular table
  (e.g. Avro nullable edge cases) is recorded as skipped in the manifest rather
  than aborting the whole generation.
"""
from __future__ import annotations

import json
import os
import pickle as pkl
from typing import Dict

import numpy as np
import pandas as pd

import sys
# Ensure the metamorphic dir (parent of _lib) is importable as the package root,
# matching the parity convention (test dir on sys.path). Works under pytest and
# when this file is run directly or via ``-m``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _lib import specs as specs_mod  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))            # .../metamorphic/_lib
METAMORPHIC_DIR = os.path.dirname(HERE)                      # .../metamorphic
TEST_DATA_DIR = os.path.join(METAMORPHIC_DIR, "test-data")
CARDS_DIR = os.path.join(METAMORPHIC_DIR, "dataset-cards")

SEED = 20240529

# extension per logical format
FORMAT_EXT = {"csv": ".csv", "tsv": ".tsv", "psv": ".psv", "avro": ".avro",
              "xlsx": ".xlsx", "pkl": ".pkl", "parquet": ".parquet"}
CSV_SEP = {"csv": ",", "tsv": "\t", "psv": "|"}


# --------------------------------------------------------------------------- #
# Format writers (each returns None on success or an error string on failure)
# --------------------------------------------------------------------------- #
def _write_csv_like(df: pd.DataFrame, path: str, sep: str):
    df.to_csv(path, sep=sep, index=False)


def _avro_base_type(dtype) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "long"
    if pd.api.types.is_float_dtype(dtype):
        return "double"
    return "string"


def _write_avro(df: pd.DataFrame, path: str):
    """Write Avro with an explicit nullable-union schema so object/null columns
    (which break pandavro's inference) are handled exactly as syngen reads them."""
    import pandavro as pdx

    out = df.copy()
    fields = []
    for col in out.columns:
        base = _avro_base_type(out[col].dtype)
        has_null = bool(out[col].isna().any())
        if base == "string":
            # stringify non-null values; NaN -> Python None (avro null branch)
            out[col] = out[col].map(lambda v: None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
            has_null = out[col].isna().any() or out[col].map(lambda v: v is None).any()
        fields.append({"name": col, "type": (["null", base] if has_null else base)})
    schema = {"type": "record", "name": "Root", "fields": fields}
    pdx.to_avro(path, out, schema=schema)


def _write_xlsx(df: pd.DataFrame, path: str):
    df.to_excel(path, index=False, engine="openpyxl")


def _write_pkl(df: pd.DataFrame, path: str):
    with open(path, "wb") as f:
        pkl.dump(df, f, protocol=pkl.HIGHEST_PROTOCOL)


def _write_parquet(df: pd.DataFrame, path: str):
    df.to_parquet(path, index=False)            # enterprise-only (pyarrow)


def write_table(df: pd.DataFrame, path: str, fmt: str):
    if fmt in CSV_SEP:
        _write_csv_like(df, path, CSV_SEP[fmt])
    elif fmt == "avro":
        _write_avro(df, path)
    elif fmt == "xlsx":
        _write_xlsx(df, path)
    elif fmt == "pkl":
        _write_pkl(df, path)
    elif fmt == "parquet":
        _write_parquet(df, path)
    else:
        raise ValueError(f"Unsupported format {fmt!r}")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_dataset(spec: specs_mod.DatasetSpec) -> Dict:
    """Build a dataset and write every (table, format). Returns a manifest entry."""
    rng = np.random.default_rng(SEED)
    tables = spec.builder(rng)
    ds_dir = os.path.join(TEST_DATA_DIR, spec.name)
    os.makedirs(ds_dir, exist_ok=True)

    entry: Dict = {"title": spec.title, "backend": spec.backend,
                   "key_focused": spec.key_focused, "tables": {}}
    for table_name, df in tables.items():
        produced, skipped = {}, {}
        for fmt in spec.formats:
            path = os.path.join(ds_dir, f"{table_name}{FORMAT_EXT[fmt]}")
            try:
                write_table(df, path, fmt)
                produced[fmt] = os.path.relpath(path, METAMORPHIC_DIR)
            except Exception as exc:                                  # noqa: BLE001
                skipped[fmt] = f"{type(exc).__name__}: {exc}"
                if os.path.exists(path):
                    os.remove(path)
        entry["tables"][table_name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "produced": produced,
            "skipped": skipped,
        }
    return entry


def card_markdown(spec: specs_mod.DatasetSpec) -> str:
    """Human-readable dataset card."""
    lines = [f"# Dataset: `{spec.name}`", "", f"**{spec.title}**", ""]
    if spec.summary:
        # plain-language overview first, so any reader gets the gist immediately
        lines += ["## Overview (in plain words)", "", spec.summary.strip(), ""]
    lines += [f"- **Backend:** {spec.backend}", f"- **Relations:** {spec.relations}",
              f"- **Key-focused tolerances:** {spec.key_focused}",
              f"- **Formats materialised:** {', '.join(spec.formats)}", ""]
    lines += ["## Explicit statistical patterns", "", spec.explicit_patterns, "",
              "## Implicit patterns", "", spec.implicit_patterns, "",
              "## Null / empty policy", "", spec.null_policy, "",
              "## Tables, columns & keys", ""]
    for tname, t in spec.tables.items():
        lines.append(f"### `{tname}`  (infer size {t.size}, epochs {t.epochs})")
        lines.append("")
        lines.append("| column | profiled kind | role |")
        lines.append("| --- | --- | --- |")
        key_cols = {c: k.type for k in t.keys for c in k.columns}
        all_cols = list(dict.fromkeys(list(t.column_kinds) + list(key_cols)
                                      + list(t.passthrough)))
        for col in all_cols:
            kind = t.column_kinds.get(col, "—")
            role = key_cols.get(col, "passthrough" if col in t.passthrough else "modelled")
            lines.append(f"| `{col}` | {kind} | {role} |")
        lines.append("")
        if t.keys:
            lines.append("**Keys:**")
            for k in t.keys:
                ref = f" -> {k.references['table']}{k.references['columns']}" if k.references else ""
                rgx = f"  regex `{list(k.regex_patterns.values())[0]}`" if k.regex_patterns else ""
                lines.append(f"- `{k.name}` **{k.type}** {k.columns}{ref}{rgx}")
            lines.append("")
    if spec.fks:
        lines.append("**Foreign keys checked:** "
                     + "; ".join(f"{f['child']}.{f['fk_col']} ⊆ "
                                 f"{f['parent']}.{f['parent_pk']}" for f in spec.fks))
        lines.append("")
    if spec.notes:
        lines += ["## Notes", "", spec.notes, ""]
    return "\n".join(lines)


def main() -> None:
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(CARDS_DIR, exist_ok=True)
    datasets = specs_mod.all_datasets()
    manifest = {"seed": SEED, "n_rows_default": specs_mod.N_ROWS, "datasets": {}}
    for name, spec in datasets.items():
        manifest["datasets"][name] = generate_dataset(spec)
        with open(os.path.join(CARDS_DIR, f"{name}.md"), "w") as fp:
            fp.write(card_markdown(spec))
    with open(os.path.join(TEST_DATA_DIR, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2, sort_keys=False)

    n_files = sum(len(t["produced"]) for d in manifest["datasets"].values()
                  for t in d["tables"].values())
    n_skip = sum(len(t["skipped"]) for d in manifest["datasets"].values()
                 for t in d["tables"].values())
    print(f"Generated {n_files} data files for {len(datasets)} datasets "
          f"into {TEST_DATA_DIR} ({n_skip} format/table combos skipped).")
    print(f"Dataset cards written to {CARDS_DIR}")


if __name__ == "__main__":
    main()
