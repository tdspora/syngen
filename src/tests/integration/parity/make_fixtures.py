"""Deterministically generate the parity fixture datasets and metadata.

Run from anywhere:

    python src/tests/integration/parity/make_fixtures.py

The generated CSVs and metadata YAMLs are committed under ``fixtures/`` so the
parity harness and the TF golden baselines are reproducible. Regenerate only
when the fixture matrix itself changes (and recapture baselines afterwards).

The fixtures intentionally cover the column families and key types that the
TF -> PyTorch migration must preserve, with ``numeric_wide.age`` (18-90) acting
as the canonical "distribution collapse" probe.
"""
import os

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")
SEED = 20240529
N_ROWS = 600


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


def _write_csv(df: pd.DataFrame, name: str) -> str:
    path = os.path.join(FIXTURES_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def _write_metadata(name: str, body: dict) -> None:
    with open(os.path.join(FIXTURES_DIR, f"{name}.yaml"), "w") as fp:
        yaml.safe_dump(body, fp, sort_keys=False)


def make_numeric_wide() -> None:
    rng = _rng()
    n = N_ROWS
    age = rng.integers(18, 91, size=n)                       # wide range 18-90
    income = np.round(rng.lognormal(mean=10.5, sigma=0.9, size=n), 2)  # skewed/outliers
    score = np.round(rng.normal(loc=0.0, scale=1.0, size=n), 4)        # ~normal
    near_constant = np.where(rng.random(n) < 0.02, 6, 5)               # near-constant
    zero_heavy = np.where(rng.random(n) < 0.45, 0.0,
                          np.round(rng.gamma(2.0, 50.0, size=n), 2))   # zero-heavy
    nullable_num = np.round(rng.normal(100, 15, size=n), 2)
    nullable_num[rng.random(n) < 0.35] = np.nan                        # null-heavy

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "age": age,
        "income": income,
        "score": score,
        "near_constant": near_constant,
        "zero_heavy": zero_heavy,
        "nullable_num": nullable_num,
    })
    _write_csv(df, "numeric_wide")
    _write_metadata("numeric_wide", {
        "numeric_wide": {
            "train_settings": {
                "source": "fixtures/numeric_wide.csv",
                "epochs": 5,
                "drop_null": False,
            },
            "infer_settings": {
                "size": n,
                "random_seed": 10,
                "run_parallel": False,
            },
            "keys": {
                "numeric_wide_pk": {"type": "PK", "columns": ["id"]},
            },
        }
    })


def make_categorical() -> None:
    rng = _rng()
    n = N_ROWS
    gender = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])
    country = rng.choice(
        ["US", "UK", "DE", "FR", "IN", "BR"], size=n,
        p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05],
    )
    city = rng.choice([f"city_{i:02d}" for i in range(40)], size=n)   # high card
    plan = rng.choice(["free", "pro", "enterprise"], size=n, p=[0.6, 0.3, 0.1]).astype(object)
    plan[rng.random(n) < 0.2] = np.nan                               # nullable categorical

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "gender": gender,
        "country": country,
        "city": city,
        "plan": plan,
    })
    _write_csv(df, "categorical")
    _write_metadata("categorical", {
        "categorical": {
            "train_settings": {
                "source": "fixtures/categorical.csv",
                "epochs": 5,
                "drop_null": False,
            },
            "infer_settings": {
                "size": n,
                "random_seed": 10,
                "run_parallel": False,
            },
            "keys": {
                "categorical_pk": {"type": "PK", "columns": ["id"]},
            },
        }
    })


def make_text_email() -> None:
    rng = _rng()
    n = N_ROWS
    first = rng.choice(["alex", "sam", "jordan", "casey", "lee", "morgan"], size=n)
    num = rng.integers(1, 9999, size=n)
    email = [f"{f}{k}@example.com" for f, k in zip(first, num)]
    short_code = [f"{c}{d:03d}" for c, d in zip(rng.choice(list("ABCDXYZ"), size=n), num % 1000)]
    bio_bits = [
        "loves data", "enjoys hiking and coffee", "writes synthetic generators",
        "plays guitar on weekends", "builds small robots", "reads science fiction",
    ]
    bio = [" ".join(rng.choice(bio_bits, size=rng.integers(1, 4))) for _ in range(n)]

    df = pd.DataFrame({
        "id": [f"USR-{i:05d}" for i in range(1, n + 1)],
        "email": email,
        "short_code": short_code,
        "bio": bio,
    })
    _write_csv(df, "text_email")
    _write_metadata("text_email", {
        "text_email": {
            "train_settings": {
                "source": "fixtures/text_email.csv",
                "epochs": 5,
                "drop_null": False,
            },
            "infer_settings": {
                "size": n,
                "random_seed": 10,
                "run_parallel": False,
            },
            "keys": {
                "text_email_pk": {
                    "type": "PK",
                    "columns": ["id"],
                    "regex_patterns": {"id": r"^USR-[0-9]{5}$"},
                },
            },
        }
    })


def make_datetime() -> None:
    rng = _rng()
    n = N_ROWS
    base = pd.Timestamp("2018-01-01")
    signup = base + pd.to_timedelta(rng.integers(0, 2000, size=n), unit="D")
    event = base + pd.to_timedelta(rng.integers(0, 2000 * 24 * 3600, size=n), unit="s")
    tz_event = (base.tz_localize("UTC")
                + pd.to_timedelta(rng.integers(0, 2000 * 3600, size=n), unit="s"))

    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "signup_date": signup.strftime("%Y-%m-%d"),
        "event_ts": event.strftime("%Y-%m-%d %H:%M:%S"),
        "tz_ts": tz_event.strftime("%Y-%m-%d %H:%M:%S%z"),
    })
    _write_csv(df, "datetime")
    _write_metadata("datetime", {
        "datetime": {
            "train_settings": {
                "source": "fixtures/datetime.csv",
                "epochs": 5,
                "drop_null": False,
            },
            "infer_settings": {
                "size": n,
                "random_seed": 10,
                "run_parallel": False,
            },
            "keys": {
                "datetime_pk": {"type": "PK", "columns": ["id"]},
            },
        }
    })


def make_keys_pair() -> None:
    """Two-table parent/child fixture exercising PK + FK + UQ."""
    rng = _rng()
    n_parent = 200
    n_child = N_ROWS

    parent = pd.DataFrame({
        "household_id": np.arange(1, n_parent + 1),
        "region": rng.choice(["north", "south", "east", "west"], size=n_parent),
        "median_income": np.round(rng.normal(60000, 15000, size=n_parent), 2),
    })
    _write_csv(parent, "keys_parent")

    child = pd.DataFrame({
        "member_id": np.arange(1, n_child + 1),                       # PK
        "household_id": rng.integers(1, n_parent + 1, size=n_child),  # FK -> parent
        "ssn": [f"UQ-{i:06d}" for i in range(1, n_child + 1)],        # UQ
        "age": rng.integers(0, 100, size=n_child),
    })
    _write_csv(child, "keys_child")

    _write_metadata("keys", {
        "keys_parent": {
            "train_settings": {
                "source": "fixtures/keys_parent.csv",
                "epochs": 5,
                "drop_null": False,
            },
            "infer_settings": {"size": n_parent, "random_seed": 10, "run_parallel": False},
            "keys": {
                "household_pk": {"type": "PK", "columns": ["household_id"]},
            },
        },
        "keys_child": {
            "train_settings": {
                "source": "fixtures/keys_child.csv",
                "epochs": 5,
                "drop_null": False,
            },
            "infer_settings": {"size": n_child, "random_seed": 10, "run_parallel": False},
            "keys": {
                "member_pk": {"type": "PK", "columns": ["member_id"]},
                "ssn_uq": {"type": "UQ", "columns": ["ssn"]},
                "household_fk": {
                    "type": "FK",
                    "columns": ["household_id"],
                    "references": {"table": "keys_parent", "columns": ["household_id"]},
                },
            },
        },
    })


def _uuids(rng: np.random.Generator, n: int) -> list:
    """Deterministic uuid4-shaped strings (8-4-4-4-12 hex)."""
    hexd = "0123456789abcdef"
    out = []
    for _ in range(n):
        chars = rng.choice(list(hexd), size=32)
        s = "".join(chars)
        out.append(f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}")
    return out


def make_mixed_complex() -> None:
    """One wide table mixing many dtypes, formats, and distributions."""
    rng = _rng()
    n = N_ROWS

    bimodal_mask = rng.random(n) < 0.5
    bimodal = np.where(bimodal_mask, rng.normal(20, 4, n), rng.normal(70, 5, n))

    zero_inflated = np.where(rng.random(n) < 0.5, 0.0, rng.gamma(2.0, 30.0, size=n))
    nullable_float = rng.normal(50, 10, size=n)
    nullable_float[rng.random(n) < 0.35] = np.nan

    base = pd.Timestamp("2019-06-01")
    created = base + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D")
    updated = base + pd.to_timedelta(rng.integers(0, 1500 * 24 * 3600, size=n), unit="s")

    names = rng.choice(["ann", "bob", "cara", "dan", "eve", "finn"], size=n)
    nums = rng.integers(1, 9999, size=n)
    note_bits = ["urgent", "follow up", "no issues", "pending review",
                 "escalated", "resolved quickly", "needs callback"]
    notes = [" ".join(rng.choice(note_bits, size=rng.integers(1, 4))) for _ in range(n)]

    df = pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "external_uuid": _uuids(rng, n),
        "uniform_int": rng.integers(0, 1001, size=n),
        "bimodal": np.round(bimodal, 3),
        "exponential": np.round(rng.exponential(scale=15.0, size=n), 3),
        "signed": np.round(rng.normal(0, 25, size=n), 3),          # negatives
        "poisson_count": rng.poisson(3.0, size=n),
        "heavy_tail": np.round(rng.pareto(2.0, size=n) * 100 + 10, 3),
        "zero_inflated": np.round(zero_inflated, 2),
        "nullable_float": np.round(nullable_float, 2),
        "is_active": rng.choice([True, False], size=n, p=[0.7, 0.3]),
        "status": rng.choice(["new", "active", "churned", "suspended"],
                             size=n, p=[0.5, 0.3, 0.15, 0.05]),
        "segment": rng.choice([f"seg_{i:02d}" for i in range(50)], size=n),
        "notes": notes,
        "contact_email": [f"{nm}{k}@corp.example.org" for nm, k in zip(names, nums)],
        "created_date": created.strftime("%d/%m/%Y"),               # day-first format
        "updated_at": updated.strftime("%Y-%m-%dT%H:%M:%S"),        # ISO 8601 with T
    })
    _write_csv(df, "mixed_complex")
    _write_metadata("mixed_complex", {
        "mixed_complex": {
            "train_settings": {
                "source": "fixtures/mixed_complex.csv",
                "epochs": 5,
                "drop_null": False,
            },
            "infer_settings": {"size": n, "random_seed": 10, "run_parallel": False},
            "keys": {
                "mixed_complex_pk": {"type": "PK", "columns": ["record_id"]},
            },
        }
    })


def make_relations_chain() -> None:
    """Four tables: regions <- stores <- sales, and sales -> products.

    Exercises a foreign-key chain plus a table (``sales``) with two FKs to
    different parents, with referential integrity in the source data.
    """
    rng = _rng()
    n_regions, n_products, n_stores, n_sales = 8, 30, 40, N_ROWS

    regions = pd.DataFrame({
        "region_id": np.arange(1, n_regions + 1),
        "region_name": [f"region_{i}" for i in range(1, n_regions + 1)],
    })
    _write_csv(regions, "regions")

    products = pd.DataFrame({
        "product_id": np.arange(1, n_products + 1),
        "category": rng.choice(["food", "tools", "apparel", "media"], size=n_products),
        "unit_price": np.round(rng.normal(40, 12, size=n_products), 2),
    })
    _write_csv(products, "products")

    stores = pd.DataFrame({
        "store_id": np.arange(1, n_stores + 1),
        "region_id": rng.integers(1, n_regions + 1, size=n_stores),        # FK -> regions
        "store_type": rng.choice(["flagship", "outlet", "kiosk"], size=n_stores),
        "sqft": np.round(rng.normal(5000, 1500, size=n_stores), 1),
    })
    _write_csv(stores, "stores")

    sales = pd.DataFrame({
        "sale_id": np.arange(1, n_sales + 1),
        "store_id": rng.integers(1, n_stores + 1, size=n_sales),           # FK -> stores
        "product_id": rng.integers(1, n_products + 1, size=n_sales),       # FK -> products
        "amount": np.round(rng.lognormal(3.0, 0.8, size=n_sales), 2),
        "qty": rng.poisson(2.0, size=n_sales) + 1,
        "sale_date": (pd.Timestamp("2021-01-01")
                      + pd.to_timedelta(rng.integers(0, 800, size=n_sales), unit="D")
                      ).strftime("%Y-%m-%d"),
    })
    _write_csv(sales, "sales")

    def _tbl(source, size, keys):
        return {
            "train_settings": {"source": source, "epochs": 5, "drop_null": False},
            "infer_settings": {"size": size, "random_seed": 10, "run_parallel": False},
            "keys": keys,
        }

    _write_metadata("relations_chain", {
        "regions": _tbl("fixtures/regions.csv", n_regions, {
            "region_pk": {"type": "PK", "columns": ["region_id"]},
        }),
        "products": _tbl("fixtures/products.csv", n_products, {
            "product_pk": {"type": "PK", "columns": ["product_id"]},
        }),
        "stores": _tbl("fixtures/stores.csv", n_stores, {
            "store_pk": {"type": "PK", "columns": ["store_id"]},
            "store_region_fk": {
                "type": "FK", "columns": ["region_id"],
                "references": {"table": "regions", "columns": ["region_id"]},
            },
        }),
        "sales": _tbl("fixtures/sales.csv", n_sales, {
            "sale_pk": {"type": "PK", "columns": ["sale_id"]},
            "sale_store_fk": {
                "type": "FK", "columns": ["store_id"],
                "references": {"table": "stores", "columns": ["store_id"]},
            },
            "sale_product_fk": {
                "type": "FK", "columns": ["product_id"],
                "references": {"table": "products", "columns": ["product_id"]},
            },
        }),
    })


def main() -> None:
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    make_numeric_wide()
    make_categorical()
    make_text_email()
    make_datetime()
    make_keys_pair()
    make_mixed_complex()
    make_relations_chain()
    print(f"Fixtures written to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
