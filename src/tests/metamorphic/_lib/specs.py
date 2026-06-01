"""Dataset registry for the metamorphic test suite.

Each :class:`DatasetSpec` describes one logical dataset: how to *build* it
deterministically (``builder`` returns ``{table_name: DataFrame}``), the
per-column *kinds* used for statistical profiling, the *keys* (PK/UQ/FK), the
target *formats* to materialise it in, and human-readable notes about the
**explicit** patterns (known parametric distributions) and **implicit** patterns
(correlations / functional dependencies / conditionals / ordering) baked in.

This module is intentionally free of any ``syngen``/``tensorflow``/``torch``
import so it can be copied verbatim into the enterprise repo and so the generator
and the metamorphic tests share a single source of truth.

Conventions (mirrors ``src/tests/integration/parity``):
* ``column_kinds`` maps a modelled column -> one of
  ``numeric|categorical|binary|text|email|datetime|uuid``. Opaque key columns
  (PK/UQ/FK) are **excluded** because they are regenerated and validated via
  ``pk_columns``/``fks`` instead.
* Builders are seeded by the caller, so regenerating yields byte-identical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Default row count for "fact" tables (>= the user-required 1000; chosen 5000 for
# strong distribution signal). Parent/dimension tables are smaller by design.
N_ROWS = 5000


# --------------------------------------------------------------------------- #
# Spec dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class KeySpec:
    name: str
    type: str                                   # "PK" | "UQ" | "FK" | "TKN"
    columns: List[str]
    references: Optional[Dict] = None           # {"table": str, "columns": [str]}
    regex_patterns: Optional[Dict[str, str]] = None

    def to_metadata(self) -> Dict:
        body: Dict = {"type": self.type, "columns": list(self.columns)}
        if self.regex_patterns:
            body["regex_patterns"] = dict(self.regex_patterns)
        if self.references:
            body["references"] = {
                "table": self.references["table"],
                "columns": list(self.references["columns"]),
            }
        return body


@dataclass
class TableSpec:
    name: str
    column_kinds: Dict[str, str]                # modelled cols -> kind (excludes keys)
    keys: List[KeySpec] = field(default_factory=list)
    size: int = N_ROWS                          # infer size
    epochs: int = 5                             # train epochs (kept low for test speed)
    uuid_columns: List[str] = field(default_factory=list)
    pk_columns: List[str] = field(default_factory=list)   # for uniqueness checks
    passthrough: List[str] = field(default_factory=list)  # enterprise-only
    email_domains: Dict[str, str] = field(default_factory=dict)

    def keys_metadata(self) -> Dict:
        return {k.name: k.to_metadata() for k in self.keys}


@dataclass
class DatasetSpec:
    name: str
    title: str
    builder: Callable[[np.random.Generator], Dict[str, pd.DataFrame]]
    tables: Dict[str, TableSpec]
    formats: List[str]                          # subset of SUPPORTED_FORMATS
    explicit_patterns: str                      # doc text for the dataset card
    implicit_patterns: str                      # doc text for the dataset card
    null_policy: str                            # doc text
    relations: str = "none"                     # doc text
    key_focused: bool = False                   # -> catastrophic_collapse_only tolerances
    backend: str = "shared"                     # "shared" | "ee"
    # for fk_validity checks: {"child","parent","fk_col","parent_pk"}
    fks: List[Dict] = field(default_factory=list)
    notes: str = ""
    summary: str = ""           # plain-language overview rendered at the top of the card

    @property
    def table_order(self) -> List[str]:
        """Tables ordered parents-before-children (insertion order is authored so)."""
        return list(self.tables.keys())


# --------------------------------------------------------------------------- #
# Deterministic helpers
# --------------------------------------------------------------------------- #
def _uuids(rng: np.random.Generator, n: int) -> List[str]:
    """Deterministic uuid4-shaped strings (8-4-4-4-12 hex)."""
    hexd = list("0123456789abcdef")
    out = []
    for _ in range(n):
        s = "".join(rng.choice(hexd, size=32))
        out.append(f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}")
    return out


# --------------------------------------------------------------------------- #
# Builders (explicit + implicit patterns)
# --------------------------------------------------------------------------- #
def build_numeric_explicit(rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    n = N_ROWS
    age = rng.integers(18, 91, size=n)                                  # uniform 18-90
    # implicit: score correlates with age (rho ~ 0.6 via shared latent)
    z = (age - age.mean()) / age.std()
    score = np.round(0.6 * z + rng.normal(0, 0.8, size=n), 4)
    # implicit: income is a power-law function of age (functional dependency) + noise
    income = np.round(np.exp(8.0 + 0.025 * age) * rng.lognormal(0, 0.25, size=n), 2)
    bimodal = np.round(np.where(rng.random(n) < 0.5,
                                rng.normal(20, 4, n), rng.normal(70, 5, n)), 3)
    exponential = np.round(rng.exponential(scale=15.0, size=n), 3)
    poisson_count = rng.poisson(3.0, size=n)
    signed = np.round(rng.normal(0, 25, size=n), 3)                     # negatives
    near_constant = np.where(rng.random(n) < 0.02, 6, 5)
    zero_inflated = np.round(np.where(rng.random(n) < 0.45, 0.0,
                                      rng.gamma(2.0, 50.0, size=n)), 2)  # >=45% zeros
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "age": age,
        "income": income,
        "score": score,
        "bimodal": bimodal,
        "exponential": exponential,
        "poisson_count": poisson_count,
        "signed": signed,
        "near_constant": near_constant,
        "zero_inflated": zero_inflated,
    })
    return {"numeric_explicit": df}


def build_categorical_explicit(rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    n = N_ROWS
    gender = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])
    is_active = rng.choice([True, False], size=n, p=[0.7, 0.3])
    country = rng.choice(["US", "UK", "DE", "FR", "IN", "BR"], size=n,
                         p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])             # Zipf-ish
    city = rng.choice([f"city_{i:02d}" for i in range(40)], size=n)      # high card
    # implicit: tier distribution is conditional on country
    tier_p = {"US": [0.2, 0.3, 0.5], "UK": [0.3, 0.4, 0.3], "DE": [0.3, 0.4, 0.3],
              "FR": [0.5, 0.3, 0.2], "IN": [0.6, 0.3, 0.1], "BR": [0.6, 0.3, 0.1]}
    tier = np.array([rng.choice(["Low", "Mid", "High"], p=tier_p[c]) for c in country])
    # implicit: plan correlates with tier
    plan = np.where(tier == "High", "enterprise",
                    np.where(tier == "Mid", "pro", "free")).astype(object)
    plan[rng.random(n) < 0.10] = np.nan                                  # ~10% null
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "gender": gender, "is_active": is_active, "country": country,
        "city": city, "tier": tier, "plan": plan,
    })
    return {"categorical_explicit": df}


def build_text_email_uuid(rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    n = N_ROWS
    first = rng.choice(["alex", "sam", "jordan", "casey", "lee", "morgan"], size=n)
    num = rng.integers(1, 9999, size=n)
    email = [f"{f}{k}@example.com" for f, k in zip(first, num)]           # implicit: name->local part
    short_code = [f"{a}{b}{c}-{d:04d}"
                  for a, b, c, d in zip(rng.choice(list("ABCDXYZ"), n),
                                        rng.choice(list("ABCDXYZ"), n),
                                        rng.choice(list("ABCDXYZ"), n), num % 10000)]
    bio_bits = ["loves data", "enjoys hiking and coffee", "writes synthetic generators",
                "plays guitar on weekends", "builds small robots", "reads science fiction"]
    bio = [" ".join(rng.choice(bio_bits, size=rng.integers(1, 4))) for _ in range(n)]
    long_note = [(" ".join(rng.choice(bio_bits, size=rng.integers(6, 12))) + ".") * 2
                 for _ in range(n)]                                       # > 200 chars -> long text
    df = pd.DataFrame({
        "id": [f"USR-{i:06d}" for i in range(1, n + 1)],                  # PK string + regex
        "email": email,
        "short_code": short_code,
        "bio": bio,
        "long_note": long_note,
        "external_uuid": _uuids(rng, n),
    })
    return {"text_email_uuid": df}


def build_datetime_patterns(rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    n = N_ROWS
    base = pd.Timestamp("2018-01-01")
    signup = base + pd.to_timedelta(rng.integers(0, 2000, size=n), unit="D")
    # implicit: event_ts is always >= signup_date (ordering constraint)
    event = signup + pd.to_timedelta(rng.integers(0, 400 * 24 * 3600, size=n), unit="s")
    tz_event = (base.tz_localize("UTC")
                + pd.to_timedelta(rng.integers(0, 2000 * 3600, size=n), unit="s"))
    created = base + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D")
    updated = base + pd.to_timedelta(rng.integers(0, 1500 * 24 * 3600, size=n), unit="s")
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "signup_date": signup.strftime("%Y-%m-%d"),
        "event_ts": event.strftime("%Y-%m-%d %H:%M:%S"),
        "tz_ts": tz_event.strftime("%Y-%m-%d %H:%M:%S%z"),
        "created_eu": created.strftime("%d/%m/%Y"),                       # day-first
        "updated_iso": updated.strftime("%Y-%m-%dT%H:%M:%S"),             # ISO 8601 with T
    })
    return {"datetime_patterns": df}


def build_mixed_wide(rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    n = N_ROWS
    uniform_int = rng.integers(0, 1001, size=n)
    # implicit functional dependency: heavy_tail derived from uniform_int
    heavy_tail = np.round(rng.pareto(2.0, size=n) * (uniform_int + 10), 3)
    nullable_float = rng.normal(50, 10, size=n)
    nullable_float[rng.random(n) < 0.35] = np.nan                        # null-heavy
    status = rng.choice(["new", "active", "churned", "suspended"], size=n,
                        p=[0.5, 0.3, 0.15, 0.05])
    # implicit conditional: amount depends on status
    status_mu = {"new": 20, "active": 80, "churned": 5, "suspended": 10}
    amount = np.round(np.array([rng.normal(status_mu[s], 8) for s in status]).clip(0), 2)
    names = rng.choice(["ann", "bob", "cara", "dan", "eve", "finn"], size=n)
    nums = rng.integers(1, 9999, size=n)
    note_bits = ["urgent", "follow up", "no issues", "pending review",
                 "escalated", "resolved quickly", "needs callback"]
    notes = [" ".join(rng.choice(note_bits, size=rng.integers(1, 4))) for _ in range(n)]
    # empty-string injection (distinct from NaN) on a free-text column
    notes = [("" if rng.random() < 0.05 else t) for t in notes]
    base = pd.Timestamp("2019-06-01")
    created = base + pd.to_timedelta(rng.integers(0, 1500, size=n), unit="D")
    updated = base + pd.to_timedelta(rng.integers(0, 1500 * 24 * 3600, size=n), unit="s")
    df = pd.DataFrame({
        "record_id": np.arange(1, n + 1),
        "external_uuid": _uuids(rng, n),
        "uniform_int": uniform_int,
        "heavy_tail": heavy_tail,
        "amount": amount,
        "nullable_float": np.round(nullable_float, 2),
        "is_active": rng.choice([True, False], size=n, p=[0.7, 0.3]),
        "status": status,
        "segment": rng.choice([f"seg_{i:02d}" for i in range(50)], size=n),
        "notes": notes,
        "contact_email": [f"{nm}{k}@corp.example.org" for nm, k in zip(names, nums)],
        "created_date": created.strftime("%d/%m/%Y"),
        "updated_at": updated.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return {"mixed_wide": df}


def _keys_relational(rng: np.random.Generator, with_nulls: bool) -> Dict[str, pd.DataFrame]:
    n_parent, n_child = 1500, N_ROWS
    parent = pd.DataFrame({
        "household_id": np.arange(1, n_parent + 1),
        "region": rng.choice(["north", "south", "east", "west"], size=n_parent),
        "median_income": np.round(rng.normal(60000, 15000, size=n_parent), 2),
    })
    household_fk = rng.integers(1, n_parent + 1, size=n_child).astype(object)
    ssn = np.array([f"UQ-{i:06d}" for i in range(1, n_child + 1)], dtype=object)
    age = rng.integers(0, 100, size=n_child).astype(float)
    if with_nulls:
        household_fk[rng.random(n_child) < 0.08] = np.nan                # nullable FK
        age[rng.random(n_child) < 0.10] = np.nan                         # nullable numeric
        # one allowed null in the UQ column
        ssn[0] = np.nan
    child = pd.DataFrame({
        "member_id": np.arange(1, n_child + 1),
        "household_id": household_fk,
        "ssn": ssn,
        "age": age,
    })
    return {"kr_parent": parent, "kr_child": child}


def build_keys_relational_clean(rng):
    return _keys_relational(rng, with_nulls=False)


def build_keys_relational_nulls(rng):
    return _keys_relational(rng, with_nulls=True)


def build_relations_chain(rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    n_regions, n_products, n_stores, n_sales = 50, 200, 800, N_ROWS
    regions = pd.DataFrame({
        "region_id": np.arange(1, n_regions + 1),
        "region_name": [f"region_{i}" for i in range(1, n_regions + 1)],
    })
    products = pd.DataFrame({
        "product_id": np.arange(1, n_products + 1),
        "category": rng.choice(["food", "tools", "apparel", "media"], size=n_products),
        "unit_price": np.round(rng.normal(40, 12, size=n_products), 2),
    })
    stores = pd.DataFrame({
        "store_id": np.arange(1, n_stores + 1),
        "region_id": rng.integers(1, n_regions + 1, size=n_stores),       # FK -> regions
        "store_type": rng.choice(["flagship", "outlet", "kiosk"], size=n_stores),
        "sqft": np.round(rng.normal(5000, 1500, size=n_stores), 1),
    })
    sales = pd.DataFrame({
        "sale_id": np.arange(1, n_sales + 1),
        "store_id": rng.integers(1, n_stores + 1, size=n_sales),          # FK -> stores
        "product_id": rng.integers(1, n_products + 1, size=n_sales),      # FK -> products
        "amount": np.round(rng.lognormal(3.0, 0.8, size=n_sales), 2),
        "qty": rng.poisson(2.0, size=n_sales) + 1,
        "sale_date": (pd.Timestamp("2021-01-01")
                      + pd.to_timedelta(rng.integers(0, 800, size=n_sales), unit="D")
                      ).strftime("%Y-%m-%d"),
    })
    return {"regions": regions, "products": products, "stores": stores, "sales": sales}


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
SHARED_DATASETS: Dict[str, DatasetSpec] = {
    "numeric_explicit": DatasetSpec(
        name="numeric_explicit",
        title="Numeric distributions (explicit) with hidden correlation & functional dependency",
        builder=build_numeric_explicit,
        tables={"numeric_explicit": TableSpec(
            name="numeric_explicit",
            column_kinds={"age": "numeric", "income": "numeric", "score": "numeric",
                          "bimodal": "numeric", "exponential": "numeric",
                          "poisson_count": "numeric", "signed": "numeric",
                          "near_constant": "categorical", "zero_inflated": "numeric"},
            keys=[KeySpec("numeric_explicit_pk", "PK", ["id"])],
            pk_columns=["id"],
        )},
        formats=["csv", "tsv", "psv", "avro", "xlsx", "pkl"],
        explicit_patterns=(
            "age ~ Uniform{18..90} (wide-range collapse probe); income ~ lognormal "
            "(heavy right tail); score ~ Normal; bimodal ~ 0.5*N(20,4)+0.5*N(70,5) "
            "(shape-collapse probe); exponential ~ Exp(15); poisson_count ~ Poisson(3); "
            "signed ~ N(0,25) (negatives); near_constant ~ 98% one value; "
            "zero_inflated ~ >=45% exact zeros."),
        implicit_patterns=(
            "score is linearly correlated with age (rho ~ 0.6); income is a power-law "
            "FUNCTION of age (income = exp(8 + 0.025*age) * lognormal noise)."),
        null_policy="No nulls.",
    ),
    "categorical_explicit": DatasetSpec(
        name="categorical_explicit",
        title="Categorical/binary patterns with a conditional dependency",
        builder=build_categorical_explicit,
        tables={"categorical_explicit": TableSpec(
            name="categorical_explicit",
            column_kinds={"gender": "binary", "is_active": "binary", "country": "categorical",
                          "city": "categorical", "tier": "categorical", "plan": "categorical"},
            keys=[KeySpec("categorical_explicit_pk", "PK", ["id"])],
            pk_columns=["id"],
        )},
        formats=["csv", "avro", "xlsx"],
        explicit_patterns=(
            "gender ~ Bernoulli(0.55); is_active ~ Bernoulli(0.7); country ~ imbalanced "
            "(Zipf-ish, 6 cats); city ~ uniform over 40 (high cardinality); tier ordinal."),
        implicit_patterns=(
            "tier is CONDITIONAL on country (e.g. IN/BR skew Low, US skews High); "
            "plan is a deterministic function of tier (then ~10% nulled)."),
        null_policy="plan ~10% NaN.",
    ),
    "text_email_uuid": DatasetSpec(
        name="text_email_uuid",
        title="Text, email, long-text and UUID with a regex string PK",
        builder=build_text_email_uuid,
        tables={"text_email_uuid": TableSpec(
            name="text_email_uuid",
            column_kinds={"email": "email", "short_code": "text", "bio": "text",
                          "long_note": "text", "external_uuid": "uuid"},
            keys=[KeySpec("text_email_uuid_pk", "PK", ["id"],
                          regex_patterns={"id": r"^USR-[0-9]{6}$"})],
            pk_columns=["id"],
            uuid_columns=["external_uuid"],
            email_domains={"email": "tdspora.ai"},      # generated domain (feature default)
        )},
        formats=["csv", "avro"],
        explicit_patterns=(
            "email = <name><1..9999>@example.com; short_code ~ ^[A-Z]{3}-\\d{4}$; "
            "bio = 1-3 phrases; long_note > 200 chars (statistical long-text path); "
            "external_uuid ~ uuid4-shaped."),
        implicit_patterns="email local-part derived from a small name vocabulary.",
        null_policy="No nulls.",
    ),
    "datetime_patterns": DatasetSpec(
        name="datetime_patterns",
        title="Multi-format datetimes with a temporal ordering constraint",
        builder=build_datetime_patterns,
        tables={"datetime_patterns": TableSpec(
            name="datetime_patterns",
            column_kinds={"signup_date": "datetime", "event_ts": "datetime",
                          "tz_ts": "datetime", "created_eu": "datetime",
                          "updated_iso": "datetime"},
            keys=[KeySpec("datetime_patterns_pk", "PK", ["id"])],
            pk_columns=["id"],
        )},
        formats=["csv", "avro", "xlsx"],
        explicit_patterns=(
            "signup_date %Y-%m-%d; event_ts %Y-%m-%d %H:%M:%S; tz_ts tz-aware %z; "
            "created_eu %d/%m/%Y (day-first); updated_iso ISO-8601 'T'."),
        implicit_patterns="event_ts >= signup_date for every row (temporal ordering).",
        null_policy="No nulls.",
    ),
    "mixed_wide": DatasetSpec(
        name="mixed_wide",
        title="Wide stress table: all dtypes + multi-format dates + nulls & empty strings",
        builder=build_mixed_wide,
        tables={"mixed_wide": TableSpec(
            name="mixed_wide",
            column_kinds={"uniform_int": "numeric", "heavy_tail": "numeric",
                          "amount": "numeric", "nullable_float": "numeric",
                          "is_active": "binary", "status": "categorical",
                          "segment": "categorical", "notes": "text",
                          "contact_email": "email", "created_date": "datetime",
                          "updated_at": "datetime", "external_uuid": "uuid"},
            keys=[KeySpec("mixed_wide_pk", "PK", ["record_id"])],
            pk_columns=["record_id"],
            uuid_columns=["external_uuid"],
            email_domains={"contact_email": "tdspora.ai"},
        )},
        formats=["csv", "avro", "xlsx", "pkl"],
        explicit_patterns=(
            "uniform_int ~ U{0..1000}; heavy_tail Pareto; amount Normal; "
            "nullable_float Normal (35% NaN); is_active boolean; status imbalanced cat; "
            "segment ~50 cats; notes free text; contact_email; created_date %d/%m/%Y; "
            "updated_at ISO-T; external_uuid uuid4-shaped."),
        implicit_patterns=(
            "heavy_tail is a FUNCTION of uniform_int; amount is CONDITIONAL on status "
            "(per-status mean)."),
        null_policy="nullable_float ~35% NaN; notes ~5% EMPTY STRING (distinct from NaN).",
    ),
    "keys_relational_clean": DatasetSpec(
        name="keys_relational_clean",
        title="Parent/child PK + FK + UQ (no nulls)",
        builder=build_keys_relational_clean,
        tables={
            "kr_parent": TableSpec(
                name="kr_parent",
                column_kinds={"region": "categorical", "median_income": "numeric"},
                keys=[KeySpec("household_pk", "PK", ["household_id"])],
                size=1500, pk_columns=["household_id"]),
            "kr_child": TableSpec(
                name="kr_child",
                column_kinds={"age": "numeric"},
                keys=[KeySpec("member_pk", "PK", ["member_id"]),
                      KeySpec("ssn_uq", "UQ", ["ssn"], regex_patterns={"ssn": r"^UQ-[0-9]{6}$"}),
                      KeySpec("household_fk", "FK", ["household_id"],
                              references={"table": "kr_parent", "columns": ["household_id"]})],
                pk_columns=["member_id", "ssn"]),
        },
        formats=["csv", "avro"],
        explicit_patterns="region 4 cats; median_income Normal; age U{0..99}.",
        implicit_patterns="referential integrity: every child.household_id exists in parent.",
        null_policy="No nulls.",
        relations="kr_parent (PK household_id) <- kr_child (FK household_id); kr_child UQ ssn.",
        key_focused=True,
        fks=[{"child": "kr_child", "parent": "kr_parent",
              "fk_col": "household_id", "parent_pk": "household_id"}],
    ),
    "keys_relational_nulls": DatasetSpec(
        name="keys_relational_nulls",
        title="Parent/child PK + FK + UQ (with nulls/empties)",
        builder=build_keys_relational_nulls,
        tables={
            "kr_parent": TableSpec(
                name="kr_parent",
                column_kinds={"region": "categorical", "median_income": "numeric"},
                keys=[KeySpec("household_pk", "PK", ["household_id"])],
                size=1500, pk_columns=["household_id"]),
            "kr_child": TableSpec(
                name="kr_child",
                column_kinds={"age": "numeric"},
                keys=[KeySpec("member_pk", "PK", ["member_id"]),
                      KeySpec("ssn_uq", "UQ", ["ssn"], regex_patterns={"ssn": r"^UQ-[0-9]{6}$"}),
                      KeySpec("household_fk", "FK", ["household_id"],
                              references={"table": "kr_parent", "columns": ["household_id"]})],
                pk_columns=["member_id", "ssn"]),
        },
        formats=["csv", "avro"],
        explicit_patterns="region 4 cats; median_income Normal; age U{0..99}.",
        implicit_patterns="referential integrity with ~8% null FK and one null UQ value.",
        null_policy="kr_child.household_id ~8% NaN; kr_child.age ~10% NaN; one NaN ssn.",
        relations="kr_parent (PK) <- kr_child (FK, nullable); kr_child UQ ssn (1 null allowed).",
        key_focused=True,
        fks=[{"child": "kr_child", "parent": "kr_parent",
              "fk_col": "household_id", "parent_pk": "household_id"}],
    ),
    "relations_chain": DatasetSpec(
        name="relations_chain",
        title="Four-table FK chain + a table with two FKs to different parents",
        builder=build_relations_chain,
        tables={
            "regions": TableSpec("regions", {"region_name": "categorical"},
                                 keys=[KeySpec("region_pk", "PK", ["region_id"])],
                                 size=50, pk_columns=["region_id"]),
            "products": TableSpec("products", {"category": "categorical", "unit_price": "numeric"},
                                  keys=[KeySpec("product_pk", "PK", ["product_id"])],
                                  size=200, pk_columns=["product_id"]),
            "stores": TableSpec(
                "stores", {"store_type": "categorical", "sqft": "numeric"},
                keys=[KeySpec("store_pk", "PK", ["store_id"]),
                      KeySpec("store_region_fk", "FK", ["region_id"],
                              references={"table": "regions", "columns": ["region_id"]})],
                size=800, pk_columns=["store_id"]),
            "sales": TableSpec(
                "sales", {"amount": "numeric", "qty": "numeric", "sale_date": "datetime"},
                keys=[KeySpec("sale_pk", "PK", ["sale_id"]),
                      KeySpec("sale_store_fk", "FK", ["store_id"],
                              references={"table": "stores", "columns": ["store_id"]}),
                      KeySpec("sale_product_fk", "FK", ["product_id"],
                              references={"table": "products", "columns": ["product_id"]})],
                pk_columns=["sale_id"]),
        },
        formats=["csv"],
        explicit_patterns="category 4 cats; unit_price/sqft/amount Normal/lognormal; qty Poisson+1.",
        implicit_patterns="full referential integrity across the chain and the two sales FKs.",
        null_policy="No nulls.",
        relations="regions <- stores <- sales; sales -> products (two FKs on sales).",
        key_focused=True,
        fks=[{"child": "stores", "parent": "regions", "fk_col": "region_id", "parent_pk": "region_id"},
             {"child": "sales", "parent": "stores", "fk_col": "store_id", "parent_pk": "store_id"},
             {"child": "sales", "parent": "products", "fk_col": "product_id", "parent_pk": "product_id"}],
    ),
}


# Plain-language overviews (rendered at the top of each dataset card) so a data
# scientist, engineer, or non-specialist can grasp the dataset in a few sentences.
SUMMARIES: Dict[str, str] = {
    "numeric_explicit": (
        "A single table of 5,000 synthetic person-like records used to stress-test how a "
        "generator handles **numbers of many different shapes at once**. In one place it "
        "mixes a wide, evenly-spread `age` (18-90), a heavily right-skewed `income`, a "
        "bell-shaped `score`, a two-humped `bimodal` column, and exponential, count "
        "(Poisson), signed (can be negative), near-constant, and zero-heavy columns. Two "
        "relationships are hidden in the data on purpose: `score` tends to rise with `age`, "
        "and `income` grows as a power-law function of `age`. There are no missing values "
        "and `id` is a unique primary key. In short: if a model flattens a distribution, "
        "narrows a range, or loses the age->income/score links, this dataset exposes it."),
    "categorical_explicit": (
        "A single table of 5,000 rows centred on **category-style columns**: a binary "
        "`gender`, a true/false `is_active`, an imbalanced `country` (a few common values, "
        "several rare ones), a high-variety `city` (~40 values), an ordered `tier`, and a "
        "`plan`. The important structure is conditional rather than per-column: a record's "
        "`tier` depends on its `country` (some countries skew to higher tiers) and `plan` "
        "follows from `tier`. About 10% of `plan` values are intentionally missing and `id` "
        "is the primary key. Use it to confirm a generator keeps every category (drops none), "
        "reproduces realistic frequencies, and preserves the country->tier->plan dependency."),
    "text_email_uuid": (
        "A single table of 5,000 rows for **free-text and identifier handling**. It contains "
        "an `email`, a structured `short_code` like `ABC-1234`, a short `bio`, a `long_note` "
        "longer than 200 characters (which routes to the statistical long-text path), and a "
        "`external_uuid`. The primary key `id` is a string that must always match the pattern "
        "`USR-000123`, and email names are drawn from a small set of first names. Use it to "
        "check that generated text stays believable in length and character set, emails keep "
        "a valid `name@domain` shape, UUIDs stay unique, and the regex-formatted key is "
        "regenerated correctly."),
    "datetime_patterns": (
        "A single table of 5,000 rows that exercises **dates and times in five formats at "
        "once**: an ISO date (`2018-05-20`), a full timestamp, a timezone-aware timestamp, a "
        "day-first European date (`20/05/2018`), and an ISO-8601 'T' timestamp. A realistic "
        "rule is built in: every row's event timestamp is on or after its signup date. `id` "
        "is the primary key and there are no missing values. Use it to confirm each date "
        "format is detected and regenerated faithfully (note: day-first dates are genuinely "
        "ambiguous and only parse partially - by design, the same way on every backend)."),
    "mixed_wide": (
        "A deliberate **'kitchen-sink' table** of 5,000 rows that combines every data type "
        "the generator supports - several numeric shapes, a boolean, low- and high-variety "
        "categories, free-text notes, an email, multiple date formats, and a UUID - to "
        "exercise the whole pipeline together. It includes messy, real-world conditions: "
        "about 35% missing values in one numeric column and ~5% empty strings (which are "
        "different from missing) in the notes. Hidden structure: `heavy_tail` is derived from "
        "`uniform_int`, and `amount` depends on the row's `status`. `record_id` is the primary "
        "key. This is the broad catch-all regression probe."),
    "keys_relational_clean": (
        "A clean **two-table parent/child** dataset that demonstrates referential integrity "
        "with no missing data. `kr_parent` (1,500 households) has a primary key; `kr_child` "
        "(5,000 members) points to it through a foreign key, carries its own primary key, and "
        "a unique `ssn` formatted like `UQ-000123`. Every child references a real parent. Use "
        "it to verify that generated keys stay unique, that foreign keys always resolve to an "
        "existing parent, and that regex-formatted keys keep their shape."),
    "keys_relational_nulls": (
        "The same parent/child shape as `keys_relational_clean`, but **intentionally messy**: "
        "about 8% of child foreign keys are missing, ~10% of a numeric column is missing, and "
        "one unique-key value is null. It checks that relationships and key rules still hold "
        "when data has gaps - present foreign keys must still resolve to a real parent, and a "
        "unique column may contain at most one null. Pair it with the clean variant to confirm "
        "null handling does not break key or relationship logic."),
    "relations_chain": (
        "A four-table dataset modelling a small **retail schema**: `regions` (50) -> `stores` "
        "(800) -> `sales` (5,000), where `sales` also references `products` (200). It exercises "
        "a chain of foreign keys plus a single table (`sales`) that has two foreign keys to "
        "two different parents, with full referential integrity in the source. Each table has "
        "its own primary key. Use it to verify multi-table generation order and that every "
        "foreign key across the chain - and both keys on `sales` - resolves to a real parent."),
}
for _name, _spec in SHARED_DATASETS.items():
    _spec.summary = SUMMARIES.get(_name, "")


def all_datasets() -> Dict[str, DatasetSpec]:
    """Datasets available in this checkout (overridden in the EE copy to add EE-only)."""
    return dict(SHARED_DATASETS)
