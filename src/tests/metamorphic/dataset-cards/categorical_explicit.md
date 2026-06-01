# Dataset: `categorical_explicit`

**Categorical/binary patterns with a conditional dependency**

## Overview (in plain words)

A single table of 5,000 rows centred on **category-style columns**: a binary `gender`, a true/false `is_active`, an imbalanced `country` (a few common values, several rare ones), a high-variety `city` (~40 values), an ordered `tier`, and a `plan`. The important structure is conditional rather than per-column: a record's `tier` depends on its `country` (some countries skew to higher tiers) and `plan` follows from `tier`. About 10% of `plan` values are intentionally missing and `id` is the primary key. Use it to confirm a generator keeps every category (drops none), reproduces realistic frequencies, and preserves the country->tier->plan dependency.

- **Backend:** shared
- **Relations:** none
- **Key-focused tolerances:** False
- **Formats materialised:** csv, avro, xlsx

## Explicit statistical patterns

gender ~ Bernoulli(0.55); is_active ~ Bernoulli(0.7); country ~ imbalanced (Zipf-ish, 6 cats); city ~ uniform over 40 (high cardinality); tier ordinal.

## Implicit patterns

tier is CONDITIONAL on country (e.g. IN/BR skew Low, US skews High); plan is a deterministic function of tier (then ~10% nulled).

## Null / empty policy

plan ~10% NaN.

## Tables, columns & keys

### `categorical_explicit`  (infer size 5000, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `gender` | binary | modelled |
| `is_active` | binary | modelled |
| `country` | categorical | modelled |
| `city` | categorical | modelled |
| `tier` | categorical | modelled |
| `plan` | categorical | modelled |
| `id` | — | PK |

**Keys:**
- `categorical_explicit_pk` **PK** ['id']
