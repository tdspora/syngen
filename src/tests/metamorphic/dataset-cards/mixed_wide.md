# Dataset: `mixed_wide`

**Wide stress table: all dtypes + multi-format dates + nulls & empty strings**

## Overview (in plain words)

A deliberate **'kitchen-sink' table** of 5,000 rows that combines every data type the generator supports - several numeric shapes, a boolean, low- and high-variety categories, free-text notes, an email, multiple date formats, and a UUID - to exercise the whole pipeline together. It includes messy, real-world conditions: about 35% missing values in one numeric column and ~5% empty strings (which are different from missing) in the notes. Hidden structure: `heavy_tail` is derived from `uniform_int`, and `amount` depends on the row's `status`. `record_id` is the primary key. This is the broad catch-all regression probe.

- **Backend:** shared
- **Relations:** none
- **Key-focused tolerances:** False
- **Formats materialised:** csv, avro, xlsx, pkl

## Explicit statistical patterns

uniform_int ~ U{0..1000}; heavy_tail Pareto; amount Normal; nullable_float Normal (35% NaN); is_active boolean; status imbalanced cat; segment ~50 cats; notes free text; contact_email; created_date %d/%m/%Y; updated_at ISO-T; external_uuid uuid4-shaped.

## Implicit patterns

heavy_tail is a FUNCTION of uniform_int; amount is CONDITIONAL on status (per-status mean).

## Null / empty policy

nullable_float ~35% NaN; notes ~5% EMPTY STRING (distinct from NaN).

## Tables, columns & keys

### `mixed_wide`  (infer size 5000, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `uniform_int` | numeric | modelled |
| `heavy_tail` | numeric | modelled |
| `amount` | numeric | modelled |
| `nullable_float` | numeric | modelled |
| `is_active` | binary | modelled |
| `status` | categorical | modelled |
| `segment` | categorical | modelled |
| `notes` | text | modelled |
| `contact_email` | email | modelled |
| `created_date` | datetime | modelled |
| `updated_at` | datetime | modelled |
| `external_uuid` | uuid | modelled |
| `record_id` | — | PK |

**Keys:**
- `mixed_wide_pk` **PK** ['record_id']
