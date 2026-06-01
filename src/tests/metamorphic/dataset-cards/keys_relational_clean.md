# Dataset: `keys_relational_clean`

**Parent/child PK + FK + UQ (no nulls)**

## Overview (in plain words)

A clean **two-table parent/child** dataset that demonstrates referential integrity with no missing data. `kr_parent` (1,500 households) has a primary key; `kr_child` (5,000 members) points to it through a foreign key, carries its own primary key, and a unique `ssn` formatted like `UQ-000123`. Every child references a real parent. Use it to verify that generated keys stay unique, that foreign keys always resolve to an existing parent, and that regex-formatted keys keep their shape.

- **Backend:** shared
- **Relations:** kr_parent (PK household_id) <- kr_child (FK household_id); kr_child UQ ssn.
- **Key-focused tolerances:** True
- **Formats materialised:** csv, avro

## Explicit statistical patterns

region 4 cats; median_income Normal; age U{0..99}.

## Implicit patterns

referential integrity: every child.household_id exists in parent.

## Null / empty policy

No nulls.

## Tables, columns & keys

### `kr_parent`  (infer size 1500, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `region` | categorical | modelled |
| `median_income` | numeric | modelled |
| `household_id` | — | PK |

**Keys:**
- `household_pk` **PK** ['household_id']

### `kr_child`  (infer size 5000, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `age` | numeric | modelled |
| `member_id` | — | PK |
| `ssn` | — | UQ |
| `household_id` | — | FK |

**Keys:**
- `member_pk` **PK** ['member_id']
- `ssn_uq` **UQ** ['ssn']  regex `^UQ-[0-9]{6}$`
- `household_fk` **FK** ['household_id'] -> kr_parent['household_id']

**Foreign keys checked:** kr_child.household_id ⊆ kr_parent.household_id
