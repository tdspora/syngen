# Dataset: `keys_relational_nulls`

**Parent/child PK + FK + UQ (with nulls/empties)**

## Overview (in plain words)

The same parent/child shape as `keys_relational_clean`, but **intentionally messy**: about 8% of child foreign keys are missing, ~10% of a numeric column is missing, and one unique-key value is null. It checks that relationships and key rules still hold when data has gaps - present foreign keys must still resolve to a real parent, and a unique column may contain at most one null. Pair it with the clean variant to confirm null handling does not break key or relationship logic.

- **Backend:** shared
- **Relations:** kr_parent (PK) <- kr_child (FK, nullable); kr_child UQ ssn (1 null allowed).
- **Key-focused tolerances:** True
- **Formats materialised:** csv, avro

## Explicit statistical patterns

region 4 cats; median_income Normal; age U{0..99}.

## Implicit patterns

referential integrity with ~8% null FK and one null UQ value.

## Null / empty policy

kr_child.household_id ~8% NaN; kr_child.age ~10% NaN; one NaN ssn.

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
