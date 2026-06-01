# Dataset: `numeric_explicit`

**Numeric distributions (explicit) with hidden correlation & functional dependency**

## Overview (in plain words)

A single table of 5,000 synthetic person-like records used to stress-test how a generator handles **numbers of many different shapes at once**. In one place it mixes a wide, evenly-spread `age` (18-90), a heavily right-skewed `income`, a bell-shaped `score`, a two-humped `bimodal` column, and exponential, count (Poisson), signed (can be negative), near-constant, and zero-heavy columns. Two relationships are hidden in the data on purpose: `score` tends to rise with `age`, and `income` grows as a power-law function of `age`. There are no missing values and `id` is a unique primary key. In short: if a model flattens a distribution, narrows a range, or loses the age->income/score links, this dataset exposes it.

- **Backend:** shared
- **Relations:** none
- **Key-focused tolerances:** False
- **Formats materialised:** csv, tsv, psv, avro, xlsx, pkl

## Explicit statistical patterns

age ~ Uniform{18..90} (wide-range collapse probe); income ~ lognormal (heavy right tail); score ~ Normal; bimodal ~ 0.5*N(20,4)+0.5*N(70,5) (shape-collapse probe); exponential ~ Exp(15); poisson_count ~ Poisson(3); signed ~ N(0,25) (negatives); near_constant ~ 98% one value; zero_inflated ~ >=45% exact zeros.

## Implicit patterns

score is linearly correlated with age (rho ~ 0.6); income is a power-law FUNCTION of age (income = exp(8 + 0.025*age) * lognormal noise).

## Null / empty policy

No nulls.

## Tables, columns & keys

### `numeric_explicit`  (infer size 5000, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `age` | numeric | modelled |
| `income` | numeric | modelled |
| `score` | numeric | modelled |
| `bimodal` | numeric | modelled |
| `exponential` | numeric | modelled |
| `poisson_count` | numeric | modelled |
| `signed` | numeric | modelled |
| `near_constant` | categorical | modelled |
| `zero_inflated` | numeric | modelled |
| `id` | — | PK |

**Keys:**
- `numeric_explicit_pk` **PK** ['id']
