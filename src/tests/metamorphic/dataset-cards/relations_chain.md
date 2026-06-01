# Dataset: `relations_chain`

**Four-table FK chain + a table with two FKs to different parents**

## Overview (in plain words)

A four-table dataset modelling a small **retail schema**: `regions` (50) -> `stores` (800) -> `sales` (5,000), where `sales` also references `products` (200). It exercises a chain of foreign keys plus a single table (`sales`) that has two foreign keys to two different parents, with full referential integrity in the source. Each table has its own primary key. Use it to verify multi-table generation order and that every foreign key across the chain - and both keys on `sales` - resolves to a real parent.

- **Backend:** shared
- **Relations:** regions <- stores <- sales; sales -> products (two FKs on sales).
- **Key-focused tolerances:** True
- **Formats materialised:** csv

## Explicit statistical patterns

category 4 cats; unit_price/sqft/amount Normal/lognormal; qty Poisson+1.

## Implicit patterns

full referential integrity across the chain and the two sales FKs.

## Null / empty policy

No nulls.

## Tables, columns & keys

### `regions`  (infer size 50, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `region_name` | categorical | modelled |
| `region_id` | — | PK |

**Keys:**
- `region_pk` **PK** ['region_id']

### `products`  (infer size 200, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `category` | categorical | modelled |
| `unit_price` | numeric | modelled |
| `product_id` | — | PK |

**Keys:**
- `product_pk` **PK** ['product_id']

### `stores`  (infer size 800, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `store_type` | categorical | modelled |
| `sqft` | numeric | modelled |
| `store_id` | — | PK |
| `region_id` | — | FK |

**Keys:**
- `store_pk` **PK** ['store_id']
- `store_region_fk` **FK** ['region_id'] -> regions['region_id']

### `sales`  (infer size 5000, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `amount` | numeric | modelled |
| `qty` | numeric | modelled |
| `sale_date` | datetime | modelled |
| `sale_id` | — | PK |
| `store_id` | — | FK |
| `product_id` | — | FK |

**Keys:**
- `sale_pk` **PK** ['sale_id']
- `sale_store_fk` **FK** ['store_id'] -> stores['store_id']
- `sale_product_fk` **FK** ['product_id'] -> products['product_id']

**Foreign keys checked:** stores.region_id ⊆ regions.region_id; sales.store_id ⊆ stores.store_id; sales.product_id ⊆ products.product_id
