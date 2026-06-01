# Dataset: `text_email_uuid`

**Text, email, long-text and UUID with a regex string PK**

## Overview (in plain words)

A single table of 5,000 rows for **free-text and identifier handling**. It contains an `email`, a structured `short_code` like `ABC-1234`, a short `bio`, a `long_note` longer than 200 characters (which routes to the statistical long-text path), and a `external_uuid`. The primary key `id` is a string that must always match the pattern `USR-000123`, and email names are drawn from a small set of first names. Use it to check that generated text stays believable in length and character set, emails keep a valid `name@domain` shape, UUIDs stay unique, and the regex-formatted key is regenerated correctly.

- **Backend:** shared
- **Relations:** none
- **Key-focused tolerances:** False
- **Formats materialised:** csv, avro

## Explicit statistical patterns

email = <name><1..9999>@example.com; short_code ~ ^[A-Z]{3}-\d{4}$; bio = 1-3 phrases; long_note > 200 chars (statistical long-text path); external_uuid ~ uuid4-shaped.

## Implicit patterns

email local-part derived from a small name vocabulary.

## Null / empty policy

No nulls.

## Tables, columns & keys

### `text_email_uuid`  (infer size 5000, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `email` | email | modelled |
| `short_code` | text | modelled |
| `bio` | text | modelled |
| `long_note` | text | modelled |
| `external_uuid` | uuid | modelled |
| `id` | — | PK |

**Keys:**
- `text_email_uuid_pk` **PK** ['id']  regex `^USR-[0-9]{6}$`
