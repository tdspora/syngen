# Dataset: `datetime_patterns`

**Multi-format datetimes with a temporal ordering constraint**

## Overview (in plain words)

A single table of 5,000 rows that exercises **dates and times in five formats at once**: an ISO date (`2018-05-20`), a full timestamp, a timezone-aware timestamp, a day-first European date (`20/05/2018`), and an ISO-8601 'T' timestamp. A realistic rule is built in: every row's event timestamp is on or after its signup date. `id` is the primary key and there are no missing values. Use it to confirm each date format is detected and regenerated faithfully (note: day-first dates are genuinely ambiguous and only parse partially - by design, the same way on every backend).

- **Backend:** shared
- **Relations:** none
- **Key-focused tolerances:** False
- **Formats materialised:** csv, avro, xlsx

## Explicit statistical patterns

signup_date %Y-%m-%d; event_ts %Y-%m-%d %H:%M:%S; tz_ts tz-aware %z; created_eu %d/%m/%Y (day-first); updated_iso ISO-8601 'T'.

## Implicit patterns

event_ts >= signup_date for every row (temporal ordering).

## Null / empty policy

No nulls.

## Tables, columns & keys

### `datetime_patterns`  (infer size 5000, epochs 5)

| column | profiled kind | role |
| --- | --- | --- |
| `signup_date` | datetime | modelled |
| `event_ts` | datetime | modelled |
| `tz_ts` | datetime | modelled |
| `created_eu` | datetime | modelled |
| `updated_iso` | datetime | modelled |
| `id` | — | PK |

**Keys:**
- `datetime_patterns_pk` **PK** ['id']
