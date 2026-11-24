# Multi-table datasets

One schema file can describe several related tables. The generator fills them in dependency order, parents
first. Child rows take their keys from rows that really exist in the parent table, and every job produces one
file per table. Single-table schema files work exactly as before.

A complete example is in `samples\schemas\procurement_multi.json`. It has five tables: suppliers, parts,
purchase orders, order lines and goods receipts (GRN).

## Running it

Run it the same way as any other schema:

```bat
rem API: upload the file, leave "records" empty (sizes are set per table)
curl -F "file=@samples\schemas\procurement_multi.json" -F mode=mock -F compress=zip http://127.0.0.1:8000/jobs
curl "http://127.0.0.1:8000/jobs/JOB_ID/preview?table=grn&rows=5"

rem CLI
python -m app.main generate --config samples\schemas\procurement_multi.json --provider mock
```

* **Output files.** Each table becomes `<table>.<csv|json|jsonl|xlsx>`. `compress=zip` packs all tables into
  one `<name>_tables.zip`, and `compress=gzip` compresses each file separately.
* **Listing and preview.** `GET /jobs/{id}/files` lists every table. `GET /jobs/{id}/preview?table=<name>`
  shows one table. In MCP, use `preview_job_output(job_id, table)`.
* **Report.** `multi_table_summary.json` gives, for each table, its row count and quality score. It also
  reports the checks described under "Checks done on every run".
* **Stop and resume.** These work as usual. Tables that already finished are kept, and a resume continues
  with the first unfinished table.

## File layout

```json
{
  "generation_instructions": "Indian automotive parts procurement.",
  "schema": {
    "tables": {
      "suppliers": {"records": 180, "primary_key": ["supplier_id"], "schema": { "...fields..." }},
      "purchase_orders": {"records": 20000, "links": {"supplier": {"table": "suppliers"}}, "schema": { "..." }},
      "po_lines": {"per_parent": {"table": "purchase_orders", "min": 1, "max": 5}, "schema": { "..." }}
    }
  }
}
```

Each table takes these settings:

| Setting | Meaning |
|---|---|
| `records` | Number of rows. A table needs either `records` or `per_parent`. |
| `per_parent` | `{"table", "min", "max", "weights"?, "where"?}`. Each parent row gets between `min` and `max` children. `weights` gives one weight per count from `min` to `max`. `where` limits which parents get children. |
| `links` | Named links to parent tables (see the next section). |
| `schema` | The fields, in the same format as a single-table schema (shorthand or `{"fields": ..., "rules": ...}`). |
| `primary_key` | Columns that together must be unique, for example `["po_no", "line_no"]`. The run checks this. |
| `instructions` | Extra LLM instructions for this table only. |

## Links, foreign keys and inherited columns

```json
"links": {
  "supplier": {"table": "suppliers"},
  "line":     {"table": "po_lines", "mode": "unique", "where": {"po_status": "closed"}},
  "po":       {"table": "purchase_orders", "match": {"po_type": "po_type"}}
}
```

A field reads a column of the linked parent row with `{"link": "<link>", "column": "<parent column>"}`. Its
type is copied from the parent column. Every column you read through one link comes from the same parent row,
for the same child row. This is what keeps composite keys aligned:

```json
"po_no":       {"link": "line", "column": "po_no"},
"line_no":     {"link": "line", "column": "line_no"},
"ordered_qty": {"link": "line", "column": "ordered_qty"}
```

Inherited fields such as a supplier name, a part name or an ordered quantity use the same syntax.

| Link setting | Meaning |
|---|---|
| `mode: "random"` (default) | Any parent row; a parent can be used many times. |
| `mode: "cover"` | Every parent row is used at least once (needs at least as many children as parents). |
| `mode: "unique"` | Each parent row is used at most once, for 1:1 or 0..1 relations such as one GRN per order line. |
| `where` | Only parent rows whose columns have these values, e.g. `{"status": ["open", "closed"]}`. |
| `match` | `{child field: parent column}`. Only parent rows whose column equals the child's own field. For example, an ASN with `po_type = legacy` links only to legacy POs. Works with mode `random`. |

`where` and `match` can use columns that are hidden in the parent's output (see `hidden` below).

In a `per_parent` table, the link to the parent is called `parent`:

```json
"po_no":   {"link": "parent", "column": "po_no"},
"line_no": {"type": "integer", "generator": "child_index"}
```

`child_index` numbers the children of each parent 1, 2, 3, and so on. To start at a different number, set
`min`.

## Planted values: exact counts, status-driven columns, formulas

These options also work in single-table schemas.

| Option | Example | Effect |
|---|---|---|
| `counts` | `"allowed_values": ["split", "full"], "counts": [45858, null]` | Exactly 45,858 rows get `split`. `null` means the rest. |
| `exact` | `"weights": [50, 40, 10], "exact": true` | The proportions are exact, not sampled. |
| `cases_by` + `cases` | `"cases_by": "status", "cases": {"open": null, "late": {"min": 3, "max": 30}}` | The value of another field decides how this field is generated. `null` leaves the field empty. A case can set any field option (`value`, `min`/`max`, `allowed_values` + `weights`, `formula`, ...). The field's own `min`/`max` and allowed values must cover every case. |
| `formula` | `"formula": "ordered_qty * unit_price"` | Computes the value from other fields of the row, including inherited ones. |
| `hidden` | `"hidden": true` | A helper column for links, cases or formulas that is left out of the output file (multi-table files only; a single table keeps it). |
| `after` | `"after": ["status"]` | Generate this field after the named fields. |

A formula uses the same safe expression language as `rules`: arithmetic, comparisons,
`a if condition else b`, and the functions `round`, `min`, `max`, `abs`, `int`, `float`, `str`, `concat`,
`add_days(date, n)`, `days_between(a, b)`, `year`, `month`, `day`, `iso_week`, `weekday`, `lower`, `upper`.

Exact counts are assigned through a seeded shuffle of the row numbers. They stay exact however the job is split
into batches or sub-jobs, and they are spread over the whole table. The same seed gives the same tables.

**A time-based story.** A combination of options can express patterns that change over time, for example
"Apex OTIF drops to 30% after week 44":

```json
"period":  {"type": "string", "formula": "'late' if iso_week(ship_date) > 44 else 'early'", "hidden": true},
"on_time": {"type": "boolean", "allowed_values": [true, false], "weights": [92, 8],
            "cases_by": "period", "cases": {"late": {"allowed_values": [true, false], "weights": [30, 70]}}}
```

The proportions inside a case are sampled. Exact counts apply to the whole table.

## Checks done on every run

* Every linked key combination exists in the parent table and satisfies the link's `where` and `match`.
  A broken link fails the job.
* `primary_key` columns are unique.
* Each table's row count is exactly the count that was planned.

## Not covered

* Documents (PDF contracts, audit reports), Excel plan workbooks and GPS tracks. These are files, not table
  rows.
* Exact counts inside a case, or tied to a time window. Inside cases, proportions are sampled.
* Modes `cover` and `unique` combined with `match`.
* A link from a table to itself. Within one table, use the `reference` generator.
