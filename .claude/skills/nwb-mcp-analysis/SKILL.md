---
name: nwb-mcp-analysis
description: NWB data exploration and analysis via the NWB MCP server tools. Use this skill whenever the user wants to query, explore, or analyze a neuroscience NWB dataset and MCP tools like get_tables, execute_query, or get_nwb_paths are available. Trigger for questions about neural recordings, spike sorting results, trials, units, timeseries, brain regions, or any NWB data — even if the user doesn't say "MCP" or "NWB" explicitly, as long as an NWB MCP server is configured.
---

# NWB MCP Analysis

Two-phase workflow: **explore with SQL first, then incorporate findings into code if the user needs more complex analysis or re-runnable code**.

---

## Phase 1: Explore

**Recommended order:**
1. `get_tables` — see what's available
2. `get_nwb_paths` — check how many files are in the dataset before diving in; the count shapes how you interpret aggregate results
3. `get_table_schema` on tables of interest
4. `execute_query` for targeted analysis

**Avoid `preview_table_values`** unless you need to see actual cell values. For schema discovery, `get_table_schema` is faster and cheaper. Even a 1-row preview can be slow on remote/cloud NWB files.

**TimeSeries tables** (those with `timestamps` and `data` columns) are sampled at high rates and, concatenated across many files, can be enormous. Before querying, warn the user and propose a strategy: (a) **slice** to specific time windows (natural fit if the user wants per-trial segments), or (b) **subsample** every Nth row for a representative overview. Ask which they prefer unless context makes it obvious (e.g. trial-aligned analysis → slicing).

**Tables are lazily loaded** — aggregations like `COUNT(*)`, `COUNT(DISTINCT ...)`, or fetching all distinct values of a column force data to be materialized and can be slow or expensive. Avoid them unless necessary; prefer targeted `WHERE` + `LIMIT` queries instead.

**Query result size** — `execute_query` raises an error when the result exceeds the server's element limit (rows × columns). Use `WHERE` conditions and `LIMIT` clauses to keep results small; aim for < 20 rows.

**Array/list columns** (e.g. `spike_times`, `waveform_mean`) can be very large even outside TimeSeries tables. Pre-filter rows using scalar columns before selecting any array column.

**`session` table** — contains experiment-level metadata (`experiment_description`, subject info, etc.). The server renames NWB's `general` group to `session`, so it may not be recognized as a metadata table from the name alone. Check it early when you need context about the experiment.

---

## Phase 2: Code (when SQL isn't enough)

Switch to Python when the analysis requires custom functions, complex data processing, or visualizations.

1. Call `nwb_file_search_code_snippet` to get the path-finding code for this dataset
2. Follow the `nwb-data-analysis` skill for `lazynwb`/polars patterns and best practices
3. **Don't hard-code data from SQL exploration** — e.g. specific session IDs or counts observed during Phase 1. Code should generalize to any set of NWB files with the same organization. Recreate filtering and selection logic in code.

---

## Notes
- The MCP server uses `lazynwb.get_sql_context()` under the hood to create a virtual SQL database from one or more NWB files, with tables concatenated across files.
- Individual NWB files may not contain all tables, and `lazynwb` normalizes schemas across files by filling missing columns with nulls. A null column in one session doesn't mean the data is absent across all sessions — and conversely, a non-null column in the inferred schema may still be null for many files.