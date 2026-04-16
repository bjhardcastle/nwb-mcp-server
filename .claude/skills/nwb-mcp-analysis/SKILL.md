---
name: nwb-mcp-analysis
description: NWB data exploration and analysis via the NWB MCP server tools. Use this skill whenever the user wants to query, explore, or analyze a neuroscience NWB dataset and MCP tools like get_tables, execute_query, or get_nwb_paths are available. Trigger for questions about neural recordings, spike sorting results, trials, units, timeseries, or brain regions.
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

**Avoid `preview_table_values`** unless you need to see actual cell values — `get_table_schema` is faster and cheaper for schema discovery.

**TimeSeries tables** (those with `timestamps` and `data` columns) are sampled at high rates and, concatenated across many files, can be enormous. Before querying, warn the user and propose a strategy: (a) **slice** to specific time windows (natural fit if the user wants per-trial segments), or (b) **subsample** every Nth row for a representative overview. Ask which they prefer unless context makes it obvious (e.g. trial-aligned analysis → slicing).

**`session` table** — check this early for experiment-level metadata. See `get_tables` for details on the naming.

**Table names from `get_tables` are full NWB paths** (e.g. `/intervals/trials`, `/processing/ecephys/LFP/LFP`) — always double-quote them in SQL: `SELECT start_time, stop_time, is_response FROM "/intervals/trials"`.

---

## Phase 2: Code (when SQL isn't enough)

Switch to Python when the analysis requires custom functions, complex data processing, or visualizations.

1. Call `nwb_file_search_code_snippet` to get the path-finding code for this dataset
2. Follow the `nwb-data-analysis` skill for `lazynwb`/polars patterns and best practices
3. **Don't hard-code data from SQL exploration** — e.g. specific session IDs or counts observed during Phase 1. Code should generalize to any set of NWB files with the same organization. Recreate filtering and selection logic in code.

---

## Notes
- The MCP server uses `lazynwb.get_sql_context()` under the hood to create a virtual SQL database from one or more NWB files, with tables concatenated across files. Schemas are normalized across files (see `get_table_schema` for implications).
- `execute_query` results are capped at `max_result_rows` (default 50). If a query exceeds this, refine with `WHERE`/`GROUP BY`/`LIMIT`. `allow_large_output=True` bypasses the cap — only use it when saving results directly to a file, since large outputs will fill the context window.