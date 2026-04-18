---
name: nwb-mcp-analysis
description: NWB data exploration and analysis via the NWB MCP server tools. Use this skill whenever the user wants to query, explore, or analyze a neuroscience NWB dataset and MCP tools like get_active_source, get_tables, execute_query, or get_nwb_paths are available. Trigger for questions about neural recordings, spike sorting results, trials, units, timeseries, or brain regions.
---

# NWB MCP Analysis

Two-phase workflow: explore with SQL first, then incorporate findings into code if the user needs more complex analysis or re-runnable code.

---

## Phase 0: Confirm The Active Source

The server may start with no dataset configured at all. Do not assume a preset `root_dir` or `dandiset_id` exists.

Recommended order before any schema or SQL work:
1. Call `get_active_source`
2. If the source is unset, choose one in chat:
   - `use_local_source(root_dir="...")`
   - `use_dandiset_source(dandiset_id="000363")`
3. If the user wants to go back to any startup preset, call `reset_active_source`

Important behavior:
- Source selection is scoped to the current chat session, not global across all clients. You can still change the active source multiple times within the same chat session if needed, accepting the cost of rescanning or reinitializing the new dataset.
- `nwb_file_search_code_snippet` depends on the active source and will fail until one is selected.
- If a dataset-dependent tool says no dataset is active, switch sources first instead of retrying the same query.

---

## Phase 1: Explore

Recommended order:
1. `get_active_source` if you have not checked the source yet
2. `get_tables` to see what is available
3. `get_nwb_paths` to check how many files are in the dataset before diving in; the count shapes how you interpret aggregate results
4. `get_table_schema` on tables of interest
5. `execute_query` for targeted analysis

Avoid `preview_table_values` unless you need actual cell values. `get_table_schema` is faster and cheaper for schema discovery.

TimeSeries tables (those with `timestamps` and `data` columns) can be huge when concatenated across many files or sampled at high rates. Before querying them, warn the user and propose a strategy:
- slice to specific time windows
- subsample every Nth row for a representative overview

Ask which approach they prefer unless the context already makes the choice obvious (e.g. slicing is a natural fit for trial-aligned analysis).

Check the `session` table early for experiment-level metadata.

Table names from `get_tables` are full NWB paths, for example `/intervals/trials` or `/processing/ecephys/LFP/LFP`. Always double-quote them in SQL:

```sql
SELECT start_time, stop_time, is_response
FROM "/intervals/trials"
```

---

## Phase 2: Code

Switch to Python when the analysis requires custom functions, complex data processing, or visualizations.

1. Call `nwb_file_search_code_snippet` to get the path-finding code for the current dataset
2. Follow the `nwb-data-analysis` skill for `lazynwb` and polars patterns
3. Do not hard-code observations from SQL exploration, such as specific session IDs or counts seen during Phase 1. Recreate the filtering and selection logic in code so it generalizes to the active dataset.

---

## Notes

- The MCP server uses `lazynwb.get_sql_context()` under the hood to create a virtual SQL database from one or more NWB files, with tables concatenated across files.
- Schemas are normalized across files, so a column can appear in the schema even when it is null for many files.
- `execute_query` results are capped at `max_result_rows` by default. If a query exceeds the cap, refine it with `WHERE`, `GROUP BY`, or `LIMIT`.
- Use `allow_large_output=True` only when the result is being written directly to a file or another tool output, not for inline chat analysis.
