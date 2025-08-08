# /// script
# dependencies = [
#   "lazynwb",
#   "fastmcp",
# ]
# ///

import os

os.environ["RUST_BACKTRACE"] = "1"  # enable Rust backtraces for lazynwb

import asyncio
import contextlib
import dataclasses
import functools
import importlib.metadata
import io
import logging
from collections.abc import AsyncIterator, Iterable

import fastmcp
import lazynwb
import polars as pl
import pydantic
import pydantic_settings
import upath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info(
    f"Starting MCP NWB Server with lazynwb v{importlib.metadata.version('lazynwb')}"
)


class ServerConfig(pydantic_settings.BaseSettings):
    """Configuration for the NWB MCP Server."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_parse_args=True,
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
    )

    root_dir: str = pydantic.Field(
        default="data",
        description="Root directory to search for NWB files",
    )
    glob_pattern: str = pydantic.Field(
        default="**/*.nwb",
        description="A glob pattern to apply (non-recursively) to `root-dir` to locate NWB files (or folders), e.g. '*.zarr.nwb'",
    )
    tables: list[str] = pydantic.Field(
        default_factory=list,
        description="List of table names to use, where each table is the name of a data object within each NWB file, e.g ['trials', 'units']",
    )
    infer_schema_length: int = pydantic.Field(
        default=1,
        description="Number of NWB files to scan to infer schema for all files",
    )
    unattended: bool = pydantic.Field(
        default=False,
        description="Run the server in unattended mode, where it does not prompt the user for input. Useful for automated tasks.",
    )
    table_element_limit: int = pydantic.Field(
        default=500,
        description="Maximum number of elements (columns x rows) allowed in a table returned by a SQL query without raising an error. This is to avoid overwhelming the context window.",
    )
    ignored_args: pydantic_settings.CliUnknownArgs


config = ServerConfig()  # type: ignore[call-arg]
logger.info(f"Configuration loaded: {config}")

UNATTENDED_RULE = (
    (
        "Never prompt the user for input or clarification. Work entirely autonomously and make principled"
        "decisions based on the available data to achieve the most accurate and useful outcome."
    )
    if config.unattended
    else ""
)

RULES = f"""
<rules>
1. Always start in No Code Mode.
2. Highlight any assumptions you make.
3. Explain how you averaged or aggregated data, if applicable.
4. Do not make things up.
5. Do not provide an excessively positive picture. Be objective. Be critical where appropriate. 
6. {UNATTENDED_RULE}
</rules>
"""

ABOUT = f"""
<mcp>
The NWB MCP server provides tools for querying a read-only virtual database of NWB data, with two modes of
operation:
1. **No Code Mode**: The agent itself performs queries. This can be used for simple analyses on smaller datasets and for
gaining an initial understanding of the structure of the data. 
2. **Code Mode**: The agent writes files in the workspace containing Python code. No Code Mode
should be used first to help plan the files that should be written.

<no_code_mode>
The agent can execute PostgreSQL queries against the NWB data using the `execute_query` tool.
a. Use standard SQL syntax, including `SELECT`, `FROM`, `WHERE`, `JOIN`, etc. and functions like `COUNT`, `AVG`, `SUM`, etc.
b. Do as much filtering as possible in the query to avoid loading all rows/columns into memory.
c. Avoid loading TimeSeries tables.
</no_code_mode>

<code_mode>
The agent can write files containing Python code to interact with the NWB files using the `lazynwb` package.
a. Use `upath` to search for NWB files: `nwb_paths = list(upath.UPath({config.root_dir!r}).glob({config.glob_pattern!r}))`
b. Use `lazynwb` to interact with tables in the NWB files:
   i. Get a polars `LazyFrame` for a table with `lazynwb.scan_nwb(nwb_paths, table_name)`.
   ii. Write queries against the `LazyFrame` using standard polars methods.
   iii. Use `.filter()` as appropriate to avoid loading all rows into memory.
   iv. Use `.select()` to choose specific columns to include in the final result and avoid loading
   unnecessary columns, particularly array- or list-like columns.
   v. Then use `.collect()` to execute the query and get a polars `DataFrame`.
</code_mode>

<nwb>
1. The `session` or `general` tables may contain useful metadata about the session, such as
   `experiment_description`.
2. Use the `default_qc` Boolean column in the `units` table, which indicates whether the unit
   passed quality control checks.
c. Be extremely cautious when loading data from TimeSeries tables (tables with `timestamps` and
   `data` columns) as they can be large.
</nwb>

</mcp>
"""


@functools.cache
def get_nwb_sources() -> list[upath.UPath]:
    """Get the list of NWB files based on the provided root directory and glob pattern."""
    # TODO make async
    logger.info(
        f"Searching for NWB files in {config.root_dir} with pattern {config.glob_pattern}"
    )
    nwb_paths = list(upath.UPath(config.root_dir).glob(config.glob_pattern))
    if not nwb_paths:
        raise ValueError(
            f"No NWB files found in {config.root_dir!r} matching pattern {config.glob_pattern!r}"
        )
    logger.info(f"Found {len(nwb_paths)} data sources")
    return nwb_paths


@dataclasses.dataclass
class AppContext:
    db: pl.SQLContext


class NWBFileSearchParameters(pydantic.BaseModel):
    """Used to find NWB files in the filesystem."""

    root_dir: str
    glob_pattern: str


def create_sql_context_non_nwb(sources: Iterable[upath.UPath]) -> pl.SQLContext:
    """Create a SQLContext for non-NWB sources."""
    sources = tuple(sources)
    if not sources:
        raise ValueError("Must provide at least one source path")
    table_name_to_path = {p.stem: p for p in sources}
    if len(table_name_to_path) != len(sources):
        raise ValueError(
            "Duplicate source names found, please ensure unique names for each source"
        )

    suffix_to_read_function = {
        ".csv": "scan_csv",
        "": "scan_delta",  # Delta uses directory structure
        ".feather": "scan_ipc",
        ".json": "scan_ndjson",
        ".ndjson": "scan_ndjson",
        ".parquet": "scan_parquet",
        ".arrow": "scan_pyarrow_dataset",
    }
    frames = {}
    for table_name, path in table_name_to_path.items():
        ext = path.suffix.lower()
        if ext not in suffix_to_read_function:
            raise ValueError(f"Unsupported file extension: {ext}")
        if ext == "" and not path.is_dir():
            raise ValueError(
                f"Received a data source with no extension but is not a directory: {path}. Unsure how to continue."
            )
        read_func = getattr(pl, suffix_to_read_function[ext])
        frames[table_name] = read_func(path.as_posix())
    return pl.SQLContext(frames=frames, eager=False)


@contextlib.asynccontextmanager
async def server_lifespan(server: fastmcp.FastMCP) -> AsyncIterator[AppContext]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize the SQL connection to the NWB files
    logger.info("Initializing SQL connection to NWB files (may take a while)")
    sources = get_nwb_sources()
    if not any(".nwb" in str(s) for s in sources):
        # If no NWB files found, create a non-NWB SQLContext
        logger.warning("No NWB files found, creating SQLContext for non-NWB sources")
        sql_context = create_sql_context_non_nwb(sources)
    else:
        sql_context = lazynwb.get_sql_context(
            nwb_sources=sources,
            infer_schema_length=config.infer_schema_length,
            table_names=config.tables or None,
            exclude_timeseries=False,
            full_path=True,  # if False, NWB objects will be referenced by their names, not full paths, e.g. 'epochs' instead of '/intervals/epochs'
            disable_progress=True,
            eager=False,  # if False, a LazyFrame is returned from `execute`
            rename_general_metadata=True,  # renames the metadata in 'general' as 'session'
        )
    logger.info("SQL connection initialized successfully")
    try:
        yield AppContext(db=sql_context)
    finally:
        await asyncio.to_thread(lazynwb.clear_cache)


server = fastmcp.FastMCP(
    name="nwb-mcp-server",
    lifespan=server_lifespan,
    # instructions=ABOUT + RULES,
    # instructions can be passed as a string, but does not appear to influence the agent (VScode)
)


@server.tool()
def get_tables(ctx: fastmcp.Context) -> list[str]:
    """Returns a list of available tables in the NWB files. Each table includes data from multiple
    NWB files, where one file includes data from a single session for a single subject.
    """
    logger.info("Fetching available tables from NWB files")
    return ctx.request_context.lifespan_context.db.tables()


@server.tool()
def get_table_schema(table_name: str, ctx: fastmcp.Context) -> dict[str, pl.DataType]:
    """Returns the schema of a specific table, mapping column names to their data types."""
    if not table_name:
        raise ValueError("Table name cannot be empty")
    logger.info(f"Fetching schema for table: {table_name}")
    query = f"SELECT * FROM {format_table_name(table_name)} LIMIT 0"
    lf: pl.LazyFrame = ctx.request_context.lifespan_context.db.execute(query)
    return lf.schema


@server.resource("config://nwb_file_search_parameters")
async def nwb_file_search_parameters(ctx: fastmcp.Context) -> NWBFileSearchParameters:
    return NWBFileSearchParameters(
        root_dir=config.root_dir,
        glob_pattern=config.glob_pattern,
    )


@server.tool(enabled=True)
async def execute_query(query: str, ctx: fastmcp.Context) -> str:
    """Executes a SQL query against a virtual read-only NWB database,
    returning results as JSON. Uses PostgreSQL syntax and functions for basic analysis.

    Queries should be designed to return a manageable amount of data for interpretation by an LLM,
    ideally less than 20 rows. If too much data is returned an error will be raised.
    """
    return await _execute_query(query, ctx)


def format_table_name(table: str) -> str:
    """Ensures the table name is properly formatted for pl.SQLContext queries."""
    if not table:
        raise ValueError("Table name cannot be empty")
    return f'"{table}"'


def format_column_names(columns: Iterable[str] | None) -> str:
    """Formats column names for SQL queries, ensuring they are properly quoted."""
    if not columns:
        return "*"
    if isinstance(columns, str):
        columns = [columns]
    return ", ".join(repr(c) for c in columns).replace("'", '"')


@server.tool(enabled=True)
async def preview_table_values(
    table: str, ctx: fastmcp.Context, columns: Iterable[str] | None = None
) -> str:
    """Returns the first row of a table to preview values. Prefer `get_table_schema` to get the
    table schema. Only use this if absolutely necessary."""
    column_query = format_column_names(columns)
    query = f"SELECT {column_query} FROM {format_table_name(table)} LIMIT 1;"
    logger.info(f"Previewing table values with: {column_query}")
    return await _execute_query(query, ctx)


async def _execute_query(query: str, ctx: fastmcp.Context) -> str:
    """Executes a SQL query against a virtual read-only NWB database,
    returning results as JSON. Uses PostgreSQL syntax and functions for basic analysis.
    """
    if not query:
        raise ValueError("SQL query cannot be empty")
    logger.info(f"Executing query: {query}")
    df: pl.DataFrame = ctx.request_context.lifespan_context.db.execute(
        query, eager=True
    )
    if df.is_empty():
        logger.warning("SQL query returned no results")
        return "[]"
    if (elements := df.shape[0] * df.shape[1]) > config.table_element_limit:
        raise ValueError(
            f"SQL query returned too many elements ({elements}): please refine the query by aggregating or filtering to avoid filling the context window with data."
        )
    logger.info(f"Query executed successfully, serializing {len(df)} rows as JSON")
    # return _to_markdown(df)
    if "obs_intervals" in df.columns:
        # lists of arrays cause JSON conversion to crash
        # arrays are ok
        # TODO report issue to polars
        # TODO generalize to cast all list[array] columns
        df = df.cast({"obs_intervals": pl.List(pl.List(pl.Float64))})
    return df.write_json()


def _to_markdown(df: pl.DataFrame) -> str:
    # https://github.com/pola-rs/polars/issues/13907#issuecomment-1904137685
    buf = io.StringIO()
    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=False,
        tbl_hide_dataframe_shape=True,
    ):
        print(df, file=buf)
    buf.seek(0)
    return buf.read()


@server.resource("dir://nwb_paths")
def nwb_paths() -> list[str]:
    """List the available NWB files."""
    return [p.as_posix() for p in get_nwb_sources()]


@server.prompt
def analysis_report_prompt(query: str) -> str:
    """User prompt for creating an analysis report."""
    if not query:
        raise ValueError("Query cannot be empty")

    return f"""
Please provide an analysis report for a scientist posing the following query:
{query!r}

{RULES}

<instructions>
1. Explore the available tables in No Code Mode to understand their schemas and relationships.
2. Develop a step-by-step plan for the analysis, prompting the user for any necessary clarifications.
    i. if the analysis is simple, you can run it in No Code Mode. 
    ii. if the analysis is complex (ie. requires custom functions, data processing, or may take a long
        time to run), switch to Code Mode and create Python code that:
        a. fetches data
        b. runs any required pre-processing
        c. generates relevant visualizations
        d. performs any statistical analyses
        e. summarizes the results
    If the analysis is expected to take a long time, incorporate a test mode that runs the
    analysis on a small subset of the data first. The user can then run the full analysis offline.
3. Execute the analysis and provide a detailed report of the findings.
</instructions>

{ABOUT}
"""


@server.prompt
def general_prompt(query: str) -> str:
    """User prompt for general queries."""
    if not query:
        raise ValueError("Query cannot be empty")

    return f"""
Please provide a detailed response to a scientist posing the following query:
{query!r}

{RULES}

<instructions>
1. Explore the available tables in No Code Mode to understand their schemas and relationships.
2. Provide a comprehensive answer to the query, including relevant data and insights.
3. If the query requires complex analysis, switch to Code Mode and create Python code that:
    i. fetches data
    ii. runs any required pre-processing
    iii. generates relevant visualizations
    iv. performs any statistical analyses
    v. summarizes the results
    If the analysis is expected to take a long time, incorporate a test mode that runs the
    analysis on a small subset of the data first. The user can then run the full analysis offline.
</instructions>

{ABOUT}
"""


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
