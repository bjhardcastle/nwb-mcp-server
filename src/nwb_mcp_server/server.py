# /// script
# dependencies = [
#   "lazynwb",
#   "fastmcp",
# ]
# ///
import asyncio
import contextlib
import dataclasses
import importlib.metadata
import io
import functools
import logging
from typing import AsyncIterator, Iterable

import lazynwb
import polars as pl
import pydantic
import pydantic_settings
import fastmcp
import upath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info(f"Starting MCP NWB Server with lazynwb v{importlib.metadata.version('lazynwb')}")

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
    no_code: bool = pydantic.Field(
        default=False,
        description="The agent itself performs analysis rather than generating code for the user to run. Works best for simple analyses on smaller datasets.",
    )
    unattended: bool = pydantic.Field(
        default=False,
        description="Run the server in unattended mode, where it does not prompt the user for input. Useful for automated tasks.",
    )
    ignored_args: pydantic_settings.CliUnknownArgs

config = ServerConfig() # type: ignore[call-arg]
logger.info(f"Configuration loaded: {config}")

@functools.cache
def get_nwb_sources() -> list[upath.UPath]:
    """Get the list of NWB files based on the provided root directory and glob pattern."""
    logger.info(f"Searching for NWB files in {config.root_dir} with pattern {config.glob_pattern}")
    nwb_paths = list(upath.UPath(config.root_dir).glob(config.glob_pattern))
    if not nwb_paths:
        raise ValueError(f"No NWB files found in {config.root_dir!r} matching pattern {config.glob_pattern!r}")
    logger.info(f"Found {len(nwb_paths)} NWB files")
    return nwb_paths

@dataclasses.dataclass
class AppContext:
    db: pl.SQLContext

class NWBFileSearchParameters(pydantic.BaseModel):
    """Used to find NWB files in the filesystem."""
    root_dir: str
    glob_pattern: str

if config.no_code:
    instructions = f"""
    This NWB MCP server provides tools for querying a read-only virtual database of NWB data.
    1. execute PostgreSQL queries against the NWB data using the `execute_query` tool.
    a. Use standard SQL syntax, including `SELECT`, `FROM`, `WHERE`, `JOIN`, etc. and functions like `COUNT`, `AVG`, `SUM`, etc.
    b. Do as much filtering as possible in the query to avoid loading all rows/columns into memory.
    """
else:
    instructions = f"""
    This NWB MCP server provides a tools for previewing and working with NWB files.
    
    When writing Python code to interact with the NWB files, please follow these steps:
    1. use upath (`universal-pathlib` Python package) to search for NWB files:
    `nwb_paths = list(upath.UPath({config.root_dir!r}).glob({config.glob_pattern!r}))`
    If the path points to S3, the user may need to supply credentials to access it.
    2. Once you have found the NWB files, use `polars` to interact with tables in them:
    a. Get a polars LazyFrame for table with `lazynwb.scan_nwb(nwb_paths, table_name)`.
    b. Write queries against the LazyFrame using standard `pl.LazyFrame` methods.
    c. Use `.filter()` as appropriate to avoid loading all rows into memory.
    d. Use `.select()` to choose specific columns to include in the final result and avoid loading
    unnecessary columns, particularly array- or list-like columns.
    e. Use `lf.collect()` to execute the query and get a polars DataFrame.
    3. Use the `default_qc` Boolean column in the `units` table, which indicates whether the unit
    passed quality control checks.
    4. The `session` or `general` tables may contain useful metadata about the session, such as `experiment_description`
    """

@contextlib.asynccontextmanager
async def server_lifespan(server: fastmcp.FastMCP) -> AsyncIterator[AppContext]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize the SQL connection to the NWB files
    logger.info("Initializing SQL connection to NWB files (may take a while)")
    sql_context = await asyncio.to_thread(
        lazynwb.get_sql_context, 
        nwb_sources=get_nwb_sources(), 
        infer_schema_length=config.infer_schema_length, 
        table_names=config.tables or None,
        exclude_timeseries=config.no_code,
        full_path=False,  # NWB objects will be referenced by their names, not full paths, e.g. 'epochs' instead of '/intervals/epochs'
        disable_progress=True,
        eager=False,  # a LazyFrame is returned from `execute`
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
    # instructions can be passed as a string, but does not appear to influence the agent (VScode) 
)

@server.tool()
async def get_tables(ctx: fastmcp.Context) -> list[str]:
    """Returns a list of available tables in the NWB files. Each table includes data from multiple
    NWB files, where one file includes data from a single session for a single subject."""
    logger.info("Fetching available tables from NWB files")
    return await asyncio.to_thread(ctx.request_context.lifespan_context.db.tables)

@server.tool()
async def get_table_schema(table_name: str, ctx: fastmcp.Context) -> dict[str, pl.DataType]:
    """Returns the schema of a specific table, mapping column names to their data types."""
    if not table_name:
        raise ValueError("Table name cannot be empty")
    logger.info(f"Fetching schema for table: {table_name}")
    query = f"SELECT * FROM {table_name} LIMIT 0"
    lf: pl.LazyFrame = await asyncio.to_thread(ctx.request_context.lifespan_context.db.execute, query)
    return lf.schema

@server.resource("config://nwb_file_search_parameters")
async def nwb_file_search_parameters(ctx: fastmcp.Context) -> NWBFileSearchParameters:
    return NWBFileSearchParameters(
        root_dir=config.root_dir,
        glob_pattern=config.glob_pattern,
    )

@server.tool(enabled=config.no_code)
async def execute_query(query: str, ctx: fastmcp.Context) -> str:
    """Executes a SQL query against a virtual read-only NWB database,
    returning results as JSON. Uses PostgreSQL syntax and functions for basic analysis."""
    return await _execute_query(query, ctx)

@server.tool(enabled=True)
async def preview_table_values(table: str, ctx: fastmcp.Context, columns: Iterable[str] | None = None) -> str:
    """Returns the first row of a table to preview values. Prefer `get_table_schema` to get the
    table schema. Only use this if absolutely necessary."""
    if not table:
        raise ValueError("Table name cannot be empty")
    if not columns:
        column_query = "*"
    else:
        if isinstance(columns, str):
            columns = [columns]
        column_query = ', '.join(repr(c) for c in columns)
    assert column_query, "column query cannot be empty"
    query = f"SELECT {column_query} FROM {table} LIMIT 1;"
    logger.info(f"Previewing table values with: {column_query}")
    return await _execute_query(query, ctx)

async def _execute_query(query: str, ctx: fastmcp.Context) -> str:
    """Executes a SQL query against a virtual read-only NWB database,
    returning results as JSON. Uses PostgreSQL syntax and functions for basic analysis."""
    if not query:
        raise ValueError("SQL query cannot be empty")
    logger.info(f"Executing query: {query}")
    df: pl.DataFrame = await asyncio.to_thread(ctx.request_context.lifespan_context.db.execute, query, eager=True)
    if df.is_empty():
        logger.warning("SQL query returned no results")
        return "[]"
    logger.info(f"Query executed successfully, serializing {len(df)} rows as JSON")
    # return _to_markdown(df)
    if 'obs_intervals' in df.columns:
        # lists of arrays cause JSON conversion to crash
        # arrays are ok
        # TODO report issue to polars
        # TODO generalize to cast all list[array] columns
        df = df.cast({'obs_intervals': pl.List(pl.List(pl.Float64))})
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
    rules = """
        <rules>
            1. Highlight any assumptions you make.
            2. Explain how you averaged or aggregated data, if applicable.
            3. Do not make things up.
            4. Do not provide an excessively positive picture. Be objective. Be critical where
               appropriate.
        </rules>
    """
    if config.unattended:
        user_input_prompt = """
        Never prompt the user for clarifying questions or input. Make principled decisions to
        achieve the most accurate and useful analysis.
        """
    else:
        user_input_prompt = """
        Ask the user clarifying questions to understand the query and plan the analysis.
        Questions should be enumerated so that the user can answer them one by one.
        """
    if config.no_code:
        
        return f"""
        Please provide an analysis report for a scientist posing the following query:
        <query>
        {query!r}
        </query>

        {rules}
        
        <instructions>
        1. {user_input_prompt}
        2. Explore the available tables to understand their schemas and relationships.
        3. Develop a step-by-step plan for the analysis.
        4. Do not create files with code to run. 
        5. Execute the analysis and provide a detailed report of the findings.
        </instructions>
        
        <mcp>
        {instructions}
        </mcp>
        """
    else:
        return f"""
        Please write analysis code in Python for a scientist posing the following query:
        <query>
        {query!r}
        </query>

        {rules}

        <instructions>
        1. {user_input_prompt}
        2. Explore the available tables to understand their schemas and relationships.
        3. Develop a step-by-step plan for the Python code that will be written.
        4. Write Python code that:
        a. fetches data
        b. runs any required pre-processing
        c. generates relevant visualizations
        d. performs any statistical analyses
        e. summarizes the results
        5. If the analysis is expected to take a long time, incorporate a test mode that runs the
        analysis on a small subset of the data first. The user can then run the full analysis offline.
        </instructions>

        <mcp>
        {instructions}
        </mcp>
        """
def main() -> None:
    server.run()
    
if __name__ == "__main__":
    main()