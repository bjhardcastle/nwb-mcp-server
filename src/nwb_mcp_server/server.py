# /// script
# dependencies = [
#   "lazynwb",
#   "fastmcp",
# ]
# ///
import asyncio
import contextlib
import dataclasses
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
logger.info("Starting MCP NWB Server")

class Settings(
    pydantic_settings.BaseSettings, 
    cli_parse_args=True, 
    cli_implicit_flags=True,
    cli_ignore_unknown_args=True,
):
    """Settings for the NWB MCP Server."""
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
    ignored_args: pydantic_settings.CliUnknownArgs

config = Settings() # type: ignore[call-arg]

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
    This server provides tools for querying a read-only virtual database of NWB data.
    1. execute PostgresSQL queries against the NWB data using the `execute_query` tool.
    a. Use standard SQL syntax, including `SELECT`, `FROM`, `WHERE`, `JOIN`, etc. and functions like `COUNT`, `AVG`, `SUM`, etc.
    b. Do as much filtering as possible in the query to avoid loading all rows/columns into memory.
    """
else:
    instructions = f"""
    This server provides a tools for previewing and working with NWB files.
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
    instructions=instructions,
)

@server.tool()
async def get_tables(ctx: fastmcp.Context) -> list[str]:
    """Returns a list of available tables in the NWB files. Each table includes data from multiple
    NWB files, where one file includes data from a single session for a single subject."""
    await ctx.info("Fetching available tables from NWB files")
    return await asyncio.to_thread(ctx.request_context.lifespan_context.db.tables)

@server.tool()
async def get_table_schema(table_name: str, ctx: fastmcp.Context) -> dict[str, pl.DataType]:
    """Returns the schema of a specific table, mapping column names to their data types."""
    if not table_name:
        raise ValueError("Table name cannot be empty")
    await ctx.info(f"Fetching schema for table: {table_name}")
    query = f"SELECT * FROM {table_name} LIMIT 0"
    lf: pl.LazyFrame = await asyncio.to_thread(ctx.request_context.lifespan_context.db.execute, query)
    return lf.schema

@server.resource("config://nwb_file_search_parameters")
async def nwb_file_search_parameters(ctx: fastmcp.Context) -> NWBFileSearchParameters:
    return NWBFileSearchParameters(
        root_dir=config.root_dir,
        glob_pattern=config.glob_pattern,
    )

@server.tool(enabled=False)
async def get_nwb_file_search_parameters(ctx: fastmcp.Context) -> NWBFileSearchParameters:
    """
    Returns a directory path and a file glob pattern for finding NWB files.
    
    >>> nwb_files = list(upath.UPath(result.root_dir).glob(result.glob_pattern))
    """
    resource = await ctx.read_resource("config://nwb_file_search_parameters")
    return NWBFileSearchParameters.model_validate_json(resource[0].content)

@server.tool(enabled=config.no_code)
async def execute_query(query: str, ctx: fastmcp.Context) -> str:
    """Executes a SQL query against a virtual read-only NWB database,
    returning results as JSON. Uses PostgreSQL syntax and functions for basic analysis."""
    return await _execute_query(query, ctx)

@server.tool(enabled=not config.no_code)
async def preview_table_values(table: str, columns: Iterable[str], ctx: fastmcp.Context) -> str:
    """Returns a preview of values from a specific table, limited to 5 rows. Likely will not include all
    unique values as that can be expensive for large tables."""
    query = f"SELECT {', '.join(columns)} FROM {table} LIMIT 5;"
    return await _execute_query(query, ctx)

async def _execute_query(query: str, ctx: fastmcp.Context) -> str:
    """Executes a SQL query against a virtual read-only NWB database,
    returning results as JSON. Uses PostgreSQL syntax and functions for basic analysis."""
    if not query:
        await ctx.error("No query provided")
        raise ValueError("Query cannot be empty")
    await ctx.info(f"Executing query: {query}")
    df: pl.DataFrame = await asyncio.to_thread(ctx.request_context.lifespan_context.db.execute, query)
    if df.is_empty():
        await ctx.warning("Query returned no results")
        return "[]"
    await ctx.info(f"Query executed successfully, serializing {len(df)} rows as JSON")
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

def main() -> None:
    server.run()
    
if __name__ == "__main__":
    main()