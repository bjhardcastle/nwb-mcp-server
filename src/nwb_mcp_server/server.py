# /// script
# dependencies = [
#   "lazynwb",
#   "fastmcp",
# ]
# ///
import argparse
import asyncio
import contextlib
import dataclasses
import io
import logging
import logging.handlers
from typing import AsyncIterator

import lazynwb
import polars as pl
import pydantic
import fastmcp
import upath

logger = logging.getLogger('__name__')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info("Starting MCP NWB Server")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='NWB MCP Server')
    parser.add_argument(
        '--root-dir', default="data", help='Root directory to search for NWB files',
    )
    parser.add_argument(
        '--glob-pattern', default="*.nwb", help='A glob pattern to apply (non-recursively) to root-dir to locate NWB files (or folders), e.g. "*.zarr.nwb"',
    )
    parser.add_argument('--tables', nargs='*', default=(), help='List of table names to use, where each table is the name of a data object within each NWB file, e.g ["trials", "units"]')
    parser.add_argument('--infer-schema-length', type=int, default=None, help='Number of NWB files to scan to infer schema for all files')
    
    args = parser.parse_args()
    return args

args = parse_args()

INSTRUCTIONS = f"""

1. Please use upath (`universal-pathlib` Python package) to search for NWB files
   in {args.root_dir!r} with the glob pattern {args.glob_pattern!r}.
2. Once you have found the NWB files, you can use the following tools to interact with them:
   - `get_tables`: Get a list of available tables in the NWB files.
   - `get_table_schema`: Get the schema of a specific table.
   - `nwb_file_search_parameters`: Get the search parameters used to find the NWB files.
"""

@dataclasses.dataclass
class AppContext:
    db: pl.SQLContext

class NWBFileSearchParameters(pydantic.BaseModel):
    """Used to find NWB files in the filesystem."""
    root_dir: str
    glob_pattern: str

@contextlib.asynccontextmanager
async def server_lifespan(server: fastmcp.FastMCP) -> AsyncIterator[AppContext]:
    """Manage server startup and shutdown lifecycle."""
    logger.info("Globbing for NWB files in search directory")
    server.nwb_sources = list(upath.UPath(server.root_dir).glob(server.glob_pattern))
    logger.info(f"Found {len(server.nwb_sources)} NWB files matching pattern '{server.root_dir}/{server.glob_pattern}'")
    # Initialize the SQL connection to the NWB files
    logger.info("Initializing SQL connection to NWB files (may take a while)")
    sql_context = await asyncio.to_thread(
        lazynwb.get_sql_context, 
        nwb_sources=server.nwb_sources, 
        infer_schema_length=server.infer_schema_length, 
        table_names=server.tables or None, 
        full_path=False, 
        disable_progress=True, 
        eager=False,
    )
    try:
        yield AppContext(db=sql_context)
    finally:
        await asyncio.to_thread(lazynwb.clear_cache)

server = fastmcp.FastMCP(
    name="nwb-mcp-server",
    lifespan=server_lifespan,
)
server.root_dir = upath.UPath(args.root_dir).resolve().as_posix()
server.glob_pattern = args.glob_pattern
server.tables = args.tables
server.infer_schema_length = args.infer_schema_length

@server.tool()
async def get_tables(ctx: fastmcp.Context) -> list[str]:
    """Get the list of SQL tables available in the NWB database. Each table corresponds to a concatenated data from multiple NWB files"""        
    await ctx.info("Fetching available tables from NWB files")
    return await asyncio.to_thread(ctx.request_context.lifespan_context.db.tables)

@server.tool()
async def get_table_schema(table_name: str, ctx: fastmcp.Context) -> dict[str, pl.DataType]:
    """Get the schema of a specific table in the NWB database, using polars ."""
    if not table_name:
        raise ValueError("Table name cannot be empty")
    await ctx.info(f"Fetching schema for table: {table_name}")
    query = f"SELECT * FROM {table_name} LIMIT 0"
    lf: pl.LazyFrame = await asyncio.to_thread(ctx.request_context.lifespan_context.db.execute, query)
    return lf.schema

@server.resource("config://nwb_file_search_parameters")
async def nwb_file_search_parameters(ctx: fastmcp.Context) -> NWBFileSearchParameters:
    return NWBFileSearchParameters(
        root_dir=server.root_dir,
        glob_pattern=server.glob_pattern,
    )

@server.tool()
async def get_nwb_file_search_parameters(ctx: fastmcp.Context) -> NWBFileSearchParameters:
    """Provides a directory path and a file glob pattern for finding NWB files. Useful for creating
    Python code that can locate the NWB files used by this server."""
    resource = await ctx.read_resource("config://nwb_file_search_parameters")
    return NWBFileSearchParameters.model_validate_json(resource[0].content)

# @server.tool()
# async def get_nwb_paths() -> list[str]:
#     """Get the list of NWB file paths that the server is currently using."""
#     return [p.as_posix() for p in server.nwb_sources]

@server.tool(enabled=False)
async def execute_query(query: str, ctx: fastmcp.Context) -> str:
    """Run a SQL query against the NWB database and return a dataframe as a JSON string."""
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

@server.prompt()
async def find_nwb_files(ctx: fastmcp.Context) -> str:
    """Generates a user message asking for a specific way to search for NWB files"""
    resource  = await ctx.read_resource("config://nwb_file_search_parameters")
    search_parameters = NWBFileSearchParameters.model_validate_json(resource[0].content)
    logger.info(f"Got {search_parameters=!r}")
    return (
        "Please use upath (`universal-pathlib` Python package) to search for NWB files "
        f"in {search_parameters.root_dir!r} with the glob pattern {search_parameters.glob_pattern!r}."
    )
    return (
        f"Please use the following code to search for NWB files:\n\n"
        f"```python\n"
        f"import upath\n"
        f"\n"
        f"nwb_root_dir = {search_parameters.root_dir!r}\n"
        f"nwb_glob_pattern = {search_parameters.glob_pattern!r}\n"
        f"\n"
        f"nwb_files = list(upath.UPath(nwb_root_dir).glob(nwb_glob_pattern))\n"
        f"print('Found {{len(nwb_files)}} NWB files matching {{nwb_root_dir}}/{{nwb_glob_pattern}}')\n"
        f"```\n\n"
    )
           
           
if __name__ == "__main__":
    server.run()