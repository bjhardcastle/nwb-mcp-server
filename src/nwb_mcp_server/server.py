# /// script
# dependencies = [
#   "lazynwb",
#   "mcp",
#   "fastmcp",
# ]
# ///
import argparse
import asyncio
import contextlib
import dataclasses
import logging
from typing import AsyncIterator

import lazynwb
import polars as pl
from mcp.server import InitializationOptions
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.stdio import stdio_server
import mcp.types as types
from mcp.server.fastmcp import FastMCP, Context
# from fastmcp import FastMCP
import upath

logger = logging.getLogger('mcp_nwb_server')
logger.info("Starting MCP NWB Server")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='NWB MCP Server')
    parser.add_argument(
        '--search-dir', default="data", help='Root directory to search for NWB files',
    )
    parser.add_argument(
        '--nwb-file-glob', default="*.nwb", help='A glob pattern to apply to search-dir to locate NWB files (or folders), e.g. "*.zarr"',
    )
    parser.add_argument('--tables', nargs='*', default=(), help='List of table names to use, where each table is the name of a data object within each NWB file, e.g ["trials", "units"]')
    parser.add_argument('--infer-schema-length', type=int, default=None, help='Number of NWB files to scan to infer schema for all files')
    
    args = parser.parse_args()
    return args

@dataclasses.dataclass
class AppContext:
    db: pl.SQLContext
    
@dataclasses.dataclass
class FileGlobParams:
    """Used to find NWB files in the filesystem."""
    search_dir: str
    nwb_file_glob: str

@contextlib.asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage server startup and shutdown lifecycle."""
    logger.info("Globbing for NWB files in search directory")
    server.nwb_sources = list(upath.UPath(server.search_dir).resolve().glob(server.nwb_file_glob))
    # Initialize the SQL connection to the NWB files
    logger.info("Initializing SQL connection to NWB files (may take a while)")
    sql_context = await asyncio.to_thread(lazynwb.get_sql_context, nwb_sources=server.nwb_sources, infer_schema_length=server.infer_schema_length, table_names=server.tables, full_path=False, disable_progress=True, eager=True)
    # TODO add server.settings.tables filtering when implemented in lazynwb
    try:
        yield AppContext(db=sql_context)
    finally:
        await asyncio.to_thread(lazynwb.clear_cache)

server = FastMCP(
    name="nwb-mcp-server",
    lifespan=server_lifespan,
)
args = parse_args()
server.search_dir = args.search_dir
server.nwb_file_glob = args.nwb_file_glob   
server.tables = args.tables
server.infer_schema_length = args.infer_schema_length


@server.tool()
async def get_tables() -> list[str]:
    """Get the list of SQL tables available in the NWB database. Each table corresponds to a concatenated data from multiple NWB files"""
    ctx: Context = server.get_context()
    await ctx.info("Fetching available tables from NWB files")
    return await asyncio.to_thread(ctx.request_context.lifespan_context.db.tables)

@server.tool()
async def get_nwb_glob_parameters() -> FileGlobParams:
    """Provides a directory path and a file glob pattern for finding NWB files. Useful for creating
    Python code that can locate the NWB files used by this server."""
    return FileGlobParams(
        search_dir=server.search_dir,
        nwb_file_glob=server.nwb_file_glob
    )
    
@server.tool()
async def get_nwb_paths() -> list[str]:
    """Get the list of NWB file paths that the server is currently using."""
    return [p.as_posix() for p in server.nwb_sources]

@server.tool()
async def execute_query(query: str) -> str:
    """Run a SQL query against the NWB database and return a dataframe as a JSON string."""
    ctx: Context = server.get_context()
    if not query:
        await ctx.error("No query provided")
        raise ValueError("Query cannot be empty")
    await ctx.info(f"Executing query: {query}")
    df: pl.DataFrame = await asyncio.to_thread(ctx.request_context.lifespan_context.db.execute, query)
    if df.is_empty():
        await ctx.warning("Query returned no results")
        return "[]"
    await ctx.info(f"Query executed successfully, serializing {len(df)} rows as JSON")
    if 'obs_intervals' in df.columns:
        # lists of arrays cause JSON conversion to crash
        # arrays are ok
        # TODO report issue to polars
        # TODO generalize to cast all list[array] columns
        df = df.cast({'obs_intervals': pl.List(pl.List(pl.Float64))})
    return df.write_ndjson()

if __name__ == "__main__":
    server.run()