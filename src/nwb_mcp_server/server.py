# /// script
# dependencies = [
#   "lazynwb",
#   "mcp",
# ]
# ///
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
import upath

logger = logging.getLogger('mcp_nwb_server')
logger.info("Starting MCP NWB Server")

@dataclasses.dataclass
class AppContext:
    db: pl.SQLContext

@contextlib.asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage server startup and shutdown lifecycle."""
    logger.info("Globbing for NWB files in search directory")
    server.nwb_sources = upath.UPath(server.search_dir).resolve().glob(server.nwb_file_glob)
    # Initialize the SQL connection to the NWB files
    sql_context = await asyncio.to_thread(lazynwb.get_sql_context, nwb_sources=server.nwb_sources, full_path=True, disable_progress=True, eager=True)
    # TODO add server.settings.tables filtering when implemented in lazynwb
    try:
        yield AppContext(db=sql_context)
    finally:
        await asyncio.to_thread(lazynwb.clear_cache)

server = FastMCP(
    name="nwb-mcp-server",
    lifespan=server_lifespan,
)

server.search_dir="data"
server.nwb_file_glob="*.nwb"
server.tables=()

@server.resource("tables://list")
async def get_tables() -> list[str]:
    """Get the list of available tables in the db."""
    ctx: Context = server.get_context()
    await ctx.info("Fetching available tables from NWB files")
    return await asyncio.to_thread(ctx.request_context.lifespan_context.db.tables)

@server.tool()
async def execute_query(query: str) -> bytes:
    """Run a SQL query against the db and return a dataframe as a JSON string."""
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
    return df.write_json()

if __name__ == "__main__":
    server.run()