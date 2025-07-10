# modified from https://github.com/modelcontextprotocol/servers-archived/blob/main/src/sqlite/src/mcp_server_sqlite/server.py

import asyncio
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

@asyncio.asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage server startup and shutdown lifecycle."""
    server.settings.nwb_sources = upath.UPath(server.settings.root_dir).glob(server.nwb_file_glob)
    # Initialize the SQL connection to the NWB files
    sql_context = await asyncio.to_thread(lazynwb.get_sql_context, nwb_sources=server.settings.nwb_sources, full_path=True, disable_progress=True, eager=True)
    # TODO add server.settings.tables filtering when implemented in lazynwb
    try:
        yield AppContext(db=sql_context)
    finally:
        # Clean up on shutdown
        await asyncio.to_thread(lazynwb.clear_cache)

server = FastMCP(
    name="nwb-mcp-server",
    lifespan=server_lifespan,
    # settings:
    search_dir="C:/Users/ben.hardcastle/github/nwb-mcp-server/data",
    nwb_file_glob="*.nwb",
    tables=(),  
)


@server.tool()
async def get_tables(ctx: Context) -> list[str]:
    """Get the list of available tables in the NWB files."""
    ctx.info("Fetching available tables from NWB files")
    return await asyncio.to_thread(ctx.request_context.lifespan_context.db.tables)


if __name__ == "__main__":
    server.run()
