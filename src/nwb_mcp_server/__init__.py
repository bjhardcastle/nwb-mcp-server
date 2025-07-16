import argparse


def parse_args():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description='NWB MCP Server')
    parser.add_argument(
        '--search-dir', default="C:/Users/ben.hardcastle/github/nwb-mcp-server/data", help='Root directory to search for NWB files',
    )
    parser.add_argument(
        '--nwb-file-glob', default="*.nwb", help='A glob pattern to apply to search-dir to locate NWB files (or folders), e.g. "*.zarr"',
    )
    parser.add_argument('--tables', nargs='*', default=(), help='Optional list of tables to use')

    args = parser.parse_args()
    print(args)
    # asyncio.run(
    #     server.main(
    #         root_dir=args.root_dir,
    #         nwb_file_glob=args.nwb_file_glob,
    #         tables=args.tables,
    #     )
    # )

# Optionally expose other important items at package level
__all__ = ["main", "server"]
