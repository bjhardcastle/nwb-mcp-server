"""
File Search MCP Server

An example MCP server that demonstrates different ways to pass parameters on startup.
This server searches for files in directories using glob patterns.
"""

import os
import argparse
from pathlib import Path
from typing import List
import glob

from mcp.server.fastmcp import FastMCP

# ========================================
# Method 1: Environment Variables
# ========================================

# Read configuration from environment variables
DEFAULT_SEARCH_DIRECTORY = os.environ.get("FILE_SEARCH_DIRECTORY", str(Path.home()))
DEFAULT_GLOB_PATTERN = os.environ.get("FILE_SEARCH_PATTERN", "*.txt")
MAX_RESULTS = int(os.environ.get("FILE_SEARCH_MAX_RESULTS", "100"))

# ========================================
# Method 2: Command Line Arguments
# ========================================

def parse_arguments():
    """Parse command line arguments for server configuration."""
    parser = argparse.ArgumentParser(description="File Search MCP Server")
    parser.add_argument(
        "--search-dir", 
        default=DEFAULT_SEARCH_DIRECTORY,
        help="Directory to search for files (default: %(default)s)"
    )
    parser.add_argument(
        "--pattern", 
        default=DEFAULT_GLOB_PATTERN,
        help="Glob pattern for file search (default: %(default)s)"
    )
    parser.add_argument(
        "--max-results", 
        type=int, 
        default=MAX_RESULTS,
        help="Maximum number of results to return (default: %(default)s)"
    )
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Search recursively in subdirectories"
    )
    return parser.parse_args()

# ========================================
# Method 3: Configuration Class
# ========================================

class ServerConfig:
    """Configuration class for the server."""
    
    def __init__(self, search_directory: str = None, glob_pattern: str = None, 
                 max_results: int = None, recursive: bool = False):
        self.search_directory = Path(search_directory or DEFAULT_SEARCH_DIRECTORY)
        self.glob_pattern = glob_pattern or DEFAULT_GLOB_PATTERN
        self.max_results = max_results or MAX_RESULTS
        self.recursive = recursive
        
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            search_directory=os.environ.get("FILE_SEARCH_DIRECTORY"),
            glob_pattern=os.environ.get("FILE_SEARCH_PATTERN"),
            max_results=int(os.environ.get("FILE_SEARCH_MAX_RESULTS", "100")),
            recursive=os.environ.get("FILE_SEARCH_RECURSIVE", "").lower() == "true"
        )

# ========================================
# Server Implementation
# ========================================

# Parse command line arguments
args = parse_arguments()

# Create configuration (you can choose which method to use)
config = ServerConfig(
    search_directory=args.search_dir,
    glob_pattern=args.pattern,
    max_results=args.max_results,
    recursive=args.recursive
)

# Create the MCP server
mcp = FastMCP("file-search")

@mcp.tool()
def search_files(pattern: str = None, directory: str = None, max_results: int = None) -> List[str]:
    """
    Search for files using glob patterns.
    
    Args:
        pattern: Glob pattern to search for (optional, uses server default if not provided)
        directory: Directory to search in (optional, uses server default if not provided)
        max_results: Maximum number of results (optional, uses server default if not provided)
    
    Returns:
        List of file paths matching the pattern
    """
    # Use provided arguments or fall back to server configuration
    search_pattern = pattern or config.glob_pattern
    search_dir = Path(directory or config.search_directory)
    max_res = max_results or config.max_results
    
    try:
        if not search_dir.exists():
            return [f"Error: Directory {search_dir} does not exist"]
        
        if not search_dir.is_dir():
            return [f"Error: {search_dir} is not a directory"]
        
        # Perform the search
        if config.recursive:
            # Use rglob for recursive search
            matches = list(search_dir.rglob(search_pattern))
        else:
            # Use glob for non-recursive search
            matches = list(search_dir.glob(search_pattern))
        
        # Convert to strings and limit results
        results = [str(match) for match in matches if match.is_file()]
        
        if len(results) > max_res:
            results = results[:max_res]
            results.append(f"... (showing first {max_res} of {len(matches)} results)")
        
        return results if results else ["No files found matching the pattern"]
        
    except Exception as e:
        return [f"Error searching files: {str(e)}"]

@mcp.tool()
def get_server_config() -> dict:
    """Get the current server configuration."""
    return {
        "search_directory": str(config.search_directory),
        "glob_pattern": config.glob_pattern,
        "max_results": config.max_results,
        "recursive": config.recursive
    }

@mcp.tool()
def list_directory(directory: str = None) -> List[str]:
    """
    List contents of a directory.
    
    Args:
        directory: Directory to list (optional, uses server default if not provided)
    
    Returns:
        List of items in the directory
    """
    target_dir = Path(directory or config.search_directory)
    
    try:
        if not target_dir.exists():
            return [f"Error: Directory {target_dir} does not exist"]
        
        if not target_dir.is_dir():
            return [f"Error: {target_dir} is not a directory"]
        
        items = []
        for item in target_dir.iterdir():
            if item.is_dir():
                items.append(f"📁 {item.name}/")
            else:
                items.append(f"📄 {item.name}")
        
        return sorted(items) if items else ["Directory is empty"]
        
    except Exception as e:
        return [f"Error listing directory: {str(e)}"]

# Add a resource to expose the search directory
@mcp.resource("file://search-directory")
def search_directory_info() -> str:
    """Information about the configured search directory."""
    return f"""
Search Directory Configuration:
- Directory: {config.search_directory}
- Default Pattern: {config.glob_pattern}
- Max Results: {config.max_results}
- Recursive: {config.recursive}
- Directory exists: {config.search_directory.exists()}
- Is directory: {config.search_directory.is_dir() if config.search_directory.exists() else 'N/A'}
"""

# ========================================
# Alternative: Creating server in a function
# ========================================

def create_server(search_dir: str, pattern: str, max_results: int = 100, recursive: bool = False) -> FastMCP:
    """
    Alternative approach: Create server with explicit parameters.
    This function can be called from other modules to create a configured server.
    """
    server_config = ServerConfig(search_dir, pattern, max_results, recursive)
    server = FastMCP("file-search-configured")
    
    @server.tool()
    def search(query_pattern: str = None) -> List[str]:
        """Search for files with the configured settings."""
        pattern_to_use = query_pattern or server_config.glob_pattern
        # Implementation would go here...
        return [f"Searching in {server_config.search_directory} for {pattern_to_use}"]
    
    return server

# ========================================
# Main execution (for CLI usage)
# ========================================

if __name__ == "__main__":
    # This allows the server to be run directly with: python file_search_server.py --search-dir /path --pattern "*.py"
    print(f"File Search Server starting with configuration:")
    print(f"  Search Directory: {config.search_directory}")
    print(f"  Glob Pattern: {config.glob_pattern}")

    
    # Run the server
    mcp.run()
