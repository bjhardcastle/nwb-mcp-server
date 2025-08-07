# NWB Model Context Protocol (MCP) Server

An MCP server for accessing NWB (Neurodata Without Borders) files, providing AI agents access to neuroscience data.

# Features
- 🚀 Rapid exploration of new datasets.
- 🧠 Prompt templates instruct agents to get the most from the tools.
- 💡 "No Code" mode allows analysis without modifying the local filesystem.
- ⚡️ Uses [lazynwb](https://github.com/NeurodataWithoutBorders/lazynwb) for efficient data access across multiple NWB files.
- ☁️ Supports local and cloud data (e.g. on AWS S3).
- 🔒 Read-only access to NWB data.
- 🛠️ Easy setup.

## Requirements

[uv](https://github.com/astral-sh/uv) is used to run the server with required dependencies.


See the [uv installation guide](https://github.com/astral-sh/uv#installation) for platform-specific instructions for a system-wide installation.

Alternatively, install with pip in your system Python environment:

```sh
pip install uv
```

## Configure Copilot Chat

To make the server available in VS Code, create a configuration file at `.vscode/mcp.json` in your
project's root directory.

### Example `.vscode/mcp.json`

```json
{
    "servers": {
        "nwb": { // the name Copilot will use to refer to this server: can be customized
            "command": "uvx",
            "args": [
                "nwb-mcp-server @ git+https://github.com/bjhardcastle/nwb-mcp-server", // will be distributed on PyPI soon
                "--root-dir", "data",       // local or cloud directory containing your NWB files
                "--glob-pattern", "*.nwb"   // pattern to match files (can be applied recursively with `**/*.nwb`)
            ]
        }
    }
}
```