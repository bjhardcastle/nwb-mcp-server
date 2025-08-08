# NWB Model Context Protocol (MCP) Server

An MCP server for accessing NWB (Neurodata Without Borders) files, providing AI agents easy access to neuroscience data.

# Features
- 🚀 Rapid exploration of new datasets
- 🗂️ Automatically-generated analysis reports
- 🧠 Prompt templates instruct agents to get the most from the tools
- 💡 "No Code" mode allows analysis without modifying the local filesystem
- ⚡️ Uses [lazynwb](https://github.com/NeurodataWithoutBorders/lazynwb) for efficient data access across multiple NWB files
- ☁️ Supports local and cloud data (e.g. on AWS S3)
- 🔒 Read-only access to NWB data
- 🛠️ Easy setup

## Requirements

### 1. uv
[`uv`](https://github.com/astral-sh/uv#readme) is used to run the server with the required dependencies, in an isolated virtual environment.

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for platform-specific instructions for a system-wide installation.

Alternatively, install with pip in your system Python environment:

```sh
pip install uv
```
### 2. Copilot Chat extension 
Available on the [VS Code extensions marketplace]
(https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat)

Similar extensions, such as Cline, may also work.

## Add server to Copilot Chat in VS Code

![Adding MCP to Copilot Chat](docs/resources/vscode_mcp_json.gif)

- ensure MCP is enabled in settings
- the first startup may take a few seconds as it downloads packages

### Configuration Parameters

| Parameter            | Description                                                                                      | Default        |
|----------------------|--------------------------------------------------------------------------------------------------|----------------|
| root_dir             | Root directory to search for NWB files (forward slash ok on Windows)                             | `"data"`        |
| glob_pattern         | A glob pattern to apply to `root_dir` to locate NWB files. Use `"**/*.nwb"` for sub-directories. | `"**/*.nwb"`    |
| tables               | Restrict the list of NWB tables to use, e.g. `["trials", "units"]`                            | `null`     |
| infer_schema_length  | Number of NWB files to scan to infer schema for all files                                        | `1`             |
| unattended           | Run the server in unattended mode (no user prompts, for automation)                              | `false`          |
| table_element_limit  | Max elements (columns x rows) allowed in a table returned by a SQL query                         | `500`            |