# /// script
# dependencies = [
#   "lazynwb",
#   "fastmcp",
# ]
# ///

import os

os.environ["RUST_BACKTRACE"] = "1"  # enable Rust backtraces for lazynwb

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import fnmatch
import importlib.metadata
import io
import logging
import threading
from collections.abc import AsyncIterator, Iterable
from typing import Any, cast

import fastmcp
import fsspec.config
import lazynwb
import lazynwb.dandi
import lazynwb.utils
import polars as pl
import pydantic
import pydantic_settings
import upath

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.info(
    f"Starting MCP NWB Server with lazynwb v{importlib.metadata.version('lazynwb')}"
)

DEFAULT_GLOB_PATTERN = "**/*.nwb"
NO_SOURCE_CONFIGURED_MESSAGE = (
    "No dataset is currently active. First select one with "
    "`use_local_source(root_dir=...)` or `use_dandiset_source(dandiset_id=...)`. "
    "If you control server startup config, you can also preset `--root_dir`/`--glob_pattern` "
    "or `--dandiset_id`."
)


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    """Describes the active data source for a session."""

    root_dir: str | None = None
    glob_pattern: str | None = None
    dandiset_id: str | None = None
    dandiset_version: str | None = None
    dandiset_path_filter: str | None = None

    def __post_init__(self) -> None:
        if self.root_dir is not None and not self.root_dir:
            raise ValueError("root_dir cannot be empty")
        if self.glob_pattern is not None and not self.glob_pattern:
            raise ValueError("glob_pattern cannot be empty")
        if self.dandiset_id is not None and not self.dandiset_id:
            raise ValueError("dandiset_id cannot be empty")
        if self.root_dir is not None and self.glob_pattern is None:
            object.__setattr__(self, "glob_pattern", DEFAULT_GLOB_PATTERN)
        if self.dandiset_id is None:
            if self.dandiset_version is not None:
                raise ValueError("dandiset_version requires dandiset_id")
            if self.dandiset_path_filter is not None:
                raise ValueError("dandiset_path_filter requires dandiset_id")

    @property
    def mode(self) -> str:
        if self.is_dandiset:
            return "dandiset"
        if self.is_filesystem:
            return "filesystem"
        return "unset"

    @property
    def is_dandiset(self) -> bool:
        return self.dandiset_id is not None

    @property
    def is_filesystem(self) -> bool:
        return self.root_dir is not None and not self.is_dandiset

    @property
    def is_configured(self) -> bool:
        return self.is_dandiset or self.is_filesystem

    def to_dict(self) -> dict[str, str | None]:
        return {
            "mode": self.mode,
            "root_dir": self.root_dir,
            "glob_pattern": self.glob_pattern,
            "dandiset_id": self.dandiset_id,
            "dandiset_version": self.dandiset_version,
            "dandiset_path_filter": self.dandiset_path_filter,
        }

    @classmethod
    def from_dandiset(
        cls,
        dandiset_id: str,
        dandiset_version: str | None = None,
        dandiset_path_filter: str | None = None,
    ) -> "SourceSpec":
        return cls(
            dandiset_id=dandiset_id,
            dandiset_version=dandiset_version,
            dandiset_path_filter=dandiset_path_filter,
        )

    @classmethod
    def from_local(
        cls, root_dir: str, glob_pattern: str = DEFAULT_GLOB_PATTERN
    ) -> "SourceSpec":
        return cls(root_dir=root_dir, glob_pattern=glob_pattern)


class ServerConfig(pydantic_settings.BaseSettings):
    """Configuration for the NWB MCP Server."""

    model_config = pydantic_settings.SettingsConfigDict(
        cli_parse_args=True,
        cli_implicit_flags=True,
        cli_ignore_unknown_args=True,
    )

    root_dir: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional root directory to search for NWB files. If omitted and no dandiset_id is"
            " set, the server starts without an active dataset and the user must choose one in chat."
        ),
    )
    glob_pattern: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional glob pattern to apply to `root-dir` to locate NWB files (or folders), e.g."
            " '*.zarr.nwb'. Defaults to '**/*.nwb' when root_dir is set."
        ),
    )
    dandiset_id: str | None = pydantic.Field(
        default=None,
        description="DANDI archive dandiset ID (e.g. '000363'). When set, overrides root_dir/glob_pattern and loads NWB files from DANDI. Note: anon has no effect for DANDI (presigned HTTPS URLs bypass fsspec S3).",
    )
    dandiset_version: str | None = pydantic.Field(
        default=None,
        description="Specific dandiset version (e.g. '0.231012.2129'). If None, uses the most recently published version.",
    )
    dandiset_path_filter: str | None = pydantic.Field(
        default=None,
        description="fnmatch-style pattern to filter DANDI assets by path, e.g. 'sub-619293/*.nwb' or 'sub-*/sub-*_ecephys.nwb'. Only used when dandiset_id is set.",
    )
    tables: list[str] = pydantic.Field(
        default_factory=list,
        description="List of table names to use, where each table is the name of a data object within each NWB file, e.g ['trials', 'units']",
    )
    infer_schema_length: int = pydantic.Field(
        default=1,
        description="Number of NWB files to scan to infer schema for all files",
    )
    anon: bool = pydantic.Field(
        default=False,
        description="Use anonymous S3 access via fsspec. Useful for accessing public S3 buckets without credentials.",
    )
    unattended: bool = pydantic.Field(
        default=False,
        description="Run the server in unattended mode, where it does not prompt the user for input. Useful for automated tasks.",
    )
    max_result_rows: int = pydantic.Field(
        default=50,
        description=(
            "Maximum number of rows returned by a SQL query. Keeps results within a manageable"
            " size for LLM context windows. Use allow_large_output on execute_query to bypass per-call."
        ),
    )
    ignored_args: pydantic_settings.CliUnknownArgs

    def default_source_spec(self) -> SourceSpec:
        if self.dandiset_id is not None:
            return SourceSpec.from_dandiset(
                dandiset_id=self.dandiset_id,
                dandiset_version=self.dandiset_version,
                dandiset_path_filter=self.dandiset_path_filter,
            )
        if self.root_dir is not None:
            return SourceSpec.from_local(
                root_dir=self.root_dir,
                glob_pattern=self.glob_pattern or DEFAULT_GLOB_PATTERN,
            )
        return SourceSpec()


@dataclasses.dataclass
class DatasetHandle:
    """Cached query handle for a concrete dataset selection."""

    source_spec: SourceSpec
    sources: list[upath.UPath]
    db: pl.SQLContext


class SourceManager:
    """Tracks the active source per MCP session and caches dataset handles."""

    def __init__(
        self,
        *,
        default_source: SourceSpec,
        infer_schema_length: int,
        tables: list[str],
    ) -> None:
        self.default_source = default_source
        self.infer_schema_length = infer_schema_length
        self.tables = tables
        self._dataset_cache: dict[SourceSpec, DatasetHandle] = {}
        self._session_sources: dict[str, SourceSpec] = {}
        self._lock = threading.RLock()

    def preload_default_dataset(self) -> DatasetHandle | None:
        if not self.default_source.is_configured:
            logger.info(
                "No startup dataset configured; sessions must select a source explicitly"
            )
            return None
        dataset = self._get_or_create_dataset(self.default_source)
        self.default_source = dataset.source_spec
        return dataset

    def get_active_source(self, session_id: str) -> SourceSpec:
        with self._lock:
            return self._session_sources.get(session_id, self.default_source)

    def get_active_dataset(self, session_id: str) -> DatasetHandle:
        active_source = self.get_active_source(session_id)
        if not active_source.is_configured:
            raise ValueError(NO_SOURCE_CONFIGURED_MESSAGE)
        return self._get_or_create_dataset(active_source)

    def set_active_source(
        self, session_id: str, source_spec: SourceSpec
    ) -> DatasetHandle:
        dataset = self._get_or_create_dataset(source_spec)
        with self._lock:
            self._session_sources[session_id] = dataset.source_spec
        return dataset

    def reset_active_source(self, session_id: str) -> DatasetHandle | None:
        with self._lock:
            self._session_sources.pop(session_id, None)
        if not self.default_source.is_configured:
            return None
        return self._get_or_create_dataset(self.default_source)

    def _get_or_create_dataset(self, requested_source: SourceSpec) -> DatasetHandle:
        with self._lock:
            dataset = self._dataset_cache.get(requested_source)
            if dataset is not None:
                return dataset

            dataset = _build_dataset_handle(
                requested_source,
                infer_schema_length=self.infer_schema_length,
                table_names=self.tables or None,
            )
            self._dataset_cache[dataset.source_spec] = dataset
            if dataset.source_spec != requested_source:
                self._dataset_cache[requested_source] = dataset
            return dataset


config = ServerConfig()  # type: ignore[call-arg]
logger.info(f"Configuration loaded: {config}")
DEFAULT_SOURCE = config.default_source_spec()

if config.anon:
    logger.info("Configuring fsspec for anonymous S3 access")
    fsspec.config.conf["s3"] = {"anon": True}

UNATTENDED_RULE = (
    (
        "Never prompt the user for input or clarification. Work entirely autonomously and make principled"
        "decisions based on the available data to achieve the most accurate and useful outcome."
    )
    if config.unattended
    else ""
)

RULES = f"""
<rules>
1. Always start in No Code Mode.
2. Highlight any assumptions you make.
3. Explain how you averaged or aggregated data, if applicable.
4. Do not make things up.
5. Do not provide an excessively positive picture. Be objective. Be critical where appropriate. 
6. {UNATTENDED_RULE}
</rules>
"""


def _build_code_mode_snippet_text(source_spec: SourceSpec) -> str:
    if not source_spec.is_configured:
        return (
            "First select a dataset with `use_local_source(root_dir=...)` or "
            "`use_dandiset_source(dandiset_id=...)`."
        )
    if source_spec.is_dandiset:
        return (
            "Use `nwb_file_search_code_snippet` to get the correct code for locating NWB files"
            f" from DANDI dandiset {source_spec.dandiset_id!r}"
            + (
                f" version {source_spec.dandiset_version!r}"
                if source_spec.dandiset_version
                else " (latest version)"
            )
            + ".\n   Presigned URLs expire after ~1 hour; re-run the snippet to refresh if needed."
        )
    return (
        "Use `upath` to search for NWB files:"
        f" `nwb_paths = list(upath.UPath({source_spec.root_dir!r}).glob({source_spec.glob_pattern!r}))`"
    )


def _build_about(source_spec: SourceSpec) -> str:
    if not source_spec.is_configured:
        return """
<mcp>
No dataset is currently selected.

Before using query tools, choose a dataset in chat with one of:
1. `use_local_source(root_dir='...')`
2. `use_dandiset_source(dandiset_id='000363')`

You can inspect the current selection with `get_active_source` and return to any preset startup
dataset with `reset_active_source`.
</mcp>
"""
    return f"""
<mcp>
The NWB MCP server provides tools for querying a read-only virtual database of NWB data, with two modes of
operation:
1. **No Code Mode**: The agent itself performs queries. This can be used for simple analyses on smaller datasets and for
gaining an initial understanding of the structure of the data.
2. **Code Mode**: The agent writes files in the workspace containing Python code. No Code Mode
should be used first to help plan the files that should be written.

<no_code_mode>
The agent can execute PostgreSQL queries against the NWB data using the `execute_query` tool.
a. Use standard SQL syntax, including `SELECT`, `FROM`, `WHERE`, `JOIN`, etc. and functions like `COUNT`, `AVG`, `SUM`, etc.
b. Do as much filtering as possible in the query to avoid loading all rows/columns into memory.
c. Avoid loading TimeSeries tables.
</no_code_mode>

<code_mode>
The agent can write files containing Python code to interact with the NWB files using the `lazynwb` package.
a. {_build_code_mode_snippet_text(source_spec)}
b. Use `lazynwb` to interact with tables in the NWB files:
   i. Get a polars `LazyFrame` for a table with `lazynwb.scan_nwb(nwb_paths, table_name)`.
   ii. Write queries against the `LazyFrame` using standard polars methods.
   iii. Use `.filter()` as appropriate to avoid loading all rows into memory.
   iv. Use `.select()` to choose specific columns to include in the final result and avoid loading
   unnecessary columns, particularly array- or list-like columns.
   v. Then use `.collect()` to execute the query and get a polars `DataFrame`.
</code_mode>

</mcp>
"""


def _build_nwb_file_search_code_snippet(source_spec: SourceSpec) -> str:
    if not source_spec.is_configured:
        raise ValueError(NO_SOURCE_CONFIGURED_MESSAGE)
    if source_spec.is_dandiset:
        filter_lines = (
            f"\nassets = [a for a in assets if fnmatch.fnmatch(a['path'], {source_spec.dandiset_path_filter!r})]"
            if source_spec.dandiset_path_filter
            else ""
        )
        return (
            f"import concurrent.futures\n"
            f"import fnmatch\n"
            f"import lazynwb.dandi\n"
            f"import upath\n"
            f"\n"
            f"dandiset_id = {source_spec.dandiset_id!r}\n"
            f"version = {source_spec.dandiset_version!r}  # None = latest\n"
            f"assets = lazynwb.dandi._get_dandiset_assets(dandiset_id, version=version)"
            f"{filter_lines}\n"
            f"executor = concurrent.futures.ThreadPoolExecutor()\n"
            f"nwb_paths = list(upath.UPath(url) for url in executor.map(\n"
            f"    lambda a: lazynwb.dandi._get_asset_s3_url(dandiset_id, a['asset_id'], version),\n"
            f"    assets,\n"
            f"))"
        )
    return f"nwb_paths = list(upath.UPath({source_spec.root_dir!r}).glob({source_spec.glob_pattern!r}))"


@dataclasses.dataclass
class AppContext:
    source_manager: SourceManager


def _get_app_context(ctx: fastmcp.Context) -> AppContext:
    request_context = ctx.request_context
    if request_context is None:
        raise RuntimeError("Request context is not available")
    return request_context.lifespan_context


def _get_dataset_for_request(ctx: fastmcp.Context) -> DatasetHandle:
    app_context = _get_app_context(ctx)
    return app_context.source_manager.get_active_dataset(ctx.session_id)


def _get_selected_source_for_request(ctx: fastmcp.Context) -> SourceSpec:
    app_context = _get_app_context(ctx)
    return app_context.source_manager.get_active_source(ctx.session_id)


def _get_active_source_for_request(ctx: fastmcp.Context) -> SourceSpec:
    selected_source = _get_selected_source_for_request(ctx)
    if not selected_source.is_configured:
        return selected_source
    return _get_dataset_for_request(ctx).source_spec


def _get_default_source_for_request(ctx: fastmcp.Context) -> SourceSpec:
    return _get_app_context(ctx).source_manager.default_source


def _format_dataset_status(
    dataset: DatasetHandle, *, default_source: SourceSpec
) -> dict[str, Any]:
    return {
        "active_source": dataset.source_spec.to_dict(),
        "is_default_source": dataset.source_spec == default_source,
        "source_count": len(dataset.sources),
    }


def _format_source_status(
    source_spec: SourceSpec,
    *,
    default_source: SourceSpec,
    source_count: int = 0,
) -> dict[str, Any]:
    return {
        "active_source": source_spec.to_dict(),
        "is_default_source": source_spec == default_source,
        "source_count": source_count,
    }


def _get_dandiset_sources(
    source_spec: SourceSpec,
) -> tuple[SourceSpec, list[upath.UPath]]:
    """Get NWB file sources from a DANDI dandiset via presigned S3 URLs."""
    assert source_spec.dandiset_id is not None
    dandiset_id = source_spec.dandiset_id
    version = source_spec.dandiset_version

    logger.info(
        f"Fetching assets from DANDI dandiset {dandiset_id!r}"
        + (f" version {version!r}" if version else " (latest version)")
    )
    if version is None:
        version = lazynwb.dandi._get_most_recent_dandiset_version(dandiset_id)
        logger.info(f"Resolved dandiset version: {version!r}")

    assets = lazynwb.dandi._get_dandiset_assets(dandiset_id, version=version)
    logger.info(f"Found {len(assets)} total assets in dandiset {dandiset_id!r}")

    if source_spec.dandiset_path_filter:
        original_count = len(assets)
        assets = [
            a
            for a in assets
            if fnmatch.fnmatchcase(a["path"], source_spec.dandiset_path_filter)
        ]
        logger.info(
            f"Filtered to {len(assets)} assets matching {source_spec.dandiset_path_filter!r}"
            f" (from {original_count})"
        )

    original_count = len(assets)
    assets = [a for a in assets if str(a["path"]).lower().endswith(".nwb")]
    logger.info(
        f"Filtered to {len(assets)} NWB assets ending in '.nwb' (from {original_count})"
    )

    if not assets:
        raise ValueError(
            f"No NWB assets found in dandiset {dandiset_id!r} version {version!r}"
            + (
                f" matching path filter {source_spec.dandiset_path_filter!r}"
                if source_spec.dandiset_path_filter
                else ""
            )
        )

    logger.info(f"Fetching presigned S3 URLs for {len(assets)} assets (parallel)")
    executor = lazynwb.utils.get_threadpool_executor()
    future_to_asset: dict[concurrent.futures.Future[str], dict] = {
        executor.submit(
            lazynwb.dandi._get_asset_s3_url, dandiset_id, asset["asset_id"], version
        ): asset
        for asset in assets
    }
    s3_urls: list[str] = []
    for future in concurrent.futures.as_completed(future_to_asset):
        asset = future_to_asset[future]
        try:
            s3_urls.append(future.result())
        except Exception as exc:
            logger.warning(
                f"Failed to get S3 URL for asset {asset.get('path')!r}: {exc!r}"
            )

    if not s3_urls:
        raise ValueError(
            f"Failed to retrieve any S3 URLs from dandiset {dandiset_id!r}"
        )

    logger.info(f"Retrieved {len(s3_urls)} S3 URLs from DANDI")
    resolved_source = dataclasses.replace(source_spec, dandiset_version=version)
    return resolved_source, [upath.UPath(url) for url in s3_urls]


def _get_local_or_remote_nwb_sources(
    source_spec: SourceSpec,
) -> tuple[SourceSpec, list[upath.UPath]]:
    """Get NWB files from the local/S3 filesystem via glob pattern."""
    if source_spec.root_dir is None or source_spec.glob_pattern is None:
        raise ValueError(NO_SOURCE_CONFIGURED_MESSAGE)
    logger.info(
        f"Searching for NWB files in {source_spec.root_dir!r} with pattern {source_spec.glob_pattern!r}"
    )
    nwb_paths = list(upath.UPath(source_spec.root_dir).glob(source_spec.glob_pattern))
    if not nwb_paths:
        raise ValueError(
            f"No NWB files found in {source_spec.root_dir!r} matching pattern {source_spec.glob_pattern!r}"
        )
    logger.info(f"Found {len(nwb_paths)} data sources")
    return source_spec, nwb_paths


def _get_nwb_sources(source_spec: SourceSpec) -> tuple[SourceSpec, list[upath.UPath]]:
    """Get NWB file sources, routing between DANDI and local/remote filesystem."""
    if source_spec.dandiset_id:
        if source_spec.root_dir is not None:
            logger.warning(
                f"Both dandiset_id={source_spec.dandiset_id!r} and root_dir={source_spec.root_dir!r} are set."
                " Using DANDI (root_dir/glob_pattern are ignored)."
            )
        return _get_dandiset_sources(source_spec)
    return _get_local_or_remote_nwb_sources(source_spec)


def create_sql_context_non_nwb(sources: Iterable[upath.UPath]) -> pl.SQLContext:
    """Create a SQLContext for non-NWB sources."""
    sources = tuple(sources)
    if not sources:
        raise ValueError("Must provide at least one source path")
    table_name_to_path = {p.stem: p for p in sources}
    if len(table_name_to_path) != len(sources):
        raise ValueError(
            "Duplicate source names found, please ensure unique names for each source"
        )

    suffix_to_read_function = {
        ".csv": "scan_csv",
        "": "scan_delta",  # Delta uses directory structure
        ".feather": "scan_ipc",
        ".json": "scan_ndjson",
        ".ndjson": "scan_ndjson",
        ".parquet": "scan_parquet",
        ".arrow": "scan_pyarrow_dataset",
    }
    frames = {}
    for table_name, path in table_name_to_path.items():
        ext = path.suffix.lower()
        if ext not in suffix_to_read_function:
            raise ValueError(f"Unsupported file extension: {ext}")
        if ext == "" and not path.is_dir():
            raise ValueError(
                f"Received a data source with no extension but is not a directory: {path}. Unsure how to continue."
            )
        read_func = getattr(pl, suffix_to_read_function[ext])
        frames[table_name] = read_func(path.as_posix())
    return pl.SQLContext(frames=frames, eager=False)


def _build_dataset_handle(
    source_spec: SourceSpec,
    *,
    infer_schema_length: int,
    table_names: list[str] | None,
) -> DatasetHandle:
    logger.info(f"Initializing SQL connection for source: {source_spec.to_dict()}")
    resolved_source_spec, sources = _get_nwb_sources(source_spec)
    if not source_spec.is_dandiset and not any(".nwb" in str(s).lower() for s in sources):
        logger.warning("No NWB files found, creating SQLContext for non-NWB sources")
        sql_context = create_sql_context_non_nwb(sources)
    else:
        sql_context = lazynwb.get_sql_context(
            nwb_sources=sources,
            infer_schema_length=infer_schema_length,
            table_names=table_names,
            exclude_timeseries=False,
            full_path=True,  # if False, NWB objects will be referenced by their names, not full paths, e.g. 'epochs' instead of '/intervals/epochs'
            disable_progress=True,
            eager=False,  # if False, a LazyFrame is returned from `execute`
            rename_general_metadata=True,  # renames the metadata in 'general' as 'session'
        )
    logger.info("SQL connection initialized successfully")
    return DatasetHandle(
        source_spec=resolved_source_spec, sources=sources, db=sql_context
    )


@contextlib.asynccontextmanager
async def server_lifespan(server: fastmcp.FastMCP) -> AsyncIterator[AppContext]:
    """Manage server startup and shutdown lifecycle."""
    source_manager = SourceManager(
        default_source=DEFAULT_SOURCE,
        infer_schema_length=config.infer_schema_length,
        tables=config.tables,
    )
    logger.info("Initializing startup dataset configuration")
    source_manager.preload_default_dataset()
    try:
        yield AppContext(source_manager=source_manager)
    finally:
        await asyncio.to_thread(lazynwb.clear_cache)


server = fastmcp.FastMCP(
    name="nwb-mcp-server",
    lifespan=server_lifespan,
    # instructions=ABOUT + RULES,
    # instructions can be passed as a string, but does not appear to influence the agent (VScode)
)


@server.tool()
def get_tables(ctx: fastmcp.Context) -> list[str]:
    """Returns a list of available tables in the NWB files. Each table includes data from multiple
    NWB files, where one file includes data from a single session for a single subject.
    Note: NWB's `general` metadata group is renamed to `session` by this server, so it may not be
    recognized as a metadata table from the name alone. Check it early when you need
    experiment-level context (e.g. `experiment_description`, subject info).
    """
    logger.info("Fetching available tables from NWB files")
    return _get_dataset_for_request(ctx).db.tables()


@server.tool()
def get_table_schema(table_name: str, ctx: fastmcp.Context) -> dict[str, pl.DataType]:
    """Returns the schema of a specific table, mapping column names to their data types.
    Schemas are normalized across all files: a column present in the schema may still be null
    for many files, and a null column in one file doesn't mean the data is absent across all files.
    """
    if not table_name:
        raise ValueError("Table name cannot be empty")
    logger.info(f"Fetching schema for table: {table_name}")
    query = f"SELECT * FROM {format_table_name(table_name)} LIMIT 0"
    lf = cast(pl.LazyFrame, _get_dataset_for_request(ctx).db.execute(query))
    return lf.schema


@server.tool()
def get_active_source(ctx: fastmcp.Context) -> dict[str, Any]:
    """Returns the session's currently active dataset configuration."""
    source_spec = _get_selected_source_for_request(ctx)
    default_source = _get_default_source_for_request(ctx)
    if not source_spec.is_configured:
        return _format_source_status(source_spec, default_source=default_source)
    dataset = _get_dataset_for_request(ctx)
    return _format_dataset_status(dataset, default_source=default_source)


@server.tool()
def use_local_source(
    root_dir: str,
    ctx: fastmcp.Context,
    glob_pattern: str = "**/*.nwb",
) -> dict[str, Any]:
    """Switch the current chat session to a local or remote filesystem dataset."""
    dataset = _get_app_context(ctx).source_manager.set_active_source(
        ctx.session_id,
        SourceSpec.from_local(root_dir=root_dir, glob_pattern=glob_pattern),
    )
    return _format_dataset_status(
        dataset, default_source=_get_default_source_for_request(ctx)
    )


@server.tool()
def use_dandiset_source(
    dandiset_id: str,
    ctx: fastmcp.Context,
    dandiset_version: str | None = None,
    dandiset_path_filter: str | None = None,
) -> dict[str, Any]:
    """Switch the current chat session to a DANDI dandiset."""
    dataset = _get_app_context(ctx).source_manager.set_active_source(
        ctx.session_id,
        SourceSpec.from_dandiset(
            dandiset_id=dandiset_id,
            dandiset_version=dandiset_version,
            dandiset_path_filter=dandiset_path_filter,
        ),
    )
    return _format_dataset_status(
        dataset, default_source=_get_default_source_for_request(ctx)
    )


@server.tool()
def reset_active_source(ctx: fastmcp.Context) -> dict[str, Any]:
    """Reset the current chat session back to the startup-configured dataset."""
    dataset = _get_app_context(ctx).source_manager.reset_active_source(ctx.session_id)
    if dataset is None:
        return _format_source_status(
            _get_selected_source_for_request(ctx),
            default_source=_get_default_source_for_request(ctx),
        )
    return _format_dataset_status(
        dataset, default_source=_get_default_source_for_request(ctx)
    )


@server.tool()
def nwb_file_search_code_snippet(ctx: fastmcp.Context) -> str:
    """Returns a code snippet for finding the user's NWB files. `upath` is a 3rd-party package that implements the pathlib
    interface for local and cloud storage: when installing, its name on PyPI is `universal-pathlib`.
    For DANDI datasets, the snippet uses `lazynwb.dandi` to fetch presigned S3 URLs.
    """
    return _build_nwb_file_search_code_snippet(_get_active_source_for_request(ctx))


@server.tool()
async def execute_query(
    query: str,
    ctx: fastmcp.Context,
    allow_large_output: bool = False,
) -> str:
    """Executes a SQL query against a virtual read-only NWB database,
    returning results as JSON. Uses PostgreSQL syntax and functions for basic analysis.

    Results are capped at max_result_rows (default 50). Refine queries with WHERE/GROUP BY/LIMIT
    to stay within this limit.

    Set allow_large_output=True ONLY when the result will be written to a file or piped to an
    external tool — do NOT use it for inline analysis, as large results will fill the context window.

    Tables are lazily loaded — aggregations like COUNT(*) or COUNT(DISTINCT ...) force full
    materialization and can be slow on large datasets. When selecting array/list columns
    (e.g. spike_times, waveform_mean), always pre-filter rows using scalar columns first.
    """
    return await _execute_query(query, ctx, allow_large_output=allow_large_output)


def format_table_name(table: str) -> str:
    """Ensures the table name is properly formatted for pl.SQLContext queries."""
    if not table:
        raise ValueError("Table name cannot be empty")
    return f'"{table}"'


def format_column_names(columns: Iterable[str] | None) -> str:
    """Formats column names for SQL queries, ensuring they are properly quoted."""
    if not columns:
        return "*"
    if isinstance(columns, str):
        columns = [columns]
    return ", ".join(repr(c) for c in columns).replace("'", '"')


@server.tool()
async def preview_table_values(
    table: str,
    ctx: fastmcp.Context,
    columns: Iterable[str] | None = None,
    n_rows: int = 1,
) -> str:
    """Returns the first row of a table to preview values. Prefer `get_table_schema` to get the
    table schema. Only use this if absolutely necessary — even a 1-row preview can be slow on
    remote/cloud NWB files."""
    column_query = format_column_names(columns)
    query = f"SELECT {column_query} FROM {format_table_name(table)} LIMIT {n_rows};"
    logger.info(f"Previewing table values with: {column_query}")
    return await _execute_query(query, ctx)


async def _execute_query(
    query: str, ctx: fastmcp.Context, allow_large_output: bool = False
) -> str:
    """Executes a SQL query against a virtual read-only NWB database,
    returning results as JSON. Uses PostgreSQL syntax and functions for basic analysis.
    """
    if not query:
        raise ValueError("SQL query cannot be empty")
    logger.info(f"Executing query: {query}")
    df: pl.DataFrame = _get_dataset_for_request(ctx).db.execute(query, eager=True)
    if df.is_empty():
        logger.warning("SQL query returned no results")
        return "[]"
    if not allow_large_output and df.shape[0] > config.max_result_rows:
        raise ValueError(
            f"Query returned {df.shape[0]} rows, exceeding the {config.max_result_rows}-row limit. "
            "Refine with WHERE/GROUP BY/LIMIT, or set allow_large_output=True only if the result "
            "will be written to a file (not read into context)."
        )
    logger.info(f"Query executed successfully, serializing {len(df)} rows as JSON")
    # return _to_markdown(df)
    if "obs_intervals" in df.columns:
        # lists of arrays cause JSON conversion to crash
        # arrays are ok
        # TODO report issue to polars
        # TODO generalize to cast all list[array] columns
        df = df.cast({"obs_intervals": pl.List(pl.List(pl.Float64))})
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
def nwb_paths(ctx: fastmcp.Context) -> list[str]:
    """List the available NWB files."""
    return [p.as_posix() for p in _get_dataset_for_request(ctx).sources]


@server.tool()
def get_nwb_paths(ctx: fastmcp.Context) -> list[str]:
    """Returns the list of all NWB file paths in the dataset (one file per session/subject).
    Call early to understand the scale of the dataset — file count shapes how to interpret
    aggregate query results (e.g. per-file breakdowns vs. overall averages).
    """
    return [p.as_posix() for p in _get_dataset_for_request(ctx).sources]


@server.prompt
def analysis_report_prompt(query: str, ctx: fastmcp.Context) -> str:
    """User prompt for creating an analysis report."""
    if not query:
        raise ValueError("Query cannot be empty")

    return f"""
Please provide an analysis report for a scientist posing the following query:
{query!r}

{RULES}

<instructions>
1. Explore the available tables in No Code Mode to understand their schemas and relationships.
2. Develop a step-by-step plan for the analysis, prompting the user for any necessary clarifications.
    i. if the analysis is simple, you can run it in No Code Mode. 
    ii. if the analysis is complex (ie. requires custom functions, data processing, or may take a long
        time to run), switch to Code Mode and create Python code that:
        a. fetches data
        b. runs any required pre-processing
        c. generates relevant visualizations
        d. performs any statistical analyses
        e. summarizes the results
    If the analysis is expected to take a long time, incorporate a test mode that runs the
    analysis on a small subset of the data first. The user can then run the full analysis offline.
3. Execute the analysis and provide a detailed report of the findings.
</instructions>

{_build_about(_get_active_source_for_request(ctx))}
"""


@server.prompt
def general_prompt(query: str, ctx: fastmcp.Context) -> str:
    """User prompt for general queries."""
    if not query:
        raise ValueError("Query cannot be empty")

    return f"""
Please provide a detailed response to a scientist posing the following query:
{query!r}

{RULES}

<instructions>
1. Explore the available tables in No Code Mode to understand their schemas and relationships.
2. Provide a comprehensive answer to the query, including relevant data and insights.
3. If the query requires complex analysis, switch to Code Mode and create Python code that:
    i. fetches data
    ii. runs any required pre-processing
    iii. generates relevant visualizations
    iv. performs any statistical analyses
    v. summarizes the results
    If the analysis is expected to take a long time, incorporate a test mode that runs the
    analysis on a small subset of the data first. The user can then run the full analysis offline.
</instructions>

{_build_about(_get_active_source_for_request(ctx))}                 
"""


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
