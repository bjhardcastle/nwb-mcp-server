import pytest


def test_import_module():
    """Test that the server module can be imported without errors."""
    try:
        import nwb_mcp_server.server
    except ImportError as e:
        pytest.fail(f"Failed to import nwb_mcp_server.server: {e}")


def test_cli_args(monkeypatch):
    import sys
    from nwb_mcp_server.server import ServerConfig

    override_values = {
        'root_dir': 'mydata',
        'glob_pattern': '*.zarr.nwb',
    }
    monkeypatch.setattr(sys, 'argv', [
        'prog',
        '--root_dir', override_values['root_dir'],
        '--glob_pattern', override_values['glob_pattern'],
    ])

    config = ServerConfig()
    assert config.root_dir == override_values['root_dir'], "CLI arguments not being parsed correctly"
    assert config.glob_pattern == override_values['glob_pattern'], "CLI arguments not being parsed correctly"


def test_dandi_cli_args(monkeypatch):
    import sys
    from nwb_mcp_server.server import ServerConfig

    monkeypatch.setattr(sys, 'argv', [
        'prog',
        '--dandiset_id', '000363',
        '--dandiset_version', '0.231012.2129',
        '--dandiset_path_filter', 'sub-*/sub-*_ecephys.nwb',
    ])
    config = ServerConfig()
    assert config.dandiset_id == '000363'
    assert config.dandiset_version == '0.231012.2129'
    assert config.dandiset_path_filter == 'sub-*/sub-*_ecephys.nwb'
    assert config.root_dir is None
    assert config.glob_pattern is None


def test_dandi_defaults(monkeypatch):
    import sys
    from nwb_mcp_server.server import ServerConfig

    monkeypatch.setattr(sys, 'argv', ['prog'])
    config = ServerConfig()
    assert config.dandiset_id is None
    assert config.dandiset_version is None
    assert config.dandiset_path_filter is None
    assert config.root_dir is None
    assert config.glob_pattern is None


def test_fnmatch_dandi_path_filter():
    import fnmatch

    assets = [
        {"path": "sub-619293/sub-619293_ses-1184980079_ecephys.nwb", "asset_id": "a"},
        {"path": "sub-619293/sub-619293_ses-1184980079_behavior.nwb", "asset_id": "b"},
        {"path": "sub-000001/sub-000001_ses-abc_ecephys.nwb", "asset_id": "c"},
    ]
    pattern = "sub-*/sub-*_ecephys.nwb"
    filtered = [a for a in assets if fnmatch.fnmatchcase(a["path"], pattern)]
    assert len(filtered) == 2
    assert all("ecephys" in a["path"] for a in filtered)


def test_source_spec_rejects_dandi_options_without_id():
    from nwb_mcp_server.server import SourceSpec

    unconfigured = SourceSpec()
    assert unconfigured.mode == "unset"
    assert unconfigured.is_configured is False

    with pytest.raises(ValueError, match="dandiset_id cannot be empty"):
        SourceSpec(dandiset_id="")

    with pytest.raises(ValueError, match="dandiset_version requires dandiset_id"):
        SourceSpec(dandiset_version="0.231012.2129")

    with pytest.raises(ValueError, match="dandiset_path_filter requires dandiset_id"):
        SourceSpec(dandiset_path_filter="sub-*/sub-*_ecephys.nwb")


def test_default_source_spec_can_be_unconfigured(monkeypatch):
    import sys
    from nwb_mcp_server.server import ServerConfig

    monkeypatch.setattr(sys, 'argv', ['prog'])
    config = ServerConfig()

    default_source = config.default_source_spec()
    assert default_source.mode == "unset"
    assert default_source.is_configured is False


def test_source_manager_caches_datasets_and_isolates_sessions(monkeypatch):
    import importlib
    from dataclasses import replace

    server_module = importlib.import_module("nwb_mcp_server.server")

    build_calls = []

    def fake_build_dataset_handle(source_spec, *, infer_schema_length, table_names):
        build_calls.append((source_spec, infer_schema_length, table_names))
        resolved_source = source_spec
        if source_spec.dandiset_id and source_spec.dandiset_version is None:
            resolved_source = replace(source_spec, dandiset_version="resolved-version")
        source_name = resolved_source.dandiset_id or resolved_source.root_dir
        return server_module.DatasetHandle(
            source_spec=resolved_source,
            sources=[server_module.upath.UPath(f"memory://{source_name}/file.nwb")],
            db={"source": source_name},
        )

    monkeypatch.setattr(server_module, "_build_dataset_handle", fake_build_dataset_handle)

    default_source = server_module.SourceSpec.from_local(root_dir="data")
    manager = server_module.SourceManager(
        default_source=default_source,
        infer_schema_length=3,
        tables=["units"],
    )

    default_dataset = manager.get_active_dataset("session-a")
    assert default_dataset.source_spec == default_source
    assert manager.get_active_source("session-b") == default_source

    dandi_request = server_module.SourceSpec.from_dandiset("000363")
    dandi_dataset = manager.set_active_source("session-a", dandi_request)
    assert dandi_dataset.source_spec.dandiset_version == "resolved-version"
    assert manager.get_active_source("session-a").dandiset_version == "resolved-version"
    assert manager.get_active_source("session-b") == default_source

    second_session_dataset = manager.set_active_source("session-b", dandi_request)
    assert second_session_dataset is dandi_dataset

    reset_dataset = manager.reset_active_source("session-a")
    assert reset_dataset is default_dataset
    assert manager.get_active_source("session-a") == default_source

    assert len(build_calls) == 2
    assert build_calls[0] == (default_source, 3, ["units"])
    assert build_calls[1] == (dandi_request, 3, ["units"])


def test_source_manager_without_default_source_requires_chat_selection(monkeypatch):
    import importlib

    server_module = importlib.import_module("nwb_mcp_server.server")

    def fake_build_dataset_handle(source_spec, *, infer_schema_length, table_names):
        return server_module.DatasetHandle(
            source_spec=source_spec,
            sources=[server_module.upath.UPath("memory://chosen/file.nwb")],
            db={"source": "chosen"},
        )

    monkeypatch.setattr(server_module, "_build_dataset_handle", fake_build_dataset_handle)

    manager = server_module.SourceManager(
        default_source=server_module.SourceSpec(),
        infer_schema_length=1,
        tables=[],
    )

    assert manager.preload_default_dataset() is None
    assert manager.get_active_source("session-a").mode == "unset"
    assert manager.reset_active_source("session-a") is None

    with pytest.raises(ValueError, match="No dataset is currently active"):
        manager.get_active_dataset("session-a")

    dataset = manager.set_active_source(
        "session-a",
        server_module.SourceSpec.from_local(root_dir="mydata"),
    )
    assert dataset.source_spec.root_dir == "mydata"


def test_build_nwb_file_search_code_snippet_uses_active_source_details():
    from nwb_mcp_server.server import SourceSpec, _build_nwb_file_search_code_snippet

    dandi_snippet = _build_nwb_file_search_code_snippet(
        SourceSpec.from_dandiset(
            dandiset_id="000363",
            dandiset_version="0.231012.2129",
            dandiset_path_filter="sub-*/sub-*_ecephys.nwb",
        )
    )
    assert "dandiset_id = '000363'" in dandi_snippet
    assert "version = '0.231012.2129'" in dandi_snippet
    assert "fnmatch.fnmatch" in dandi_snippet

    local_snippet = _build_nwb_file_search_code_snippet(
        SourceSpec.from_local(root_dir="mydata", glob_pattern="*.zarr.nwb")
    )
    assert "upath.UPath('mydata').glob('*.zarr.nwb')" in local_snippet

    with pytest.raises(ValueError, match="No dataset is currently active"):
        _build_nwb_file_search_code_snippet(SourceSpec())


def test_get_dandiset_sources_filters_to_nwb_assets(monkeypatch):
    import importlib

    server_module = importlib.import_module("nwb_mcp_server.server")

    monkeypatch.setattr(
        server_module.lazynwb.dandi,
        "_get_dandiset_assets",
        lambda dandiset_id, version=None: [
            {"path": "sub-1/sub-1_ecephys.nwb", "asset_id": "nwb-1"},
            {"path": "README.md", "asset_id": "readme"},
            {"path": "sub-1/sub-1_ecephys.zarr", "asset_id": "zarr"},
        ],
    )
    monkeypatch.setattr(
        server_module.lazynwb.dandi,
        "_get_asset_s3_url",
        lambda dandiset_id, asset_id, version=None: f"https://example.test/{asset_id}",
    )

    resolved_source, sources = server_module._get_dandiset_sources(
        server_module.SourceSpec.from_dandiset(
            dandiset_id="000363",
            dandiset_version="0.231012.2129",
        )
    )

    assert resolved_source.dandiset_version == "0.231012.2129"
    assert [str(path) for path in sources] == ["https://example.test/nwb-1"]


def test_build_dataset_handle_treats_dandiset_blob_urls_as_nwb(monkeypatch):
    import importlib

    server_module = importlib.import_module("nwb_mcp_server.server")

    requested_source = server_module.SourceSpec.from_dandiset(
        dandiset_id="000363",
        dandiset_version="0.231012.2129",
    )
    opaque_blob_url = "https://dandiarchive.s3.amazonaws.com/blobs/45d/3e8/blob-id"

    monkeypatch.setattr(
        server_module,
        "_get_nwb_sources",
        lambda source_spec: (source_spec, [server_module.upath.UPath(opaque_blob_url)]),
    )

    captured = {}

    def fake_get_sql_context(**kwargs):
        captured["kwargs"] = kwargs
        return {"kind": "nwb"}

    def fail_non_nwb(_sources):
        raise AssertionError("Should not create a non-NWB SQL context for DANDI blob URLs")

    monkeypatch.setattr(server_module.lazynwb, "get_sql_context", fake_get_sql_context)
    monkeypatch.setattr(server_module, "create_sql_context_non_nwb", fail_non_nwb)

    dataset = server_module._build_dataset_handle(
        requested_source,
        infer_schema_length=2,
        table_names=["units"],
    )

    assert dataset.db == {"kind": "nwb"}
    assert captured["kwargs"]["nwb_sources"] == [server_module.upath.UPath(opaque_blob_url)]
    assert captured["kwargs"]["infer_schema_length"] == 2
    assert captured["kwargs"]["table_names"] == ["units"]


if __name__ == "__main__":
    pytest.main([__file__])
