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
    assert config.root_dir == 'data'  # unchanged default


def test_dandi_defaults(monkeypatch):
    import sys
    from nwb_mcp_server.server import ServerConfig

    monkeypatch.setattr(sys, 'argv', ['prog'])
    config = ServerConfig()
    assert config.dandiset_id is None
    assert config.dandiset_version is None
    assert config.dandiset_path_filter is None


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


if __name__ == "__main__":
    pytest.main([__file__])