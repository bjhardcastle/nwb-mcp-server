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
        '--no_code', # flag should not need a value
    ])
    
    config = ServerConfig()
    assert config.root_dir == override_values['root_dir'], "CLI arguments not being parsed correctly"
    assert config.glob_pattern == override_values['glob_pattern'], "CLI arguments not being parsed correctly"
    assert config.no_code is True, "Boolean flag argument not being parsed correctly"
    
if __name__ == "__main__":
    pytest.main([__file__])