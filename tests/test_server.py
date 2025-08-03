import pytest

def test_import_module():
    """Test that the server module can be imported without errors."""
    try:
        import nwb_mcp_server.server
    except ImportError as e:
        pytest.fail(f"Failed to import nwb_mcp_server.server: {e}")