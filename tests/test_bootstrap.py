"""
Tests for Bootstrap and Main Entry Point (main.py)

Tests cover:
- Python version checking
- Virtual environment detection and creation
- Dependency installation checking
- Bootstrap flow
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Note: We import main.py functions directly for unit testing
# Integration tests would run the actual script


@pytest.mark.unit
class TestPythonVersionCheck:
    """Test Python version validation."""
    
    def test_check_python_version_valid(self):
        """Test that current Python version passes check."""
        # Import and execute module
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # This should not raise
        try:
            main_module.check_python_version()
        except SystemExit:
            pytest.fail("check_python_version() should not exit with valid Python")
    
    def test_check_python_version_invalid(self, monkeypatch):
        """Test that old Python version fails check."""
        # Create a mock version_info with proper structure
        from collections import namedtuple
        VersionInfo = namedtuple('version_info', ['major', 'minor', 'micro', 'releaselevel', 'serial'])
        mock_version = VersionInfo(3, 7, 0, 'final', 0)
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        
        # Mock version_info in the main module's context
        monkeypatch.setattr("sys.version_info", mock_version)
        
        spec.loader.exec_module(main_module)
        
        # Should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            main_module.check_python_version()
        
        assert exc_info.value.code == 1


@pytest.mark.unit
class TestVirtualEnvironment:
    """Test virtual environment detection and paths."""
    
    def test_get_venv_python_unix(self, monkeypatch):
        """Test venv Python path on Unix systems."""
        monkeypatch.setattr(sys, "platform", "linux")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        venv_python = main_module.get_venv_python()
        
        assert venv_python.name == "python"
        assert "bin" in str(venv_python)
    
    def test_get_venv_python_windows(self, monkeypatch):
        """Test venv Python path on Windows."""
        monkeypatch.setattr(sys, "platform", "win32")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        venv_python = main_module.get_venv_python()
        
        assert venv_python.name == "python.exe"
        assert "Scripts" in str(venv_python)
    
    def test_is_in_venv_false(self, monkeypatch):
        """Test venv detection when not in venv."""
        # Remove venv indicators
        if hasattr(sys, 'real_prefix'):
            monkeypatch.delattr(sys, 'real_prefix')
        
        monkeypatch.setattr(sys, 'prefix', '/usr')
        monkeypatch.setattr(sys, 'base_prefix', '/usr')
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        assert main_module.is_in_venv() is False
    
    def test_is_in_venv_true_with_base_prefix(self, monkeypatch):
        """Test venv detection with base_prefix."""
        monkeypatch.setattr(sys, 'prefix', '/path/to/venv')
        monkeypatch.setattr(sys, 'base_prefix', '/usr')
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        assert main_module.is_in_venv() is True


@pytest.mark.unit
class TestDependencyChecking:
    """Test dependency installation checking."""
    
    def test_check_dependencies_installed_missing_venv(self, temp_dir, monkeypatch):
        """Test dependency check when venv doesn't exist."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Point to nonexistent venv
        fake_venv = temp_dir / ".venv_nonexistent"
        monkeypatch.setattr(main_module, "VENV_DIR", fake_venv)
        
        assert main_module.check_dependencies_installed() is False
    
    @pytest.mark.slow
    def test_check_dependencies_installed_with_packages(self, temp_dir, monkeypatch):
        """Test dependency check with mock installed packages."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Mock subprocess to return package list
        import subprocess
        
        def mock_run(*args, **kwargs):
            result = MagicMock()
            result.stdout = "chromadb==0.4.0\nlangchain==0.1.0\nrich==13.0.0"
            result.returncode = 0
            return result
        
        monkeypatch.setattr(subprocess, "run", mock_run)
        
        # Create fake venv python
        fake_venv = temp_dir / ".venv"
        fake_venv.mkdir()
        (fake_venv / "bin").mkdir()
        fake_python = fake_venv / "bin" / "python"
        fake_python.write_text("#!/bin/sh\necho mock")
        
        monkeypatch.setattr(main_module, "VENV_DIR", fake_venv)
        
        assert main_module.check_dependencies_installed() is True


@pytest.mark.integration
class TestBootstrapIntegration:
    """Integration tests for bootstrap process."""
    
    @pytest.mark.slow
    def test_bootstrap_creates_venv(self, temp_dir, monkeypatch):
        """Test that bootstrap creates virtual environment."""
        # This is a slow integration test
        # In practice, you'd mock subprocess calls or use a fixture
        # For now, this is a placeholder
        pass
    
    @pytest.mark.slow
    def test_bootstrap_installs_dependencies(self, temp_dir, monkeypatch):
        """Test that bootstrap installs dependencies."""
        # This is a slow integration test
        pass


# ============================================================================
# Command-line Argument Tests
# ============================================================================

@pytest.mark.unit
class TestCommandLineArgs:
    """Test command-line argument handling."""
    
    def test_reset_flag_removes_venv(self, temp_dir, monkeypatch):
        """Test that --reset flag removes venv."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Create fake venv
        fake_venv = temp_dir / ".venv"
        fake_venv.mkdir()
        (fake_venv / "test.txt").write_text("test")
        
        monkeypatch.setattr(main_module, "VENV_DIR", fake_venv)
        monkeypatch.setattr(sys, "argv", ["main.py", "--reset"])
        
        # Should exit after removing venv
        with pytest.raises(SystemExit) as exc_info:
            main_module.handle_reset()
        
        assert exc_info.value.code == 0
        assert not fake_venv.exists()
