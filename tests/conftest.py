"""
Pytest Configuration and Shared Fixtures

This module contains pytest fixtures that are shared across all test files.
Fixtures defined here are automatically available in all test modules.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator
import pytest

from src.config import Config


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.
    
    Yields:
        Path to temporary directory
        
    Cleanup:
        Automatically removed after test
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_config_dir(temp_dir: Path, monkeypatch) -> Path:
    """
    Create a temporary directory and set it as the config location.
    
    Args:
        temp_dir: Temporary directory fixture
        monkeypatch: Pytest monkeypatch fixture
        
    Returns:
        Path to temporary config directory
    """
    # Monkey patch the config module to use temp directory
    import src.config as config_module
    
    config_file = temp_dir / "config.yaml"
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file)
    monkeypatch.setattr(config_module, "DATA_DIR", data_dir)
    
    return temp_dir


# ============================================================================
# Sample Directory Structure Fixtures
# ============================================================================

@pytest.fixture
def sample_study_structure(temp_dir: Path) -> dict[str, Path]:
    """
    Create a sample study directory structure for testing.
    
    Creates:
        temp_dir/
        ├── CS101/
        │   ├── lecture1.pdf (empty placeholder)
        │   ├── notes.txt
        │   └── assignments/
        │       └── hw1.docx (empty placeholder)
        ├── Math202/
        │   ├── textbook.pdf (empty placeholder)
        │   └── exercises.txt
        └── EmptyFolder/
    
    Returns:
        Dictionary mapping directory names to paths
    """
    # Create directories
    cs101 = temp_dir / "CS101"
    cs101.mkdir()
    
    math202 = temp_dir / "Math202"
    math202.mkdir()
    
    empty = temp_dir / "EmptyFolder"
    empty.mkdir()
    
    assignments = cs101 / "assignments"
    assignments.mkdir()
    
    # Create files
    (cs101 / "lecture1.pdf").write_text("PDF content placeholder")
    (cs101 / "notes.txt").write_text("CS101 lecture notes\nPython basics")
    (assignments / "hw1.docx").write_text("DOCX content placeholder")
    
    (math202 / "textbook.pdf").write_text("Math textbook placeholder")
    (math202 / "exercises.txt").write_text("Math exercises\nCalculus problems")
    
    return {
        "root": temp_dir,
        "cs101": cs101,
        "math202": math202,
        "empty": empty,
        "assignments": assignments,
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_config() -> Config:
    """
    Create a default configuration for testing.
    
    Returns:
        Config instance with default values
    """
    return Config()


@pytest.fixture
def valid_config(temp_dir: Path) -> Config:
    """
    Create a valid configuration for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Valid Config instance
    """
    config = Config(
        llm_provider="openai",
        llm_model="gpt-4o",
        llm_api_key="sk-test-key-12345",
        directories=[str(temp_dir)],
    )
    return config


@pytest.fixture
def config_dict() -> dict:
    """
    Create a configuration dictionary for testing.
    
    Returns:
        Dictionary with valid configuration
    """
    return {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "llm_api_key": "sk-test-key-12345",
        "llm_temperature": 0.7,
        "llm_max_tokens": 2000,
        "embedding_provider": "sentence-transformers",
        "embedding_model": "all-MiniLM-L6-v2",
        "directories": ["/test/dir1", "/test/dir2"],
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "file_types": [".pdf", ".docx", ".txt"],
        "top_k": 5,
        "similarity_threshold": 0.7,
        "track_tokens": True,
        "show_sources": True,
        "verbose": False,
    }


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def clean_env(monkeypatch) -> None:
    """
    Clean environment variables that might interfere with tests.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    # Remove API keys from environment
    env_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_openai_env(monkeypatch) -> str:
    """
    Set up mock OpenAI API key in environment.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
        
    Returns:
        The mock API key
    """
    api_key = "sk-test-openai-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key


@pytest.fixture
def mock_gemini_env(monkeypatch) -> str:
    """
    Set up mock Gemini API key in environment.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture
        
    Returns:
        The mock API key
    """
    api_key = "test-gemini-key"
    monkeypatch.setenv("GOOGLE_API_KEY", api_key)
    return api_key


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_console(mocker):
    """
    Mock Rich console for testing CLI output.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock console object
    """
    return mocker.patch("src.cli.console")


@pytest.fixture
def mock_prompt(mocker):
    """
    Mock Rich Prompt for testing user input.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock Prompt object
    """
    return mocker.patch("src.setup_wizard.Prompt")


@pytest.fixture
def mock_confirm(mocker):
    """
    Mock Rich Confirm for testing yes/no prompts.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock Confirm object
    """
    return mocker.patch("src.setup_wizard.Confirm")
