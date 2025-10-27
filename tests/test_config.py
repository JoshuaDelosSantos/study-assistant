"""
Tests for Configuration Module (src/config.py)

Tests cover:
- Config dataclass creation and validation
- Configuration file I/O
- Environment variable handling
- Default values and model selection
"""

import pytest
from pathlib import Path
import yaml

from src.config import (
    Config,
    config_exists,
    load_config,
    save_config,
    get_default_config,
    get_api_key_from_env,
    ensure_data_dir,
)


# ============================================================================
# Config Creation and Validation Tests
# ============================================================================

@pytest.mark.unit
class TestConfigCreation:
    """Test Config dataclass creation and defaults."""
    
    def test_default_config(self, default_config):
        """Test default configuration values."""
        assert default_config.llm_provider == "openai"
        assert default_config.llm_model is None
        assert default_config.llm_api_key == ""
        assert default_config.embedding_provider == "sentence-transformers"
        assert default_config.chunk_size == 1000
        assert default_config.chunk_overlap == 200
        assert default_config.top_k == 5
        assert isinstance(default_config.directories, list)
        assert len(default_config.directories) == 0
    
    def test_gemini_config_creation(self, temp_dir):
        """Test creating configuration with Gemini provider."""
        config = Config(
            llm_provider="gemini",
            llm_model="gemini-2.5-flash",
            llm_api_key="test-api-key",
            directories=[str(temp_dir)],
        )
        
        assert config.llm_provider == "gemini"
        assert config.llm_model == "gemini-2.5-flash"
    
    def test_get_model_default_openai(self):
        """Test default model for OpenAI."""
        config = Config(llm_provider="openai")
        assert config.get_model_default() == "gpt-4o"
    
    def test_get_model_default_gemini(self):
        """Test getting default model for Gemini provider."""
        config = Config(llm_provider="gemini")
        
        assert config.get_model_default() == "gemini-2.5-flash"
    
    def test_get_model_default_unknown_provider(self):
        """Test default model for unknown provider."""
        config = Config(llm_provider="unknown")
        assert config.get_model_default() == "gpt-4o"


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self, temp_dir):
        """Test validation of valid configuration."""
        config = Config(
            llm_provider="openai",
            llm_api_key="sk-test-key",
            directories=[str(temp_dir)],
        )
        
        is_valid, error = config.validate()
        assert is_valid is True
        assert error is None
    
    def test_invalid_provider(self):
        """Test validation fails for invalid provider."""
        config = Config(
            llm_provider="invalid",
            llm_api_key="key",
            directories=["/tmp"],
        )
        
        is_valid, error = config.validate()
        assert is_valid is False
        assert "Invalid LLM provider" in error
    
    def test_missing_api_key(self, temp_dir):
        """Test validation fails without API key."""
        config = Config(
            llm_provider="openai",
            llm_api_key="",
            directories=[str(temp_dir)],
        )
        
        is_valid, error = config.validate()
        assert is_valid is False
        assert "API key is required" in error
    
    def test_missing_directories(self):
        """Test validation fails without directories."""
        config = Config(
            llm_provider="openai",
            llm_api_key="key",
            directories=[],
        )
        
        is_valid, error = config.validate()
        assert is_valid is False
        assert "At least one directory" in error
    
    def test_nonexistent_directory(self):
        """Test validation fails for nonexistent directory."""
        config = Config(
            llm_provider="openai",
            llm_api_key="key",
            directories=["/nonexistent/path/12345"],
        )
        
        is_valid, error = config.validate()
        assert is_valid is False
        assert "does not exist" in error
    
    def test_invalid_temperature(self, temp_dir):
        """Test validation fails for invalid temperature."""
        config = Config(
            llm_provider="openai",
            llm_api_key="key",
            directories=[str(temp_dir)],
            llm_temperature=1.5,  # > 1.0
        )
        
        is_valid, error = config.validate()
        assert is_valid is False
        assert "Temperature" in error
    
    def test_invalid_chunk_size(self, temp_dir):
        """Test validation fails for invalid chunk size."""
        config = Config(
            llm_provider="openai",
            llm_api_key="key",
            directories=[str(temp_dir)],
            chunk_size=-100,
        )
        
        is_valid, error = config.validate()
        assert is_valid is False
        assert "Chunk size" in error
    
    def test_chunk_overlap_exceeds_size(self, temp_dir):
        """Test validation fails when overlap exceeds chunk size."""
        config = Config(
            llm_provider="openai",
            llm_api_key="key",
            directories=[str(temp_dir)],
            chunk_size=100,
            chunk_overlap=200,
        )
        
        is_valid, error = config.validate()
        assert is_valid is False
        assert "overlap must be less than chunk size" in error


# ============================================================================
# Config Serialization Tests
# ============================================================================

@pytest.mark.unit
class TestConfigSerialization:
    """Test Config to/from dictionary conversion."""
    
    def test_to_dict(self, valid_config):
        """Test Config to dictionary conversion."""
        config_dict = valid_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["llm_provider"] == "openai"
        assert config_dict["llm_model"] == "gpt-4o"
        assert config_dict["llm_api_key"] == "sk-test-key-12345"
    
    def test_from_dict(self, config_dict, temp_dir):
        """Test Config from dictionary creation."""
        # Update with valid directory
        config_dict["directories"] = [str(temp_dir)]
        
        config = Config.from_dict(config_dict)
        
        assert config.llm_provider == config_dict["llm_provider"]
        assert config.llm_model == config_dict["llm_model"]
        assert config.llm_api_key == config_dict["llm_api_key"]
        assert config.directories == config_dict["directories"]
    
    def test_from_dict_filters_invalid_keys(self):
        """Test that from_dict filters out invalid keys."""
        data = {
            "llm_provider": "openai",
            "llm_api_key": "key",
            "invalid_key": "should_be_ignored",
            "another_invalid": 123,
        }
        
        config = Config.from_dict(data)
        
        # Should not raise error, invalid keys ignored
        assert config.llm_provider == "openai"
        assert not hasattr(config, "invalid_key")


# ============================================================================
# Config File I/O Tests
# ============================================================================

@pytest.mark.unit
class TestConfigIO:
    """Test configuration file loading and saving."""
    
    def test_config_exists_false(self, temp_config_dir):
        """Test config_exists returns False when no config."""
        assert config_exists() is False
    
    def test_config_exists_true(self, temp_config_dir, valid_config):
        """Test config_exists returns True after saving."""
        save_config(valid_config)
        assert config_exists() is True
    
    def test_save_and_load_config(self, temp_config_dir, valid_config):
        """Test saving and loading configuration."""
        # Save
        save_config(valid_config)
        
        # Load
        loaded_config = load_config()
        
        assert loaded_config.llm_provider == valid_config.llm_provider
        assert loaded_config.llm_model == valid_config.llm_model
        assert loaded_config.llm_api_key == valid_config.llm_api_key
        assert loaded_config.directories == valid_config.directories
    
    def test_save_invalid_config_fails(self, temp_config_dir):
        """Test that saving invalid config raises error."""
        invalid_config = Config(
            llm_provider="invalid",
            llm_api_key="key",
        )
        
        with pytest.raises(ValueError, match="Cannot save invalid configuration"):
            save_config(invalid_config)
    
    def test_load_nonexistent_config_fails(self, temp_config_dir):
        """Test loading nonexistent config raises error."""
        with pytest.raises(FileNotFoundError):
            load_config()
    
    def test_load_invalid_yaml_fails(self, temp_config_dir):
        """Test loading invalid YAML raises error."""
        import src.config as config_module
        
        # Write invalid YAML
        config_module.CONFIG_FILE.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config()
    
    def test_load_empty_config_fails(self, temp_config_dir):
        """Test loading empty config raises error."""
        import src.config as config_module
        
        # Write empty file
        config_module.CONFIG_FILE.write_text("")
        
        with pytest.raises(ValueError, match="empty"):
            load_config()


# ============================================================================
# Environment Variable Tests
# ============================================================================

@pytest.mark.unit
class TestEnvironmentVariables:
    """Test environment variable handling."""
    
    def test_get_api_key_from_env_openai(self, mock_openai_env):
        """Test getting OpenAI API key from environment."""
        api_key = get_api_key_from_env("openai")
        assert api_key == mock_openai_env
    
    def test_get_api_key_from_env_gemini(self, mock_gemini_env):
        """Test getting Gemini API key from environment."""
        api_key = get_api_key_from_env("gemini")
        assert api_key == mock_gemini_env
    
    def test_get_api_key_from_env_none(self, clean_env):
        """Test getting API key returns None when not set."""
        api_key = get_api_key_from_env("openai")
        assert api_key is None
    
    def test_get_api_key_unknown_provider(self, clean_env):
        """Test getting API key for unknown provider."""
        api_key = get_api_key_from_env("unknown")
        assert api_key is None


# ============================================================================
# Utility Function Tests
# ============================================================================

@pytest.mark.unit
class TestUtilities:
    """Test utility functions."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        
        assert isinstance(config, Config)
        assert config.llm_provider == "openai"
        assert config.llm_api_key == ""
    
    def test_ensure_data_dir(self, temp_config_dir):
        """Test data directory creation."""
        import src.config as config_module
        
        # Remove data dir
        if config_module.DATA_DIR.exists():
            config_module.DATA_DIR.rmdir()
        
        # Ensure it's created
        data_dir = ensure_data_dir()
        
        assert data_dir.exists()
        assert data_dir.is_dir()
    
    def test_ensure_data_dir_idempotent(self, temp_config_dir):
        """Test that ensure_data_dir can be called multiple times."""
        dir1 = ensure_data_dir()
        dir2 = ensure_data_dir()
        
        assert dir1 == dir2
        assert dir1.exists()
