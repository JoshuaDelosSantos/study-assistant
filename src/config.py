"""
Configuration Module - Application configuration management

This module handles loading, saving, and validating application configuration.
Configuration is stored in YAML format and includes:
- LLM provider and model settings
- API keys
- Document directories
- Indexing parameters
- RAG settings
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import yaml


# ============================================================================
# Constants
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================================
# Configuration Data Class
# ============================================================================

@dataclass
class Config:
    """
    Application configuration.
    
    Attributes:
        llm_provider: LLM provider ('openai' or 'gemini')
        llm_model: Specific model to use
        llm_api_key: API key for the LLM provider
        llm_temperature: Temperature for LLM responses (0.0-1.0)
        llm_max_tokens: Maximum tokens in LLM response
        
        embedding_provider: Provider for embeddings
        embedding_model: Specific embedding model
        
        directories: List of directories to index
        
        chunk_size: Size of document chunks (characters)
        chunk_overlap: Overlap between chunks (characters)
        file_types: List of file extensions to process
        
        top_k: Number of chunks to retrieve for RAG
        similarity_threshold: Minimum similarity score (0.0-1.0)
        
        track_tokens: Whether to track token usage
        show_sources: Whether to show source citations
        verbose: Enable verbose logging
    """
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    llm_api_key: str = ""
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    
    # Embedding Configuration
    embedding_provider: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Document Directories
    directories: List[str] = field(default_factory=list)
    
    # Indexing Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    file_types: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".pptx", ".txt", ".md"])
    
    # RAG Configuration
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Application Settings
    track_tokens: bool = True
    show_sources: bool = True
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary containing configuration values
            
        Returns:
            Config instance
        """
        # Filter out keys that aren't part of the Config dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered_data)
    
    def get_model_default(self) -> str:
        """
        Get the default model for the selected provider.
        
        Returns:
            Default model name
        """
        if self.llm_provider == "openai":
            return "gpt-4o"
        elif self.llm_provider == "gemini":
            return "gemini-1.5-flash"
        else:
            return "gpt-4o"
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check LLM provider
        if self.llm_provider not in ["openai", "gemini"]:
            return False, f"Invalid LLM provider: {self.llm_provider}"
        
        # Check API key
        if not self.llm_api_key:
            return False, "API key is required"
        
        # Check directories
        if not self.directories:
            return False, "At least one directory must be configured"
        
        # Validate directory paths
        for directory in self.directories:
            path = Path(directory).expanduser()
            if not path.exists():
                return False, f"Directory does not exist: {directory}"
            if not path.is_dir():
                return False, f"Not a directory: {directory}"
        
        # Validate numeric ranges
        if not 0.0 <= self.llm_temperature <= 1.0:
            return False, "Temperature must be between 0.0 and 1.0"
        
        if not 0.0 <= self.similarity_threshold <= 1.0:
            return False, "Similarity threshold must be between 0.0 and 1.0"
        
        if self.chunk_size <= 0:
            return False, "Chunk size must be positive"
        
        if self.chunk_overlap < 0:
            return False, "Chunk overlap cannot be negative"
        
        if self.chunk_overlap >= self.chunk_size:
            return False, "Chunk overlap must be less than chunk size"
        
        if self.top_k <= 0:
            return False, "top_k must be positive"
        
        return True, None


# ============================================================================
# Configuration I/O
# ============================================================================

def config_exists() -> bool:
    """
    Check if a configuration file exists.
    
    Returns:
        True if config file exists, False otherwise
    """
    return CONFIG_FILE.exists()


def load_config() -> Config:
    """
    Load configuration from file.
    
    Returns:
        Config instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not config_exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")
    
    try:
        with open(CONFIG_FILE, "r") as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise ValueError("Configuration file is empty")
        
        config = Config.from_dict(data)
        
        # Validate configuration
        is_valid, error = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {error}")
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


def save_config(config: Config) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate before saving
    is_valid, error = config.validate()
    if not is_valid:
        raise ValueError(f"Cannot save invalid configuration: {error}")
    
    # Convert to dictionary
    data = config.to_dict()
    
    # Write to file
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Config:
    """
    Get a default configuration instance.
    
    Returns:
        Config with default values
    """
    return Config()


# ============================================================================
# Environment Variables
# ============================================================================

def get_api_key_from_env(provider: str) -> Optional[str]:
    """
    Attempt to get API key from environment variables.
    
    Args:
        provider: Provider name ('openai' or 'gemini')
        
    Returns:
        API key if found, None otherwise
    """
    env_vars = {
        "openai": ["OPENAI_API_KEY"],
        "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    }
    
    for var_name in env_vars.get(provider, []):
        api_key = os.environ.get(var_name)
        if api_key:
            return api_key
    
    return None


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_data_dir() -> Path:
    """
    Ensure the data directory exists.
    
    Returns:
        Path to data directory
    """
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    # For testing configuration functionality
    config = get_default_config()
    print("Default configuration:")
    print(yaml.dump(config.to_dict(), default_flow_style=False))
