"""
Setup Wizard Module - Interactive configuration setup

This module provides guided setup flows for first-time users.
It offers two modes:
- Quick Start: Minimal configuration (provider, API key, directories)
- First-Time Setup: Comprehensive configuration with all options

The wizard validates inputs and saves the configuration automatically.
"""

from pathlib import Path
from typing import Optional, List

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table

from src.config import Config, save_config, get_api_key_from_env


# ============================================================================
# Constants
# ============================================================================

console = Console()


# ============================================================================
# Main Setup Flow
# ============================================================================

def run_setup_wizard() -> Optional[Config]:
    """
    Run the interactive setup wizard.
    
    Returns:
        Configured Config instance, or None if user cancelled
    """
    console.print("[bold cyan]Setup Wizard[/bold cyan]\n")
    
    # Choose setup mode
    console.print("Choose your setup mode:\n")
    console.print("  [1] Quick Start - Get started in under a minute")
    console.print("  [2] First-Time Setup - Configure all options")
    console.print("  [3] Exit\n")
    
    choice = Prompt.ask(
        "Select mode",
        choices=["1", "2", "3"],
        default="1"
    )
    
    if choice == "3":
        return None
    
    console.print()
    
    # Run appropriate setup flow
    if choice == "1":
        config = quick_start_setup()
    else:
        config = first_time_setup()
    
    if config is None:
        return None
    
    # Save configuration
    try:
        save_config(config)
        console.print("\n[bold green]Configuration saved successfully![/bold green]\n")
        return config
    except Exception as e:
        console.print(f"\n[bold red]Error saving configuration: {e}[/bold red]")
        return None


# ============================================================================
# Quick Start Setup
# ============================================================================

def quick_start_setup() -> Optional[Config]:
    """
    Quick start setup flow.
    
    Asks only essential questions:
    - LLM provider and API key
    - Directories to index
    
    Uses sensible defaults for everything else.
    
    Returns:
        Configured Config instance, or None if user cancelled
    """
    console.print("[bold]Quick Start Setup[/bold]")
    console.print("[dim]Answer a few quick questions to get started[/dim]\n")
    
    config = Config()
    
    # Step 1: Choose LLM provider
    provider = choose_llm_provider()
    if provider is None:
        return None
    
    config.llm_provider = provider
    config.llm_model = config.get_model_default()
    
    # Step 2: Get API key
    api_key = get_api_key(provider)
    if api_key is None:
        return None
    
    config.llm_api_key = api_key
    
    # Step 3: Configure directories
    directories = get_directories()
    if not directories:
        console.print("[yellow]No directories configured. Setup cancelled.[/yellow]")
        return None
    
    config.directories = directories
    
    # Show summary
    display_config_summary(config, is_quick_start=True)
    
    # Confirm
    if not Confirm.ask("\nProceed with this configuration?", default=True):
        return None
    
    return config


# ============================================================================
# First-Time Setup
# ============================================================================

def first_time_setup() -> Optional[Config]:
    """
    Comprehensive first-time setup flow.
    
    Walks through all configuration options with explanations.
    
    Returns:
        Configured Config instance, or None if user cancelled
    """
    console.print("[bold]First-Time Setup[/bold]")
    console.print("[dim]Comprehensive configuration with all options[/dim]\n")
    
    config = Config()
    
    # LLM Configuration
    if not configure_llm(config):
        return None
    
    # Directories
    directories = get_directories()
    if not directories:
        console.print("[yellow]No directories configured. Setup cancelled.[/yellow]")
        return None
    config.directories = directories
    
    # Advanced options
    console.print("\n[bold cyan]Advanced Options[/bold cyan]\n")
    
    if Confirm.ask("Configure advanced options?", default=False):
        configure_advanced_options(config)
    
    # Show summary
    display_config_summary(config, is_quick_start=False)
    
    # Confirm
    if not Confirm.ask("\nProceed with this configuration?", default=True):
        return None
    
    return config


# ============================================================================
# Configuration Steps
# ============================================================================

def choose_llm_provider() -> Optional[str]:
    """
    Prompt user to choose LLM provider.
    
    Returns:
        Provider name ('openai' or 'gemini'), or None if cancelled
    """
    console.print("[bold]Step 1: Choose LLM Provider[/bold]\n")
    
    # Display provider options
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Option", style="cyan", width=8)
    table.add_column("Provider", width=15)
    table.add_column("Default Model", width=20)
    table.add_column("Notes", width=30)
    
    table.add_row("1", "OpenAI", "gpt-4o", "High quality, moderate cost")
    table.add_row("2", "Google Gemini", "gemini-1.5-flash", "Fast, cost-effective")
    
    console.print(table)
    console.print()
    
    choice = Prompt.ask(
        "Select provider",
        choices=["1", "2"],
        default="1"
    )
    
    return "openai" if choice == "1" else "gemini"


def validate_api_key_format(provider: str, api_key: str) -> tuple[bool, Optional[str]]:
    """
    Validate API key format (quick local check).
    
    Args:
        provider: Provider name
        api_key: API key to validate
        
    Returns:
        (is_valid, error_message)
    """
    if not api_key or not api_key.strip():
        return False, "API key cannot be empty"
    
    api_key = api_key.strip()
    
    if provider == "openai":
        # OpenAI keys start with 'sk-' and have minimum length
        if not api_key.startswith("sk-"):
            return False, "OpenAI API keys should start with 'sk-'"
        if len(api_key) < 20:
            return False, "OpenAI API key appears too short"
    elif provider == "gemini":
        # Gemini keys are typically 39 characters long
        if len(api_key) < 20:
            return False, "Gemini API key appears too short"
    
    return True, None


def validate_api_key_live(provider: str, api_key: str) -> tuple[bool, Optional[str]]:
    """
    Validate API key by making a minimal test call.
    
    Args:
        provider: Provider name
        api_key: API key to validate
        
    Returns:
        (is_valid, error_message)
    """
    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=10.0)
            # Minimal request to test key - just list models
            client.models.list()
            return True, None
            
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # List models to verify key
            list(genai.list_models())
            return True, None
            
        return False, "Unknown provider"
        
    except Exception as e:
        error_str = str(e).lower()
        if 'api key' in error_str or 'api_key' in error_str or 'auth' in error_str or '401' in error_str or 'unauthorized' in error_str:
            return False, "Invalid API key"
        elif 'network' in error_str or 'connection' in error_str or 'timeout' in error_str:
            return False, "Network error - could not validate key"
        elif 'quota' in error_str or 'billing' in error_str:
            return False, "API key valid but account has billing issues"
        else:
            return False, f"Validation failed: {str(e)[:100]}"


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key from user or environment with validation.
    
    Args:
        provider: Provider name
        
    Returns:
        API key, or None if not provided
    """
    console.print(f"\n[bold]Step 2: API Key[/bold]\n")
    
    # Check environment first
    env_key = get_api_key_from_env(provider)
    if env_key:
        console.print(f"[green]API key found in environment variables[/green]")
        if Confirm.ask("Use this key?", default=True):
            # Validate environment key
            console.print("[cyan]Validating API key...[/cyan]")
            is_valid, error = validate_api_key_live(provider, env_key)
            
            if is_valid:
                console.print("[green]API key validated successfully[/green]")
                return env_key
            else:
                console.print(f"[yellow]Warning: {error}[/yellow]")
                console.print("[yellow]Environment key validation failed, please enter a different key[/yellow]\n")
    
    # Provide instructions
    if provider == "openai":
        console.print("[dim]Get your API key from: https://platform.openai.com/api-keys[/dim]")
    else:
        console.print("[dim]Get your API key from: https://makersuite.google.com/app/apikey[/dim]")
    
    console.print()
    
    # Retry loop for manual entry
    max_attempts = 3
    for attempt in range(max_attempts):
        # Prompt for key
        api_key = Prompt.ask(
            "Enter your API key",
            password=True
        )
        
        if not api_key or not api_key.strip():
            console.print("[yellow]No API key provided[/yellow]")
            return None
        
        api_key = api_key.strip()
        
        # Step 1: Format validation (quick, no network)
        is_valid_format, format_error = validate_api_key_format(provider, api_key)
        if not is_valid_format:
            console.print(f"[red]{format_error}[/red]")
            if attempt < max_attempts - 1:
                if not Confirm.ask("Try again?", default=True):
                    return None
            continue
        
        # Step 2: Live validation (requires network)
        console.print("[cyan]Validating API key...[/cyan]")
        is_valid_live, live_error = validate_api_key_live(provider, api_key)
        
        if is_valid_live:
            console.print("[green]API key validated successfully[/green]")
            return api_key
        else:
            console.print(f"[red]{live_error}[/red]")
            
            # Offer to skip validation on network errors
            if 'network' in live_error.lower():
                console.print("[yellow]Unable to validate due to network issue[/yellow]")
                if Confirm.ask("Skip validation and use this key anyway?", default=False):
                    return api_key
            
            # Retry prompt
            if attempt < max_attempts - 1:
                if not Confirm.ask("Try again?", default=True):
                    return None
    
    console.print(f"[red]Maximum validation attempts ({max_attempts}) exceeded[/red]")
    return None


def get_directories() -> List[str]:
    """
    Get list of directories to index from user.
    
    Offers two modes:
    - Parent directory scan (recommended)
    - Manual entry
    
    Returns:
        List of directory paths
    """
    console.print(f"\n[bold]Step 3: Document Directories[/bold]\n")
    console.print("Choose directory selection mode:\n")
    console.print("  [1] Scan parent directory (recommended)")
    console.print("  [2] Enter paths manually\n")
    
    choice = Prompt.ask(
        "Select mode",
        choices=["1", "2"],
        default="1"
    )
    
    console.print()
    
    if choice == "1":
        return scan_parent_directory()
    else:
        return get_directories_manual()


def scan_parent_directory() -> List[str]:
    """
    Scan parent directory and let user select subdirectories.
    
    Returns:
        List of selected directory paths
    """
    # Suggest parent of current directory
    project_root = Path(__file__).parent.parent.resolve()
    suggested_parent = project_root.parent
    
    console.print("[dim]Suggested: Scan the parent directory containing this project[/dim]")
    console.print(f"[dim]Location: {suggested_parent}[/dim]\n")
    
    # Get parent directory
    parent_input = Prompt.ask(
        "Enter parent directory to scan",
        default=str(suggested_parent)
    )
    
    parent_path = Path(parent_input).expanduser().resolve()
    
    # Validate parent directory
    if not parent_path.exists():
        console.print(f"[red]Error: Directory does not exist: {parent_path}[/red]")
        return []
    
    if not parent_path.is_dir():
        console.print(f"[red]Error: Not a directory: {parent_path}[/red]")
        return []
    
    console.print(f"\n[cyan]Scanning {parent_path}...[/cyan]\n")
    
    # Find subdirectories with document files
    subdirs = discover_document_directories(parent_path)
    
    if not subdirs:
        console.print("[yellow]No subdirectories with documents found[/yellow]")
        if Confirm.ask("Enter directories manually instead?", default=True):
            return get_directories_manual()
        return []
    
    # Display found directories
    console.print(f"[green]Found {len(subdirs)} directories with documents:[/green]\n")
    
    # Let user select directories
    selected = select_directories(subdirs)
    
    if not selected:
        console.print("\n[yellow]No directories selected[/yellow]")
        if Confirm.ask("Try again?", default=True):
            return get_directories()
        return []
    
    return selected


def discover_document_directories(parent_path: Path, max_depth: int = 2) -> List[tuple[Path, int]]:
    """
    Discover subdirectories containing document files.
    
    Args:
        parent_path: Parent directory to scan
        max_depth: Maximum depth to scan (default: 2 levels)
        
    Returns:
        List of tuples (directory_path, document_count)
    """
    from src.config import Config
    
    config = Config()  # Get default file types
    valid_extensions = set(config.file_types)
    
    discovered = []
    project_dir = Path(__file__).parent.parent.resolve()
    
    try:
        # Scan subdirectories
        for item in parent_path.iterdir():
            # Skip hidden directories, project directory, and common non-content dirs
            if not item.is_dir():
                continue
            
            name_lower = item.name.lower()
            
            # Skip hidden, project, and system directories
            if (item.name.startswith('.') or 
                item == project_dir or
                name_lower in ['node_modules', 'venv', '.venv', '__pycache__', 
                               'cache', 'temp', 'tmp', '.git']):
                continue
            
            # Count document files in this directory (recursively up to max_depth)
            doc_count = count_documents_recursive(item, valid_extensions, current_depth=1, max_depth=max_depth)
            
            # Only include if it has documents
            if doc_count > 0:
                discovered.append((item, doc_count))
        
        # Sort by document count (descending)
        discovered.sort(key=lambda x: x[1], reverse=True)
        
    except PermissionError as e:
        console.print(f"[yellow]Warning: Permission denied accessing some directories[/yellow]")
    
    return discovered


def count_documents_recursive(directory: Path, valid_extensions: set, current_depth: int, max_depth: int) -> int:
    """
    Count document files in directory recursively.
    
    Args:
        directory: Directory to scan
        valid_extensions: Set of valid file extensions
        current_depth: Current recursion depth
        max_depth: Maximum depth to recurse
        
    Returns:
        Number of document files found
    """
    count = 0
    
    try:
        for item in directory.iterdir():
            if item.is_file():
                if item.suffix.lower() in valid_extensions:
                    count += 1
            elif item.is_dir() and current_depth < max_depth:
                # Skip hidden and system directories
                if not item.name.startswith('.'):
                    count += count_documents_recursive(
                        item, valid_extensions, current_depth + 1, max_depth
                    )
    except PermissionError:
        pass  # Skip directories we can't access
    
    return count


def select_directories(subdirs: List[tuple[Path, int]]) -> List[str]:
    """
    Let user select directories from a list.
    
    Args:
        subdirs: List of (directory_path, document_count) tuples
        
    Returns:
        List of selected directory paths as strings
    """
    # Display directories with indices
    for i, (path, count) in enumerate(subdirs, 1):
        console.print(f"  [{i:2d}] {path.name:40s} ({count} documents)")
    
    console.print()
    console.print("[dim]Enter numbers separated by commas (e.g., 1,3,5)[/dim]")
    console.print("[dim]Or type 'all' to select all directories[/dim]\n")
    
    selection = Prompt.ask("Select directories", default="all")
    
    if selection.lower() == "all":
        selected = [str(path) for path, _ in subdirs]
    else:
        # Parse comma-separated numbers
        try:
            indices = [int(x.strip()) for x in selection.split(",")]
            selected = []
            
            for idx in indices:
                if 1 <= idx <= len(subdirs):
                    path, _ = subdirs[idx - 1]
                    selected.append(str(path))
                else:
                    console.print(f"[yellow]Warning: Invalid selection {idx}, skipping[/yellow]")
            
        except ValueError:
            console.print("[red]Error: Invalid input format[/red]")
            return []
    
    # Show selected
    console.print(f"\n[green]Selected {len(selected)} directories:[/green]")
    for path in selected:
        console.print(f"  - {path}")
    
    return selected


def get_directories_manual() -> List[str]:
    """
    Get list of directories via manual entry (original method).
    
    Returns:
        List of directory paths
    """
    console.print("[dim]Enter the paths to directories containing your study materials[/dim]")
    console.print("[dim]You can enter multiple directories, one at a time[/dim]\n")
    
    directories = []
    
    while True:
        if directories:
            console.print(f"\n[green]Configured directories: {len(directories)}[/green]")
            for i, d in enumerate(directories, 1):
                console.print(f"  {i}. {d}")
            console.print()
        
        # Prompt for directory
        directory = Prompt.ask(
            "Enter directory path (or press Enter to finish)",
            default=""
        )
        
        if not directory:
            break
        
        # Expand user path and validate
        path = Path(directory).expanduser().resolve()
        
        if not path.exists():
            console.print(f"[yellow]Warning: Directory does not exist: {path}[/yellow]")
            if not Confirm.ask("Add anyway?", default=False):
                continue
        elif not path.is_dir():
            console.print(f"[red]Error: Not a directory: {path}[/red]")
            continue
        
        # Add to list
        directory_str = str(path)
        if directory_str not in directories:
            directories.append(directory_str)
            console.print(f"[green]Added: {directory_str}[/green]")
        else:
            console.print(f"[yellow]Already added: {directory_str}[/yellow]")
    
    return directories


def configure_llm(config: Config) -> bool:
    """
    Configure LLM settings.
    
    Args:
        config: Config instance to modify
        
    Returns:
        True if successful, False if cancelled
    """
    # Provider
    provider = choose_llm_provider()
    if provider is None:
        return False
    
    config.llm_provider = provider
    
    # Model
    console.print(f"\n[bold]Model Selection[/bold]\n")
    default_model = config.get_model_default()
    
    models = {
        "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
        "gemini": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
    }
    
    console.print(f"Available models for {provider}:")
    for model in models[provider]:
        marker = " (default)" if model == default_model else ""
        console.print(f"  - {model}{marker}")
    
    console.print()
    
    model = Prompt.ask(
        "Select model",
        default=default_model
    )
    
    config.llm_model = model
    
    # API Key
    api_key = get_api_key(provider)
    if api_key is None:
        return False
    
    config.llm_api_key = api_key
    
    return True


def configure_advanced_options(config: Config) -> None:
    """
    Configure advanced options.
    
    Args:
        config: Config instance to modify
    """
    # Chunk size
    chunk_size = Prompt.ask(
        "\nChunk size (characters)",
        default=str(config.chunk_size)
    )
    config.chunk_size = int(chunk_size)
    
    # Chunk overlap
    chunk_overlap = Prompt.ask(
        "Chunk overlap (characters)",
        default=str(config.chunk_overlap)
    )
    config.chunk_overlap = int(chunk_overlap)
    
    # Top K
    top_k = Prompt.ask(
        "Number of chunks to retrieve (top_k)",
        default=str(config.top_k)
    )
    config.top_k = int(top_k)
    
    # Temperature
    temperature = Prompt.ask(
        "LLM temperature (0.0-1.0)",
        default=str(config.llm_temperature)
    )
    config.llm_temperature = float(temperature)


# ============================================================================
# Display Functions
# ============================================================================

def display_config_summary(config: Config, is_quick_start: bool = False) -> None:
    """
    Display configuration summary.
    
    Args:
        config: Configuration to display
        is_quick_start: Whether this is from quick start mode
    """
    console.print("\n[bold cyan]Configuration Summary[/bold cyan]\n")
    
    # Create summary table
    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan", width=20)
    table.add_column("Value", width=50)
    
    table.add_row("LLM Provider", config.llm_provider)
    table.add_row("Model", config.llm_model or "default")
    table.add_row("API Key", f"{'*' * 8}...{config.llm_api_key[-4:]}" if len(config.llm_api_key) > 4 else "***")
    table.add_row("Directories", f"{len(config.directories)} configured")
    
    if not is_quick_start:
        table.add_row("Chunk Size", str(config.chunk_size))
        table.add_row("Top K", str(config.top_k))
        table.add_row("Temperature", str(config.llm_temperature))
    
    console.print(table)
    
    # Show directories
    console.print("\n[bold]Directories:[/bold]")
    for directory in config.directories:
        console.print(f"  - {directory}")


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    # For testing setup wizard independently
    result = run_setup_wizard()
    if result:
        console.print("\n[green]Setup completed successfully![/green]")
    else:
        console.print("\n[yellow]Setup cancelled[/yellow]")
