"""
CLI Module - Command-line interface and main application loop

This module provides the terminal-based user interface for the study assistant.
It handles:
- Initial application startup
- Configuration detection
- Setup wizard invocation
- Main query loop
- Command processing
- Graceful shutdown
"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from src.config import Config, load_config, config_exists
from src.setup_wizard import run_setup_wizard


# ============================================================================
# Constants
# ============================================================================

APP_TITLE = "Study Assistant"
APP_VERSION = "1.0.0"

console = Console()


# ============================================================================
# Application Lifecycle
# ============================================================================

def display_banner() -> None:
    """Display the application banner."""
    banner = f"""
[bold cyan]{APP_TITLE}[/bold cyan] [dim]v{APP_VERSION}[/dim]
Your personal AI study companion
    """
    console.print(Panel(banner.strip(), border_style="cyan"))


def run_application() -> None:
    """
    Main application entry point.
    
    Handles initial setup detection and launches appropriate flow:
    - If no config exists: Run setup wizard
    - If config exists: Load and start main loop
    """
    display_banner()
    console.print()
    
    # Check if configuration exists
    if not config_exists():
        console.print("[yellow]No configuration found. Let's get you set up.[/yellow]\n")
        
        # Run setup wizard
        config = run_setup_wizard()
        
        if config is None:
            console.print("\n[red]Setup cancelled. Exiting.[/red]")
            sys.exit(0)
    else:
        # Load existing configuration
        try:
            config = load_config()
            console.print("[green]Configuration loaded successfully[/green]\n")
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            console.print("[yellow]Please run setup again.[/yellow]\n")
            
            # Offer to run setup wizard
            retry = Prompt.ask(
                "Would you like to reconfigure?",
                choices=["y", "n"],
                default="y"
            )
            
            if retry.lower() == "y":
                config = run_setup_wizard()
                if config is None:
                    sys.exit(0)
            else:
                sys.exit(1)
    
    # Start main application loop
    run_main_loop(config)


def run_main_loop(config: Config) -> None:
    """
    Main application loop.
    
    Handles user queries and commands until exit.
    
    Args:
        config: Application configuration
    """
    console.print("[cyan]Starting Study Assistant...[/cyan]\n")
    
    # TODO: Initialize components (vector store, agent, etc.)
    # This will be implemented in later phases
    
    console.print("[bold green]Ready![/bold green]")
    console.print("Ask me anything about your study materials.")
    console.print("[dim]Type 'help' for commands, 'q' or 'exit' to quit.[/dim]\n")
    
    while True:
        try:
            # Get user input
            query = Prompt.ask("[bold cyan]>[/bold cyan]").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ["q", "quit", "exit"]:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            
            elif query.lower() == "help":
                display_help()
            
            elif query.lower() == "status":
                display_status(config)
            
            elif query.lower() == "reindex":
                console.print("[yellow]Reindexing not yet implemented[/yellow]")
                # TODO: Implement reindexing
            
            else:
                # Process query
                # TODO: Implement query processing via RAG agent
                console.print("[yellow]Query processing not yet implemented[/yellow]")
                console.print(f"[dim]You asked: {query}[/dim]\n")
        
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Use 'q' or 'exit' to quit.[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


# ============================================================================
# Helper Functions
# ============================================================================

def display_help() -> None:
    """Display available commands."""
    help_text = """
[bold]Available Commands:[/bold]

  [cyan]help[/cyan]      - Show this help message
  [cyan]status[/cyan]    - Display configuration and system status
  [cyan]reindex[/cyan]   - Rebuild document index
  [cyan]q, exit[/cyan]   - Quit the application

[bold]Usage:[/bold]

  Simply type your question and press Enter to get answers from your study materials.
  
[bold]Examples:[/bold]

  > What are the key concepts in my CS101 notes?
  > Explain the difference between supervised and unsupervised learning
  > Summarise the main points from my calculus lectures
    """
    console.print(Panel(help_text.strip(), title="Help", border_style="cyan"))
    console.print()


def display_status(config: Config) -> None:
    """
    Display current configuration and system status.
    
    Args:
        config: Application configuration
    """
    status_text = f"""
[bold]Configuration:[/bold]

  LLM Provider:  {config.llm_provider}
  Model:         {config.llm_model or 'default'}
  Directories:   {len(config.directories)} configured
  
[bold]System Status:[/bold]

  Vector Store:  [yellow]Not initialized[/yellow]
  Documents:     [yellow]Not indexed[/yellow]
  Agent:         [yellow]Not initialized[/yellow]
    """
    console.print(Panel(status_text.strip(), title="Status", border_style="cyan"))
    console.print()


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    # For testing the CLI independently
    run_application()
