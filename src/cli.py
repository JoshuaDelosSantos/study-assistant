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
from rich.prompt import Prompt, Confirm

from src.config import Config, load_config, config_exists
from src.setup_wizard import run_setup_wizard
from src.vector_store import VectorStore
from src.indexer import DocumentIndexer


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
Terminal based question answering assistant for your study materials
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
    
    # Initialize vector store
    console.print("[cyan]Initializing vector store...[/cyan]")
    vector_store = VectorStore(config)
    
    if not vector_store.initialize():
        console.print("[red]Failed to initialize vector store[/red]")
        sys.exit(1)
    
    # Check schema version
    needs_reindex, reason = vector_store.needs_reindex()
    if needs_reindex:
        console.print(f"\n[yellow]Configuration has changed:[/yellow]")
        console.print(f"[yellow]{reason}[/yellow]\n")
        
        if Confirm.ask("Reindex documents now?", default=True):
            reindex_documents(vector_store, config)
        else:
            console.print("[yellow]Warning: Using existing index with different configuration[/yellow]\n")
    
    # Check if we need to index documents
    stats = vector_store.get_stats()
    if stats.total_documents == 0:
        console.print("[yellow]No documents indexed yet[/yellow]")
        
        if Confirm.ask("Index documents now?", default=True):
            indexer = DocumentIndexer(config, vector_store)
            indexer.index_all()
            console.print()
    
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
                display_status(config, vector_store)
            
            elif query.lower() == "reindex":
                reindex_documents(vector_store, config)
            
            else:
                # Process query
                # TODO: Implement query processing via RAG agent (Phase 3)
                console.print("[yellow]Query processing not yet implemented (Phase 3)[/yellow]")
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


def display_status(config: Config, vector_store: Optional[VectorStore] = None) -> None:
    """
    Display current configuration and system status.
    
    Args:
        config: Application configuration
        vector_store: Optional initialized vector store
    """
    status_lines = [
        "[bold]Configuration:[/bold]",
        "",
        f"  LLM Provider:  {config.llm_provider}",
        f"  Model:         {config.llm_model or 'default'}",
        f"  Directories:   {len(config.directories)} configured",
        "",
        "[bold]System Status:[/bold]",
        ""
    ]
    
    if vector_store:
        try:
            stats = vector_store.get_stats()
            status_lines.extend([
                f"  Vector Store:      [green]Initialized[/green]",
                f"  Documents:         {stats.indexed_files_count} files, {stats.total_chunks} chunks",
                f"  Embedding Model:   {stats.embedding_model} ({stats.embedding_dimension}d)",
                f"  Schema Version:    {stats.schema_version}",
                f"  Chunk Size:        {stats.chunk_size} chars (overlap: {stats.chunk_overlap})",
            ])
            
            if stats.last_indexed:
                try:
                    from datetime import datetime
                    indexed_dt = datetime.fromisoformat(stats.last_indexed)
                    indexed_str = indexed_dt.strftime("%Y-%m-%d %H:%M:%S")
                    status_lines.append(f"  Last Indexed:      {indexed_str}")
                except Exception:
                    status_lines.append(f"  Last Indexed:      {stats.last_indexed}")
            
            # Check for config mismatches
            needs_reindex, reason = vector_store.needs_reindex()
            if needs_reindex:
                status_lines.extend([
                    "",
                    "[yellow]⚠️  Configuration Mismatch:[/yellow]",
                    f"[yellow]   {reason}[/yellow]",
                    "[yellow]   Run 'reindex' to update[/yellow]"
                ])
                
        except Exception as e:
            status_lines.append(f"  Vector Store:  [red]Error: {e}[/red]")
    else:
        status_lines.extend([
            "  Vector Store:  [yellow]Not initialized[/yellow]",
            "  Documents:     [yellow]Not indexed[/yellow]",
        ])
    
    status_text = "\n".join(status_lines)
    console.print(Panel(status_text, title="Status", border_style="cyan"))
    console.print()


def reindex_documents(vector_store: VectorStore, config: Config) -> None:
    """
    Reindex all documents from scratch.
    
    This is a destructive operation that deletes the existing collection
    and rebuilds it from the configured directories.
    
    Args:
        vector_store: Initialized vector store
        config: Application configuration
    """
    console.print("\n[bold yellow]⚠️  Reindexing Warning[/bold yellow]")
    console.print("This will delete all existing indexed documents and rebuild from scratch.")
    console.print()
    
    if not Confirm.ask("Are you sure you want to continue?", default=False):
        console.print("[yellow]Reindex cancelled[/yellow]\n")
        return
    
    console.print("\n[cyan]Clearing existing index...[/cyan]")
    vector_store.clear()
    
    # Reinitialize collection
    console.print("[cyan]Recreating collection...[/cyan]")
    if not vector_store.initialize():
        console.print("[red]Failed to recreate collection[/red]\n")
        return
    
    # Index documents
    console.print("[cyan]Indexing documents...[/cyan]\n")
    indexer = DocumentIndexer(config, vector_store)
    stats = indexer.index_all()
    
    console.print("\n[green]Reindexing complete![/green]\n")


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    # For testing the CLI independently
    run_application()
