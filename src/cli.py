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
from src.llm_providers import create_provider
from src.agent import RAGAgent


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
    
    # Initialize LLM provider
    console.print("[cyan]Initializing LLM provider...[/cyan]")
    try:
        llm_provider = create_provider(
            provider_type=config.llm_provider,
            api_key=config.llm_api_key,
            model=config.llm_model or config.get_model_default(),
            temperature=config.llm_temperature
        )
        console.print(f"[green]LLM provider initialized: {config.llm_provider}/{config.llm_model or config.get_model_default()}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to initialize LLM provider: {e}[/red]")
        console.print("[yellow]Please check your API key and configuration[/yellow]")
        sys.exit(1)
    
    # Initialize RAG agent
    console.print("[cyan]Initializing RAG agent...[/cyan]")
    agent = RAGAgent(config, vector_store, llm_provider)
    
    console.print("[bold green]Ready![/bold green]")
    console.print("Ask me anything about your study materials.")
    console.print("[dim]Type 'help' for commands, 'q' or 'exit' to quit.[/dim]\n")
    
    while True:
        try:
            # Get user input with turn counter
            turn_number = len(agent.conversation_history) + 1
            query = Prompt.ask(f"[bold cyan][{turn_number}]>[/bold cyan]").strip()
            
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
            
            elif query.lower() == "history":
                display_conversation_history(agent)
            
            elif query.lower() == "clear":
                clear_conversation_history(agent)
            
            else:
                # Process query using RAG agent
                process_query(agent, query, config)
        
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Use 'q' or 'exit' to quit.[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


# ============================================================================
# Helper Functions
# ============================================================================

def process_query(agent: RAGAgent, query: str, config: Config) -> None:
    """
    Process a user query using the RAG agent and display results.
    
    Args:
        agent: Initialized RAG agent
        query: User query string
        config: Application configuration
    """
    console.print()
    
    try:
        # Process query
        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            response = agent.process_query(query)
        
        # Display answer
        console.print("[bold cyan]Answer:[/bold cyan]")
        console.print(response.answer)
        console.print()
        
        # Display sources if enabled
        if config.show_sources and response.sources:
            console.print("[bold cyan]Sources:[/bold cyan]")
            for i, source in enumerate(response.sources, 1):
                if source.page is not None:
                    # PDF with page number
                    console.print(
                        f"  {i}. [dim]{source.filename}[/dim] "
                        f"(Page {source.page}, Chunk {source.chunk_index}) "
                        f"[dim]- similarity: {source.similarity:.2f}[/dim]"
                    )
                else:
                    # Other document types
                    console.print(
                        f"  {i}. [dim]{source.filename}[/dim] "
                        f"(Chunk {source.chunk_index}) "
                        f"[dim]- similarity: {source.similarity:.2f}[/dim]"
                    )
            console.print()
        
        # Display token usage if enabled
        if config.track_tokens and response.tokens_used:
            console.print(
                f"[dim]Tokens: {response.prompt_tokens} prompt + "
                f"{response.completion_tokens} completion = "
                f"{response.tokens_used} total[/dim]"
            )
            console.print()
    
    except Exception as e:
        console.print(f"[red]Error processing query: {e}[/red]")
        if config.verbose:
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        console.print()


def display_help() -> None:
    """Display available commands."""
    help_text = """
[bold]Available Commands:[/bold]

  [cyan]help[/cyan]      - Show this help message
  [cyan]status[/cyan]    - Display configuration and system status
  [cyan]history[/cyan]   - View conversation history
  [cyan]clear[/cyan]     - Clear conversation history
  [cyan]reindex[/cyan]   - Rebuild document index
  [cyan]q, exit[/cyan]   - Quit the application

[bold]Usage:[/bold]

  Simply type your question and press Enter to get answers from your study materials.
  The assistant maintains conversation history, so you can ask follow-up questions.
  
[bold]Examples:[/bold]

  > What are the key concepts in my CS101 notes?
  > Explain the difference between supervised and unsupervised learning
  > How does that relate to what you just explained?  [dim](uses conversation history)[/dim]
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


def display_conversation_history(agent: RAGAgent) -> None:
    """
    Display the conversation history.
    
    Args:
        agent: RAG agent with conversation history
    """
    if not agent.conversation_history:
        console.print("[yellow]No conversation history yet. Ask a question to start![/yellow]\n")
        return
    
    console.print(f"\n[bold cyan]Conversation History ({len(agent.conversation_history)} turns):[/bold cyan]\n")
    
    for i, turn in enumerate(agent.conversation_history.turns, 1):
        # Display query
        console.print(f"[bold cyan][{i}] Q:[/bold cyan] {turn.query}")
        
        # Display answer (truncated if too long)
        answer = turn.answer
        if len(answer) > 200:
            answer = answer[:200] + "..."
        console.print(f"[bold green]    A:[/bold green] {answer}")
        
        # Display metadata
        metadata_parts = []
        if turn.sources:
            metadata_parts.append(f"{len(turn.sources)} sources")
        if turn.tokens_used:
            metadata_parts.append(f"{turn.tokens_used} tokens")
        
        if metadata_parts:
            console.print(f"[dim]    ({', '.join(metadata_parts)})[/dim]")
        
        console.print()
    
    console.print()


def clear_conversation_history(agent: RAGAgent) -> None:
    """
    Clear the conversation history.
    
    Args:
        agent: RAG agent with conversation history
    """
    if not agent.conversation_history:
        console.print("[yellow]Conversation history is already empty.[/yellow]\n")
        return
    
    turn_count = len(agent.conversation_history)
    agent.conversation_history.clear()
    console.print(f"[green]✓ Cleared {turn_count} turn(s) from conversation history[/green]\n")

# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    # For testing the CLI independently
    run_application()
