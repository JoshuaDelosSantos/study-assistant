"""
Conversation history management for multi-turn interactions.

This module provides a simple rolling window conversation history that
stores recent turns (query-answer pairs) to enable context-aware responses.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Source:
    """Source document reference."""
    file_path: str
    chunk_index: int
    page_number: Optional[int] = None
    relevance_score: Optional[float] = None


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    query: str
    answer: str
    sources: List[Source] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: Optional[int] = None


class ConversationHistory:
    """
    Manages conversation history with rolling window.
    
    Keeps the most recent N turns in memory to provide context for
    multi-turn conversations. Older turns are automatically discarded
    when the limit is reached.
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation history.
        
        Args:
            max_turns: Maximum number of turns to keep (default: 10)
        """
        self.turns: List[ConversationTurn] = []
        self.max_turns = max_turns
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """
        Add a turn to the history and enforce max_turns limit.
        
        Args:
            turn: The conversation turn to add
        """
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)  # Remove oldest turn
    
    def get_recent_context(self, n: int = 3) -> str:
        """
        Get the last n turns formatted for prompt context.
        
        Args:
            n: Number of recent turns to include (default: 3)
            
        Returns:
            Formatted string with recent conversation history,
            or empty string if no history exists
        """
        if not self.turns:
            return ""
        
        recent = self.turns[-n:] if n < len(self.turns) else self.turns
        context_parts = []
        
        for turn in recent:
            context_parts.append(f"Q: {turn.query}\nA: {turn.answer}")
        
        return "\n\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns.clear()
    
    def __len__(self) -> int:
        """Return number of turns in history."""
        return len(self.turns)
    
    def __bool__(self) -> bool:
        """Return True if history contains any turns."""
        return len(self.turns) > 0
