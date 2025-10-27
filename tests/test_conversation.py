"""
Tests for conversation history management.
"""

import pytest
from datetime import datetime
from src.conversation import ConversationHistory, ConversationTurn, Source


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""
    
    def test_create_turn_minimal(self):
        """Test creating turn with minimal required fields."""
        turn = ConversationTurn(
            query="What is Python?",
            answer="Python is a programming language."
        )
        assert turn.query == "What is Python?"
        assert turn.answer == "Python is a programming language."
        assert turn.sources == []
        assert isinstance(turn.timestamp, datetime)
        assert turn.tokens_used is None
    
    def test_create_turn_with_sources(self):
        """Test creating turn with source citations."""
        sources = [
            Source(file_path="test.pdf", chunk_index=0, page_number=1),
            Source(file_path="test.pdf", chunk_index=1, page_number=2)
        ]
        turn = ConversationTurn(
            query="Test query",
            answer="Test answer",
            sources=sources
        )
        assert len(turn.sources) == 2
        assert turn.sources[0].page_number == 1
    
    def test_create_turn_with_tokens(self):
        """Test creating turn with token usage."""
        turn = ConversationTurn(
            query="Test",
            answer="Answer",
            tokens_used=150
        )
        assert turn.tokens_used == 150


class TestConversationHistory:
    """Tests for ConversationHistory class."""
    
    def test_init_default_max_turns(self):
        """Test initialization with default max_turns."""
        history = ConversationHistory()
        assert history.max_turns == 10
        assert len(history) == 0
        assert not history  # __bool__ returns False when empty
    
    def test_init_custom_max_turns(self):
        """Test initialization with custom max_turns."""
        history = ConversationHistory(max_turns=5)
        assert history.max_turns == 5
    
    def test_add_single_turn(self):
        """Test adding a single turn."""
        history = ConversationHistory()
        turn = ConversationTurn(query="Q1", answer="A1")
        
        history.add_turn(turn)
        
        assert len(history) == 1
        assert history  # __bool__ returns True when not empty
        assert history.turns[0].query == "Q1"
    
    def test_add_multiple_turns(self):
        """Test adding multiple turns."""
        history = ConversationHistory()
        
        for i in range(5):
            turn = ConversationTurn(query=f"Q{i}", answer=f"A{i}")
            history.add_turn(turn)
        
        assert len(history) == 5
        assert history.turns[0].query == "Q0"
        assert history.turns[4].query == "Q4"
    
    def test_rolling_window_enforcement(self):
        """Test that history enforces max_turns limit (rolling window)."""
        history = ConversationHistory(max_turns=3)
        
        # Add 5 turns (exceeds max_turns)
        for i in range(5):
            turn = ConversationTurn(query=f"Q{i}", answer=f"A{i}")
            history.add_turn(turn)
        
        # Should only keep last 3 turns
        assert len(history) == 3
        assert history.turns[0].query == "Q2"  # Oldest is Q2
        assert history.turns[1].query == "Q3"
        assert history.turns[2].query == "Q4"  # Newest is Q4
    
    def test_rolling_window_removes_oldest_first(self):
        """Test that oldest turns are removed first when limit reached."""
        history = ConversationHistory(max_turns=2)
        
        turn1 = ConversationTurn(query="First", answer="A1")
        turn2 = ConversationTurn(query="Second", answer="A2")
        turn3 = ConversationTurn(query="Third", answer="A3")
        
        history.add_turn(turn1)
        history.add_turn(turn2)
        assert len(history) == 2
        
        history.add_turn(turn3)
        assert len(history) == 2
        assert history.turns[0].query == "Second"  # First was removed
        assert history.turns[1].query == "Third"
    
    def test_get_recent_context_empty_history(self):
        """Test get_recent_context with no history."""
        history = ConversationHistory()
        context = history.get_recent_context()
        
        assert context == ""
    
    def test_get_recent_context_single_turn(self):
        """Test get_recent_context with one turn."""
        history = ConversationHistory()
        history.add_turn(ConversationTurn(query="What is AI?", answer="AI is artificial intelligence."))
        
        context = history.get_recent_context()
        
        assert "Q: What is AI?" in context
        assert "A: AI is artificial intelligence." in context
    
    def test_get_recent_context_multiple_turns(self):
        """Test get_recent_context with multiple turns."""
        history = ConversationHistory()
        history.add_turn(ConversationTurn(query="Q1", answer="A1"))
        history.add_turn(ConversationTurn(query="Q2", answer="A2"))
        history.add_turn(ConversationTurn(query="Q3", answer="A3"))
        
        context = history.get_recent_context(n=2)
        
        # Should only include last 2 turns
        assert "Q: Q2" in context
        assert "Q: Q3" in context
        assert "Q: Q1" not in context
    
    def test_get_recent_context_default_n(self):
        """Test get_recent_context with default n=3."""
        history = ConversationHistory()
        for i in range(5):
            history.add_turn(ConversationTurn(query=f"Q{i}", answer=f"A{i}"))
        
        context = history.get_recent_context()  # Default n=3
        
        # Should include last 3 turns
        assert "Q: Q2" in context
        assert "Q: Q3" in context
        assert "Q: Q4" in context
        assert "Q: Q0" not in context
        assert "Q: Q1" not in context
    
    def test_get_recent_context_n_larger_than_history(self):
        """Test get_recent_context when n > number of turns."""
        history = ConversationHistory()
        history.add_turn(ConversationTurn(query="Q1", answer="A1"))
        history.add_turn(ConversationTurn(query="Q2", answer="A2"))
        
        context = history.get_recent_context(n=10)
        
        # Should return all available turns
        assert "Q: Q1" in context
        assert "Q: Q2" in context
    
    def test_get_recent_context_formatting(self):
        """Test that context is formatted correctly with newlines."""
        history = ConversationHistory()
        history.add_turn(ConversationTurn(query="First question", answer="First answer"))
        history.add_turn(ConversationTurn(query="Second question", answer="Second answer"))
        
        context = history.get_recent_context()
        
        # Check proper formatting with double newlines between turns
        expected = "Q: First question\nA: First answer\n\nQ: Second question\nA: Second answer"
        assert context == expected
    
    def test_clear_history(self):
        """Test clearing conversation history."""
        history = ConversationHistory()
        history.add_turn(ConversationTurn(query="Q1", answer="A1"))
        history.add_turn(ConversationTurn(query="Q2", answer="A2"))
        
        assert len(history) == 2
        
        history.clear()
        
        assert len(history) == 0
        assert not history
        assert history.get_recent_context() == ""
    
    def test_clear_empty_history(self):
        """Test clearing already empty history (should not error)."""
        history = ConversationHistory()
        history.clear()  # Should not raise
        assert len(history) == 0
    
    def test_len_operator(self):
        """Test __len__ operator."""
        history = ConversationHistory()
        assert len(history) == 0
        
        history.add_turn(ConversationTurn(query="Q", answer="A"))
        assert len(history) == 1
        
        history.add_turn(ConversationTurn(query="Q", answer="A"))
        assert len(history) == 2
    
    def test_bool_operator(self):
        """Test __bool__ operator."""
        history = ConversationHistory()
        assert not history  # Empty history is False
        
        history.add_turn(ConversationTurn(query="Q", answer="A"))
        assert history  # Non-empty history is True
        
        history.clear()
        assert not history  # Cleared history is False again


class TestConversationHistoryIntegration:
    """Integration tests for realistic usage scenarios."""
    
    def test_typical_conversation_flow(self):
        """Test a typical multi-turn conversation."""
        history = ConversationHistory(max_turns=10)
        
        # Turn 1
        history.add_turn(ConversationTurn(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI...",
            tokens_used=120
        ))
        
        # Turn 2 (follow-up)
        history.add_turn(ConversationTurn(
            query="What are the main types?",
            answer="The main types are supervised, unsupervised, and reinforcement learning.",
            tokens_used=95
        ))
        
        # Turn 3 (another follow-up)
        history.add_turn(ConversationTurn(
            query="Explain supervised learning",
            answer="Supervised learning uses labeled training data...",
            tokens_used=150
        ))
        
        assert len(history) == 3
        
        # Get context for next query
        context = history.get_recent_context(n=2)
        assert "What are the main types?" in context
        assert "Explain supervised learning" in context
        assert "What is machine learning?" not in context  # Too old (n=2)
    
    def test_long_conversation_with_rolling_window(self):
        """Test conversation that exceeds max_turns."""
        history = ConversationHistory(max_turns=5)
        
        # Simulate 10-turn conversation
        for i in range(10):
            history.add_turn(ConversationTurn(
                query=f"Question {i}",
                answer=f"Answer {i}"
            ))
        
        # Should only have last 5 turns
        assert len(history) == 5
        assert history.turns[0].query == "Question 5"
        assert history.turns[4].query == "Question 9"
        
        # Context should only include recent turns
        context = history.get_recent_context(n=3)
        assert "Question 7" in context
        assert "Question 8" in context
        assert "Question 9" in context
    
    def test_conversation_with_sources(self):
        """Test conversation with source citations."""
        history = ConversationHistory()
        
        sources = [
            Source(file_path="textbook.pdf", chunk_index=0, page_number=42, relevance_score=0.95),
            Source(file_path="notes.md", chunk_index=5, relevance_score=0.87)
        ]
        
        history.add_turn(ConversationTurn(
            query="What is gradient descent?",
            answer="Gradient descent is an optimization algorithm...",
            sources=sources
        ))
        
        assert len(history.turns[0].sources) == 2
        assert history.turns[0].sources[0].page_number == 42
        assert history.turns[0].sources[1].file_path == "notes.md"
