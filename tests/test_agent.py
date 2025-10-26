"""
Tests for RAG Agent implementation.
"""

import pytest
from unittest.mock import Mock
from src.agent import RAGAgent, AgentResponse, Source
from src.config import Config
from src.vector_store import QueryResult
from src.llm_providers import LLMResponse, TokenUsage


@pytest.fixture
def mock_vector_store(mocker):
    """Create a mock vector store."""
    mock_store = Mock()
    
    # Default query results
    mock_store.query.return_value = [
        QueryResult(
            id="chunk1",
            text="Python is a high-level programming language.",
            metadata={"filename": "python_basics.txt", "chunk_index": 0, "source": "txt"},
            distance=0.2
        ),
        QueryResult(
            id="chunk2",
            text="Python supports multiple programming paradigms.",
            metadata={"filename": "python_basics.txt", "chunk_index": 1, "source": "txt"},
            distance=0.3
        ),
        QueryResult(
            id="chunk3",
            text="Python has dynamic typing and garbage collection.",
            metadata={"filename": "python_advanced.pdf", "chunk_index": 0, "page": 5, "source": "pdf"},
            distance=0.4
        ),
    ]
    
    return mock_store


@pytest.fixture
def mock_llm_provider(mocker):
    """Create a mock LLM provider."""
    mock_provider = Mock()
    
    # Default generation response
    mock_provider.generate.return_value = LLMResponse(
        content="Python is a versatile programming language with many features.",
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        model="gpt-4o"
    )
    
    # Default token counting (4 chars per token)
    mock_provider.count_tokens.side_effect = lambda text: max(1, len(text) // 4)
    
    # Default max tokens
    mock_provider.get_max_tokens.return_value = 128000
    
    return mock_provider


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return Config(
        llm_provider="openai",
        llm_model="gpt-4o",
        llm_api_key="test-key",
        llm_max_tokens=2000,
        top_k=5,
        directories=["/test/dir"]
    )


@pytest.fixture
def rag_agent(test_config, mock_vector_store, mock_llm_provider):
    """Create a RAG agent with mocked dependencies."""
    return RAGAgent(
        config=test_config,
        vector_store=mock_vector_store,
        llm_provider=mock_llm_provider
    )


class TestRAGAgent:
    """Tests for RAG Agent."""
    
    def test_initialization(self, test_config, mock_vector_store, mock_llm_provider):
        """Test agent initialization."""
        agent = RAGAgent(test_config, mock_vector_store, mock_llm_provider)
        assert agent.config == test_config
        assert agent.vector_store == mock_vector_store
        assert agent.llm_provider == mock_llm_provider
    
    def test_format_sources_with_pdf(self, rag_agent):
        """Test formatting sources with PDF (has page number)."""
        results = [
            QueryResult(
                id="chunk1",
                text="Test content",
                metadata={"filename": "test.pdf", "chunk_index": 0, "page": 5},
                distance=0.2
            )
        ]
        
        sources = rag_agent._format_sources(results)
        
        assert len(sources) == 1
        assert sources[0].filename == "test.pdf"
        assert sources[0].chunk_index == 0
        assert sources[0].page == 5
        assert abs(sources[0].similarity - 0.8) < 0.01  # 1.0 - 0.2
    
    def test_format_sources_without_pdf(self, rag_agent):
        """Test formatting sources for non-PDF (no page number)."""
        results = [
            QueryResult(
                id="chunk1",
                text="Test content",
                metadata={"filename": "test.txt", "chunk_index": 2},
                distance=0.3
            )
        ]
        
        sources = rag_agent._format_sources(results)
        
        assert len(sources) == 1
        assert sources[0].filename == "test.txt"
        assert sources[0].chunk_index == 2
        assert sources[0].page is None
        assert abs(sources[0].similarity - 0.7) < 0.01  # 1.0 - 0.3
    
    def test_build_context_with_pdf(self, rag_agent):
        """Test context building with PDF sources."""
        results = [
            QueryResult(
                id="chunk1",
                text="Python is great.",
                metadata={"filename": "book.pdf", "chunk_index": 0, "page": 5},
                distance=0.2
            ),
            QueryResult(
                id="chunk2",
                text="Python is versatile.",
                metadata={"filename": "book.pdf", "chunk_index": 1, "page": 6},
                distance=0.3
            )
        ]
        
        context = rag_agent._build_context(results)
        
        assert "[book.pdf, Page 5, Chunk 0]" in context
        assert "[book.pdf, Page 6, Chunk 1]" in context
        assert "Python is great." in context
        assert "Python is versatile." in context
    
    def test_build_context_without_pdf(self, rag_agent):
        """Test context building with non-PDF sources."""
        results = [
            QueryResult(
                id="chunk1",
                text="Python is great.",
                metadata={"filename": "notes.txt", "chunk_index": 0},
                distance=0.2
            )
        ]
        
        context = rag_agent._build_context(results)
        
        assert "[notes.txt, Chunk 0]" in context
        assert "Python is great." in context
        assert "Page" not in context  # No page reference for non-PDFs
    
    def test_build_context_empty(self, rag_agent):
        """Test context building with no results."""
        context = rag_agent._build_context([])
        assert context == ""
    
    def test_calculate_token_budget(self, rag_agent):
        """Test token budget calculation."""
        query = "What is Python?" * 10  # ~40 chars = ~10 tokens
        
        budget = rag_agent._calculate_token_budget(query)
        
        # Budget should be: 128000 (max) - 10 (query) - 100 (system) - 2000 (generation)
        # = 125890
        assert budget > 100000
        assert budget < 130000
    
    def test_calculate_token_budget_minimum(self, rag_agent, mock_llm_provider):
        """Test token budget doesn't go negative."""
        # Set very low max tokens
        mock_llm_provider.get_max_tokens.return_value = 100
        
        query = "Test query"
        budget = rag_agent._calculate_token_budget(query)
        
        # Should return 0 or positive, not negative
        assert budget >= 0
    
    def test_fit_context_to_budget_no_truncation(self, rag_agent):
        """Test context fitting when within budget."""
        context = "Short context"
        budget = 1000
        
        result = rag_agent._fit_context_to_budget(context, budget)
        
        assert result == context
    
    def test_fit_context_to_budget_with_truncation(self, rag_agent):
        """Test context truncation when exceeding budget."""
        context = "A" * 1000  # Long context
        budget = 10  # Small budget (10 tokens = ~40 chars)
        
        result = rag_agent._fit_context_to_budget(context, budget)
        
        assert len(result) < len(context)
        assert len(result) <= budget * 4  # Rough char estimate
    
    def test_fit_context_to_budget_zero_budget(self, rag_agent):
        """Test context fitting with zero budget."""
        context = "Some context"
        budget = 0
        
        result = rag_agent._fit_context_to_budget(context, budget)
        
        assert result == ""
    
    def test_build_prompt_with_context(self, rag_agent):
        """Test prompt building with context."""
        query = "What is Python?"
        context = "[test.txt, Chunk 0]\nPython is a programming language."
        
        prompt = rag_agent._build_prompt(query, context)
        
        assert "study assistant" in prompt.lower()
        assert query in prompt
        assert context in prompt
        assert "Context:" in prompt
    
    def test_build_prompt_without_context(self, rag_agent):
        """Test prompt building without context."""
        query = "What is Python?"
        context = ""
        
        prompt = rag_agent._build_prompt(query, context)
        
        assert "study assistant" in prompt.lower()
        assert query in prompt
        assert "No relevant context" in prompt
        assert "general knowledge" in prompt.lower()
    
    def test_process_query_success(self, rag_agent, mock_vector_store, mock_llm_provider):
        """Test successful query processing."""
        query = "What is Python?"
        
        response = rag_agent.process_query(query)
        
        # Verify response structure
        assert isinstance(response, AgentResponse)
        assert response.answer == "Python is a versatile programming language with many features."
        assert len(response.sources) == 3
        assert response.tokens_used == 70
        assert response.prompt_tokens == 50
        assert response.completion_tokens == 20
        
        # Verify vector store was queried
        mock_vector_store.query.assert_called_once_with(
            query_text=query,
            n_results=5
        )
        
        # Verify LLM was called
        mock_llm_provider.generate.assert_called_once()
    
    def test_process_query_with_sources(self, rag_agent):
        """Test query processing includes correct sources."""
        query = "Test query"
        
        response = rag_agent.process_query(query)
        
        # Check first source (from mock_vector_store fixture)
        assert response.sources[0].filename == "python_basics.txt"
        assert response.sources[0].chunk_index == 0
        assert response.sources[0].page is None
        
        # Check third source (PDF with page)
        assert response.sources[2].filename == "python_advanced.pdf"
        assert response.sources[2].chunk_index == 0
        assert response.sources[2].page == 5
    
    def test_process_query_no_results(self, rag_agent, mock_vector_store, mock_llm_provider):
        """Test query processing when no results found."""
        mock_vector_store.query.return_value = []
        
        query = "Test query"
        response = rag_agent.process_query(query)
        
        assert isinstance(response, AgentResponse)
        assert len(response.sources) == 0
        assert "No relevant context" in mock_llm_provider.generate.call_args[1]["prompt"]
    
    def test_process_query_vector_store_error(self, rag_agent, mock_vector_store):
        """Test query processing when vector store fails."""
        mock_vector_store.query.side_effect = Exception("Vector store error")
        
        with pytest.raises(Exception, match="Vector store error"):
            rag_agent.process_query("Test query")
    
    def test_process_query_llm_error(self, rag_agent, mock_llm_provider):
        """Test query processing when LLM fails."""
        mock_llm_provider.generate.side_effect = Exception("LLM error")
        
        with pytest.raises(Exception, match="LLM error"):
            rag_agent.process_query("Test query")
    
    def test_process_query_no_usage_info(self, rag_agent, mock_llm_provider):
        """Test query processing when LLM doesn't return usage info."""
        mock_llm_provider.generate.return_value = LLMResponse(
            content="Test response",
            usage=None,
            model="gpt-4o"
        )
        
        response = rag_agent.process_query("Test query")
        
        assert response.tokens_used is None
        assert response.prompt_tokens is None
        assert response.completion_tokens is None


class TestSource:
    """Tests for Source dataclass."""
    
    def test_creation_with_page(self):
        """Test creating Source with page number."""
        source = Source(
            filename="test.pdf",
            chunk_index=5,
            page=10,
            similarity=0.85
        )
        assert source.filename == "test.pdf"
        assert source.chunk_index == 5
        assert source.page == 10
        assert source.similarity == 0.85
    
    def test_creation_without_page(self):
        """Test creating Source without page number."""
        source = Source(
            filename="test.txt",
            chunk_index=3,
            similarity=0.90
        )
        assert source.filename == "test.txt"
        assert source.chunk_index == 3
        assert source.page is None
        assert source.similarity == 0.90


class TestAgentResponse:
    """Tests for AgentResponse dataclass."""
    
    def test_creation_with_tokens(self):
        """Test creating AgentResponse with token info."""
        sources = [Source("test.txt", 0)]
        response = AgentResponse(
            answer="Test answer",
            sources=sources,
            context_used="Test context",
            tokens_used=100,
            prompt_tokens=70,
            completion_tokens=30
        )
        assert response.answer == "Test answer"
        assert response.sources == sources
        assert response.context_used == "Test context"
        assert response.tokens_used == 100
        assert response.prompt_tokens == 70
        assert response.completion_tokens == 30
    
    def test_creation_without_tokens(self):
        """Test creating AgentResponse without token info."""
        sources = [Source("test.txt", 0)]
        response = AgentResponse(
            answer="Test answer",
            sources=sources,
            context_used="Test context"
        )
        assert response.answer == "Test answer"
        assert response.tokens_used is None
        assert response.prompt_tokens is None
        assert response.completion_tokens is None
