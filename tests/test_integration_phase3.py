"""
Integration tests for Phase 3: LLM Provider + RAG Agent end-to-end.
"""

import pytest
from pathlib import Path
from src.config import Config
from src.vector_store import VectorStore
from src.indexer import DocumentIndexer
from src.llm_providers import create_provider
from src.agent import RAGAgent


class TestPhase3Integration:
    """Integration tests for Phase 3 functionality."""
    
    def test_end_to_end_rag_pipeline(
        self,
        temp_dir,
        sample_txt_file,
        mock_openai_env,
        mocker
    ):
        """Test complete RAG pipeline from indexing to query answering."""
        # Setup configuration
        config = Config(
            llm_provider="openai",
            llm_model="gpt-4o",
            llm_api_key=mock_openai_env,
            directories=[str(temp_dir)],
            chunk_size=200,
            chunk_overlap=50,
            top_k=3
        )
        
        # Initialize vector store and index documents
        vector_store = VectorStore(config)
        assert vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats = indexer.index_all()
        
        # Verify indexing succeeded
        assert stats.files_processed > 0
        assert stats.chunks_created > 0
        
        # Initialize LLM provider with mocked API
        llm_provider = create_provider(
            provider_type="openai",
            api_key=mock_openai_env,
            model="gpt-4o"
        )
        
        # Mock the LLM response
        from src.llm_providers import LLMResponse, TokenUsage
        mock_response = LLMResponse(
            content="This is a test document for the study assistant.",
            usage=TokenUsage(
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70
            ),
            model="gpt-4o"
        )
        mocker.patch.object(llm_provider, 'generate', return_value=mock_response)
        
        # Initialize RAG agent
        agent = RAGAgent(config, vector_store, llm_provider)
        
        # Process a query
        response = agent.process_query("What is this document about?")
        
        # Verify response structure
        assert response.answer == "This is a test document for the study assistant."
        assert len(response.sources) > 0
        assert response.tokens_used == 70
        assert response.prompt_tokens == 50
        assert response.completion_tokens == 20
        
        # Verify sources have correct metadata
        for source in response.sources:
            assert isinstance(source.filename, str)
            assert len(source.filename) > 0
            assert isinstance(source.chunk_index, int)
            assert source.chunk_index >= 0
            assert 0.0 <= source.similarity <= 1.0
    
    def test_rag_pipeline_with_gemini(
        self,
        temp_dir,
        sample_txt_file,
        mock_gemini_env,
        mocker
    ):
        """Test RAG pipeline with Gemini provider."""
        # Setup configuration for Gemini
        config = Config(
            llm_provider="gemini",
            llm_model="gemini-1.5-flash",
            llm_api_key=mock_gemini_env,
            directories=[str(temp_dir)],
            top_k=2
        )
        
        # Initialize and index
        vector_store = VectorStore(config)
        assert vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        indexer.index_all()
        
        # Initialize Gemini provider with mocked API
        llm_provider = create_provider(
            provider_type="gemini",
            api_key=mock_gemini_env,
            model="gemini-1.5-flash"
        )
        
        # Mock the LLM response
        from src.llm_providers import LLMResponse, TokenUsage
        mock_response = LLMResponse(
            content="Test response from Gemini",
            usage=TokenUsage(
                prompt_tokens=45,
                completion_tokens=15,
                total_tokens=60
            ),
            model="gemini-1.5-flash"
        )
        mocker.patch.object(llm_provider, 'generate', return_value=mock_response)
        
        # Initialize RAG agent
        agent = RAGAgent(config, vector_store, llm_provider)
        
        # Process query
        response = agent.process_query("Test query")
        
        # Verify response
        assert response.answer == "Test response from Gemini"
        assert len(response.sources) > 0
        assert response.tokens_used == 60
    
    def test_rag_pipeline_with_pdf_sources(
        self,
        temp_dir,
        sample_pdf_file,
        mock_openai_env,
        mocker
    ):
        """Test RAG pipeline correctly handles PDF sources with page numbers."""
        # Setup configuration
        config = Config(
            llm_provider="openai",
            llm_model="gpt-4o",
            llm_api_key=mock_openai_env,
            directories=[str(temp_dir)],
            file_types=[".pdf"],
            top_k=3
        )
        
        # Initialize and index
        vector_store = VectorStore(config)
        assert vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats = indexer.index_all()
        
        # Skip if PDF couldn't be processed (reportlab not available)
        if stats.files_processed == 0:
            pytest.skip("PDF processing not available")
        
        # Initialize provider
        llm_provider = create_provider(
            provider_type="openai",
            api_key=mock_openai_env,
            model="gpt-4o"
        )
        
        # Mock LLM response
        from src.llm_providers import LLMResponse, TokenUsage
        mock_response = LLMResponse(
            content="PDF content summary",
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            ),
            model="gpt-4o"
        )
        mocker.patch.object(llm_provider, 'generate', return_value=mock_response)
        
        # Initialize agent
        agent = RAGAgent(config, vector_store, llm_provider)
        
        # Process query
        response = agent.process_query("What's in the PDF?")
        
        # Verify PDF sources have page numbers
        assert len(response.sources) > 0
        for source in response.sources:
            assert source.filename.endswith('.pdf')  # Check it's a PDF file
            assert source.page is not None  # PDFs should have page numbers
            assert isinstance(source.page, int)
            assert source.page > 0
    
    def test_rag_pipeline_no_relevant_documents(
        self,
        temp_dir,
        sample_txt_file,
        mock_openai_env,
        mocker
    ):
        """Test RAG pipeline behavior when no relevant documents found."""
        # Setup configuration
        config = Config(
            llm_provider="openai",
            llm_model="gpt-4o",
            llm_api_key=mock_openai_env,
            directories=[str(temp_dir)],
            similarity_threshold=0.99  # Very high threshold to get no results
        )
        
        # Initialize and index
        vector_store = VectorStore(config)
        assert vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        indexer.index_all()
        
        # Mock vector store to return empty results
        mocker.patch.object(vector_store, 'query', return_value=[])
        
        # Initialize provider
        llm_provider = create_provider(
            provider_type="openai",
            api_key=mock_openai_env,
            model="gpt-4o"
        )
        
        # Mock LLM response for no-context scenario
        from src.llm_providers import LLMResponse, TokenUsage
        mock_response = LLMResponse(
            content="I don't have relevant information in your study materials.",
            usage=TokenUsage(
                prompt_tokens=40,
                completion_tokens=15,
                total_tokens=55
            ),
            model="gpt-4o"
        )
        mocker.patch.object(llm_provider, 'generate', return_value=mock_response)
        
        # Initialize agent
        agent = RAGAgent(config, vector_store, llm_provider)
        
        # Process query
        response = agent.process_query("Completely unrelated query about quantum physics")
        
        # Verify response indicates no relevant context
        assert response.answer == "I don't have relevant information in your study materials."
        assert len(response.sources) == 0
        
        # Verify prompt indicates no context was found
        call_args = llm_provider.generate.call_args
        prompt = call_args[1]['prompt']
        assert "No relevant context" in prompt or "general knowledge" in prompt.lower()
