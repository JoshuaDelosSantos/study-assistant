"""
Tests for Vector Store Module (src/vector_store.py)

Tests cover:
- Vector store initialization
- Document addition and retrieval
- Query functionality
- Metadata filtering
- Statistics retrieval
- Schema versioning
- Reindex detection
- Persistence across sessions
"""

import pytest
from pathlib import Path
import json

from src.config import Config
from src.vector_store import (
    VectorStore,
    Document,
    QueryResult,
    StoreStats,
    generate_document_id,
    SCHEMA_VERSION
)


# ============================================================================
# Vector Store Initialization Tests
# ============================================================================

@pytest.mark.unit
class TestVectorStoreInitialization:
    """Test vector store initialization."""
    
    def test_initialization_success(self, temp_data_dir, valid_config):
        """Test successful vector store initialization."""
        store = VectorStore(valid_config)
        
        assert store.initialize() is True
        assert store.client is not None
        assert store.collection is not None
        assert store.embedding_function is not None
    
    def test_data_directory_created(self, temp_dir, monkeypatch, valid_config):
        """Test that data directory is created if missing."""
        import src.config as config_module
        import src.vector_store as vector_store_module
        
        data_dir = temp_dir / "new_data"
        monkeypatch.setattr(config_module, "DATA_DIR", data_dir)
        monkeypatch.setattr(vector_store_module, "DATA_DIR", data_dir)
        
        assert not data_dir.exists()
        
        store = VectorStore(valid_config)
        store.initialize()
        
        assert data_dir.exists()
    
    def test_collection_persists(self, temp_data_dir, valid_config):
        """Test that collection persists across instances."""
        # Create first store and add document
        store1 = VectorStore(valid_config)
        store1.initialize()
        
        doc = Document(
            id="test_1",
            text="Test document",
            metadata={"source": "test.txt"}
        )
        store1.add_documents([doc])
        
        # Create second store and verify document exists
        store2 = VectorStore(valid_config)
        store2.initialize()
        
        stats = store2.get_stats()
        assert stats.total_documents == 1
    
    def test_embedding_function_sentence_transformers(self, temp_data_dir):
        """Test sentence-transformers embedding function creation."""
        config = Config(
            embedding_provider="sentence-transformers",
            embedding_model="all-MiniLM-L6-v2",
            directories=["."]
        )
        
        store = VectorStore(config)
        store.initialize()
        
        assert store.embedding_function is not None
    
    def test_embedding_function_openai(self, temp_data_dir):
        """Test OpenAI embedding function creation."""
        config = Config(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            llm_api_key="sk-test-key",
            directories=["."]
        )
        
        store = VectorStore(config)
        store.initialize()
        
        assert store.embedding_function is not None


# ============================================================================
# Document Addition Tests
# ============================================================================

@pytest.mark.unit
class TestDocumentAddition:
    """Test adding documents to vector store."""
    
    def test_add_single_document(self, temp_data_dir, valid_config):
        """Test adding a single document."""
        store = VectorStore(valid_config)
        store.initialize()
        
        doc = Document(
            id="doc_1",
            text="This is a test document about machine learning.",
            metadata={
                "source": "/test/file.txt",
                "filename": "file.txt",
                "chunk_index": 0
            }
        )
        
        store.add_documents([doc])
        
        stats = store.get_stats()
        assert stats.total_documents == 1
    
    def test_add_multiple_documents(self, temp_data_dir, valid_config):
        """Test adding multiple documents in batch."""
        store = VectorStore(valid_config)
        store.initialize()
        
        docs = [
            Document(
                id=f"doc_{i}",
                text=f"Document {i} content",
                metadata={"source": f"file{i}.txt"}
            )
            for i in range(10)
        ]
        
        store.add_documents(docs)
        
        stats = store.get_stats()
        assert stats.total_documents == 10
    
    def test_add_documents_with_metadata(self, temp_data_dir, valid_config):
        """Test that metadata is stored correctly."""
        store = VectorStore(valid_config)
        store.initialize()
        
        doc = Document(
            id="doc_meta",
            text="Document with rich metadata",
            metadata={
                "source": "/test/document.pdf",
                "filename": "document.pdf",
                "page": 5,
                "chunk_index": 2,
                "file_type": ".pdf"
            }
        )
        
        store.add_documents([doc])
        
        # Query to retrieve and verify metadata
        results = store.query("document", n_results=1)
        assert len(results) == 1
        assert results[0].metadata["page"] == 5
        assert results[0].metadata["chunk_index"] == 2
    
    def test_add_empty_list(self, temp_data_dir, valid_config):
        """Test adding empty document list."""
        store = VectorStore(valid_config)
        store.initialize()
        
        # Should not raise error
        store.add_documents([])
        
        stats = store.get_stats()
        assert stats.total_documents == 0
    
    def test_add_documents_not_initialized(self, valid_config):
        """Test that adding documents fails if not initialized."""
        store = VectorStore(valid_config)
        
        doc = Document(id="test", text="test", metadata={})
        
        with pytest.raises(RuntimeError, match="not initialized"):
            store.add_documents([doc])


# ============================================================================
# Query Tests
# ============================================================================

@pytest.mark.unit
class TestQuery:
    """Test querying vector store."""
    
    def test_query_returns_results(self, temp_data_dir, valid_config):
        """Test that query returns relevant results."""
        store = VectorStore(valid_config)
        store.initialize()
        
        # Add documents
        docs = [
            Document(id="doc_1", text="Python programming tutorial", metadata={"topic": "python"}),
            Document(id="doc_2", text="Machine learning basics", metadata={"topic": "ml"}),
            Document(id="doc_3", text="Python data structures", metadata={"topic": "python"}),
        ]
        store.add_documents(docs)
        
        # Query
        results = store.query("python programming", n_results=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, QueryResult) for r in results)
    
    def test_query_with_metadata_filter(self, temp_data_dir, valid_config):
        """Test querying with metadata filtering."""
        store = VectorStore(valid_config)
        store.initialize()
        
        # Add documents with different types
        docs = [
            Document(id="doc_1", text="PDF content", metadata={"file_type": ".pdf"}),
            Document(id="doc_2", text="Text content", metadata={"file_type": ".txt"}),
            Document(id="doc_3", text="More PDF content", metadata={"file_type": ".pdf"}),
        ]
        store.add_documents(docs)
        
        # Query with filter
        results = store.query("content", n_results=10, metadata_filter={"file_type": ".pdf"})
        
        # All results should be PDFs
        for result in results:
            assert result.metadata["file_type"] == ".pdf"
    
    def test_query_empty_store(self, temp_data_dir, valid_config):
        """Test querying an empty store."""
        store = VectorStore(valid_config)
        store.initialize()
        
        results = store.query("test query", n_results=5)
        
        assert len(results) == 0
    
    def test_query_not_initialized(self, valid_config):
        """Test that querying fails if not initialized."""
        store = VectorStore(valid_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            store.query("test")
    
    def test_query_result_structure(self, temp_data_dir, valid_config):
        """Test that query results have correct structure."""
        store = VectorStore(valid_config)
        store.initialize()
        
        doc = Document(
            id="doc_1",
            text="Test document for query",
            metadata={"source": "test.txt"}
        )
        store.add_documents([doc])
        
        results = store.query("document", n_results=1)
        
        assert len(results) == 1
        result = results[0]
        
        assert hasattr(result, "id")
        assert hasattr(result, "text")
        assert hasattr(result, "metadata")
        assert hasattr(result, "distance")
        assert isinstance(result.distance, float)


# ============================================================================
# Statistics Tests
# ============================================================================

@pytest.mark.unit
class TestStatistics:
    """Test statistics retrieval."""
    
    def test_get_stats_empty_store(self, temp_data_dir, valid_config):
        """Test stats for empty store."""
        store = VectorStore(valid_config)
        store.initialize()
        
        stats = store.get_stats()
        
        assert isinstance(stats, StoreStats)
        assert stats.total_documents == 0
        assert stats.total_chunks == 0
        assert stats.schema_version == SCHEMA_VERSION
    
    def test_get_stats_with_documents(self, temp_data_dir, valid_config):
        """Test stats after adding documents."""
        store = VectorStore(valid_config)
        store.initialize()
        
        # Add 5 documents
        docs = [
            Document(id=f"doc_{i}", text=f"Content {i}", metadata={"source": f"file{i}.txt"})
            for i in range(5)
        ]
        store.add_documents(docs)
        
        stats = store.get_stats()
        
        assert stats.total_documents == 5
        assert stats.total_chunks == 5
        assert stats.indexed_files_count == 5
    
    def test_get_stats_includes_config(self, temp_data_dir, valid_config):
        """Test that stats include configuration details."""
        store = VectorStore(valid_config)
        store.initialize()
        
        stats = store.get_stats()
        
        assert stats.embedding_provider != ""
        assert stats.embedding_model != ""
        assert stats.chunk_size > 0
        assert stats.chunk_overlap >= 0
    
    def test_get_stats_not_initialized(self, valid_config):
        """Test that get_stats fails if not initialized."""
        store = VectorStore(valid_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            store.get_stats()


# ============================================================================
# Schema Version Tests
# ============================================================================

@pytest.mark.unit
class TestSchemaVersioning:
    """Test schema version checking."""
    
    def test_check_schema_version_matches(self, temp_data_dir, valid_config):
        """Test schema version check when config matches."""
        store = VectorStore(valid_config)
        store.initialize()
        
        matches, stored_config = store.check_schema_version()
        
        assert matches is True
        assert stored_config["schema_version"] == SCHEMA_VERSION
    
    def test_check_schema_version_mismatch_chunk_size(self, temp_data_dir, valid_config):
        """Test schema version check with different chunk size."""
        # Initialize with one config
        store1 = VectorStore(valid_config)
        store1.initialize()
        
        # Create new config with different chunk size
        new_config = Config(
            llm_provider=valid_config.llm_provider,
            llm_api_key=valid_config.llm_api_key,
            directories=valid_config.directories,
            chunk_size=2000  # Different from default
        )
        
        store2 = VectorStore(new_config)
        store2.initialize()
        
        matches, stored_config = store2.check_schema_version()
        
        assert matches is False
    
    def test_check_schema_version_mismatch_model(self, temp_data_dir, valid_config):
        """Test schema version check with different embedding model."""
        store1 = VectorStore(valid_config)
        store1.initialize()
        
        new_config = Config(
            llm_provider=valid_config.llm_provider,
            llm_api_key=valid_config.llm_api_key,
            directories=valid_config.directories,
            embedding_provider="sentence-transformers",
            embedding_model="all-mpnet-base-v2"  # Different valid model
        )
        
        store2 = VectorStore(new_config)
        store2.initialize()
        
        matches, stored_config = store2.check_schema_version()
        
        assert matches is False


# ============================================================================
# Reindex Detection Tests
# ============================================================================

@pytest.mark.unit
class TestReindexDetection:
    """Test reindex need detection."""
    
    def test_needs_reindex_false_when_matching(self, temp_data_dir, valid_config):
        """Test that reindex not needed when config matches."""
        store = VectorStore(valid_config)
        store.initialize()
        
        needs_reindex, reason = store.needs_reindex()
        
        assert needs_reindex is False
        assert reason == ""
    
    def test_needs_reindex_true_when_mismatch(self, temp_data_dir, valid_config):
        """Test that reindex needed when config changes."""
        store1 = VectorStore(valid_config)
        store1.initialize()
        
        # Change config
        new_config = Config(
            llm_provider=valid_config.llm_provider,
            llm_api_key=valid_config.llm_api_key,
            directories=valid_config.directories,
            chunk_size=2000
        )
        
        store2 = VectorStore(new_config)
        store2.initialize()
        
        needs_reindex, reason = store2.needs_reindex()
        
        assert needs_reindex is True
        assert "chunk size" in reason.lower()
    
    def test_needs_reindex_reason_includes_details(self, temp_data_dir, valid_config):
        """Test that reindex reason includes specific changes."""
        store1 = VectorStore(valid_config)
        store1.initialize()
        
        new_config = Config(
            llm_provider=valid_config.llm_provider,
            llm_api_key=valid_config.llm_api_key,
            directories=valid_config.directories,
            chunk_size=2000,
            chunk_overlap=300
        )
        
        store2 = VectorStore(new_config)
        store2.initialize()
        
        needs_reindex, reason = store2.needs_reindex()
        
        assert needs_reindex is True
        assert "1000" in reason  # Old chunk size
        assert "2000" in reason  # New chunk size


# ============================================================================
# Collection Management Tests
# ============================================================================

@pytest.mark.unit
class TestCollectionManagement:
    """Test collection management operations."""
    
    def test_clear_collection(self, temp_data_dir, valid_config):
        """Test clearing collection."""
        store = VectorStore(valid_config)
        store.initialize()
        
        # Add documents
        docs = [
            Document(id=f"doc_{i}", text=f"Content {i}", metadata={"source": f"file{i}.txt"})
            for i in range(5)
        ]
        store.add_documents(docs)
        
        # Clear collection
        store.clear()
        
        # Reinitialize and check it's empty
        store.initialize()
        stats = store.get_stats()
        
        assert stats.total_documents == 0
    
    def test_clear_not_initialized(self, valid_config):
        """Test that clear fails if not initialized."""
        store = VectorStore(valid_config)
        
        with pytest.raises(RuntimeError, match="not initialized"):
            store.clear()
    
    def test_clear_nonexistent_collection(self, temp_data_dir, valid_config):
        """Test clearing when collection doesn't exist."""
        store = VectorStore(valid_config)
        store.initialize()
        
        # Clear should succeed even if collection doesn't exist
        store.clear()
        store.clear()  # Should not raise


# ============================================================================
# Utility Function Tests
# ============================================================================

@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_generate_document_id_consistent(self, temp_dir):
        """Test that document ID is consistent for same file."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("content")
        
        id1 = generate_document_id(file_path, 0)
        id2 = generate_document_id(file_path, 0)
        
        assert id1 == id2
    
    def test_generate_document_id_different_indices(self, temp_dir):
        """Test that different indices produce different IDs."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("content")
        
        id1 = generate_document_id(file_path, 0)
        id2 = generate_document_id(file_path, 1)
        
        assert id1 != id2
    
    def test_generate_document_id_different_files(self, temp_dir):
        """Test that different files produce different IDs."""
        file1 = temp_dir / "test1.txt"
        file2 = temp_dir / "test2.txt"
        file1.write_text("content")
        file2.write_text("content")
        
        id1 = generate_document_id(file1, 0)
        id2 = generate_document_id(file2, 0)
        
        assert id1 != id2
    
    def test_generate_document_id_format(self, temp_dir):
        """Test that document ID has expected format."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("content")
        
        doc_id = generate_document_id(file_path, 5)
        
        # Should be hash_index format
        assert "_" in doc_id
        parts = doc_id.split("_")
        assert len(parts) == 2
        assert parts[1] == "5"


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestVectorStoreIntegration:
    """Integration tests for vector store."""
    
    def test_full_workflow(self, temp_data_dir, valid_config):
        """Test complete workflow: init → add → query → stats."""
        store = VectorStore(valid_config)
        
        # Initialize
        assert store.initialize() is True
        
        # Add documents
        docs = [
            Document(
                id="ml_1",
                text="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "ml.txt", "topic": "ml"}
            ),
            Document(
                id="python_1",
                text="Python is a popular programming language.",
                metadata={"source": "python.txt", "topic": "python"}
            ),
        ]
        store.add_documents(docs)
        
        # Query
        results = store.query("machine learning", n_results=1)
        assert len(results) == 1
        assert "machine learning" in results[0].text.lower()
        
        # Stats
        stats = store.get_stats()
        assert stats.total_documents == 2
        
        # Clear
        store.clear()
        store.initialize()
        stats = store.get_stats()
        assert stats.total_documents == 0
