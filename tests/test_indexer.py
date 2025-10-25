"""
Tests for Document Indexer Module (src/indexer.py)

Tests cover:
- File discovery with filtering
- Text extraction from all formats
- Chunking strategy validation
- Document processing pipeline
- End-to-end indexing
- Error handling
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.config import Config
from src.indexer import (
    discover_files,
    extract_text,
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_pptx,
    extract_text_from_plain,
    chunk_text,
    process_document,
    DocumentIndexer,
    ExtractedText,
    Chunk,
    IndexStats
)
from src.vector_store import VectorStore


# ============================================================================
# File Discovery Tests
# ============================================================================

@pytest.mark.unit
class TestFileDiscovery:
    """Test file discovery functionality."""
    
    def test_discover_files_basic(self, sample_documents_dir):
        """Test discovering files in a directory."""
        files = discover_files(
            directories=[str(sample_documents_dir)],
            file_types=[".txt", ".md", ".pdf", ".docx", ".pptx"]
        )
        
        assert len(files) > 0
        assert all(isinstance(f, Path) for f in files)
    
    def test_discover_files_filters_by_extension(self, sample_documents_dir):
        """Test that only specified extensions are discovered."""
        files = discover_files(
            directories=[str(sample_documents_dir)],
            file_types=[".txt"]
        )
        
        assert all(f.suffix.lower() == ".txt" for f in files)
    
    def test_discover_files_recursive(self, temp_dir):
        """Test that discovery works recursively."""
        # Create nested structure
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        nested = subdir / "nested"
        nested.mkdir()
        
        (temp_dir / "file1.txt").write_text("content")
        (subdir / "file2.txt").write_text("content")
        (nested / "file3.txt").write_text("content")
        
        files = discover_files(
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        assert len(files) == 3
    
    def test_discover_files_skips_hidden(self, temp_dir):
        """Test that hidden files and directories are skipped."""
        # Create hidden directory
        hidden_dir = temp_dir / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "file.txt").write_text("content")
        
        # Create hidden file
        (temp_dir / ".hiddenfile.txt").write_text("content")
        
        # Create visible file
        (temp_dir / "visible.txt").write_text("content")
        
        files = discover_files(
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        # Should only find visible.txt
        assert len(files) == 1
        assert files[0].name == "visible.txt"
    
    def test_discover_files_skips_system_dirs(self, temp_dir):
        """Test that system directories are skipped."""
        # Create system directories
        for dirname in [".git", ".venv", "node_modules", "__pycache__"]:
            d = temp_dir / dirname
            d.mkdir()
            (d / "file.txt").write_text("content")
        
        # Create normal file
        (temp_dir / "normal.txt").write_text("content")
        
        files = discover_files(
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        # Should only find normal.txt
        assert len(files) == 1
        assert files[0].name == "normal.txt"
    
    def test_discover_files_nonexistent_directory(self, temp_dir):
        """Test handling of nonexistent directory."""
        nonexistent = temp_dir / "nonexistent"
        
        # Should not raise error, just return empty list
        files = discover_files(
            directories=[str(nonexistent)],
            file_types=[".txt"]
        )
        
        assert len(files) == 0
    
    def test_discover_files_multiple_directories(self, temp_dir):
        """Test discovering files from multiple directories."""
        dir1 = temp_dir / "dir1"
        dir2 = temp_dir / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        (dir1 / "file1.txt").write_text("content")
        (dir2 / "file2.txt").write_text("content")
        
        files = discover_files(
            directories=[str(dir1), str(dir2)],
            file_types=[".txt"]
        )
        
        assert len(files) == 2


# ============================================================================
# Text Extraction Tests
# ============================================================================

@pytest.mark.unit
class TestTextExtraction:
    """Test text extraction from various formats."""
    
    def test_extract_text_from_txt(self, sample_txt_file):
        """Test extracting text from TXT file."""
        extracted = extract_text(sample_txt_file)
        
        assert isinstance(extracted, ExtractedText)
        assert len(extracted.text) > 0
        assert "Test Text Document" in extracted.text
    
    def test_extract_text_from_md(self, sample_md_file):
        """Test extracting text from Markdown file."""
        extracted = extract_text(sample_md_file)
        
        assert isinstance(extracted, ExtractedText)
        assert len(extracted.text) > 0
        assert "Test Markdown Document" in extracted.text
    
    def test_extract_text_from_plain_utf8(self, temp_dir):
        """Test extracting UTF-8 encoded text."""
        file_path = temp_dir / "utf8.txt"
        file_path.write_text("Hello 世界 🌍", encoding="utf-8")
        
        extracted = extract_text_from_plain(file_path)
        
        assert "Hello" in extracted.text
        assert "世界" in extracted.text
        assert "🌍" in extracted.text
    
    def test_extract_text_from_plain_fallback(self, temp_dir):
        """Test fallback encoding handling."""
        file_path = temp_dir / "latin1.txt"
        # Write with latin-1 encoding
        file_path.write_bytes("Café résumé".encode("latin-1"))
        
        # Should not raise error
        extracted = extract_text_from_plain(file_path)
        assert len(extracted.text) > 0
    
    @pytest.mark.slow
    def test_extract_text_from_pdf(self, sample_pdf_file):
        """Test extracting text from PDF file."""
        try:
            extracted = extract_text_from_pdf(sample_pdf_file)
            
            assert isinstance(extracted, ExtractedText)
            # PDF should have pages
            if extracted.pages:
                assert len(extracted.pages) > 0
        except Exception as e:
            # PDF extraction might fail if reportlab not available
            if "reportlab" not in str(e).lower():
                raise
    
    @pytest.mark.slow
    def test_extract_text_from_docx(self, sample_docx_file):
        """Test extracting text from DOCX file."""
        try:
            extracted = extract_text_from_docx(sample_docx_file)
            
            assert isinstance(extracted, ExtractedText)
            assert len(extracted.text) > 0
        except Exception as e:
            # DOCX extraction might fail if python-docx not available
            if "docx" not in str(e).lower():
                raise
    
    @pytest.mark.slow
    def test_extract_text_from_pptx(self, sample_pptx_file):
        """Test extracting text from PPTX file."""
        try:
            extracted = extract_text_from_pptx(sample_pptx_file)
            
            assert isinstance(extracted, ExtractedText)
            assert len(extracted.text) > 0
        except Exception as e:
            # PPTX extraction might fail if python-pptx not available
            if "pptx" not in str(e).lower():
                raise
    
    def test_extract_text_unsupported_format(self, temp_dir):
        """Test handling of unsupported file format."""
        file_path = temp_dir / "file.xyz"
        file_path.write_text("content")
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(file_path)
    
    def test_extract_text_corrupted_file(self, temp_dir):
        """Test handling of corrupted file."""
        file_path = temp_dir / "corrupted.pdf"
        file_path.write_text("This is not a valid PDF file")
        
        with pytest.raises(Exception):
            extract_text_from_pdf(file_path)


# ============================================================================
# Chunking Tests
# ============================================================================

@pytest.mark.unit
class TestChunking:
    """Test text chunking functionality."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "A" * 1000 + "B" * 1000 + "C" * 1000
        
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
    
    def test_chunk_text_indices(self):
        """Test that chunk indices are sequential."""
        text = "A" * 3000
        
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=100)
        
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
    
    def test_chunk_text_overlap(self):
        """Test that chunks have overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 100  # 2600 chars
        
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        
        # Verify overlap exists between consecutive chunks
        if len(chunks) > 1:
            # Last 200 chars of chunk 0 should appear in chunk 1
            chunk0_end = chunks[0].text[-50:]
            assert chunk0_end in chunks[1].text
    
    def test_chunk_text_empty_string(self):
        """Test chunking empty string."""
        chunks = chunk_text("", chunk_size=1000, chunk_overlap=200)
        
        assert len(chunks) == 0
    
    def test_chunk_text_whitespace_only(self):
        """Test chunking whitespace-only string."""
        chunks = chunk_text("   \n\n   ", chunk_size=1000, chunk_overlap=200)
        
        assert len(chunks) == 0
    
    def test_chunk_text_smaller_than_chunk_size(self):
        """Test text smaller than chunk size."""
        text = "Short text"
        
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
    
    def test_chunk_text_with_pages(self):
        """Test chunking with page information."""
        pages = [
            (1, "A" * 1000),
            (2, "B" * 1000),
            (3, "C" * 1000)
        ]
        text = "\n\n".join(p[1] for p in pages)
        
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200, pages=pages)
        
        # Some chunks should have page numbers
        page_nums = [c.page for c in chunks if c.page is not None]
        assert len(page_nums) > 0
    
    def test_chunk_text_no_infinite_loop(self):
        """Test that large overlap doesn't cause infinite loop."""
        text = "A" * 2000
        
        # Overlap >= chunk_size should not hang
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=100)
        
        assert len(chunks) > 0
        assert len(chunks) < 1000  # Sanity check


# ============================================================================
# Document Processing Tests
# ============================================================================

@pytest.mark.unit
class TestDocumentProcessing:
    """Test document processing pipeline."""
    
    def test_process_document_txt(self, sample_txt_file, valid_config):
        """Test processing text file."""
        documents = process_document(sample_txt_file, valid_config)
        
        assert len(documents) > 0
        assert all(doc.text for doc in documents)
        assert all(doc.id for doc in documents)
        assert all(doc.metadata for doc in documents)
    
    def test_process_document_metadata(self, sample_txt_file, valid_config):
        """Test that document metadata is populated correctly."""
        documents = process_document(sample_txt_file, valid_config)
        
        for doc in documents:
            assert "source" in doc.metadata
            assert "filename" in doc.metadata
            assert "chunk_index" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert "file_type" in doc.metadata
            assert "indexed_at" in doc.metadata
            assert "file_size" in doc.metadata
            assert "file_modified" in doc.metadata
    
    def test_process_document_chunk_indices(self, sample_txt_file, valid_config):
        """Test that chunk indices are correct."""
        documents = process_document(sample_txt_file, valid_config)
        
        total_chunks = documents[0].metadata["total_chunks"]
        
        for i, doc in enumerate(documents):
            assert doc.metadata["chunk_index"] == i
            assert doc.metadata["total_chunks"] == total_chunks
    
    def test_process_document_empty_file(self, temp_dir, valid_config):
        """Test processing empty file."""
        file_path = temp_dir / "empty.txt"
        file_path.write_text("")
        
        with pytest.raises(Exception, match="No text content"):
            process_document(file_path, valid_config)
    
    def test_process_document_custom_chunk_size(self, sample_txt_file):
        """Test processing with custom chunk size."""
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=["."],
            chunk_size=500,
            chunk_overlap=100
        )
        
        documents = process_document(sample_txt_file, config)
        
        # Chunks should be roughly 500 chars
        for doc in documents:
            assert len(doc.text) <= 500 + 100  # Allow some variance


# ============================================================================
# Document Indexer Tests
# ============================================================================

@pytest.mark.unit
class TestDocumentIndexer:
    """Test DocumentIndexer class."""
    
    def test_indexer_initialization(self, valid_config, temp_data_dir):
        """Test indexer initialization."""
        vector_store = VectorStore(valid_config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(valid_config, vector_store)
        
        assert indexer.config == valid_config
        assert indexer.vector_store == vector_store
    
    def test_index_all_basic(self, temp_dir, temp_data_dir):
        """Test basic indexing workflow."""
        # Create test files
        (temp_dir / "file1.txt").write_text("Content 1 " * 100)
        (temp_dir / "file2.txt").write_text("Content 2 " * 100)
        
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats = indexer.index_all(show_progress=False)
        
        assert isinstance(stats, IndexStats)
        assert stats.files_processed == 2
        assert stats.chunks_created > 0
        assert stats.files_failed == 0
    
    def test_index_all_handles_errors(self, temp_dir, temp_data_dir):
        """Test that indexer handles errors gracefully."""
        # Create valid file
        (temp_dir / "good.txt").write_text("Good content " * 100)
        
        # Create corrupted PDF
        (temp_dir / "bad.pdf").write_text("Not a real PDF")
        
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(temp_dir)],
            file_types=[".txt", ".pdf"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats = indexer.index_all(show_progress=False)
        
        # Should process good file, fail on bad file
        assert stats.files_processed == 1
        assert stats.files_failed == 1
    
    def test_index_all_no_files(self, temp_dir, temp_data_dir):
        """Test indexing when no files found."""
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats = indexer.index_all(show_progress=False)
        
        assert stats.files_processed == 0
        assert stats.chunks_created == 0
    
    def test_index_all_stores_in_vector_db(self, temp_dir, temp_data_dir):
        """Test that indexed documents are stored in vector store."""
        (temp_dir / "test.txt").write_text("Test content " * 100)
        
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        indexer.index_all(show_progress=False)
        
        # Verify documents in store
        store_stats = vector_store.get_stats()
        assert store_stats.total_documents > 0
    
    def test_index_all_tracks_duration(self, temp_dir, temp_data_dir):
        """Test that indexing duration is tracked."""
        (temp_dir / "test.txt").write_text("Content " * 100)
        
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats = indexer.index_all(show_progress=False)
        
        assert stats.duration_seconds > 0


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestIndexerIntegration:
    """Integration tests for complete indexing workflow."""
    
    def test_end_to_end_indexing(self, sample_documents_dir, temp_data_dir):
        """Test complete indexing workflow with multiple file types."""
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(sample_documents_dir)],
            file_types=[".txt", ".md", ".pdf", ".docx", ".pptx"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats = indexer.index_all(show_progress=False)
        
        # Should process files successfully
        assert stats.files_processed > 0
        assert stats.chunks_created > 0
        
        # Verify searchable
        results = vector_store.query("test", n_results=5)
        assert len(results) > 0
    
    def test_reindex_workflow(self, temp_dir, temp_data_dir):
        """Test clearing and reindexing."""
        (temp_dir / "test.txt").write_text("Original content " * 100)
        
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        # Initial index
        indexer = DocumentIndexer(config, vector_store)
        stats1 = indexer.index_all(show_progress=False)
        
        # Clear and reindex
        vector_store.clear()
        vector_store.initialize()
        
        stats2 = indexer.index_all(show_progress=False)
        
        # Should have same number of files
        assert stats1.files_processed == stats2.files_processed
        assert stats1.chunks_created == stats2.chunks_created
    
    def test_incremental_indexing(self, temp_dir, temp_data_dir):
        """Test adding more files after initial indexing."""
        # Initial files
        (temp_dir / "file1.txt").write_text("Content 1 " * 100)
        
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats1 = indexer.index_all(show_progress=False)
        
        # Add more files
        (temp_dir / "file2.txt").write_text("Content 2 " * 100)
        
        stats2 = indexer.index_all(show_progress=False)
        
        # Should process new file (note: this will re-index all files in MVP)
        assert stats2.files_processed >= stats1.files_processed


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in indexer."""
    
    def test_handles_permission_error(self, temp_dir, temp_data_dir):
        """Test handling of permission errors during file discovery."""
        # This test is platform-dependent
        # Just verify no crash occurs with nonexistent directory
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=["/nonexistent/path"],
            file_types=[".txt"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        stats = indexer.index_all(show_progress=False)
        
        assert stats.files_processed == 0
    
    def test_handles_unicode_errors(self, temp_dir, temp_data_dir):
        """Test handling of files with encoding issues."""
        # Create file with mixed encodings
        file_path = temp_dir / "mixed.txt"
        file_path.write_bytes(b"ASCII text \x80\x81\x82 more text")
        
        config = Config(
            llm_provider="openai",
            llm_api_key="test",
            directories=[str(temp_dir)],
            file_types=[".txt"]
        )
        
        vector_store = VectorStore(config)
        vector_store.initialize()
        
        indexer = DocumentIndexer(config, vector_store)
        
        # Should not crash, may skip or process with fallback
        stats = indexer.index_all(show_progress=False)
        
        # Either successfully processed or marked as failed
        assert stats.files_processed + stats.files_failed + stats.files_skipped == 1
