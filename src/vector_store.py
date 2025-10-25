"""
Vector Store Module - ChromaDB wrapper for document embeddings

This module provides a high-level interface to ChromaDB for storing and
querying document embeddings. It handles:
- Collection initialisation and management
- Document storage with metadata
- Semantic similarity search
- Schema versioning and validation
- Persistence across sessions
"""

import json
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.config import Config, DATA_DIR


# ============================================================================
# Constants
# ============================================================================

COLLECTION_NAME = "study_documents"
SCHEMA_VERSION = "2.0"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Document:
    """
    Document to be stored in vector database.
    
    Attributes:
        id: Unique identifier for the document chunk
        text: The actual text content
        metadata: Metadata dict (source, filename, page, etc.)
        embedding: Optional pre-computed embedding vector
    """
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class QueryResult:
    """
    Result from a similarity search query.
    
    Attributes:
        id: Document chunk ID
        text: Document text content
        metadata: Document metadata
        distance: Similarity distance (lower = more similar)
    """
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float


@dataclass
class StoreStats:
    """
    Statistics about the vector store.
    
    Attributes:
        total_documents: Total number of document chunks
        total_chunks: Alias for total_documents (for clarity)
        indexed_files_count: Number of unique files indexed
        schema_version: Current schema version
        embedding_provider: Embedding provider name
        embedding_model: Embedding model name
        embedding_dimension: Embedding vector dimension
        chunk_size: Chunk size used during indexing
        chunk_overlap: Chunk overlap used during indexing
        last_indexed: ISO timestamp of last indexing operation
    """
    total_documents: int = 0
    total_chunks: int = 0
    indexed_files_count: int = 0
    schema_version: str = SCHEMA_VERSION
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    chunk_size: int = 0
    chunk_overlap: int = 0
    last_indexed: Optional[str] = None


# ============================================================================
# Vector Store Class
# ============================================================================

class VectorStore:
    """
    High-level interface to ChromaDB for document storage and retrieval.
    
    This class manages a ChromaDB collection for storing document embeddings
    and provides methods for adding documents, querying, and managing the
    collection lifecycle.
    """
    
    def __init__(self, config: Config):
        """
        Initialize vector store with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self.embedding_function = None
        
    def initialize(self) -> bool:
        """
        Initialize ChromaDB client and collection.
        
        Creates data directory if needed, sets up ChromaDB client,
        and loads or creates the document collection.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Ensure data directory exists
            DATA_DIR.mkdir(exist_ok=True)
            
            # Create ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=str(DATA_DIR),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Set up embedding function based on config
            self.embedding_function = self._create_embedding_function()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_function
                )
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_function,
                    metadata=self._create_initial_metadata()
                )
            
            return True
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            return False
    
    def _create_embedding_function(self):
        """
        Create appropriate embedding function based on config.
        
        Returns:
            ChromaDB embedding function instance
        """
        provider = self.config.embedding_provider
        model = self.config.embedding_model
        
        if provider == "sentence-transformers":
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model
            )
        elif provider == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.config.llm_api_key,
                model_name=model
            )
        elif provider == "gemini":
            # Gemini doesn't have built-in embedding function yet
            # Fall back to sentence-transformers for now
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            # Default to sentence-transformers
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
    
    def _create_initial_metadata(self) -> Dict[str, str]:
        """
        Create initial metadata for new collection.
        
        Returns:
            Metadata dictionary
        """
        indexing_config = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
        }
        
        return {
            "schema_version": SCHEMA_VERSION,
            "indexing_config": json.dumps(indexing_config),
            "indexed_files_count": "0",
            "total_chunks": "0",
            "last_indexed": datetime.now().isoformat()
        }
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the collection in batch.
        
        Args:
            documents: List of Document objects to add
            
        Raises:
            RuntimeError: If store not initialized
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        if not documents:
            return
        
        # Extract components for batch add
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection (ChromaDB will generate embeddings)
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        # Update collection metadata
        self._update_collection_stats()
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """
        Query the collection for similar documents.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            metadata_filter: Optional metadata filter (e.g., {"file_type": ".pdf"})
            
        Returns:
            List of QueryResult objects
            
        Raises:
            RuntimeError: If store not initialized
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        # Query collection
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=metadata_filter
        )
        
        # Convert to QueryResult objects
        query_results = []
        
        # Results structure: {ids: [[...]], documents: [[...]], metadatas: [[...]], distances: [[...]]}
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                query_results.append(QueryResult(
                    id=results["ids"][0][i],
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    distance=results["distances"][0][i]
                ))
        
        return query_results
    
    def get_stats(self) -> StoreStats:
        """
        Get statistics about the vector store.
        
        Returns:
            StoreStats object with collection statistics
            
        Raises:
            RuntimeError: If store not initialized
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        # Get collection count
        count = self.collection.count()
        
        # Get metadata
        metadata = self.collection.metadata or {}
        
        # Parse indexing config
        indexing_config = {}
        if "indexing_config" in metadata:
            try:
                indexing_config = json.loads(metadata["indexing_config"])
            except json.JSONDecodeError:
                pass
        
        # Count unique files
        indexed_files_count = int(metadata.get("indexed_files_count", 0))
        
        stats = StoreStats(
            total_documents=count,
            total_chunks=count,
            indexed_files_count=indexed_files_count,
            schema_version=metadata.get("schema_version", "unknown"),
            embedding_provider=indexing_config.get("embedding_provider", ""),
            embedding_model=indexing_config.get("embedding_model", ""),
            embedding_dimension=self._get_embedding_dimension(),
            chunk_size=indexing_config.get("chunk_size", 0),
            chunk_overlap=indexing_config.get("chunk_overlap", 0),
            last_indexed=metadata.get("last_indexed")
        )
        
        return stats
    
    def _get_embedding_dimension(self) -> int:
        """
        Get embedding dimension from embedding function.
        
        Returns:
            Embedding dimension or 0 if cannot determine
        """
        # Try to get dimension from embedding function
        if self.embedding_function is None:
            return 0
        
        # For sentence transformers, we can test with dummy text
        try:
            dummy_embedding = self.embedding_function(["test"])
            if dummy_embedding and len(dummy_embedding) > 0:
                return len(dummy_embedding[0])
        except Exception:
            pass
        
        # Return default dimensions based on model
        model = self.config.embedding_model
        if "MiniLM" in model:
            return 384
        elif "openai" in self.config.embedding_provider.lower():
            return 1536
        elif "gemini" in self.config.embedding_provider.lower():
            return 768
        
        return 0
    
    def check_schema_version(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if collection schema version matches current version.
        
        Returns:
            Tuple of (matches, stored_config)
            - matches: True if schema version and config match
            - stored_config: Dictionary with stored configuration
            
        Raises:
            RuntimeError: If store not initialized
        """
        if self.collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        metadata = self.collection.metadata or {}
        
        # Parse stored config
        stored_config = {
            "schema_version": metadata.get("schema_version", "unknown")
        }
        
        if "indexing_config" in metadata:
            try:
                indexing_config = json.loads(metadata["indexing_config"])
                stored_config.update(indexing_config)
            except json.JSONDecodeError:
                pass
        
        # Check if matches current config
        matches = (
            stored_config.get("schema_version") == SCHEMA_VERSION and
            stored_config.get("chunk_size") == self.config.chunk_size and
            stored_config.get("chunk_overlap") == self.config.chunk_overlap and
            stored_config.get("embedding_provider") == self.config.embedding_provider and
            stored_config.get("embedding_model") == self.config.embedding_model
        )
        
        return matches, stored_config
    
    def needs_reindex(self) -> Tuple[bool, str]:
        """
        Check if reindexing is needed due to configuration changes.
        
        Returns:
            Tuple of (needs_reindex, reason)
            - needs_reindex: True if reindex required
            - reason: Human-readable explanation
            
        Raises:
            RuntimeError: If store not initialized
        """
        matches, stored_config = self.check_schema_version()
        
        if matches:
            return False, ""
        
        # Build reason string
        reasons = []
        
        if stored_config.get("schema_version") != SCHEMA_VERSION:
            reasons.append(f"Schema version changed: {stored_config.get('schema_version')} → {SCHEMA_VERSION}")
        
        if stored_config.get("chunk_size") != self.config.chunk_size:
            reasons.append(f"Chunk size changed: {stored_config.get('chunk_size')} → {self.config.chunk_size}")
        
        if stored_config.get("chunk_overlap") != self.config.chunk_overlap:
            reasons.append(f"Chunk overlap changed: {stored_config.get('chunk_overlap')} → {self.config.chunk_overlap}")
        
        if stored_config.get("embedding_provider") != self.config.embedding_provider:
            reasons.append(f"Embedding provider changed: {stored_config.get('embedding_provider')} → {self.config.embedding_provider}")
        
        if stored_config.get("embedding_model") != self.config.embedding_model:
            reasons.append(f"Embedding model changed: {stored_config.get('embedding_model')} → {self.config.embedding_model}")
        
        return True, "; ".join(reasons)
    
    def clear(self) -> None:
        """
        Delete the collection entirely.
        
        This is a destructive operation that removes all indexed documents.
        
        Raises:
            RuntimeError: If store not initialized
        """
        if self.client is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
            self.collection = None
        except Exception as e:
            # Collection might not exist, which is fine
            pass
    
    def _update_collection_stats(self) -> None:
        """
        Update collection metadata with current statistics.
        
        This is called after adding documents to keep stats up to date.
        """
        if self.collection is None:
            return
        
        # Count unique files from metadata
        try:
            all_items = self.collection.get()
            unique_files = set()
            
            if all_items["metadatas"]:
                for metadata in all_items["metadatas"]:
                    if "source" in metadata:
                        unique_files.add(metadata["source"])
            
            # Update metadata
            current_metadata = self.collection.metadata or {}
            indexing_config = json.loads(current_metadata.get("indexing_config", "{}"))
            
            updated_metadata = {
                "schema_version": SCHEMA_VERSION,
                "indexing_config": json.dumps({
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap,
                    "embedding_provider": self.config.embedding_provider,
                    "embedding_model": self.config.embedding_model,
                }),
                "indexed_files_count": str(len(unique_files)),
                "total_chunks": str(self.collection.count()),
                "last_indexed": datetime.now().isoformat()
            }
            
            self.collection.modify(metadata=updated_metadata)
            
        except Exception as e:
            # Don't fail if metadata update fails
            pass


# ============================================================================
# Utility Functions
# ============================================================================

def generate_document_id(file_path: Path, chunk_index: int) -> str:
    """
    Generate unique document ID from file path and chunk index.
    
    Uses SHA256 hash of file path (not contents) to create stable IDs
    across indexing runs for the same file.
    
    Args:
        file_path: Path to the source file
        chunk_index: Index of the chunk within the file (0-based)
        
    Returns:
        Unique document ID string
    """
    # Hash the file path (not contents) for stable IDs
    path_str = str(file_path.resolve())
    file_hash = hashlib.sha256(path_str.encode()).hexdigest()[:16]
    
    return f"{file_hash}_{chunk_index}"


# ============================================================================
# Module Testing
# ============================================================================

if __name__ == "__main__":
    # For testing vector store independently
    from src.config import get_default_config
    
    config = get_default_config()
    config.directories = ["."]  # Dummy directory
    
    store = VectorStore(config)
    
    if store.initialize():
        print("Vector store initialized successfully")
        stats = store.get_stats()
        print(f"Total documents: {stats.total_documents}")
        print(f"Schema version: {stats.schema_version}")
    else:
        print("Failed to initialize vector store")
