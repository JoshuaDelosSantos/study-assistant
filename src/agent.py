"""
RAG Agent Implementation

This module implements the Retrieval-Augmented Generation (RAG) agent that
coordinates vector search, context assembly, and LLM generation to answer queries.
"""

from dataclasses import dataclass
from typing import List, Optional
import logging

from src.config import Config
from src.vector_store import VectorStore, QueryResult
from src.llm_providers import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Source document information for a query result."""
    filename: str
    chunk_index: int
    page: Optional[int] = None  # Only for PDFs
    similarity: float = 0.0


@dataclass
class AgentResponse:
    """Response from the RAG agent."""
    answer: str
    sources: List[Source]
    context_used: str
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class RAGAgent:
    """
    Retrieval-Augmented Generation Agent.
    
    Coordinates retrieval from vector store and generation from LLM to
    answer user queries with context from indexed documents.
    """
    
    def __init__(
        self,
        config: Config,
        vector_store: VectorStore,
        llm_provider: LLMProvider
    ):
        """
        Initialize the RAG agent.
        
        Args:
            config: Application configuration
            vector_store: Vector store for document retrieval
            llm_provider: LLM provider for generation
        """
        self.config = config
        self.vector_store = vector_store
        self.llm_provider = llm_provider
    
    def _format_sources(self, results: List[QueryResult]) -> List[Source]:
        """
        Format query results into source information.
        
        Args:
            results: Query results from vector store
        
        Returns:
            List of formatted source objects
        """
        sources = []
        for result in results:
            metadata = result.metadata or {}
            source = Source(
                filename=metadata.get("filename", "Unknown"),
                chunk_index=metadata.get("chunk_index", 0),
                page=metadata.get("page"),  # Only present for PDFs
                similarity=1.0 - result.distance  # Convert distance to similarity
            )
            sources.append(source)
        return sources
    
    def _build_context(self, results: List[QueryResult]) -> str:
        """
        Build context string from query results.
        
        Args:
            results: Query results from vector store
        
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result.metadata or {}
            filename = metadata.get("filename", "Unknown")
            chunk_index = metadata.get("chunk_index", 0)
            
            # Format source reference
            if "page" in metadata:
                # PDF with page number
                source_ref = f"[{filename}, Page {metadata['page']}, Chunk {chunk_index}]"
            else:
                # Other document types
                source_ref = f"[{filename}, Chunk {chunk_index}]"
            
            context_parts.append(f"{source_ref}\n{result.text}")
        
        return "\n\n".join(context_parts)
    
    def _calculate_token_budget(self, query: str) -> int:
        """
        Calculate available tokens for context based on model limits.
        
        Reserves tokens for:
        - Query text
        - System prompt and template
        - Generation (max_tokens from config)
        
        Args:
            query: User query text
        
        Returns:
            Number of tokens available for context
        """
        max_model_tokens = self.llm_provider.get_max_tokens()
        max_generation = self.config.llm_max_tokens or 2000
        
        # Count tokens in query
        query_tokens = self.llm_provider.count_tokens(query)
        
        # Estimate system prompt tokens (conservative estimate)
        system_prompt_tokens = 100
        
        # Calculate budget
        budget = max_model_tokens - query_tokens - system_prompt_tokens - max_generation
        
        # Ensure positive budget
        return max(0, budget)
    
    def _fit_context_to_budget(self, context: str, budget: int) -> str:
        """
        Truncate context to fit within token budget.
        
        Args:
            context: Full context string
            budget: Maximum tokens allowed
        
        Returns:
            Truncated context that fits within budget
        """
        if budget <= 0:
            return ""
        
        # Count tokens in full context
        context_tokens = self.llm_provider.count_tokens(context)
        
        if context_tokens <= budget:
            return context
        
        # Truncate by characters (approximate)
        # Use 4 chars per token as rough estimate
        target_chars = budget * 4
        truncated = context[:target_chars]
        
        # Find last complete chunk boundary to avoid cutting mid-sentence
        last_double_newline = truncated.rfind("\n\n")
        if last_double_newline > 0:
            truncated = truncated[:last_double_newline]
        
        logger.warning(
            f"Context truncated from {context_tokens} to ~{budget} tokens "
            f"to fit within model limits"
        )
        
        return truncated
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build the final prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Formatted prompt string
        """
        if not context:
            return (
                f"You are a helpful study assistant. Answer the following question:\n\n"
                f"Question: {query}\n\n"
                f"Note: No relevant context was found in the indexed documents. "
                f"Provide a helpful response based on your general knowledge, but "
                f"indicate that the answer is not based on the user's study materials."
            )
        
        return (
            f"You are a helpful study assistant. Use the context below to answer "
            f"the question. If the context doesn't contain relevant information, "
            f"say so and provide a general answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
    
    def process_query(self, query: str) -> AgentResponse:
        """
        Process a user query using RAG pipeline.
        
        Pipeline:
        1. Retrieve relevant chunks from vector store
        2. Calculate token budget for context
        3. Build and truncate context to fit budget
        4. Generate response using LLM
        5. Format and return response with sources
        
        Args:
            query: User query string
        
        Returns:
            AgentResponse with answer and metadata
        
        Raises:
            Exception: If retrieval or generation fails
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant chunks
        try:
            results = self.vector_store.query(
                query_text=query,
                n_results=self.config.top_k
            )
            logger.info(f"Retrieved {len(results)} results from vector store")
        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
            raise
        
        # Step 2: Calculate token budget
        token_budget = self._calculate_token_budget(query)
        logger.debug(f"Token budget for context: {token_budget}")
        
        # Step 3: Build context
        full_context = self._build_context(results)
        context = self._fit_context_to_budget(full_context, token_budget)
        
        # Step 4: Build prompt and generate response
        prompt = self._build_prompt(query, context)
        
        try:
            llm_response: LLMResponse = self.llm_provider.generate(
                prompt=prompt,
                max_tokens=self.config.llm_max_tokens
            )
            logger.info("LLM generation successful")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
        
        # Step 5: Format response
        sources = self._format_sources(results)
        
        response = AgentResponse(
            answer=llm_response.content,
            sources=sources,
            context_used=context,
            tokens_used=llm_response.usage.total_tokens if llm_response.usage else None,
            prompt_tokens=llm_response.usage.prompt_tokens if llm_response.usage else None,
            completion_tokens=llm_response.usage.completion_tokens if llm_response.usage else None
        )
        
        logger.info(
            f"Query processed successfully. "
            f"Sources: {len(sources)}, "
            f"Tokens: {response.tokens_used or 'N/A'}"
        )
        
        return response
