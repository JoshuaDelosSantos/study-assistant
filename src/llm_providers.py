"""
LLM Provider Abstraction Layer

This module provides a unified interface for interacting with different LLM providers
(OpenAI, Google Gemini) with consistent token counting and response handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage statistics for an LLM request."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Unified response format from LLM providers."""
    content: str
    usage: Optional[TokenUsage] = None
    model: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model: str, temperature: float = 0.7):
        """
        Initialize the LLM provider.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "gemini-1.5-flash")
            temperature: Sampling temperature for generation (0.0-1.0)
        """
        self.model = model
        self.temperature = temperature
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate (None for provider default)
        
        Returns:
            LLMResponse with generated content and metadata
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.
        
        Args:
            text: Text to tokenize
        
        Returns:
            Number of tokens (minimum 1 for non-empty text)
        """
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """
        Get the maximum context window size for this model.
        
        Returns:
            Maximum number of tokens supported by the model
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""
    
    # Known model context limits
    MODEL_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
    }
    
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.7):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model identifier
            temperature: Sampling temperature
        """
        super().__init__(model, temperature)
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai>=1.12.0"
            )
        
        # Initialize tokenizer with fallback
        try:
            import tiktoken
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                logger.warning(
                    f"Model {model} not recognized by tiktoken, "
                    f"falling back to cl100k_base encoding"
                )
                self.encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            raise ImportError(
                "tiktoken package not installed. "
                "Install with: pip install tiktoken>=0.5.2"
            )
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=usage,
                model=response.model
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if not text:
            return 0
        try:
            token_count = len(self.encoding.encode(text))
            return max(1, token_count)  # Ensure minimum 1 token for non-empty text
        except Exception as e:
            logger.warning(f"Token counting error: {e}, using fallback")
            # Fallback: ~4 chars per token, minimum 1
            return max(1, len(text) // 4)
    
    def get_max_tokens(self) -> int:
        """Get maximum context window size."""
        return self.MODEL_LIMITS.get(self.model, 4096)  # Conservative default


class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation."""
    
    # Known model context limits
    MODEL_LIMITS = {
        "gemini-1.5-pro": 2000000,
        "gemini-1.5-flash": 1000000,
        "gemini-1.0-pro": 32760,
    }
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google API key
            model: Model identifier
            temperature: Sampling temperature
        """
        super().__init__(model, temperature)
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model_obj = genai.GenerativeModel(model)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Install with: pip install google-generativeai>=0.3.0"
            )
    
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate response using Gemini API."""
        try:
            generation_config = {
                "temperature": self.temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            
            response = self.model_obj.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Gemini returns token usage in response
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = TokenUsage(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    completion_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count
                )
            
            return LLMResponse(
                content=response.text,
                usage=usage,
                model=self.model
            )
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's count_tokens API."""
        if not text:
            return 0
        try:
            result = self.model_obj.count_tokens(text)
            return max(1, result.total_tokens)  # Ensure minimum 1 token
        except Exception as e:
            logger.warning(f"Token counting error: {e}, using fallback")
            # Fallback: ~4 chars per token, minimum 1
            return max(1, len(text) // 4)
    
    def get_max_tokens(self) -> int:
        """Get maximum context window size."""
        return self.MODEL_LIMITS.get(self.model, 32760)  # Conservative default


def create_provider(provider_type: str, api_key: str, model: str, temperature: float = 0.7) -> LLMProvider:
    """
    Factory function to create the appropriate LLM provider.
    
    Args:
        provider_type: "openai" or "gemini"
        api_key: API key for the provider
        model: Model identifier
        temperature: Sampling temperature
    
    Returns:
        Initialized LLM provider instance
    
    Raises:
        ValueError: If provider_type is not supported
    """
    provider_type = provider_type.lower()
    
    if provider_type == "openai":
        return OpenAIProvider(api_key=api_key, model=model, temperature=temperature)
    elif provider_type == "gemini":
        return GeminiProvider(api_key=api_key, model=model, temperature=temperature)
    else:
        raise ValueError(
            f"Unsupported provider: {provider_type}. "
            f"Supported providers: openai, gemini"
        )
