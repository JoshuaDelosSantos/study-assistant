"""
Tests for LLM provider implementations.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.llm_providers import (
    LLMProvider,
    LLMResponse,
    TokenUsage,
    OpenAIProvider,
    GeminiProvider,
    create_provider
)


class TestOpenAIProvider:
    """Tests for OpenAI provider implementation."""
    
    def test_initialization(self, mock_openai_env):
        """Test OpenAI provider initializes correctly."""
        provider = OpenAIProvider(
            api_key="test-key",
            model="gpt-4o",
            temperature=0.5
        )
        assert provider.model == "gpt-4o"
        assert provider.temperature == 0.5
        assert provider.client is not None
        assert provider.encoding is not None
    
    def test_unknown_model_fallback(self, mock_openai_env, mocker):
        """Test fallback to cl100k_base for unknown models."""
        import tiktoken
        
        # Mock encoding_for_model to raise KeyError
        mocker.patch.object(
            tiktoken,
            'encoding_for_model',
            side_effect=KeyError("unknown model")
        )
        
        # Mock get_encoding to return a valid encoding
        mock_encoding = Mock()
        mocker.patch.object(tiktoken, 'get_encoding', return_value=mock_encoding)
        
        provider = OpenAIProvider(api_key="test-key", model="unknown-model")
        assert provider.encoding == mock_encoding
    
    def test_generate(self, mock_openai_env, mocker):
        """Test generation with OpenAI API."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        mock_response.model = "gpt-4o"
        
        mocker.patch.object(
            provider.client.chat.completions,
            'create',
            return_value=mock_response
        )
        
        response = provider.generate("Test prompt", max_tokens=100)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.model == "gpt-4o"
    
    def test_count_tokens(self, mock_openai_env):
        """Test token counting with tiktoken."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        
        # Test normal text
        count = provider.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)
        
        # Test empty string
        assert provider.count_tokens("") == 0
        
        # Test minimum token floor
        count = provider.count_tokens("a")
        assert count >= 1
    
    def test_count_tokens_fallback(self, mock_openai_env, mocker):
        """Test token counting fallback on error."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        
        # Mock encoding to raise exception
        mocker.patch.object(
            provider.encoding,
            'encode',
            side_effect=Exception("Encoding error")
        )
        
        # Should use fallback (len/4, minimum 1)
        count = provider.count_tokens("Hello, world!")
        assert count >= 1
        assert isinstance(count, int)
    
    def test_get_max_tokens(self, mock_openai_env):
        """Test getting maximum token limits."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-4o")
        assert provider.get_max_tokens() == 128000
        
        provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
        assert provider.get_max_tokens() == 16385
        
        # Unknown model should return conservative default
        provider = OpenAIProvider(api_key="test-key", model="unknown-model")
        assert provider.get_max_tokens() == 4096


class TestGeminiProvider:
    """Tests for Gemini provider implementation."""
    
    def test_initialization(self, mock_gemini_env):
        """Test Gemini provider initializes correctly."""
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-1.5-flash",
            temperature=0.5
        )
        assert provider.model == "gemini-1.5-flash"
        assert provider.temperature == 0.5
        assert provider.model_obj is not None
    
    def test_generate(self, mock_gemini_env, mocker):
        """Test generation with Gemini API."""
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-flash")
        
        # Mock the API response
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_response.usage_metadata = Mock(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15
        )
        
        mocker.patch.object(
            provider.model_obj,
            'generate_content',
            return_value=mock_response
        )
        
        response = provider.generate("Test prompt", max_tokens=100)
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.model == "gemini-1.5-flash"
    
    def test_count_tokens(self, mock_gemini_env, mocker):
        """Test token counting with Gemini API."""
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-flash")
        
        # Mock count_tokens response
        mock_result = Mock(total_tokens=5)
        mocker.patch.object(
            provider.model_obj,
            'count_tokens',
            return_value=mock_result
        )
        
        count = provider.count_tokens("Hello, world!")
        assert count == 5
        
        # Test empty string
        assert provider.count_tokens("") == 0
        
        # Test minimum token floor
        mock_result.total_tokens = 0
        count = provider.count_tokens("a")
        assert count >= 1
    
    def test_count_tokens_fallback(self, mock_gemini_env, mocker):
        """Test token counting fallback on error."""
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-flash")
        
        # Mock count_tokens to raise exception
        mocker.patch.object(
            provider.model_obj,
            'count_tokens',
            side_effect=Exception("API error")
        )
        
        # Should use fallback (len/4, minimum 1)
        count = provider.count_tokens("Hello, world!")
        assert count >= 1
        assert isinstance(count, int)
    
    def test_get_max_tokens(self, mock_gemini_env):
        """Test getting maximum token limits."""
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-flash")
        assert provider.get_max_tokens() == 1000000
        
        provider = GeminiProvider(api_key="test-key", model="gemini-1.5-pro")
        assert provider.get_max_tokens() == 2000000
        
        # Unknown model should return conservative default
        provider = GeminiProvider(api_key="test-key", model="unknown-model")
        assert provider.get_max_tokens() == 32760


class TestCreateProvider:
    """Tests for provider factory function."""
    
    def test_create_openai_provider(self, mock_openai_env):
        """Test creating OpenAI provider."""
        provider = create_provider(
            provider_type="openai",
            api_key="test-key",
            model="gpt-4o",
            temperature=0.5
        )
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o"
        assert provider.temperature == 0.5
    
    def test_create_gemini_provider(self, mock_gemini_env):
        """Test creating Gemini provider."""
        provider = create_provider(
            provider_type="gemini",
            api_key="test-key",
            model="gemini-1.5-flash",
            temperature=0.5
        )
        assert isinstance(provider, GeminiProvider)
        assert provider.model == "gemini-1.5-flash"
        assert provider.temperature == 0.5
    
    def test_case_insensitive(self, mock_openai_env):
        """Test provider type is case-insensitive."""
        provider = create_provider(
            provider_type="OPENAI",
            api_key="test-key",
            model="gpt-4o"
        )
        assert isinstance(provider, OpenAIProvider)
    
    def test_unsupported_provider(self):
        """Test error on unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_provider(
                provider_type="unsupported",
                api_key="test-key",
                model="model"
            )


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""
    
    def test_creation(self):
        """Test creating TokenUsage instance."""
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""
    
    def test_creation_with_usage(self):
        """Test creating LLMResponse with usage."""
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        response = LLMResponse(
            content="Test response",
            usage=usage,
            model="gpt-4o"
        )
        assert response.content == "Test response"
        assert response.usage == usage
        assert response.model == "gpt-4o"
    
    def test_creation_without_usage(self):
        """Test creating LLMResponse without usage."""
        response = LLMResponse(content="Test response")
        assert response.content == "Test response"
        assert response.usage is None
        assert response.model is None
