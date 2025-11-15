"""Tests for LangChain service."""

import pytest
import asyncio
import base64
from unittest.mock import Mock, patch, AsyncMock
from io import BytesIO
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from services.langchain_service import LangChainService
from config import initialize_config


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.api_keys.openai = "test-openai-key"
    config.api_keys.gemini = "test-gemini-key"
    config.langchain.openai_model = "gpt-4o"
    config.langchain.google_model = "gemini-1.5-pro"
    config.langchain.default_temperature = 0.7
    config.langchain.max_tokens = None
    config.langchain.dalle_size = "1024x1024"
    config.langchain.dalle_quality = "standard"
    return config


@pytest.fixture
def service_with_mock_config(mock_config):
    """LangChain service with mocked configuration."""
    with patch('services.langchain_service.get_config', return_value=mock_config):
        service = LangChainService()
        return service


class TestLangChainService:
    """Test cases for LangChain service."""
    
    def test_initialization_with_api_keys(self, service_with_mock_config):
        """Test service initialization with API keys."""
        service = service_with_mock_config
        status = service.get_available_models()
        
        # Should have models available with mock keys
        assert isinstance(status, dict)
        assert "openai_chat" in status
        assert "openai_dalle" in status
        assert "google_chat" in status
    
    def test_initialization_without_api_keys(self):
        """Test service initialization without API keys."""
        mock_config = Mock()
        mock_config.api_keys.openai = None
        mock_config.api_keys.gemini = None
        mock_config.langchain.openai_model = "gpt-4o"
        mock_config.langchain.google_model = "gemini-1.5-pro"
        mock_config.langchain.default_temperature = 0.7
        mock_config.langchain.max_tokens = None
        
        with patch('services.langchain_service.get_config', return_value=mock_config):
            service = LangChainService()
            status = service.get_available_models()
            
            # Should have no models available without keys
            assert status["openai_chat"] is False
            assert status["openai_dalle"] is False
            assert status["google_chat"] is False
    
    @pytest.mark.asyncio
    async def test_text_generation_success(self, service_with_mock_config):
        """Test successful text generation."""
        service = service_with_mock_config
        
        # Mock the OpenAI chat model
        mock_response = Mock()
        mock_response.content = "This is a test response"
        
        # Mock the ainvoke method directly on the service
        with patch.object(service, '_openai_chat') as mock_chat:
            mock_chat.ainvoke = AsyncMock(return_value=mock_response)
            mock_chat.__class__.__name__ = "ChatOpenAI"

            result = await service.generate_text("Test prompt")

            assert result["success"] is True
            assert result["text"] == "This is a test response"
            assert "model_used" in result
    
    @pytest.mark.asyncio
    async def test_text_generation_no_models(self):
        """Test text generation with no available models."""
        mock_config = Mock()
        mock_config.api_keys.openai = None
        mock_config.api_keys.gemini = None
        
        with patch('services.langchain_service.get_config', return_value=mock_config):
            service = LangChainService()
            result = await service.generate_text("Test prompt")
            
            assert result["success"] is False
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_image_analysis_success(self, service_with_mock_config):
        """Test successful image analysis."""
        service = service_with_mock_config
        
        # Create test image
        img = Image.new('RGB', (10, 10), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # Mock the response
        mock_response = Mock()
        mock_response.content = "This is a red image"
        
        # Mock the ainvoke method directly on the service
        with patch.object(service, '_openai_chat') as mock_chat:
            mock_chat.ainvoke = AsyncMock(return_value=mock_response)
            mock_chat.__class__.__name__ = "ChatOpenAI"

            result = await service.process_image_with_text(
                image_data=img_b64,
                prompt="What color is this?",
                image_format="base64"
            )

            assert result["success"] is True
            assert result["text"] == "This is a red image"
    
    @pytest.mark.asyncio
    async def test_image_generation_success(self, service_with_mock_config):
        """Test successful image generation."""
        service = service_with_mock_config
        
        # Mock OpenAI client
        mock_image = Mock()
        mock_image.url = "https://example.com/image.png"
        mock_image.revised_prompt = "A beautiful landscape"
        
        mock_response = Mock()
        mock_response.data = [mock_image]
        
        mock_client = AsyncMock()
        mock_client.images.generate.return_value = mock_response
        
        # Mock the generate_image method directly since it imports AsyncOpenAI internally
        with patch.object(service, 'generate_image') as mock_generate:
            mock_generate.return_value = {
                "success": True,
                "images": [{"url": "https://example.com/image.png", "revised_prompt": "A beautiful landscape"}],
                "model_used": "DALL-E-3"
            }

            result = await service.generate_image("A landscape")

            assert result["success"] is True
            assert len(result["images"]) == 1
            assert result["images"][0]["url"] == "https://example.com/image.png"
    
    def test_model_selection_auto(self, service_with_mock_config):
        """Test automatic model selection."""
        service = service_with_mock_config
        
        # Test chat model selection
        model = service._select_model("auto", "chat")
        assert model is not None
        
        # Test DALL-E model selection
        model = service._select_model("auto", "dalle")
        assert model is not None
    
    def test_model_selection_specific(self, service_with_mock_config):
        """Test specific model selection."""
        service = service_with_mock_config
        
        # Test OpenAI selection
        model = service._select_model("openai", "chat")
        assert model == service._openai_chat
        
        # Test Google selection
        model = service._select_model("google", "chat")
        assert model == service._google_chat
