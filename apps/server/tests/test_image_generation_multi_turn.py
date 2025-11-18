"""Tests for image generation with multi-turn chat support."""

import pytest
from unittest.mock import Mock

from services.inference.models import UnifiedRequest, Message, MessageRole, Attachment, AttachmentType
from api.models import RegistryRequest


class TestImageGenerationMultiTurn:
    """Test image generation with multi-turn conversation support."""
    
    def test_registry_request_image_generation_multi_turn(self):
        """Test RegistryRequest for image generation with multi-turn messages."""
        messages = [
            {"role": "system", "content": "You are a creative image generator"},
            {"role": "user", "content": "Create a landscape"},
            {"role": "assistant", "content": "I'll create a beautiful landscape for you"},
            {"role": "user", "content": "Make it more colorful and add a sunset"}
        ]
        
        request = RegistryRequest(
            model="google/gemini-2.5-flash-image",
            messages=messages,
            temperature=0.8
        )
        
        assert request.messages is not None
        assert len(request.messages) == 4
        assert request.messages[0].role == MessageRole.SYSTEM
        assert request.messages[-1].content == "Make it more colorful and add a sunset"
        assert request.prompt is None
    
    def test_unified_request_image_generation_multi_turn(self):
        """Test UnifiedRequest for image generation with multi-turn messages."""
        messages = [
            Message(role=MessageRole.USER, content="Generate an image"),
            Message(role=MessageRole.ASSISTANT, content="What kind of image?"),
            Message(role=MessageRole.USER, content="A futuristic city")
        ]
        
        request = UnifiedRequest(
            messages=messages,
            model="google/gemini-2.5-flash-image",
            response_format="image"
        )
        
        assert request.is_multi_turn()
        assert request.messages == messages
        assert request.response_format == "image"
    
    def test_unified_request_image_generation_with_reference_multi_turn(self):
        """Test UnifiedRequest for image generation with reference image in multi-turn."""
        # Mock reference image
        reference_image = Mock(spec=Attachment)
        reference_image.attachment_type = AttachmentType.IMAGE
        reference_image.mime_type = "image/png"
        reference_image.filename = "reference.png"
        
        messages = [
            Message(role=MessageRole.USER, content="I want to generate an image"),
            Message(role=MessageRole.ASSISTANT, content="What style are you looking for?"),
            Message(role=MessageRole.USER, content="Create something similar to the reference image I'm providing")
        ]
        
        request = UnifiedRequest(
            messages=messages,
            attachments=[reference_image],
            model="google/gemini-2.5-flash-image",
            response_format="image",
            extras={
                "aspect_ratio": "16:9",
                "response_modalities": ["Text", "Image"]
            }
        )
        
        assert request.is_multi_turn()
        assert len(request.attachments) == 1
        assert request.attachments[0] == reference_image
        assert request.extras["aspect_ratio"] == "16:9"
    
    def test_backward_compatibility_single_turn_image_generation(self):
        """Test that single-turn image generation still works."""
        request = UnifiedRequest(
            prompt="Generate a beautiful sunset over mountains",
            model="google/gemini-2.5-flash-image",
            response_format="image",
            extras={"aspect_ratio": "4:3"}
        )
        
        assert not request.is_multi_turn()
        assert request.prompt == "Generate a beautiful sunset over mountains"
        assert request.messages is None
    
    def test_validation_image_generation_no_prompt_or_messages(self):
        """Test validation fails for image generation without prompt or messages."""
        with pytest.raises(ValueError, match="Either 'prompt' or 'messages' must be provided"):
            UnifiedRequest(
                model="google/gemini-2.5-flash-image",
                response_format="image"
            )
    
    def test_api_request_format_image_generation_multi_turn(self):
        """Test the API request format for multi-turn image generation."""
        # This would be the JSON payload sent to /api/ai/chat
        api_payload = {
            "model": "google/gemini-2.5-flash-image",
            "messages": [
                {"role": "system", "content": "You are a creative AI artist"},
                {"role": "user", "content": "I need help creating art"},
                {"role": "assistant", "content": "I'd be happy to help! What kind of art?"},
                {"role": "user", "content": "Create a cyberpunk cityscape with neon lights"}
            ],
            "temperature": 0.9,
            "extras": {
                "aspect_ratio": "21:9",
                "response_modalities": ["Text", "Image"]
            }
        }
        
        # Validate it can be parsed as RegistryRequest
        request = RegistryRequest(**api_payload)
        
        assert request.model == "google/gemini-2.5-flash-image"
        assert len(request.messages) == 4
        assert request.messages[-1].content == "Create a cyberpunk cityscape with neon lights"
        assert request.temperature == 0.9
    
    def test_openai_image_generation_multi_turn_compatibility(self):
        """Test that OpenAI image generation models work with multi-turn (even though they use prompt)."""
        # Note: OpenAI DALL-E models typically use the prompt field, but our system should
        # be able to extract the prompt from messages for backward compatibility
        messages = [
            Message(role=MessageRole.USER, content="Create an image"),
            Message(role=MessageRole.ASSISTANT, content="What should I create?"),
            Message(role=MessageRole.USER, content="A serene lake at dawn")
        ]
        
        request = UnifiedRequest(
            messages=messages,
            model="openai/dall-e-3",
            response_format="image"
        )
        
        assert request.is_multi_turn()
        assert request.messages == messages
        # The OpenAI provider would need to extract the last user message as the prompt


if __name__ == "__main__":
    pytest.main([__file__])
