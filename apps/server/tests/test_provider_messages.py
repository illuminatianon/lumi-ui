"""Tests for provider message building with multi-turn support."""

import pytest
from unittest.mock import Mock, AsyncMock

from services.inference.models import UnifiedRequest, Message, MessageRole, Attachment, AttachmentType
from services.inference.providers.openai import OpenAIProvider
from services.inference.providers.google import GoogleProvider


class TestOpenAIProviderMessages:
    """Test OpenAI provider message building."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = OpenAIProvider(config={"api_key": "test-key"})
    
    def test_build_messages_single_turn(self):
        """Test building messages from single-turn request."""
        request = UnifiedRequest(
            prompt="Hello, world!",
            system_message="You are helpful",
            model="gpt-4o"
        )
        
        messages = self.provider._build_messages(request)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello, world!"}
        ]
        assert messages == expected
    
    def test_build_messages_single_turn_no_system(self):
        """Test building messages from single-turn request without system message."""
        request = UnifiedRequest(
            prompt="Hello, world!",
            model="gpt-4o"
        )
        
        messages = self.provider._build_messages(request)
        
        expected = [
            {"role": "user", "content": "Hello, world!"}
        ]
        assert messages == expected
    
    def test_build_messages_multi_turn(self):
        """Test building messages from multi-turn request."""
        messages_input = [
            Message(role=MessageRole.SYSTEM, content="You are helpful"),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi! How can I help?"),
            Message(role=MessageRole.USER, content="What's the weather?")
        ]
        request = UnifiedRequest(
            messages=messages_input,
            model="gpt-4o"
        )
        
        messages = self.provider._build_messages(request)
        
        expected = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's the weather?"}
        ]
        assert messages == expected
    
    def test_build_messages_with_attachments_single_turn(self):
        """Test building messages with attachments for single-turn request."""
        # Mock attachment
        attachment = Mock(spec=Attachment)
        attachment.attachment_type = AttachmentType.IMAGE
        attachment.mime_type = "image/png"
        attachment.to_base64.return_value = "base64data"
        
        request = UnifiedRequest(
            prompt="What's in this image?",
            attachments=[attachment],
            model="gpt-4o"
        )
        
        messages = self.provider._build_messages_with_attachments(request)
        
        expected = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,base64data"}
                    }
                ]
            }
        ]
        assert messages == expected
    
    def test_build_messages_with_attachments_multi_turn(self):
        """Test building messages with attachments for multi-turn request."""
        # Mock attachment
        attachment = Mock(spec=Attachment)
        attachment.attachment_type = AttachmentType.IMAGE
        attachment.mime_type = "image/png"
        attachment.to_base64.return_value = "base64data"
        
        messages_input = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi!"),
            Message(role=MessageRole.USER, content="What's in this image?")
        ]
        request = UnifiedRequest(
            messages=messages_input,
            attachments=[attachment],
            model="gpt-4o"
        )
        
        messages = self.provider._build_messages_with_attachments(request)
        
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,base64data"}
                    }
                ]
            }
        ]
        assert messages == expected


class TestGoogleProviderMessages:
    """Test Google provider message building."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = GoogleProvider(config={"api_key": "test-key"})
    
    def test_build_contents_single_turn(self):
        """Test building contents from single-turn request."""
        request = UnifiedRequest(
            prompt="Hello, world!",
            system_message="You are helpful",
            model="gemini-1.5-pro"
        )
        
        contents = self.provider._build_contents(request)
        
        expected = [
            {"parts": [{"text": "System: You are helpful\n\nUser: Hello, world!"}]}
        ]
        assert contents == expected
    
    def test_build_contents_multi_turn(self):
        """Test building contents from multi-turn request."""
        messages_input = [
            Message(role=MessageRole.SYSTEM, content="You are helpful"),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi! How can I help?"),
            Message(role=MessageRole.USER, content="What's the weather?")
        ]
        request = UnifiedRequest(
            messages=messages_input,
            model="gemini-1.5-pro"
        )

        contents = self.provider._build_contents(request)

        expected = [
            {"role": "user", "parts": [{"text": "You are helpful"}]},  # system -> user
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi! How can I help?"}]},  # assistant -> model
            {"role": "user", "parts": [{"text": "What's the weather?"}]}
        ]
        assert contents == expected

    def test_build_contents_with_attachments_multi_turn(self):
        """Test building contents with attachments for multi-turn request."""
        # Mock attachment
        attachment = Mock(spec=Attachment)
        attachment.attachment_type = AttachmentType.IMAGE
        attachment.mime_type = "image/png"
        attachment.to_base64.return_value = "base64data"

        messages_input = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi!"),
            Message(role=MessageRole.USER, content="What's in this image?")
        ]
        request = UnifiedRequest(
            messages=messages_input,
            attachments=[attachment],
            model="gemini-2.5-flash-image"
        )

        contents = self.provider._build_contents_with_attachments(request)

        expected = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi!"}]},
            {
                "role": "user",
                "parts": [
                    {"text": "What's in this image?"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "base64data"
                        }
                    }
                ]
            }
        ]
        assert contents == expected

    def test_build_contents_with_attachments_single_turn(self):
        """Test building contents with attachments for single-turn request."""
        # Mock attachment
        attachment = Mock(spec=Attachment)
        attachment.attachment_type = AttachmentType.IMAGE
        attachment.mime_type = "image/png"
        attachment.to_base64.return_value = "base64data"

        request = UnifiedRequest(
            prompt="Generate an image like this",
            attachments=[attachment],
            model="gemini-2.5-flash-image"
        )

        contents = self.provider._build_contents_with_attachments(request)

        expected = [
            {
                "parts": [
                    {"text": "Generate an image like this"},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": "base64data"
                        }
                    }
                ]
            }
        ]
        assert contents == expected


if __name__ == "__main__":
    pytest.main([__file__])
