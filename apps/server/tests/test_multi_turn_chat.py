"""Tests for multi-turn chat functionality."""

import pytest
from pydantic import ValidationError

from services.inference.models import UnifiedRequest, Message, MessageRole
from api.models import RegistryRequest


class TestMessageModel:
    """Test the Message model."""
    
    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
    
    def test_message_roles(self):
        """Test all message roles."""
        system_msg = Message(role=MessageRole.SYSTEM, content="You are helpful")
        user_msg = Message(role=MessageRole.USER, content="Hello")
        assistant_msg = Message(role=MessageRole.ASSISTANT, content="Hi there!")
        
        assert system_msg.role == MessageRole.SYSTEM
        assert user_msg.role == MessageRole.USER
        assert assistant_msg.role == MessageRole.ASSISTANT


class TestUnifiedRequest:
    """Test the UnifiedRequest model with multi-turn support."""
    
    def test_single_turn_request(self):
        """Test backward compatibility with single-turn requests."""
        request = UnifiedRequest(
            prompt="Hello, world!",
            model="openai/gpt-4o"
        )
        assert request.prompt == "Hello, world!"
        assert request.messages is None
        assert not request.is_multi_turn()
    
    def test_single_turn_with_system_message(self):
        """Test single-turn with system message."""
        request = UnifiedRequest(
            prompt="Hello",
            system_message="You are helpful",
            model="openai/gpt-4o"
        )
        assert request.prompt == "Hello"
        assert request.system_message == "You are helpful"
        assert not request.is_multi_turn()
    
    def test_multi_turn_request(self):
        """Test multi-turn conversation request."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful"),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi! How can I help?"),
            Message(role=MessageRole.USER, content="What's the weather?")
        ]
        request = UnifiedRequest(
            messages=messages,
            model="openai/gpt-4o"
        )
        assert request.messages == messages
        assert request.prompt is None
        assert request.is_multi_turn()
    
    def test_validation_no_prompt_or_messages(self):
        """Test validation fails when neither prompt nor messages provided."""
        with pytest.raises(ValidationError, match="Either 'prompt' or 'messages' must be provided"):
            UnifiedRequest(model="openai/gpt-4o")
    
    def test_validation_both_prompt_and_messages(self):
        """Test validation fails when both prompt and messages provided."""
        messages = [Message(role=MessageRole.USER, content="Hello")]
        with pytest.raises(ValidationError, match="Cannot provide both 'prompt' and 'messages'"):
            UnifiedRequest(
                prompt="Hello",
                messages=messages,
                model="openai/gpt-4o"
            )
    
    def test_empty_messages_treated_as_none(self):
        """Test that empty messages list is treated as no messages."""
        with pytest.raises(ValidationError, match="Either 'prompt' or 'messages' must be provided"):
            UnifiedRequest(
                messages=[],
                model="openai/gpt-4o"
            )


class TestRegistryRequest:
    """Test the RegistryRequest model with multi-turn support."""
    
    def test_single_turn_registry_request(self):
        """Test backward compatibility with single-turn registry requests."""
        request = RegistryRequest(
            model="openai/gpt-4o",
            prompt="Hello, world!"
        )
        assert request.prompt == "Hello, world!"
        assert request.messages is None
    
    def test_multi_turn_registry_request(self):
        """Test multi-turn registry request."""
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        request = RegistryRequest(
            model="openai/gpt-4o",
            messages=messages
        )
        assert request.messages == messages
        assert request.prompt is None
    
    def test_registry_validation_no_prompt_or_messages(self):
        """Test validation fails when neither prompt nor messages provided."""
        with pytest.raises(ValidationError, match="Either 'prompt' or 'messages' must be provided"):
            RegistryRequest(model="openai/gpt-4o")
    
    def test_registry_validation_both_prompt_and_messages(self):
        """Test validation fails when both prompt and messages provided."""
        messages = [Message(role=MessageRole.USER, content="Hello")]
        with pytest.raises(ValidationError, match="Cannot provide both 'prompt' and 'messages'"):
            RegistryRequest(
                model="openai/gpt-4o",
                prompt="Hello",
                messages=messages
            )


if __name__ == "__main__":
    pytest.main([__file__])
