"""Integration tests for multi-turn chat API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

from main import app
from services.inference.models import Message, MessageRole


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_inference_service():
    """Mock inference service."""
    mock_service = Mock()
    mock_response = Mock()
    mock_response.content = "Test response"
    mock_response.token_usage = None
    mock_response.finish_reason = "stop"
    mock_service.process_registry_request = AsyncMock(return_value=mock_response)
    return mock_service


class TestMultiTurnChatAPI:
    """Test multi-turn chat API endpoints."""
    
    def test_single_turn_request_backward_compatibility(self, client, mock_inference_service):
        """Test that single-turn requests still work (backward compatibility)."""
        with patch('main.inference_service', mock_inference_service):
            response = client.post("/api/ai/chat", json={
                "model": "openai/gpt-4o",
                "prompt": "Hello, world!",
                "system_message": "You are helpful"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content"] == "Test response"
        assert data["model_used"] == "openai/gpt-4o"
        assert data["provider"] == "openai"
        
        # Verify the service was called with correct config
        mock_inference_service.process_registry_request.assert_called_once()
        call_args = mock_inference_service.process_registry_request.call_args[0][0]
        assert call_args["model"] == "openai/gpt-4o"
        assert call_args["prompt"] == "Hello, world!"
        assert call_args["system_message"] == "You are helpful"
        assert call_args.get("messages") is None
    
    def test_multi_turn_request(self, client, mock_inference_service):
        """Test multi-turn conversation request."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "What's the weather?"}
        ]
        
        with patch('main.inference_service', mock_inference_service):
            response = client.post("/api/ai/chat", json={
                "model": "openai/gpt-4o",
                "messages": messages
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["content"] == "Test response"
        
        # Verify the service was called with correct config
        mock_inference_service.process_registry_request.assert_called_once()
        call_args = mock_inference_service.process_registry_request.call_args[0][0]
        assert call_args["model"] == "openai/gpt-4o"
        assert call_args.get("prompt") is None
        assert call_args.get("system_message") is None
        assert len(call_args["messages"]) == 4
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][0]["content"] == "You are helpful"
    
    def test_validation_error_no_prompt_or_messages(self, client, mock_inference_service):
        """Test validation error when neither prompt nor messages provided."""
        with patch('main.inference_service', mock_inference_service):
            response = client.post("/api/ai/chat", json={
                "model": "openai/gpt-4o"
            })
        
        assert response.status_code == 422  # Validation error
    
    def test_validation_error_both_prompt_and_messages(self, client, mock_inference_service):
        """Test validation error when both prompt and messages provided."""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch('main.inference_service', mock_inference_service):
            response = client.post("/api/ai/chat", json={
                "model": "openai/gpt-4o",
                "prompt": "Hello",
                "messages": messages
            })
        
        assert response.status_code == 422  # Validation error
    
    def test_multi_turn_with_parameters(self, client, mock_inference_service):
        """Test multi-turn request with additional parameters."""
        messages = [
            {"role": "user", "content": "Write a story"}
        ]
        
        with patch('main.inference_service', mock_inference_service):
            response = client.post("/api/ai/chat", json={
                "model": "openai/gpt-4o",
                "messages": messages,
                "temperature": 0.8,
                "max_tokens": 500
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify parameters were passed through
        call_args = mock_inference_service.process_registry_request.call_args[0][0]
        assert call_args["temperature"] == 0.8
        assert call_args["max_tokens"] == 500
    
    def test_service_unavailable(self, client):
        """Test response when inference service is unavailable."""
        with patch('main.inference_service', None):
            response = client.post("/api/ai/chat", json={
                "model": "openai/gpt-4o",
                "prompt": "Hello"
            })
        
        assert response.status_code == 200  # API returns 200 but with error in response
        data = response.json()
        assert data["success"] is False
        assert "not available" in data["error"]
    
    def test_service_error_handling(self, client, mock_inference_service):
        """Test error handling when service throws exception."""
        mock_inference_service.process_registry_request.side_effect = Exception("Service error")
        
        with patch('main.inference_service', mock_inference_service):
            response = client.post("/api/ai/chat", json={
                "model": "openai/gpt-4o",
                "prompt": "Hello"
            })
        
        assert response.status_code == 200  # API returns 200 but with error in response
        data = response.json()
        assert data["success"] is False
        assert "Service error" in data["error"]


if __name__ == "__main__":
    pytest.main([__file__])
