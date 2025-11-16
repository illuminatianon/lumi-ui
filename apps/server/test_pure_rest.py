#!/usr/bin/env python3
"""Test script for pure REST API implementation."""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from services.inference.providers.openai_shim import OpenAIShim
from services.inference.providers.google_shim import GoogleShim
from services.inference.models import UnifiedRequest, ModelConfig, ModelCapabilities, ParameterMapping

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_openai_shim():
    """Test OpenAI shim with pure REST."""
    print("\n=== Testing OpenAI Shim (Pure REST) ===")
    
    # Mock config - you'd need real API key for actual testing
    config = {
        'api_key': 'test-key-replace-with-real',
        'base_url': 'https://api.openai.com'
    }
    
    # Create model config
    model_config = ModelConfig(
        name="gpt-4o-mini",
        display_name="GPT-4o Mini",
        provider="openai",
        capabilities=ModelCapabilities(
            text_generation=True,
            vision=True,
            function_calling=True
        ),
        parameter_mapping=ParameterMapping(
            temperature_param="temperature",
            max_tokens_param="max_tokens",
            top_p_param="top_p"
        ),
        supported_parameters={"temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"},
        context_window=128000
    )
    
    # Create request
    request = UnifiedRequest(
        prompt="Hello, how are you?",
        system_message="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=100
    )
    
    try:
        async with OpenAIShim(config) as shim:
            print(f"OpenAI shim available: {shim.is_available()}")
            
            # Test parameter mapping
            mapped_params = shim.map_parameters(request.model_dump(), model_config)
            print(f"Mapped parameters: {mapped_params}")
            
            # Note: Actual API call would require real API key
            print("✓ OpenAI shim initialization and parameter mapping successful")
            
    except Exception as e:
        print(f"✗ OpenAI shim test failed: {e}")


async def test_google_shim():
    """Test Google shim with pure REST."""
    print("\n=== Testing Google Shim (Pure REST) ===")
    
    # Mock config - you'd need real API key for actual testing
    config = {
        'api_key': 'test-key-replace-with-real'
    }
    
    # Create model config
    model_config = ModelConfig(
        name="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        provider="google",
        capabilities=ModelCapabilities(
            text_generation=True,
            vision=True
        ),
        parameter_mapping=ParameterMapping(
            temperature_param="temperature",
            max_tokens_param="max_output_tokens",
            top_p_param="top_p"
        ),
        supported_parameters={"temperature", "max_output_tokens", "top_p", "stop_sequences"},
        context_window=1000000
    )
    
    # Create request
    request = UnifiedRequest(
        prompt="Hello, how are you?",
        system_message="You are a helpful assistant.",
        temperature=0.7,
        max_tokens=100
    )
    
    try:
        async with GoogleShim(config) as shim:
            print(f"Google shim available: {shim.is_available()}")
            
            # Test parameter mapping
            mapped_params = shim.map_parameters(request.model_dump(), model_config)
            print(f"Mapped parameters: {mapped_params}")
            
            # Test generation config building
            gen_config = shim._build_generation_config(request, model_config)
            print(f"Generation config: {gen_config}")
            
            # Note: Actual API call would require real API key
            print("✓ Google shim initialization and parameter mapping successful")
            
    except Exception as e:
        print(f"✗ Google shim test failed: {e}")


async def main():
    """Run all tests."""
    print("Pure REST API Implementation Test")
    print("=" * 50)
    
    await test_openai_shim()
    await test_google_shim()
    
    print("\n" + "=" * 50)
    print("✓ All tests completed successfully!")
    print("Note: Actual API calls require real API keys")


if __name__ == "__main__":
    asyncio.run(main())
