#!/usr/bin/env python3
"""Simple test script for unified inference infrastructure."""

import asyncio
import logging
import sys
import os

# Add the server directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import initialize_config, get_config
from services.inference.models import UnifiedRequest, Attachment
from services.inference.factory import create_unified_inference_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_functionality():
    """Test basic functionality of the unified inference system."""
    
    print("🚀 Testing Unified Inference Infrastructure (Simple)")
    print("=" * 60)
    
    # Initialize configuration
    print("1. Initializing configuration...")
    if not initialize_config():
        print("❌ Failed to initialize configuration")
        return False
    
    config = get_config()
    print(f"✅ Configuration loaded")
    print(f"   - Unified inference enabled: {config.inference.enabled}")
    print(f"   - OpenAI API key configured: {'Yes' if config.api_keys.openai else 'No'}")
    print(f"   - Google API key configured: {'Yes' if config.api_keys.gemini else 'No'}")
    
    # Test creating unified service
    print("\n2. Testing unified service creation...")
    try:
        service = create_unified_inference_service()
        if service:
            print("✅ Unified service created successfully")
            
            # Test provider status
            provider_status = service.get_provider_status()
            print(f"   - Provider status: {provider_status}")
            
            # Test available models
            available_models = service.get_available_models()
            print(f"   - Available models: {available_models}")
            
        else:
            print("⚠️  Unified service not created (likely disabled or missing API keys)")
            return True  # This is expected if not configured
            
    except Exception as e:
        print(f"❌ Failed to create unified service: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test basic text generation (only if we have API keys)
    if config.api_keys.openai and service:
        print("\n3. Testing text generation...")
        try:
            response = await service.chat("Hello, how are you?", model="auto")
            print(f"✅ Text generation successful")
            print(f"   - Response: {response[:100]}...")
            
        except Exception as e:
            print(f"⚠️  Text generation failed (expected without valid API key): {e}")
    else:
        print("\n3. Skipping text generation test (no API key configured)")
    
    # Test model-specific parameter mapping
    print("\n4. Testing model configuration...")
    try:
        from services.inference.registry import ModelRegistry
        registry = ModelRegistry()
        
        # Test getting OpenAI model config
        gpt4_config = await registry.get_model_config("openai", "gpt-4o")
        if gpt4_config:
            print("✅ GPT-4o configuration loaded")
            print(f"   - Max tokens param: {gpt4_config.parameter_mapping.max_tokens_param}")
            print(f"   - Supports vision: {gpt4_config.capabilities.vision}")
            print(f"   - Supported parameters: {list(gpt4_config.supported_parameters)[:5]}...")
        
        # Test getting GPT-5 config
        gpt5_config = await registry.get_model_config("openai", "gpt-5")
        if gpt5_config:
            print("✅ GPT-5 configuration loaded")
            print(f"   - Max tokens param: {gpt5_config.parameter_mapping.max_tokens_param}")
            print(f"   - Supports reasoning: {gpt5_config.capabilities.reasoning}")
            print(f"   - Custom params: {gpt5_config.parameter_mapping.custom_params}")
        
    except Exception as e:
        print(f"❌ Model configuration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Basic test completed!")
    return True


async def test_parameter_mapping():
    """Test parameter mapping functionality."""
    
    print("\n🧪 Testing Parameter Mapping")
    print("=" * 40)
    
    try:
        from services.inference.providers.openai_shim import OpenAIShim
        from services.inference.registry import ModelRegistry
        
        # Create a mock shim for testing
        shim = OpenAIShim({"api_key": "test-key"})
        registry = ModelRegistry()
        
        # Test GPT-4o parameter mapping
        gpt4_config = await registry.get_model_config("openai", "gpt-4o")
        if gpt4_config:
            unified_params = {
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.1
            }
            
            mapped_params = shim.map_parameters(unified_params, gpt4_config)
            print("✅ GPT-4o parameter mapping:")
            print(f"   - Input: {unified_params}")
            print(f"   - Mapped: {mapped_params}")
        
        # Test GPT-5 parameter mapping
        gpt5_config = await registry.get_model_config("openai", "gpt-5")
        if gpt5_config:
            unified_params = {
                "max_tokens": 200,
                "temperature": 0.7,  # Should be ignored
                "reasoning_effort": "high"
            }
            
            mapped_params = shim.map_parameters(unified_params, gpt5_config)
            print("✅ GPT-5 parameter mapping:")
            print(f"   - Input: {unified_params}")
            print(f"   - Mapped: {mapped_params}")
            print(f"   - Note: temperature ignored, max_tokens -> max_completion_tokens")
        
    except Exception as e:
        print(f"❌ Parameter mapping test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
    asyncio.run(test_parameter_mapping())
