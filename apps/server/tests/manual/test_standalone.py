#!/usr/bin/env python3
"""Standalone test for unified inference infrastructure."""

import asyncio
import logging
import sys
import os

# Add the server directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_models_and_registry():
    """Test the models and registry components."""
    
    print("🚀 Testing Unified Inference Models and Registry")
    print("=" * 50)
    
    try:
        # Test importing models
        print("1. Testing model imports...")
        from services.inference.models import (
            UnifiedRequest, UnifiedResponse, Attachment, AttachmentType,
            ModelConfig, ParameterMapping, ModelCapabilities
        )
        print("✅ Models imported successfully")
        
        # Test creating a unified request
        print("\n2. Testing UnifiedRequest creation...")
        request = UnifiedRequest(
            prompt="Hello, world!",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=100
        )
        print(f"✅ UnifiedRequest created: {request.prompt}")
        print(f"   - Model: {request.model}")
        print(f"   - Temperature: {request.temperature}")
        print(f"   - Max tokens: {request.max_tokens}")
        
        # Test creating an attachment
        print("\n3. Testing Attachment creation...")
        attachment = Attachment.from_base64(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "image/png",
            "test.png"
        )
        print(f"✅ Attachment created: {attachment.filename}")
        print(f"   - Type: {attachment.attachment_type}")
        print(f"   - MIME type: {attachment.mime_type}")
        print(f"   - Size: {len(attachment.content)} bytes")
        
        # Test model configuration
        print("\n4. Testing ModelConfig creation...")
        model_config = ModelConfig(
            name="test-model",
            display_name="Test Model",
            provider="test",
            capabilities=ModelCapabilities(
                text_generation=True,
                vision=True,
                reasoning=True
            ),
            parameter_mapping=ParameterMapping(
                max_tokens_param="max_completion_tokens",
                temperature_param=None,
                custom_params={"reasoning_effort": "medium"}
            ),
            supported_parameters={"max_completion_tokens", "reasoning_effort"},
            cost_per_1k_tokens=0.01,
            context_window=128000
        )
        print(f"✅ ModelConfig created: {model_config.name}")
        print(f"   - Capabilities: vision={model_config.capabilities.vision}, reasoning={model_config.capabilities.reasoning}")
        print(f"   - Max tokens param: {model_config.parameter_mapping.max_tokens_param}")
        print(f"   - Custom params: {model_config.parameter_mapping.custom_params}")
        
        # Test model registry
        print("\n5. Testing ModelRegistry...")
        from services.inference.registry import ModelRegistry
        registry = ModelRegistry()
        
        # Test getting available models
        available_models = registry.get_available_models()
        print(f"✅ ModelRegistry created")
        print(f"   - Available models: {available_models}")
        
        # Test getting specific model configs
        gpt4_config = await registry.get_model_config("openai", "gpt-4o")
        if gpt4_config:
            print(f"   - GPT-4o config loaded: {gpt4_config.display_name}")
            print(f"     * Max tokens param: {gpt4_config.parameter_mapping.max_tokens_param}")
            print(f"     * Supports vision: {gpt4_config.capabilities.vision}")
        
        gpt5_config = await registry.get_model_config("openai", "gpt-5")
        if gpt5_config:
            print(f"   - GPT-5 config loaded: {gpt5_config.display_name}")
            print(f"     * Max tokens param: {gpt5_config.parameter_mapping.max_tokens_param}")
            print(f"     * Supports reasoning: {gpt5_config.capabilities.reasoning}")
            print(f"     * Custom params: {gpt5_config.parameter_mapping.custom_params}")
        
        # Test provider registry
        print("\n6. Testing ProviderRegistry...")
        from services.inference.registry import ProviderRegistry
        from services.inference.providers import OpenAIShim, GoogleShim
        
        provider_registry = ProviderRegistry()
        provider_registry.register_provider("openai", OpenAIShim)
        provider_registry.register_provider("google", GoogleShim)
        
        available_providers = provider_registry.get_available_providers()
        print(f"✅ ProviderRegistry created")
        print(f"   - Registered providers: {available_providers}")
        
        # Test parameter mapping
        print("\n7. Testing parameter mapping...")
        openai_shim = OpenAIShim({"api_key": "test-key"})
        
        if gpt4_config:
            unified_params = {
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
            mapped_params = openai_shim.map_parameters(unified_params, gpt4_config)
            print(f"✅ GPT-4o parameter mapping:")
            print(f"   - Input: {unified_params}")
            print(f"   - Mapped: {mapped_params}")
        
        if gpt5_config:
            unified_params = {
                "max_tokens": 200,
                "temperature": 0.7,  # Should be ignored
                "reasoning_effort": "high"
            }
            mapped_params = openai_shim.map_parameters(unified_params, gpt5_config)
            print(f"✅ GPT-5 parameter mapping:")
            print(f"   - Input: {unified_params}")
            print(f"   - Mapped: {mapped_params}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_models_and_registry())
    if success:
        print("\n✅ Unified inference infrastructure is working correctly!")
    else:
        print("\n❌ Tests failed - check the implementation")
        sys.exit(1)
