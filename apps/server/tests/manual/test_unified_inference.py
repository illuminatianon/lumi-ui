#!/usr/bin/env python3
"""Test script for unified inference infrastructure."""

import asyncio
import logging
import sys
import os

# Add the server directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import initialize_config, get_config
from services.inference.factory import get_unified_inference_service
from services.inference.models import UnifiedRequest, Attachment
from services import get_inference_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_unified_inference():
    """Test the unified inference infrastructure."""

    print("🚀 Testing Unified Inference Infrastructure")
    print("=" * 50)

    # Initialize configuration
    print("1. Initializing configuration...")
    if not initialize_config():
        print("❌ Failed to initialize configuration")
        return False

    config = get_config()
    print(f"✅ Configuration loaded")
    print(f"   - Unified inference enabled: {config.inference.enabled}")
    print(
        f"   - OpenAI API key configured: {'Yes' if config.api_keys.openai else 'No'}"
    )
    print(
        f"   - Google API key configured: {'Yes' if config.api_keys.gemini else 'No'}"
    )

    # Test compatibility service
    print("\n2. Testing compatibility service...")
    try:
        service = get_inference_service()
        print(f"✅ Compatibility service created")
        print(f"   - Using unified inference: {service.use_unified}")

        # Test model status
        models = service.get_available_models()
        print(f"   - Available models: {models}")

    except Exception as e:
        print(f"❌ Failed to create compatibility service: {e}")
        return False

    # Test text generation
    print("\n3. Testing text generation...")
    try:
        result = await service.generate_text(
            prompt="Hello, how are you?", model="auto", temperature=0.7
        )

        if result["success"]:
            print(f"✅ Text generation successful")
            print(f"   - Model used: {result['model_used']}")
            print(f"   - Response: {result['text'][:100]}...")
        else:
            print(f"❌ Text generation failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        print(f"❌ Text generation error: {e}")

    # Test unified service directly (if enabled)
    if config.inference.enabled:
        print("\n4. Testing unified service directly...")
        try:
            unified_service = get_unified_inference_service()
            if unified_service:
                print("✅ Unified service created")

                # Test simple chat
                response = await unified_service.chat("What is 2+2?")
                print(f"   - Chat response: {response[:100]}...")
                import pdb

                pdb.set_trace()

                # Test model availability
                available_models = unified_service.get_available_models()
                print(f"   - Available models: {available_models}")

                provider_status = unified_service.get_provider_status()
                print(f"   - Provider status: {provider_status}")

            else:
                print("⚠️  Unified service not available (likely missing API keys)")

        except Exception as e:
            print(f"❌ Unified service error: {e}")
    else:
        print("\n4. Unified inference disabled in configuration")

    print("\n" + "=" * 50)
    print("🎉 Test completed!")
    return True


async def test_model_specific_parameters():
    """Test model-specific parameter handling."""

    print("\n🧪 Testing Model-Specific Parameters")
    print("=" * 50)

    config = get_config()
    if not config.inference.enabled or not config.api_keys.openai:
        print(
            "⚠️  Skipping model-specific tests (unified inference disabled or no OpenAI key)"
        )
        return

    try:
        unified_service = get_unified_inference_service()
        if not unified_service:
            print("⚠️  Unified service not available")
            return

        # Test GPT-4o with standard parameters
        print("1. Testing GPT-4o with standard parameters...")
        request = UnifiedRequest(
            prompt="Explain quantum computing in one sentence.",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=100,
        )

        response = await unified_service.process_request(request)
        print(f"✅ GPT-4o response: {response.content[:100]}...")
        print(f"   - Model used: {response.model_used}")
        print(f"   - Provider: {response.provider}")

        # Test GPT-5 with reasoning parameters (if available)
        print("\n2. Testing GPT-5 with reasoning parameters...")
        try:
            request = UnifiedRequest(
                prompt="Solve this step by step: If a train travels 60 mph for 2 hours, how far does it go?",
                model="gpt-5",
                reasoning_effort="medium",
                max_tokens=200,
            )

            response = await unified_service.process_request(request)
            print(f"✅ GPT-5 response: {response.content[:100]}...")
            print(f"   - Model used: {response.model_used}")
            print(f"   - Provider: {response.provider}")

        except Exception as e:
            print(f"⚠️  GPT-5 test failed (model may not be available): {e}")

    except Exception as e:
        print(f"❌ Model-specific parameter test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_unified_inference())
    asyncio.run(test_model_specific_parameters())
