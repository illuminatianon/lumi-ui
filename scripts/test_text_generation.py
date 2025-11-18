#!/usr/bin/env python3
"""
Text Generation Test Script

This script tests text generation capabilities using both OpenAI and Google models
through the unified inference service.

Usage:
    python scripts/test_text_generation.py

Requirements:
    - Set OPENAI_API_KEY environment variable for OpenAI models
    - Set GOOGLE_API_KEY environment variable for Google models
    - Or configure API keys in the Lumi configuration system
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the server directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "server"))

from config import initialize_config, get_config
from services.inference.factory import create_unified_inference_service
from services.inference.models import UnifiedRequest, Attachment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_test_image(image_name: str) -> Attachment:
    """Load a test image as an Attachment object."""
    script_dir = Path(__file__).parent
    image_path = script_dir / image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    return Attachment.from_file(str(image_path))


async def test_openai_text_generation(service):
    """Test OpenAI text generation models."""
    print("\n🤖 Testing OpenAI Models")
    print("=" * 50)
    
    models_to_test = ["openai/gpt-4o", "openai/gpt-5"]
    prompt = "Explain quantum computing in simple terms, in about 100 words."
    
    for model in models_to_test:
        print(f"\n--- Testing {model} ---")
        try:
            # GPT-5 requires special parameters and higher token limit
            if model == "openai/gpt-5":
                request = UnifiedRequest(
                    prompt=prompt,
                    model=model,
                    max_tokens=500,  # Higher limit to account for reasoning tokens
                    reasoning_effort="medium"  # Required for GPT-5
                )
            else:
                request = UnifiedRequest(
                    prompt=prompt,
                    model=model,
                    temperature=0.7,
                    max_tokens=150
                )
            
            response = await service.process_request(request)

            if response and response.content:
                print(f"✅ {model} Response:")
                print(f"   Content: {response.content[:200]}...")
                print(f"   Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
                print(f"   Model used: {response.model_used}")
            else:
                print(f"❌ {model} Failed: No content in response")
                
        except Exception as e:
            print(f"❌ {model} Error: {e}")


async def test_google_text_generation(service):
    """Test Google text generation models."""
    print("\n🧠 Testing Google Models")
    print("=" * 50)
    
    models_to_test = ["google/gemini-2.5-flash"]
    prompt = "Write a creative short story about a robot learning to paint, in about 100 words."
    
    for model in models_to_test:
        print(f"\n--- Testing {model} ---")
        try:
            request = UnifiedRequest(
                prompt=prompt,
                model=model,
                temperature=0.8,
                max_tokens=150
            )
            
            response = await service.process_request(request)

            if response and response.content:
                print(f"✅ {model} Response:")
                print(f"   Content: {response.content[:200]}...")
                print(f"   Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
                print(f"   Model used: {response.model_used}")
            else:
                print(f"❌ {model} Failed: No content in response")
                
        except Exception as e:
            print(f"❌ {model} Error: {e}")


async def test_vision_capabilities(service):
    """Test vision capabilities with image attachments."""
    print("\n👁️ Testing Vision Capabilities")
    print("=" * 50)

    # Load the test image
    try:
        vision_image = load_test_image("vision_test.jpg")
        print(f"✅ Loaded test image: {vision_image.filename} ({vision_image.mime_type})")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Test vision-capable models
    vision_models = ["openai/gpt-4o", "google/gemini-2.5-flash"]
    vision_prompt = "What do you see in this image? Describe it in detail."

    for model in vision_models:
        print(f"\n--- Testing {model} with vision ---")
        try:
            request = UnifiedRequest(
                prompt=vision_prompt,
                model=model,
                attachments=[vision_image],
                temperature=0.3,
                max_tokens=300
            )

            response = await service.process_request(request)

            if response and response.content:
                print(f"✅ {model} Vision Response:")
                print(f"   Content: {response.content[:200]}...")
                print(f"   Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
                print(f"   Model used: {response.model_used}")
                print(f"   Attachments processed: {len(response.attachments_processed)}")
            else:
                print(f"❌ {model} Vision Failed: No content in response")

        except Exception as e:
            print(f"❌ {model} Vision Error: {e}")


async def test_auto_model_selection(service):
    """Test automatic model selection."""
    print("\n🎯 Testing Auto Model Selection")
    print("=" * 50)

    prompt = "What are the main differences between Python and JavaScript?"

    try:
        request = UnifiedRequest(
            prompt=prompt,
            model="auto",  # Let the service choose the best model
            temperature=0.5,
            max_tokens=200
        )

        response = await service.process_request(request)

        if response and response.content:
            print(f"✅ Auto Selection Response:")
            print(f"   Model chosen: {response.model_used}")
            print(f"   Content: {response.content[:200]}...")
            print(f"   Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        else:
            print(f"❌ Auto Selection Failed: No content in response")

    except Exception as e:
        print(f"❌ Auto Selection Error: {e}")


async def main():
    """Main test function."""
    print("🚀 Lumi Text Generation Test Script")
    print("=" * 60)
    
    # Initialize configuration
    print("1. Initializing configuration...")
    if not initialize_config():
        print("❌ Failed to initialize configuration")
        return
    
    config = get_config()
    
    # Enable inference if not already enabled
    if not config.inference.enabled:
        print("⚠️  Unified inference is disabled. Enabling for this test...")
        config.inference.enabled = True
    
    # Create unified inference service
    print("2. Creating unified inference service...")
    service = create_unified_inference_service()
    
    if not service:
        print("❌ Failed to create unified inference service")
        print("   Make sure you have API keys configured:")
        print("   - Set OPENAI_API_KEY environment variable")
        print("   - Set GOOGLE_API_KEY environment variable")
        return
    
    # Check provider availability
    print("3. Checking provider availability...")
    provider_status = service.get_provider_status()
    print(f"   Provider status: {provider_status}")
    
    available_models = service.get_available_models()
    print(f"   Available models: {available_models}")
    
    # Run tests
    try:
        await test_openai_text_generation(service)
        await test_google_text_generation(service)
        await test_vision_capabilities(service)
        await test_auto_model_selection(service)

        print("\n🎉 All text generation and vision tests completed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
