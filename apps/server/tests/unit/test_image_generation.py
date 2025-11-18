"""Test script for image generation models."""

import asyncio
import os
from typing import Dict, Any

from services.inference.image_validation import ImageGenerationValidator
from services.inference.service import UnifiedInferenceService
from services.inference.models import UnifiedRequest, InferenceConfig
# Model configs now come from providers directly


async def test_parameter_validation():
    """Test parameter validation for both models."""
    print("=== Testing Parameter Validation ===")
    
    # Test Gemini 2.5 Flash Image validation
    print("\n--- Gemini 2.5 Flash Image ---")
    params = {
        "aspect_ratio": "16:9",
        "quality": "standard",
        "response_modalities": ["Image"]
    }
    validated, warnings = ImageGenerationValidator.validate_parameters("gemini-2.5-flash-image", params)
    print(f"Original: {params}")
    print(f"Validated: {validated}")
    print(f"Warnings: {warnings}")
    
    # Test GPT Image 1 validation
    print("\n--- GPT Image 1 ---")
    params = {
        "aspect_ratio": "3:2",
        "quality": "hd",
        "style": "vivid"
    }
    validated, warnings = ImageGenerationValidator.validate_parameters("gpt-image-1", params)
    print(f"Original: {params}")
    print(f"Validated: {validated}")
    print(f"Warnings: {warnings}")
    
    # Test invalid parameters
    print("\n--- Invalid Parameters ---")
    params = {
        "aspect_ratio": "21:9",  # Not supported by GPT Image 1
        "quality": "ultra",      # Not supported
        "style": "anime"         # Not supported
    }
    validated, warnings = ImageGenerationValidator.validate_parameters("gpt-image-1", params)
    print(f"Original: {params}")
    print(f"Validated: {validated}")
    print(f"Warnings: {warnings}")


async def test_model_configurations():
    """Test model configuration loading."""
    print("\n=== Testing Model Configurations ===")
    
    # Import providers to get model configs
    from services.inference.providers.google import GoogleProvider
    from services.inference.providers.openai import OpenAIProvider

    # Test Gemini config from provider
    google_models = GoogleProvider.get_supported_models()
    if "gemini-2.5-flash-image" in google_models:
        gemini_config = google_models["gemini-2.5-flash-image"]
        print("\n--- Gemini 2.5 Flash Image Config ---")
        print(f"Name: {gemini_config.name}")
        print(f"Display Name: {gemini_config.display_name}")
        print(f"Image Generation: {gemini_config.capabilities.image_generation}")
        print(f"Image Editing: {gemini_config.capabilities.image_editing}")
        print(f"Vision: {gemini_config.capabilities.vision}")

    # Test OpenAI models
    openai_models = OpenAIProvider.get_supported_models()
    if "dall-e-3" in openai_models:
        dalle_config = openai_models["dall-e-3"]
        print("\n--- DALL-E 3 Config ---")
        print(f"Name: {dalle_config.name}")
        print(f"Display Name: {dalle_config.display_name}")
        print(f"Image Generation: {dalle_config.capabilities.image_generation}")
        print(f"Text Generation: {dalle_config.capabilities.text_generation}")


async def test_unified_request_creation():
    """Test creating unified requests for image generation."""
    print("\n=== Testing Unified Request Creation ===")
    
    # Test Gemini request
    print("\n--- Gemini Request ---")
    gemini_request = UnifiedRequest(
        prompt="A serene mountain landscape with a crystal clear lake",
        model="gemini-2.5-flash-image",
        response_format="image",
        extras={
            "aspect_ratio": "16:9",
            "response_modalities": ["Text", "Image"]
        }
    )
    print(f"Model: {gemini_request.model}")
    print(f"Prompt: {gemini_request.prompt}")
    print(f"Extras: {gemini_request.extras}")
    
    # Test GPT Image 1 request
    print("\n--- GPT Image 1 Request ---")
    gpt_request = UnifiedRequest(
        prompt="A futuristic cityscape at sunset",
        model="gpt-image-1",
        response_format="image",
        extras={
            "aspect_ratio": "3:2",
            "quality": "hd",
            "style": "vivid"
        }
    )
    print(f"Model: {gpt_request.model}")
    print(f"Prompt: {gpt_request.prompt}")
    print(f"Extras: {gpt_request.extras}")


async def test_aspect_ratio_mapping():
    """Test aspect ratio to size mapping."""
    print("\n=== Testing Aspect Ratio Mapping ===")
    
    test_cases = [
        ("1:1", ["1024x1024", "512x512"]),
        ("16:9", ["1344x768", "1920x1080"]),
        ("3:2", ["1536x1024", "1248x832"]),
        ("21:9", ["1536x672"])
    ]
    
    for aspect_ratio, supported_sizes in test_cases:
        mapped_size = ImageGenerationValidator._map_aspect_ratio_to_size(aspect_ratio, supported_sizes)
        print(f"Aspect Ratio: {aspect_ratio} -> Size: {mapped_size} (from {supported_sizes})")


def test_model_capabilities():
    """Test model capabilities retrieval."""
    print("\n=== Testing Model Capabilities ===")
    
    models = ["gemini-2.5-flash-image", "gpt-image-1", "dall-e-3"]
    
    for model in models:
        capabilities = ImageGenerationValidator.get_model_capabilities(model)
        if capabilities:
            print(f"\n--- {model} ---")
            print(f"Aspect Ratios: {capabilities['supported_aspect_ratios']}")
            print(f"Quality Levels: {capabilities['supported_quality_levels']}")
            print(f"Styles: {capabilities['supported_styles']}")
            print(f"Reference Images: {capabilities['supports_reference_images']}")
        else:
            print(f"\n--- {model} --- NOT FOUND")


async def main():
    """Run all tests."""
    print("🧪 Image Generation Implementation Tests")
    print("=" * 50)
    
    try:
        await test_parameter_validation()
        await test_model_configurations()
        await test_unified_request_creation()
        await test_aspect_ratio_mapping()
        test_model_capabilities()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
