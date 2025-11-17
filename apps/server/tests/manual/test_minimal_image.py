#!/usr/bin/env python3
"""Minimal test script for Gemini 2.5 Flash Image generation."""

import asyncio
import base64
import os
from pathlib import Path

from services.inference.service import UnifiedInferenceService
from services.inference.models import UnifiedRequest, InferenceConfig
from config import initialize_config, get_config, get_api_key


async def test_gemini_image_generation():
    """Test Gemini 2.5 Flash Image generation and save result."""

    # Initialize Lumi configuration system
    print("🔧 Loading Lumi configuration...")
    if not initialize_config():
        print("❌ Error: Failed to initialize Lumi configuration system")
        return

    # Get configuration
    lumi_config = get_config()

    # Get API key from Lumi config
    api_key = get_api_key("gemini")
    if not api_key:
        print("❌ Error: Gemini API key not found in Lumi configuration")
        print("Please set your Google API key in the Lumi config:")
        print("1. Through the UI: Settings > API Keys > Gemini")
        print("2. Or set environment variable: export GEMINI_API_KEY='your-api-key-here'")
        print("3. Or edit the config file directly")
        return

    print("🚀 Testing Gemini 2.5 Flash Image generation...")
    print(f"📋 Using API key from Lumi config: {api_key[:8]}...{api_key[-4:]}")

    # Create inference configuration from Lumi config
    config = InferenceConfig(
        enabled=True,
        providers={
            "google": {
                "enabled": True,
                "api_key": api_key,
                "default_model": "gemini-2.5-flash-image"
            }
        },
        default_provider="google",
        fallback_providers=[]
    )
    
    # Initialize the inference service
    service = UnifiedInferenceService(config)
    
    # Create image generation request using registry format
    request_config = {
        "prompt": "A beautiful sunset over a serene mountain lake with vibrant orange and pink colors reflecting in the water",
        "model": "google/gemini-2.5-flash-image",  # Registry format: provider/model
        "aspect_ratio": "16:9",
        "response_modalities": ["Image"]  # Only image, no text
    }
    
    try:
        print("🎨 Generating image...")
        print(f"📝 Prompt: {request_config['prompt']}")
        print(f"📐 Aspect ratio: {request_config['aspect_ratio']}")

        # Make the request using registry resolver
        response = await service.process_registry_request(request_config)
        
        if response.images:
            print(f"✅ Image generated successfully!")
            print(f"🤖 Model used: {response.model_used}")
            print(f"🔧 Provider: {response.provider}")
            
            # Get the first image
            image_data = response.images[0]
            
            # Handle different image formats
            if image_data.startswith("data:image/"):
                # Extract base64 data from data URL
                header, base64_data = image_data.split(",", 1)
                image_bytes = base64.b64decode(base64_data)
                print(f"📊 Image format: {header}")
            else:
                # Assume it's a direct URL or base64
                try:
                    image_bytes = base64.b64decode(image_data)
                except:
                    print(f"❌ Error: Unexpected image format: {image_data[:50]}...")
                    return
            
            # Save to file
            output_path = Path("test.png")
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            
            print(f"💾 Image saved to: {output_path.absolute()}")
            print(f"📏 Image size: {len(image_bytes)} bytes")
            
            # Print metadata if available
            if response.metadata:
                print(f"📋 Metadata: {response.metadata}")
            
            # Print text content if available
            if response.content:
                print(f"📝 Generated text: {response.content}")
                
        else:
            print("❌ No images returned in response")
            if response.content:
                print(f"📝 Response content: {response.content}")
    
    except Exception as e:
        print(f"❌ Error during image generation: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function."""
    print("🧪 Minimal Gemini 2.5 Flash Image Test")
    print("=" * 40)
    
    await test_gemini_image_generation()
    
    print("\n✨ Test completed!")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("services").exists():
        print("❌ Error: Please run this script from the apps/server directory")
        print("cd apps/server && python test_minimal_image.py")
        exit(1)
    
    asyncio.run(main())
