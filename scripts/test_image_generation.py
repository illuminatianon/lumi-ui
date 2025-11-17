#!/usr/bin/env python3
"""
Image Generation Test Script

This script tests image generation capabilities using the Gemini 2.5 Flash Image model
through the unified inference service.

Usage:
    python scripts/test_image_generation.py

Requirements:
    - Set GOOGLE_API_KEY environment variable for Google models
    - Or configure API keys in the Lumi configuration system

Generated images will be saved to the 'generated_images' directory.
"""

import asyncio
import logging
import sys
import os
import base64
from pathlib import Path
from datetime import datetime

# Add the server directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "apps" / "server"))

from config import initialize_config, get_config
from services.inference.factory import create_unified_inference_service
from services.inference.models import UnifiedRequest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def save_image_from_base64(base64_data: str, filename: str, output_dir: Path) -> str:
    """Save a base64 encoded image to a file."""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image/'):
            base64_data = base64_data.split(',', 1)[1]
        
        # Decode base64 data
        image_data = base64.b64decode(base64_data)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Save image
        output_path = output_dir / filename
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return str(output_path)
    except Exception as e:
        logger.error(f"Failed to save image {filename}: {e}")
        return None


async def test_basic_image_generation(service, output_dir: Path):
    """Test basic image generation with Gemini 2.5 Flash Image."""
    print("\n🎨 Testing Basic Image Generation")
    print("=" * 50)
    
    prompts = [
        "A serene mountain landscape with a crystal clear lake reflecting the sky",
        "A futuristic robot painting on a canvas in an art studio",
        "A cozy coffee shop on a rainy day with warm lighting"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}: {prompt[:50]}... ---")
        try:
            request = UnifiedRequest(
                prompt=prompt,
                model="auto",  # Let the service choose the best image generation model
                response_format="image",
                extras={
                    "aspect_ratio": "16:9",
                    "response_modalities": ["Text", "Image"]
                }
            )
            
            response = await service.process_request(request)
            
            if response and response.images:
                print(f"✅ Image generated successfully!")
                print(f"   Model used: {response.model_used}")
                print(f"   Text description: {response.content[:100] if response.content else 'None'}...")

                # Save generated images
                for j, image_data in enumerate(response.images):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"test_{i}_{j+1}_{timestamp}.png"
                    saved_path = save_image_from_base64(image_data, filename, output_dir)
                    if saved_path:
                        print(f"   💾 Image saved: {saved_path}")
                    else:
                        print(f"   ❌ Failed to save image {j+1}")
            else:
                print(f"❌ Image generation failed: No images in response")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            # Debug: Let's see what models are actually available
            if "not compatible" in str(e):
                print(f"   Debug: Available models: {service.get_available_models()}")
                print(f"   Debug: Request model: {request.model}")
                print(f"   Debug: Request format: {request.response_format}")


async def test_different_aspect_ratios(service, output_dir: Path):
    """Test image generation with different aspect ratios."""
    print("\n📐 Testing Different Aspect Ratios")
    print("=" * 50)
    
    aspect_ratios = ["1:1", "16:9", "9:16", "3:2", "2:3"]
    base_prompt = "A beautiful sunset over the ocean with gentle waves"
    
    for aspect_ratio in aspect_ratios:
        print(f"\n--- Testing aspect ratio: {aspect_ratio} ---")
        try:
            request = UnifiedRequest(
                prompt=f"{base_prompt} (aspect ratio {aspect_ratio})",
                model="auto",  # Let the service choose the best image generation model
                response_format="image",
                extras={
                    "aspect_ratio": aspect_ratio,
                    "response_modalities": ["Image"]
                }
            )
            
            response = await service.process_request(request)
            
            if response and response.images:
                print(f"✅ Generated image with {aspect_ratio} aspect ratio")

                # Save the image
                for j, image_data in enumerate(response.images):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"aspect_{aspect_ratio.replace(':', 'x')}_{timestamp}.png"
                    saved_path = save_image_from_base64(image_data, filename, output_dir)
                    if saved_path:
                        print(f"   💾 Image saved: {saved_path}")
            else:
                print(f"❌ Failed: No images in response")
                
        except Exception as e:
            print(f"❌ Error: {e}")


async def test_creative_prompts(service, output_dir: Path):
    """Test image generation with creative and complex prompts."""
    print("\n🎭 Testing Creative Prompts")
    print("=" * 50)
    
    creative_prompts = [
        "A steampunk airship floating above a Victorian city at golden hour, detailed brass gears and copper pipes visible",
        "An underwater city with bioluminescent coral buildings and schools of colorful fish swimming between the structures",
        "A magical library where books float in the air and their pages glow with different colored lights"
    ]
    
    for i, prompt in enumerate(creative_prompts, 1):
        print(f"\n--- Creative Test {i} ---")
        print(f"Prompt: {prompt}")
        try:
            request = UnifiedRequest(
                prompt=prompt,
                model="auto",  # Let the service choose the best image generation model
                response_format="image",
                extras={
                    "aspect_ratio": "16:9",
                    "response_modalities": ["Text", "Image"]
                }
            )
            
            response = await service.process_request(request)
            
            if response and response.images:
                print(f"✅ Creative image generated!")
                if response.content:
                    print(f"   Description: {response.content[:150]}...")

                # Save the image
                for j, image_data in enumerate(response.images):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"creative_{i}_{timestamp}.png"
                    saved_path = save_image_from_base64(image_data, filename, output_dir)
                    if saved_path:
                        print(f"   💾 Image saved: {saved_path}")
            else:
                print(f"❌ Failed: No images in response")
                
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main test function."""
    print("🚀 Lumi Image Generation Test Script")
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
        print("   Make sure you have Google API key configured:")
        print("   - Set GOOGLE_API_KEY environment variable")
        return
    
    # Check if image generation models are available
    print("3. Checking image generation capabilities...")
    available_models = service.get_available_models()
    print(f"   Available models: {available_models}")
    
    # Create output directory
    output_dir = Path("generated_images")
    print(f"4. Images will be saved to: {output_dir.absolute()}")
    
    # Run tests
    try:
        await test_basic_image_generation(service, output_dir)
        await test_different_aspect_ratios(service, output_dir)
        await test_creative_prompts(service, output_dir)
        
        print("\n🎉 All image generation tests completed!")
        print(f"📁 Check the '{output_dir}' directory for generated images")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
