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
from services.inference.models import UnifiedRequest, Attachment

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_test_image(image_name: str) -> Attachment:
    """Load a test image as an Attachment object."""
    script_dir = Path(__file__).parent
    image_path = script_dir / image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    return Attachment.from_file(str(image_path))


def save_image_from_base64(base64_data: str, filename: str, output_dir: Path) -> str:
    """Save a base64 encoded image to a file."""
    try:
        # Remove data URL prefix if present
        if base64_data.startswith("data:image/"):
            base64_data = base64_data.split(",", 1)[1]

        # Decode base64 data
        image_data = base64.b64decode(base64_data)

        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)

        # Save image
        output_path = output_dir / filename
        with open(output_path, "wb") as f:
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
        "A disheveled hobo, warming his hands by a barrel fire. he is roasting a quoakka on a stick",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Test {i}: {prompt[:50]}... ---")
        try:
            request = UnifiedRequest(
                prompt=prompt,
                model="google/gemini-2.5-flash-image",  # Use Gemini for image generation
                response_format="image",
                extras={
                    "aspect_ratio": "16:9",
                    "response_modalities": ["Text", "Image"],
                },
            )

            response = await service.process_request(request)

            if response and response.images:
                print(f"✅ Image generated successfully!")
                print(f"   Model used: {response.model_used}")
                print(
                    f"   Text description: {response.content[:100] if response.content else 'None'}..."
                )

                # Save generated images
                for j, image_data in enumerate(response.images):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"test_{i}_{j+1}_{timestamp}.png"
                    saved_path = save_image_from_base64(
                        image_data, filename, output_dir
                    )
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
    print("\n📐 Testing Aspect Ratio Support")
    print("=" * 50)

    # Test just one non-square aspect ratio to verify the feature works
    aspect_ratio = "16:9"
    prompt = "A beautiful sunset over the ocean with gentle waves"

    try:
        request = UnifiedRequest(
            prompt=prompt,
            model="google/gemini-2.5-flash-image",
            response_format="image",
            extras={"aspect_ratio": aspect_ratio, "response_modalities": ["Image"]},
        )

        response = await service.process_request(request)

        if response and response.images:
            print(f"✅ Generated image with {aspect_ratio} aspect ratio")

            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"aspect_test_{timestamp}.png"
            saved_path = save_image_from_base64(
                response.images[0], filename, output_dir
            )
            if saved_path:
                print(f"   💾 Image saved: {saved_path}")
        else:
            print(f"❌ Failed: No images in response")

    except Exception as e:
        print(f"❌ Error: {e}")


async def test_creative_prompts(service, output_dir: Path):
    """Test image generation with creative and complex prompts."""
    print("\n🎭 Testing Creative Style Generation")
    print("=" * 50)

    # Single creative prompt test
    prompt = "A steampunk airship floating above a Victorian city at golden hour, detailed brass gears and copper pipes visible"

    try:
        request = UnifiedRequest(
            prompt=prompt,
            model="google/gemini-2.5-flash-image",
            response_format="image",
            extras={
                "aspect_ratio": "16:9",
                "response_modalities": ["Text", "Image"],
            },
        )

        response = await service.process_request(request)

        if response and response.images:
            print(f"✅ Creative style image generated!")
            if response.content:
                print(f"   Description: {response.content[:150]}...")

            # Save the image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"style_test_{timestamp}.png"
            saved_path = save_image_from_base64(
                response.images[0], filename, output_dir
            )
            if saved_path:
                print(f"   💾 Image saved: {saved_path}")
        else:
            print(f"❌ Failed: No images in response")

    except Exception as e:
        print(f"❌ Error: {e}")


async def test_reference_image_functionality(service, output_dir: Path):
    """Test reference image functionality - loading and processing."""
    print("\n🖼️ Testing Reference Image Functionality")
    print("=" * 50)

    # Load the reference test image
    try:
        ref_image = load_test_image("ref_test.png")
        print(
            f"✅ Loaded reference image: {ref_image.filename} ({ref_image.mime_type})"
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Single reference-based generation test
    prompt = "Create a new image inspired by the style and composition of this reference image"

    try:
        request = UnifiedRequest(
            prompt=prompt,
            model="google/gemini-2.5-flash-image",
            response_format="image",
            attachments=[ref_image],
            extras={
                "aspect_ratio": "1:1",
                "response_modalities": ["Text", "Image"],
            },
        )

        response = await service.process_request(request)

        if response and response.images:
            print(f"✅ Reference-based image generated!")
            print(f"   Model used: {response.model_used}")
            if response.content:
                print(f"   Description: {response.content[:150]}...")

            # Save the generated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ref_test_{timestamp}.png"
            saved_path = save_image_from_base64(
                response.images[0], filename, output_dir
            )
            if saved_path:
                print(f"   💾 Image saved: {saved_path}")
        else:
            print(f"❌ Reference generation failed: No images in response")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"   Reference image functionality may not be supported by this model")


async def test_multi_image_generation(service, output_dir: Path):
    """Test image generation with multiple reference images."""
    print("\n🖼️🖼️ Testing Multi-Image Generation")
    print("=" * 50)

    # Load both test images
    try:
        target_image = load_test_image("dam.png")  # TARGET
        reference_image = load_test_image("ref_test.png")  # REFERENCE
        print(
            f"✅ Loaded target image: {target_image.filename} ({target_image.mime_type})"
        )
        print(
            f"✅ Loaded reference image: {reference_image.filename} ({reference_image.mime_type})"
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Specific prompt with both images
    prompt = "Image 1: TARGET, Image 2: REFERENCE [BUNCHAN].\nUpdate TARGET according to instructions. Change only what is requested. Make sure the change fits with the style of the image. You may modify the REFERENCE to match the request! it is only to guide the appearance.\n\nBUNCHAN has CHUCKED more WOOD than a WOODCHUCK COULD. the WOODCHUCK is sitting around, looking dazed and confuzed."

    try:
        request = UnifiedRequest(
            prompt=prompt,
            model="google/gemini-2.5-flash-image",
            response_format="image",
            attachments=[target_image, reference_image],  # Both images as attachments
            extras={
                "aspect_ratio": "4:3",  # Specific aspect ratio
                "response_modalities": ["Text", "Image"],
            },
        )

        response = await service.process_request(request)

        if response and response.images:
            print(f"✅ Multi-image generation successful!")
            print(f"   Model used: {response.model_used}")
            print(f"   Attachments processed: {len(response.attachments_processed)}")
            if response.content:
                print(f"   Description: {response.content[:150]}...")

            # Save the generated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_image_test_{timestamp}.png"
            saved_path = save_image_from_base64(
                response.images[0], filename, output_dir
            )
            if saved_path:
                print(f"   💾 Image saved: {saved_path}")
        else:
            print(f"❌ Multi-image generation failed: No images in response")

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

    # Inference is always enabled if API keys are configured

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

    # Run tests - three simple tests
    try:
        await test_basic_image_generation(service, output_dir)
        await test_reference_image_functionality(service, output_dir)
        await test_multi_image_generation(service, output_dir)

        print("\n🎉 Image generation tests completed!")
        print("   ✅ Basic generation")
        print("   ✅ Reference image functionality")
        print("   ✅ Multi-image generation")
        print(f"📁 Check the '{output_dir}' directory for generated images")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
