"""Example usage of LangChain integrations."""

import asyncio
import base64
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from services.langchain_service import LangChainService
from config import initialize_config


async def example_text_generation():
    """Example of text generation."""
    print("\n=== Text Generation Example ===")
    
    service = LangChainService()
    
    # Simple text generation
    result = await service.generate_text(
        prompt="Write a short poem about artificial intelligence.",
        model="auto"
    )
    
    if result["success"]:
        print(f"Model used: {result['model_used']}")
        print(f"Generated text:\n{result['text']}")
    else:
        print(f"Error: {result.get('error')}")


async def example_text_with_system_message():
    """Example of text generation with system message."""
    print("\n=== Text Generation with System Message ===")
    
    service = LangChainService()
    
    result = await service.generate_text(
        prompt="What is the capital of France?",
        system_message="You are a helpful geography teacher. Always provide additional context.",
        model="auto",
        temperature=0.3
    )
    
    if result["success"]:
        print(f"Model used: {result['model_used']}")
        print(f"Generated text:\n{result['text']}")
    else:
        print(f"Error: {result.get('error')}")


async def example_image_analysis():
    """Example of image analysis (requires a test image)."""
    print("\n=== Image Analysis Example ===")
    
    service = LangChainService()
    
    # Create a simple test image (1x1 pixel red image)
    from PIL import Image
    import io
    
    # Create a simple red square
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Convert to base64
    img_b64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    
    result = await service.process_image_with_text(
        image_data=img_b64,
        prompt="What color is this image? Describe what you see.",
        model="auto"
    )
    
    if result["success"]:
        print(f"Model used: {result['model_used']}")
        print(f"Analysis result:\n{result['text']}")
    else:
        print(f"Error: {result.get('error')}")


async def example_image_generation():
    """Example of image generation."""
    print("\n=== Image Generation Example ===")
    
    service = LangChainService()
    
    result = await service.generate_image(
        prompt="A serene mountain landscape with a crystal clear lake reflecting the sky",
        size="1024x1024",
        quality="standard"
    )
    
    if result["success"]:
        print(f"Model used: {result['model_used']}")
        print(f"Generated {len(result['images'])} image(s)")
        for i, img in enumerate(result['images']):
            print(f"Image {i+1} URL: {img['url']}")
            if img.get('revised_prompt'):
                print(f"Revised prompt: {img['revised_prompt']}")
    else:
        print(f"Error: {result.get('error')}")


async def example_model_status():
    """Example of checking model availability."""
    print("\n=== Model Status Example ===")
    
    service = LangChainService()
    status = service.get_available_models()
    
    print("Available models:")
    for model, available in status.items():
        print(f"  {model}: {'✓' if available else '✗'}")


async def main():
    """Run all examples."""
    print("LangChain Integration Examples")
    print("=" * 40)
    
    # Initialize configuration
    if not initialize_config():
        print("Failed to initialize configuration")
        return
    
    # Run examples
    await example_model_status()
    await example_text_generation()
    await example_text_with_system_message()
    await example_image_analysis()
    
    # Only run image generation if OpenAI is available
    service = LangChainService()
    if service.get_available_models()["openai_dalle"]:
        await example_image_generation()
    else:
        print("\n=== Image Generation Example ===")
        print("Skipped: OpenAI DALL-E not available (API key required)")


if __name__ == "__main__":
    asyncio.run(main())
