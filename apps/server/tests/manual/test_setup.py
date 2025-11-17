#!/usr/bin/env python3
"""Quick test script to verify LangChain setup."""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import initialize_config
from services.langchain_service import LangChainService


async def test_setup():
    """Test the LangChain setup."""
    print("Testing LangChain Setup")
    print("=" * 30)
    
    # Initialize configuration
    print("1. Initializing configuration...")
    if initialize_config():
        print("   ✓ Configuration initialized successfully")
    else:
        print("   ✗ Configuration initialization failed")
        return False
    
    # Initialize LangChain service
    print("2. Initializing LangChain service...")
    try:
        service = LangChainService()
        print("   ✓ LangChain service initialized successfully")
    except Exception as e:
        print(f"   ✗ LangChain service initialization failed: {e}")
        return False
    
    # Check model availability
    print("3. Checking model availability...")
    status = service.get_available_models()
    
    models = [
        ("OpenAI Chat", status["openai_chat"]),
        ("OpenAI DALL-E", status["openai_dalle"]),
        ("Google Gemini", status["google_chat"])
    ]
    
    available_count = 0
    for model_name, available in models:
        if available:
            print(f"   ✓ {model_name}: Available")
            available_count += 1
        else:
            print(f"   ✗ {model_name}: Not available (API key required)")
    
    if available_count == 0:
        print("\n⚠️  No models are available. Please set API keys in configuration:")
        print("   - Set OPENAI_API_KEY environment variable for OpenAI models")
        print("   - Set GOOGLE_API_KEY environment variable for Google models")
        print("   - Or configure them in the config files")
        return False
    
    # Test basic functionality (only if models are available)
    if status["openai_chat"] or status["google_chat"]:
        print("4. Testing text generation...")
        try:
            result = await service.generate_text(
                prompt="Say 'Hello, LangChain setup is working!'",
                model="auto"
            )
            
            if result["success"]:
                print(f"   ✓ Text generation successful")
                print(f"   Model used: {result['model_used']}")
                print(f"   Response: {result['text'][:100]}...")
            else:
                print(f"   ✗ Text generation failed: {result.get('error')}")
        except Exception as e:
            print(f"   ✗ Text generation test failed: {e}")
    
    print("\n" + "=" * 30)
    print("Setup test completed!")
    
    if available_count > 0:
        print("✓ LangChain is ready to use!")
        print(f"  Available models: {available_count}/3")
    else:
        print("⚠️  Setup incomplete - no API keys configured")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_setup())
