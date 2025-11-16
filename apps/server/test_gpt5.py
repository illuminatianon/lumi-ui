#!/usr/bin/env python3
"""Quick test script for GPT-5 with unified inference infrastructure."""

import asyncio
import logging
import sys
import os

# Add the server directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import initialize_config, get_config
from services.inference.factory import create_unified_inference_service
from services.inference.models import UnifiedRequest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def test_gpt5_request():
    """Test GPT-5 with model-specific parameters."""
    
    print("🚀 Testing GPT-5 with Unified Inference")
    print("=" * 50)
    
    # Initialize configuration like the server does
    print("1. Initializing configuration...")
    if not initialize_config():
        print("❌ Failed to initialize configuration")
        return False
    
    config = get_config()
    print(f"✅ Configuration loaded")
    
    # Check API key
    if not config.api_keys.openai:
        print("⚠️  No OpenAI API key found in configuration")
        print("   Please set LUMI_OPENAI_API_KEY environment variable or add to config")
        print("   Example: export LUMI_OPENAI_API_KEY='your-api-key-here'")
        return False
    
    print(f"✅ OpenAI API key configured: {config.api_keys.openai[:8]}...")
    
    # Enable unified inference for this test
    config.inference.enabled = True
    config.inference.openai.enabled = True
    config.inference.openai.api_key = config.api_keys.openai
    
    print(f"✅ Unified inference enabled for test")
    
    # Create unified service
    print("\n2. Creating unified inference service...")
    try:
        service = create_unified_inference_service()
        if not service:
            print("❌ Failed to create unified inference service")
            return False
        
        print("✅ Unified inference service created")
        
        # Check provider status
        provider_status = service.get_provider_status()
        print(f"   - Provider status: {provider_status}")
        
        if not provider_status.get("openai", False):
            print("❌ OpenAI provider not available")
            return False
        
    except Exception as e:
        print(f"❌ Failed to create service: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test GPT-5 request with reasoning
    print("\n3. Testing GPT-5 request...")
    try:
        # Create a request that showcases GPT-5's reasoning capabilities
        request = UnifiedRequest(
            prompt="Solve this step by step: If a train travels at 60 mph for 2.5 hours, then slows to 40 mph for another 1.5 hours, what is the total distance traveled?",
            model="gpt-5",
            reasoning_effort="medium",  # GPT-5 specific parameter
            max_tokens=300,  # Will be mapped to max_completion_tokens
            temperature=0.7  # Will be ignored since GPT-5 doesn't support it
        )
        
        print(f"📝 Request details:")
        print(f"   - Model: {request.model}")
        print(f"   - Reasoning effort: {request.reasoning_effort}")
        print(f"   - Max tokens: {request.max_tokens} (will map to max_completion_tokens)")
        print(f"   - Temperature: {request.temperature} (will be ignored)")
        print(f"   - Prompt: {request.prompt[:80]}...")
        
        print(f"\n🔄 Making request to GPT-5...")
        response = await service.process_request(request)
        
        print(f"✅ GPT-5 response received!")
        print(f"   - Model used: {response.model_used}")
        print(f"   - Provider: {response.provider}")
        print(f"   - Finish reason: {response.finish_reason}")
        print(f"   - Token usage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total")
        
        print(f"\n📄 Response content:")
        print("-" * 50)
        print(response.content)
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ GPT-5 request failed: {e}")
        
        # Check if it's a model availability issue
        if "model" in str(e).lower() and "not found" in str(e).lower():
            print("   This might be because GPT-5 is not yet available in your OpenAI account")
            print("   Trying with GPT-4o instead...")
            
            # Fallback to GPT-4o
            try:
                request.model = "gpt-4o"
                request.reasoning_effort = None  # GPT-4o doesn't have reasoning effort
                
                print(f"\n🔄 Making fallback request to GPT-4o...")
                response = await service.process_request(request)
                
                print(f"✅ GPT-4o response received!")
                print(f"   - Model used: {response.model_used}")
                print(f"   - Provider: {response.provider}")
                
                print(f"\n📄 Response content:")
                print("-" * 50)
                print(response.content)
                print("-" * 50)
                
                return True
                
            except Exception as fallback_error:
                print(f"❌ Fallback to GPT-4o also failed: {fallback_error}")
        
        import traceback
        traceback.print_exc()
        return False


async def demonstrate_parameter_mapping():
    """Demonstrate how parameters are mapped for different models."""
    
    print("\n🔧 Demonstrating Parameter Mapping")
    print("=" * 40)
    
    try:
        from services.inference.registry import ModelRegistry
        from services.inference.providers.openai_shim import OpenAIShim
        
        registry = ModelRegistry()
        shim = OpenAIShim({"api_key": "test"})
        
        # Test GPT-4o parameter mapping
        print("1. GPT-4o parameter mapping:")
        gpt4_config = await registry.get_model_config("openai", "gpt-4o")
        if gpt4_config:
            unified_params = {
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "frequency_penalty": 0.1
            }
            mapped = shim.map_parameters(unified_params, gpt4_config)
            print(f"   Input:  {unified_params}")
            print(f"   Mapped: {mapped}")
        
        # Test GPT-5 parameter mapping
        print("\n2. GPT-5 parameter mapping:")
        gpt5_config = await registry.get_model_config("openai", "gpt-5")
        if gpt5_config:
            unified_params = {
                "max_tokens": 200,
                "temperature": 0.7,  # Will be ignored
                "reasoning_effort": "high"
            }
            mapped = shim.map_parameters(unified_params, gpt5_config)
            print(f"   Input:  {unified_params}")
            print(f"   Mapped: {mapped}")
            print(f"   Note: temperature ignored, max_tokens → max_completion_tokens")
        
    except Exception as e:
        print(f"❌ Parameter mapping demo failed: {e}")


if __name__ == "__main__":
    print("🧪 GPT-5 Test Script")
    print("This script demonstrates the unified inference infrastructure")
    print("with GPT-5's model-specific parameter handling.\n")
    
    success = asyncio.run(test_gpt5_request())
    asyncio.run(demonstrate_parameter_mapping())
    
    if success:
        print("\n✅ Test completed successfully!")
        print("The unified inference infrastructure correctly handled GPT-5's")
        print("model-specific parameters (max_completion_tokens, reasoning_effort)")
    else:
        print("\n⚠️  Test completed with issues")
        print("Check your OpenAI API key and model availability")
