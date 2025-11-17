#!/usr/bin/env python3
"""Debug test script for Gemini 2.5 Flash Image generation."""

import asyncio
import json
import os
from pathlib import Path

from services.inference.providers.google_shim import GoogleShim
from services.inference.models import UnifiedRequest, ModelConfig, ModelCapabilities


async def test_direct_gemini_call():
    """Test direct Gemini API call to debug response format."""
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return
    
    print("🔍 Debug: Testing direct Gemini 2.5 Flash Image API call...")
    print(f"📋 Using API key: {api_key[:8]}...{api_key[-4:]}")
    
    # Create Google shim
    config = {"api_key": api_key}
    shim = GoogleShim(config)
    
    # Create model config
    model_config = ModelConfig(
        name="gemini-2.5-flash-image",
        display_name="Gemini 2.5 Flash Image",
        provider="google",
        capabilities=ModelCapabilities(
            image_generation=True,
            text_generation=True,
            supports_multimodal=True
        ),
        parameter_mapping={},
        supported_parameters=set(),
        cost_per_1k_tokens=0.03,
        context_window=8192
    )
    
    # Create request
    request = UnifiedRequest(
        prompt="A simple red circle on a white background",
        model="gemini-2.5-flash-image",
        response_format="image",
        extras={
            "aspect_ratio": "1:1",
            "response_modalities": ["Image"]
        }
    )
    
    print(f"📝 Prompt: {request.prompt}")
    print(f"📐 Aspect ratio: {request.extras['aspect_ratio']}")
    
    try:
        # Make the request
        print("🎨 Making API request...")
        
        # Build the payload manually to debug
        parts = [{"text": request.prompt}]
        generation_config = {
            "imageConfig": {"aspectRatio": "1:1"},
            "responseModalities": ["Image"]
        }
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config
        }
        
        print(f"📤 Request payload: {json.dumps(payload, indent=2)}")
        
        # Make raw HTTP request
        url = f'/v1beta/models/{model_config.name}:generateContent'
        params = {'key': api_key}
        
        response = await shim.client.post(url, json=payload, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"📥 Raw response: {json.dumps(data, indent=2)}")
        
        # Check response structure
        if 'candidates' in data and data['candidates']:
            candidate = data['candidates'][0]
            print(f"🔍 Candidate structure: {json.dumps(candidate, indent=2)}")
            
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                print(f"🔍 Parts count: {len(parts)}")
                
                for i, part in enumerate(parts):
                    print(f"🔍 Part {i}: {list(part.keys())}")
                    if 'inline_data' in part:
                        inline_data = part['inline_data']
                        print(f"🔍 Inline data keys: {list(inline_data.keys())}")
                        if 'data' in inline_data:
                            data_len = len(inline_data['data'])
                            print(f"🔍 Image data length: {data_len} characters")
                            print(f"🔍 MIME type: {inline_data.get('mime_type', 'unknown')}")
            else:
                print("❌ No content/parts in candidate")
        else:
            print("❌ No candidates in response")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await shim.client.aclose()


async def main():
    """Main function."""
    print("🔍 Gemini 2.5 Flash Image Debug Test")
    print("=" * 40)
    
    await test_direct_gemini_call()
    
    print("\n✨ Debug test completed!")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not Path("services").exists():
        print("❌ Error: Please run this script from the apps/server directory")
        print("cd apps/server && python test_debug_image.py")
        exit(1)
    
    asyncio.run(main())
