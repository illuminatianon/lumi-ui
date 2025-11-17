# Image Generation Implementation Guide

## Overview

This document describes the implementation of image generation support for **Gemini 2.5 Flash Image** and **GPT Image 1** models in the Lumi inference system.

## Supported Models

### Gemini 2.5 Flash Image
- **Provider**: Google
- **Capabilities**: Text-to-image, image editing, style transfer, inpainting
- **Aspect Ratios**: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- **Quality**: Standard (only)
- **Special Features**: 
  - Reference image support
  - Response modalities (text + image)
  - Conversational editing

### GPT Image 1
- **Provider**: OpenAI
- **Capabilities**: Text-to-image
- **Aspect Ratios**: 1:1, 3:2, 2:3
- **Quality**: Standard, HD
- **Styles**: Vivid, Natural
- **Special Features**: 
  - High-quality generation
  - Revised prompt feedback

## API Usage

### Basic Image Generation

```python
# Gemini 2.5 Flash Image
request = ImageGenerationRequest(
    prompt="A serene mountain landscape with a crystal clear lake",
    model="gemini-2.5-flash-image",
    aspect_ratio="16:9",
    response_modalities=["Text", "Image"]
)

# GPT Image 1
request = ImageGenerationRequest(
    prompt="A futuristic cityscape at sunset",
    model="gpt-image-1",
    aspect_ratio="3:2",
    quality="hd",
    style="vivid"
)
```

### Advanced Features

```python
# Image editing with Gemini (reference image + prompt)
request = ImageGenerationRequest(
    prompt="Add a wizard hat to the cat in this image",
    model="gemini-2.5-flash-image",
    reference_images=["base64-encoded-image"],
    aspect_ratio="1:1"
)

# Style control with GPT Image 1
request = ImageGenerationRequest(
    prompt="A peaceful garden scene",
    model="gpt-image-1",
    quality="hd",
    style="natural"  # or "vivid"
)
```

## Parameter Validation

The system automatically validates parameters and provides helpful warnings:

```python
from services.inference.image_validation import ImageGenerationValidator

# Validate parameters for a specific model
validated_params, warnings = ImageGenerationValidator.validate_parameters(
    "gpt-image-1", 
    {"aspect_ratio": "21:9", "quality": "ultra"}
)
# Result: aspect_ratio corrected to "1:1", quality to "standard"
# Warnings: ["Aspect ratio '21:9' not supported...", "Quality 'ultra' not supported..."]
```

## Model Capabilities

```python
# Get model capabilities
capabilities = ImageGenerationValidator.get_model_capabilities("gemini-2.5-flash-image")
print(capabilities["supported_aspect_ratios"])  # All 10 aspect ratios
print(capabilities["supports_reference_images"])  # True

capabilities = ImageGenerationValidator.get_model_capabilities("gpt-image-1")
print(capabilities["supported_styles"])  # ["vivid", "natural"]
print(capabilities["supports_reference_images"])  # False
```

## Response Format

```json
{
  "images": [
    {
      "url": "https://example.com/image.png",
      "revised_prompt": "Enhanced prompt (GPT models)",
      "metadata": {"quality": "hd", "style": "vivid"}
    }
  ],
  "model_used": "gpt-image-1",
  "success": true,
  "text_content": "Generated description (Gemini models)",
  "generation_metadata": {
    "aspect_ratio": "3:2",
    "quality": "hd",
    "style": "vivid"
  }
}
```

## Implementation Files

- `config/inference_models.py` - Model configurations
- `services/inference/registry.py` - Model registry
- `services/inference/providers/google_shim.py` - Gemini implementation
- `services/inference/providers/openai_shim.py` - GPT Image 1 implementation
- `services/inference/image_validation.py` - Parameter validation
- `api/models.py` - API request/response models

## Testing

Run the comprehensive tests:

```bash
cd apps/server
source venv/bin/activate
python tests/unit/test_image_generation.py
python tests/integration/test_image_api_integration.py
```

## Next Steps

1. **API Endpoints**: Create FastAPI endpoints for image generation
2. **Authentication**: Add API key validation for providers
3. **Rate Limiting**: Implement rate limiting for image generation
4. **Caching**: Add response caching for repeated requests
5. **Monitoring**: Add usage tracking and analytics
