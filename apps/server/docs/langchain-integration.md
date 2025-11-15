# LangChain Integration

This document describes the LangChain integration in the Lumi UI backend, providing AI capabilities for text generation, image analysis, and image generation.

## Features

- **Text Generation**: Generate text using OpenAI GPT models or Google Gemini
- **Image Analysis**: Analyze images with text prompts using vision models
- **Image Generation**: Generate images using OpenAI DALL-E
- **Multi-provider Support**: Automatic fallback between OpenAI and Google models
- **Configuration Management**: Integrated with the existing configuration system

## Setup

### 1. Install Dependencies

Dependencies are already included in `pyproject.toml`:
- `langchain>=1.0.0`
- `langchain-openai>=1.0.0`
- `langchain-google-genai>=3.0.0`
- `pillow>=12.0.0`

### 2. Configure API Keys

Set your API keys using environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

Or configure them in the configuration files (see Configuration section).

### 3. Test Setup

Run the setup test script:

```bash
cd apps/server
python test_setup.py
```

## API Endpoints

### Model Status
```
GET /api/ai/status
```
Returns the availability status of all AI models.

### Text Generation
```
POST /api/ai/generate-text
```
Generate text using AI models.

**Request Body:**
```json
{
  "prompt": "Write a short story about AI",
  "model": "auto",
  "system_message": "You are a creative writer",
  "temperature": 0.7
}
```

### Image Analysis
```
POST /api/ai/analyze-image
```
Analyze an image with a text prompt.

**Request Body:**
```json
{
  "image_data": "base64-encoded-image",
  "prompt": "What do you see in this image?",
  "model": "auto"
}
```

### Image Upload Analysis
```
POST /api/ai/analyze-image-upload
```
Upload and analyze an image file.

**Form Data:**
- `file`: Image file
- `prompt`: Text prompt
- `model`: Model preference (optional)

### Image Generation
```
POST /api/ai/generate-image
```
Generate images using DALL-E.

**Request Body:**
```json
{
  "prompt": "A serene mountain landscape",
  "size": "1024x1024",
  "quality": "standard",
  "n": 1
}
```

## Configuration

The LangChain integration uses the existing configuration system. Configuration options are available in the `langchain` section:

```json
{
  "langchain": {
    "default_model": "auto",
    "openai_model": "gpt-4o",
    "google_model": "gemini-1.5-pro",
    "default_temperature": 0.7,
    "max_tokens": null,
    "dalle_size": "1024x1024",
    "dalle_quality": "standard"
  }
}
```

### Configuration Options

- `default_model`: Default model preference ("openai", "google", "auto")
- `openai_model`: OpenAI model to use (e.g., "gpt-4o", "gpt-3.5-turbo")
- `google_model`: Google model to use (e.g., "gemini-1.5-pro")
- `default_temperature`: Default temperature for text generation (0.0-2.0)
- `max_tokens`: Maximum tokens for responses (null for model default)
- `dalle_size`: Default image size for DALL-E
- `dalle_quality`: Default image quality for DALL-E

## Usage Examples

### Python Service Usage

```python
from services.langchain_service import LangChainService

service = LangChainService()

# Text generation
result = await service.generate_text("Hello, world!")
print(result["text"])

# Image analysis
result = await service.process_image_with_text(
    image_data="base64-image-data",
    prompt="What's in this image?"
)
print(result["text"])

# Image generation
result = await service.generate_image("A beautiful sunset")
print(result["images"][0]["url"])
```

### API Usage

```bash
# Check model status
curl http://localhost:8000/api/ai/status

# Generate text
curl -X POST http://localhost:8000/api/ai/generate-text \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, AI!"}'

# Generate image
curl -X POST http://localhost:8000/api/ai/generate-image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A robot painting"}'
```

## Testing

Run the test suite:

```bash
cd apps/server
python -m pytest tests/test_langchain_service.py -v
```

Run examples:

```bash
cd apps/server
python examples/langchain_examples.py
```

## Troubleshooting

### No Models Available
- Ensure API keys are set correctly
- Check that the configuration system is initialized
- Verify network connectivity

### Import Errors
- Ensure all dependencies are installed in the virtual environment
- Check that the Python path includes the server directory

### API Key Issues
- Verify API keys are valid and have sufficient credits
- Check environment variable names (OPENAI_API_KEY, GOOGLE_API_KEY)
- Ensure keys have the necessary permissions for the models being used
