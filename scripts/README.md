# Lumi Scripts

This directory contains utility scripts for testing and demonstrating Lumi's AI capabilities.

## Available Scripts

### 🤖 Text Generation Testing
**File:** `test_text_generation.py`

Tests text generation capabilities using both OpenAI and Google models through the unified inference service.

**Features:**
- Tests OpenAI models (GPT-4o, GPT-5)
- Tests Google models (Gemini 1.5 Pro, Gemini 1.5 Flash)
- Tests vision capabilities with image attachments (`vision_test.jpg`)
- Tests automatic model selection
- Shows token usage and model performance

**Usage:**
```bash
python scripts/test_text_generation.py
```

### 🎨 Image Generation Testing
**File:** `test_image_generation.py`

Tests image generation capabilities using the Gemini 2.5 Flash Image model.

**Features:**
- Tests basic image generation
- Tests aspect ratio support (16:9)
- Tests creative style generation
- Tests reference image functionality with `ref_test.png`
- Saves generated images to `generated_images/` directory

**Features:**
- Basic image generation with descriptive prompts
- Different aspect ratio testing (1:1, 16:9, 9:16, 3:2, 2:3)
- Creative and complex prompt testing
- Automatic image saving to `generated_images/` directory

**Usage:**
```bash
python scripts/test_image_generation.py
```

## Test Images

The scripts use two test images for testing image attachment functionality:

- **`vision_test.jpg`** - Used for testing vision capabilities in text models (960x782 JPEG of a kitten with a cowboy hat)
- **`ref_test.png`** - Used for testing reference image functionality in image generation models (864x1184 PNG)

These images are included in the repository and automatically loaded by the test scripts.

## Setup Requirements

### API Keys
You need to configure API keys for the services you want to test:

**Option 1: Environment Variables**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

**Option 2: Lumi Configuration**
Set API keys through the Lumi configuration system (recommended for persistent setup).

### Dependencies
Make sure you have the Lumi server dependencies installed:
```bash
cd apps/server
pip install -r requirements.txt
```

## Output

### Text Generation Script
- Displays responses from different models
- Shows token usage statistics
- Compares model performance and capabilities

### Image Generation Script
- Saves generated images to `generated_images/` directory
- Creates timestamped filenames for easy organization
- Shows text descriptions when available from the model

## Notes

- Scripts automatically enable the unified inference system for testing
- Both scripts include comprehensive error handling and status reporting
- Generated images are saved as PNG files with descriptive filenames
- All scripts use the production Lumi configuration and service infrastructure

## Troubleshooting

**"Failed to create unified inference service"**
- Check that your API keys are properly configured
- Ensure the Lumi configuration system is working correctly

**"No models available"**
- Verify your API keys are valid and have sufficient credits
- Check network connectivity to the AI service providers

**Image generation fails**
- Ensure you have a valid Google API key with access to Gemini models
- Check that the prompt doesn't violate content policies
