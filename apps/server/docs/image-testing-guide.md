# Image Generation Test Guide

## Quick Test Script

Use `tests/manual/test_minimal_image.py` to test Gemini 2.5 Flash Image generation.

### Setup

1. **Get a Google API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key
   - Copy the key

2. **Configure API Key** (choose one method):

   **Option A: Through Lumi UI** (Recommended):
   - Start the Lumi server
   - Go to Settings > API Keys
   - Add your Gemini API key

   **Option B: Environment Variable**:
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```

   **Option C: Direct Config File**:
   - Edit `~/.config/lumi-ui/config.hjson`
   - Add: `"api_keys": {"gemini": "your-api-key-here"}`

3. **Run the Test**:
   ```bash
   cd apps/server
   source venv/bin/activate
   python tests/manual/test_minimal_image.py
   ```

### Expected Output

```
🧪 Minimal Gemini 2.5 Flash Image Test
========================================
🔧 Loading Lumi configuration...
🚀 Testing Gemini 2.5 Flash Image generation...
📋 Using API key from Lumi config: AIzaSyC1...xyz4
🎨 Generating image...
📝 Prompt: A beautiful sunset over a serene mountain lake with vibrant orange and pink colors reflecting in the water
📐 Aspect ratio: 16:9
✅ Image generated successfully!
🤖 Model used: gemini-2.5-flash-image
🔧 Provider: google
📊 Image format: data:image/png;base64
💾 Image saved to: /path/to/test.png
📏 Image size: 245678 bytes
📋 Metadata: {'aspect_ratio': '16:9', 'response_modalities': ['Image']}

✨ Test completed!
```

### What the Script Does

1. **Loads Lumi Configuration**: Initializes the full Lumi config system and loads API keys
2. **Gets API Key**: Retrieves the Gemini API key from Lumi configuration
3. **Creates Request**: Builds a UnifiedRequest for Gemini 2.5 Flash Image
4. **Generates Image**: Calls the inference service to generate an image
5. **Saves Result**: Decodes the base64 image and saves it as `test.png`

### Parameters Used

- **Model**: `gemini-2.5-flash-image`
- **Aspect Ratio**: `16:9` (widescreen)
- **Response Mode**: Image only (no text description)
- **Prompt**: Beautiful sunset landscape

### Customization

You can modify the script to test different parameters:

```python
# Change aspect ratio
"aspect_ratio": "1:1"  # Square
"aspect_ratio": "9:16" # Portrait

# Include text description
"response_modalities": ["Text", "Image"]

# Different prompt
prompt="A futuristic cityscape with flying cars at night"
```

### Troubleshooting

- **API Key Error**: Make sure Gemini API key is configured in Lumi (see setup options above)
- **Config Error**: Ensure Lumi configuration system can initialize (check `~/.config/lumi-ui/`)
- **Import Errors**: Run `pip install -e .` in the venv
- **Permission Error**: Make sure the script is executable (`chmod +x tests/manual/test_minimal_image.py`)
- **Network Error**: Check internet connection and API key validity

### File Structure

```
apps/server/
├── tests/manual/test_minimal_image.py          # The test script
├── test.png                       # Generated image (after running)
├── services/inference/            # Inference implementation
├── config/                        # Configuration models
└── venv/                         # Virtual environment
```
