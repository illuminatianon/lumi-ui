# Testing the Unified Inference Infrastructure

This directory contains test scripts to demonstrate and validate the unified inference infrastructure, particularly the GPT-5 model-specific parameter handling.

## Quick Start

### 1. Set up your API key

```bash
# Option 1: Environment variable (recommended)
export LUMI_OPENAI_API_KEY='your-openai-api-key-here'

# Option 2: Use existing OpenAI key
export OPENAI_API_KEY='your-openai-api-key-here'
```

### 2. Run the GPT-5 test

```bash
cd apps/server
source venv/bin/activate
python test_gpt5.py
```

## Test Scripts

### `test_gpt5.py` - GPT-5 Parameter Mapping Demo

**Purpose**: Demonstrates the unified inference infrastructure with GPT-5's model-specific parameters.

**What it does**:
- Loads configuration like the server does
- Creates a unified inference service
- Makes a GPT-5 request with `reasoning_effort` parameter
- Shows parameter mapping: `max_tokens` → `max_completion_tokens`
- Demonstrates that `temperature` is ignored for GPT-5
- Falls back to GPT-4o if GPT-5 is not available

**Example output**:
```
✅ GPT-5 response received!
   - Model used: gpt-5-2025-08-07
   - Provider: openai
   - Finish reason: stop
   - Token usage: 48 prompt + 200 completion = 248 total

Parameter mapping:
   Input:  {'max_tokens': 200, 'temperature': 0.7, 'reasoning_effort': 'high'}
   Mapped: {'max_completion_tokens': 200, 'stream': False, 'reasoning_effort': 'high'}
   Note: temperature ignored, max_tokens → max_completion_tokens
```

### `test_standalone.py` - Core Infrastructure Test

**Purpose**: Tests the core components without requiring API keys.

**What it tests**:
- Model imports and creation
- Attachment handling
- Model configuration loading
- Provider registry functionality
- Parameter mapping for different models

### `setup_test.py` - Environment Setup Helper

**Purpose**: Helps set up the test environment and explains configuration options.

**Features**:
- Checks for API keys
- Validates dependencies
- Shows configuration examples
- Provides usage instructions

## Configuration

### Environment Variables

```bash
# Primary API key (recommended)
export LUMI_OPENAI_API_KEY='sk-proj-...'

# Alternative API key name
export OPENAI_API_KEY='sk-proj-...'

# Google API key (optional)
export LUMI_GEMINI_API_KEY='your-google-key'
```

### Config File

Create `~/.config/lumi/config.hjson`:

```hjson
{
  "api_keys": {
    "openai": "your-openai-api-key-here",
    "gemini": "your-google-api-key-here"
  },
  
  "inference": {
    "enabled": true,
    "default_provider": "auto",
    "fallback_providers": ["openai", "google"],
    
    "openai": {
      "enabled": true,
      "default_model": "gpt-4o"
    }
  }
}
```

## Key Features Demonstrated

### 1. Model-Specific Parameter Handling

The infrastructure automatically handles different parameter requirements:

- **GPT-4o**: Uses `max_tokens`, supports `temperature`
- **GPT-5**: Uses `max_completion_tokens`, ignores `temperature`, supports `reasoning_effort`
- **o1 models**: Use `max_completion_tokens`, no `temperature` support

### 2. Unified Interface

Same code works across different models:

```python
# Works with any model - parameters are automatically mapped
request = UnifiedRequest(
    prompt="Solve this problem...",
    model="gpt-5",  # or "gpt-4o", "gemini-1.5-pro", etc.
    max_tokens=300,  # Automatically mapped to correct parameter
    reasoning_effort="medium"  # Only used if model supports it
)
```

### 3. Provider Abstraction

Switch between providers seamlessly:

```python
# Same interface works with OpenAI, Google, etc.
response = await service.process_request(request)
```

## Troubleshooting

### "No OpenAI API key found"

Set the environment variable:
```bash
export LUMI_OPENAI_API_KEY='your-key-here'
```

### "GPT-5 not available"

GPT-5 may not be available in all OpenAI accounts yet. The test will automatically fall back to GPT-4o.

### "Module not found" errors

Make sure you're in the virtual environment:
```bash
source venv/bin/activate
pip install -e .
```

### LangChain compatibility issues

The test scripts avoid the LangChain compatibility issues by importing the unified inference components directly.

## Expected Results

When working correctly, you should see:

1. ✅ Configuration loaded successfully
2. ✅ Unified inference service created
3. ✅ Provider status shows OpenAI as available
4. ✅ GPT-5 request succeeds (or falls back to GPT-4o)
5. ✅ Parameter mapping demonstration shows correct transformations
6. 📄 Actual LLM response to the test question

This confirms that the unified inference infrastructure is working correctly and properly handling model-specific parameters.
