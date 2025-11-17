# Tests Directory

This directory contains all test files for the Lumi server, organized by test type.

## Directory Structure

### `unit/`
Unit tests for individual components and functions:
- `test_config.py` - Configuration system tests
- `test_image_generation.py` - Image generation parameter validation tests

### `integration/`
Integration tests that test multiple components working together:
- `test_image_api_integration.py` - API model validation and integration tests

### `manual/`
Manual test scripts for development and debugging:
- `test_gpt5.py` - GPT-5 parameter mapping demonstration
- `test_minimal_image.py` - Minimal Gemini image generation test
- `test_debug_image.py` - Debug script for Gemini image generation
- `test_standalone.py` - Core infrastructure test without API keys
- `test_unified_inference.py` - Unified inference infrastructure test
- `test_unified_simple.py` - Simple unified inference test
- `test_pure_rest.py` - Pure REST API implementation test
- `test_img.py` - Basic image generation test
- `test_setup.py` - LangChain setup verification (legacy)

## Running Tests

### Automated Tests
```bash
# Run all tests
pnpm test

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Manual Tests
```bash
# Run individual manual tests
python tests/manual/test_gpt5.py
python tests/manual/test_minimal_image.py
```

## Test Guidelines

- **Unit tests**: Use pytest framework, mock external dependencies
- **Integration tests**: Test real component interactions, may use test APIs
- **Manual tests**: Standalone scripts for development and debugging
