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
- Directory cleaned up - manual test scripts have been moved to dedicated scripts outside the tests directory
- This directory is reserved for future manual testing scripts

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
# Manual test scripts have been moved to dedicated scripts
# See scripts/ directory for API testing scripts
```

## Test Guidelines

- **Unit tests**: Use pytest framework, mock external dependencies
- **Integration tests**: Test real component interactions, may use test APIs
- **Manual tests**: Standalone scripts for development and debugging
