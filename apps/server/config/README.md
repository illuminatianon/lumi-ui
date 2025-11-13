# Lumi UI Configuration System

This directory contains the complete backend configuration management system for Lumi UI, implementing the specifications from `docs/config-handling.md`.

## Overview

The configuration system provides:

- **Secure Configuration Management**: HJSON-based configuration with API key security
- **Application State Persistence**: UI state and user data persistence across sessions
- **Directory Management**: Automatic creation and management of `~/.lumi/` structure
- **Safe API Endpoints**: Frontend-safe configuration access without exposing sensitive data
- **Validation & Error Handling**: Robust validation with graceful degradation
- **Environment Overrides**: Development-friendly environment variable support

## Architecture

```
config/
├── __init__.py          # Main module interface and global instances
├── models.py            # Pydantic models for configuration and state
├── filesystem.py        # Directory and file management
├── loader.py            # Configuration loading and saving logic
├── validator.py         # Configuration validation and error handling
├── state_manager.py     # Application state management
├── api.py              # FastAPI endpoints for safe configuration access
└── README.md           # This file
```

## Key Components

### 1. Configuration Models (`models.py`)

Defines Pydantic models for:
- `LumiConfig`: Main configuration structure
- `LumiState`: Application state structure
- API request/response models for safe frontend access

### 2. Filesystem Management (`filesystem.py`)

Handles:
- `~/.lumi/` directory structure creation
- File permissions (600 for files, 700 for directories)
- Temporary file cleanup
- Disk space monitoring

### 3. Configuration Loader (`loader.py`)

Manages:
- Configuration loading with precedence (defaults → user config → env overrides)
- HJSON file reading/writing
- Configuration merging and updates

### 4. State Manager (`state_manager.py`)

Provides:
- Recent items tracking (conversations, searches, actions)
- Bookmark management
- Usage metrics and application statistics
- UI state persistence

### 5. Validation (`validator.py`)

Implements:
- API key format validation
- Configuration value validation
- Error reporting and sanitization for logging

### 6. API Endpoints (`api.py`)

Exposes safe endpoints:
- `GET /api/config/user` - Get user preferences
- `PUT /api/config/user` - Update user preferences
- `GET /api/config/app` - Get application settings
- `GET /api/config/ui-state` - Get UI state
- `PUT /api/config/ui-state` - Update UI state
- `GET /api/config/health` - Configuration system health

## Usage

### Initialization

```python
from config import initialize_config

# Initialize the configuration system
success = initialize_config()
if not success:
    print("Failed to initialize configuration")
```

### Accessing Configuration

```python
from config import get_config, get_api_key

# Get full configuration
config = get_config()
print(f"Theme: {config.user.theme}")

# Get specific API key
openai_key = get_api_key("openai")
```

### Updating Configuration

```python
from config import update_user_config

# Update user preferences
success = update_user_config({
    "theme": "light",
    "language": "es"
})
```

### State Management

```python
from config import get_state_manager

state_manager = get_state_manager()

# Add recent search
state_manager.add_recent_search("machine learning")

# Add bookmark
state_manager.add_bookmark({
    "title": "Example",
    "url": "https://example.com"
})
```

## Directory Structure

The system creates and manages:

```
~/.lumi/
├── config.hjson         # Main configuration (API keys, preferences)
├── state.hjson          # Application state (UI state, recent items)
├── cache/               # Cached data
├── tmp/                 # Temporary files
└── logs/                # Application logs (if enabled)
```

## Security Features

- **File Permissions**: Configuration files are created with 600 permissions
- **API Key Protection**: API keys never exposed to frontend
- **Input Validation**: All configuration updates are validated
- **Safe Endpoints**: Only non-sensitive configuration exposed via API

## Environment Variables

Development overrides supported:

```bash
LUMI_OPENAI_API_KEY=sk-...     # Override OpenAI API key
LUMI_GEMINI_API_KEY=...        # Override Gemini API key
LUMI_THEME=light               # Override theme
LUMI_LANGUAGE=es               # Override language
```

## Testing

Run the test suite:

```bash
cd apps/server
source venv/bin/activate
python test_config.py
```

## Integration with FastAPI

The configuration system is automatically integrated with the main FastAPI application:

```python
# In main.py
from config import initialize_config, config_router

# Initialize on startup
initialize_config()

# Include API routes
app.include_router(config_router)
```

## Error Handling

The system provides graceful degradation:
- Falls back to defaults if configuration files are missing/corrupted
- Continues operation with warnings for non-critical validation errors
- Provides detailed error messages for debugging

## Performance Considerations

- Configuration is loaded once at startup and cached
- State updates are batched and written asynchronously when possible
- Temporary file cleanup runs periodically to prevent disk space issues
- Directory structure validation is performed only when needed
