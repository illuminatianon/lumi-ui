# Lumi UI Configuration Handling Guide

This guide provides comprehensive documentation for global application configuration handling in Lumi UI. The application uses HJSON format for all configuration files and maintains a structured home directory for global state management.

## Overview

Lumi UI uses a hierarchical configuration system that separates:

- **Global Application Configuration**: User preferences, application settings, and persistent state
- **Development Configuration**: Environment variables for development and deployment (existing .env files)
- **Runtime Configuration**: Temporary data, cache, and scratch space

## Specification vs Examples

This document contains both **required specifications** and **illustrative examples**:

- **SPECIFICATION** sections define required behavior and structure
- **EXAMPLE** sections show possible implementations and configurations
- Implementation should focus on specifications first, with examples serving as guidance

## Directory Structure (SPECIFICATION)

The application **MUST** maintain the following directory structure in the user's home directory:

```
~/.lumi/
├── config.hjson              # REQUIRED: Main configuration file
├── state.hjson               # REQUIRED: Application state persistence
├── cache/                    # REQUIRED: Cached data directory
├── tmp/                      # REQUIRED: Temporary files and scratch space
└── logs/                     # OPTIONAL: Application logs (if logging enabled)
```

**Required Subdirectories:**

- `cache/` - For caching API responses and computed data
- `tmp/` - For temporary files, uploads, exports, and scratch space

**Optional Subdirectories:**

- `logs/` - Created only if logging is enabled
- Additional subdirectories may be created as needed by the application

## Configuration Files

### Root Configuration (`~/.lumi/config.hjson`) - SPECIFICATION

The main configuration file **MUST** contain the following required sections using HJSON format:

#### Required Configuration Structure:

```hjson
{
  // REQUIRED: API Keys for external services
  apiKeys: {
    openai: null               // OpenAI API key (string or null)
    gemini: null               // Google Gemini API key (string or null)
    // Additional API keys can be added as needed
  }

  // REQUIRED: User preferences
  user: {
    theme: "dark"              // "light" | "dark" | "auto"
    language: "en"             // ISO 639-1 language code
    // Additional user preferences as needed
  }

  // REQUIRED: Directory paths (relative to ~/.lumi/)
  paths: {
    cache: "cache"             // Cache directory
    tmp: "tmp"                 // Temporary files directory
    logs: "logs"               // Logs directory (if logging enabled)
  }
}
```

#### Example Extended Configuration:

The following shows additional configuration options that **MAY** be implemented:

```hjson
{
  // REQUIRED sections (as above)
  apiKeys: {
    openai: null
    gemini: null
  }

  user: {
    theme: "dark"
    language: "en"
    timezone: "UTC"            // EXAMPLE: IANA timezone identifier
    dateFormat: "YYYY-MM-DD"   // EXAMPLE: Date display format
    timeFormat: "24h"          // EXAMPLE: "12h" | "24h"
  }

  paths: {
    cache: "cache"
    tmp: "tmp"
    logs: "logs"
  }

  // EXAMPLE: Application behavior settings
  app: {
    autoSave: true             // Auto-save user data
    autoSaveInterval: 30000    // Auto-save interval in milliseconds
    maxRecentItems: 10         // Maximum recent items to remember
  }

  // EXAMPLE: API and connectivity settings
  api: {
    baseUrl: "http://localhost:8000"  // API base URL
    timeout: 30000             // Request timeout in milliseconds
    retryAttempts: 3           // Number of retry attempts
    enableCache: true          // Enable API response caching
  }

  // EXAMPLE: Security settings
  security: {
    enableLogging: true        // Enable application logging
    logLevel: "info"           // "debug" | "info" | "warn" | "error"
  }
}
```

### Application State (`~/.lumi/state.hjson`) - SPECIFICATION

The state file **MUST** be created to persist application state across restarts. For a standalone local webapp, this focuses on UI state and application data rather than user sessions.

#### Example State Structure:

```hjson
{
  // EXAMPLE: Application session (local only, no auth)
  session: {
    lastActive: "2025-11-09T12:00:00Z"
    appVersion: "1.0.0"        // Version when last used
  }

  // EXAMPLE: Recent items and history
  recent: {
    conversations: []          // Recent conversations/chats
    searches: []               // Recent search queries
    actions: []                // Recent user actions
  }

  // EXAMPLE: UI state (persisted across restarts)
  ui: {
    sidebarCollapsed: false    // Sidebar collapse state
    activeTab: "home"          // Currently active tab
    selectedTheme: "dark"      // Currently selected theme
    panelSizes: {              // Resizable panel sizes
      sidebar: 250
      main: 800
      details: 300
    }
    windowState: {             // Window position/size
      width: 1200
      height: 800
      maximized: false
    }
  }

  // EXAMPLE: User data (local preferences)
  userData: {
    bookmarks: []              // User bookmarks
    favorites: []              // Favorite items
    customSettings: {}         // User-specific custom settings
  }

  // EXAMPLE: Application metrics
  metrics: {
    startupCount: 0            // Number of times app has been started
    totalUsageTime: 0          // Total usage time in milliseconds
    lastVersion: "1.0.0"       // Last version used
    crashCount: 0              // Number of crashes
  }
}
```

## Configuration Management (SPECIFICATION)

### Loading Configuration

The application **MUST** load configuration in the following order:

1. **Default configuration** (hardcoded fallbacks for all required fields)
2. **User configuration** from `~/.lumi/config.hjson` (if exists)
3. **Environment variable overrides** (for development, optional)
4. **Command-line argument overrides** (optional)

### Required Default Values

The application **MUST** provide sensible defaults for all configuration values:

```hjson
{
  apiKeys: {
    openai: null
    gemini: null
  }
  user: {
    theme: "dark"
    language: "en"
  }
  paths: {
    cache: "cache"
    tmp: "tmp"
    logs: "logs"
  }
}
```

### Configuration Validation

The application **SHOULD** validate configuration values to ensure:

- Type safety for known fields
- API key format validation (if provided)
- Path validation for directory references
- Graceful handling of unknown fields (preserve but ignore)

## Directory Management (SPECIFICATION)

### Automatic Directory Creation

The application **MUST** create the required directory structure on first run:

```typescript
// SPECIFICATION: Required directory initialization
const homeDir = os.homedir();
const lumiDir = path.join(homeDir, ".lumi");

const requiredDirectories = [
  lumiDir, // REQUIRED: Main .lumi directory
  path.join(lumiDir, "cache"), // REQUIRED: Cache directory
  path.join(lumiDir, "tmp"), // REQUIRED: Temporary files directory
];

// OPTIONAL: Additional subdirectories as needed
const optionalDirectories = [
  path.join(lumiDir, "logs"), // If logging enabled
  path.join(lumiDir, "tmp", "uploads"), // If file uploads needed
  path.join(lumiDir, "tmp", "exports"), // If file exports needed
];
```

### Cache Management

The cache directory **SHOULD** be managed with basic cleanup:

- **RECOMMENDED**: Implement size limits to prevent unlimited growth
- **RECOMMENDED**: Clean up old cache files periodically
- **OPTIONAL**: Advanced cache invalidation strategies

### Temporary File Management

The tmp directory **SHOULD** be managed with:

- **RECOMMENDED**: Automatic cleanup of old files (e.g., >24 hours)
- **RECOMMENDED**: Size monitoring to prevent disk space issues
- **OPTIONAL**: Organized subdirectories for different file types

## HJSON Format Benefits

Using HJSON provides several advantages over JSON:

- **Comments**: Inline documentation and explanations
- **Relaxed syntax**: Trailing commas, unquoted keys
- **Multi-line strings**: Better readability for long values
- **Human-friendly**: Easier to edit manually
- **JSON compatible**: Can be parsed as JSON when needed

## Migration and Upgrades

### Configuration Migration

When upgrading between versions:

1. Backup existing configuration
2. Apply schema migrations
3. Validate migrated configuration
4. Update version metadata

### Backward Compatibility

The configuration system maintains backward compatibility by:

- Preserving unknown fields
- Providing sensible defaults for new fields
- Logging migration warnings
- Supporting multiple configuration versions

## Security Considerations (SPECIFICATION)

### File Permissions

Configuration files **MUST** be created with restricted permissions:

- `config.hjson`: 600 (read/write owner only) - **REQUIRED**
- `state.hjson`: 600 (read/write owner only) - **REQUIRED**
- `~/.lumi/` directory: 700 (full access owner only) - **REQUIRED**
- Cache directory: 700 (full access owner only) - **REQUIRED**
- Tmp directory: 700 (full access owner only) - **REQUIRED**

### API Key Security

API keys **MUST** be handled securely in this standalone local webapp:

- **REQUIRED**: Store API keys in plain text in config.hjson (user responsible for file permissions)
- **REQUIRED**: Never log API keys in application logs
- **REQUIRED**: Never transmit API keys except to their intended services
- **REQUIRED**: Never send API keys to the frontend/client
- **RECOMMENDED**: Validate API key format before storage
- **OPTIONAL**: Consider system keychain integration for enhanced security

### Client-Server Configuration Boundary

Since this is a local webapp with a Vue frontend and FastAPI backend, configuration access **MUST** be controlled:

#### Backend (Server) Access:

- **FULL ACCESS**: All configuration including API keys
- **RESPONSIBILITY**: Make API calls using stored keys
- **SECURITY**: Never expose API keys in API responses

#### Frontend (Client) Access:

- **LIMITED ACCESS**: Only safe configuration values
- **NO ACCESS**: API keys, sensitive paths, security settings
- **MECHANISM**: Dedicated API endpoints for safe config values

## Client-Side Configuration Handling (SPECIFICATION)

### Safe Configuration API

The backend **MUST** provide API endpoints for the frontend to access safe configuration values:

```python
# EXAMPLE: FastAPI endpoints for safe config access
@app.get("/api/config/user")
async def get_user_config():
    """Return safe user configuration for frontend."""
    return {
        "theme": config.user.theme,
        "language": config.user.language,
        "timezone": config.user.timezone,  # if implemented
    }

@app.get("/api/config/app")
async def get_app_config():
    """Return safe application configuration for frontend."""
    return {
        "version": "1.0.0",
        "features": {
            "enableExperimentalFeatures": config.features.enableExperimentalFeatures,
            # Only expose non-sensitive feature flags
        }
    }

@app.put("/api/config/user")
async def update_user_config(user_config: UserConfigUpdate):
    """Update user configuration from frontend."""
    # Validate and update only safe user preferences
    # Never allow API key updates from frontend
    pass
```

### Configuration Categories

Configuration values **MUST** be categorized for client access:

#### SAFE (Client Accessible):

- User preferences: theme, language, display settings
- UI state: window size, panel positions, sidebar state
- Application features: non-sensitive feature flags
- Display settings: date format, time format

#### RESTRICTED (Server Only):

- API keys: OpenAI, Gemini, any external service keys
- File system paths: cache, tmp, logs directories
- Security settings: session timeouts, logging levels
- Internal configuration: retry attempts, timeouts

### Frontend Configuration Management

The Vue frontend **SHOULD** implement configuration management:

```typescript
// EXAMPLE: Frontend config store (Pinia)
export const useConfigStore = defineStore("config", {
  state: () => ({
    user: {
      theme: "dark",
      language: "en",
    },
    app: {
      version: "1.0.0",
      features: {},
    },
    ui: {
      sidebarCollapsed: false,
      activeTab: "home",
    },
  }),

  actions: {
    async loadConfig() {
      // Load safe config from backend API
      const userConfig = await api.get("/api/config/user");
      const appConfig = await api.get("/api/config/app");
      this.user = userConfig.data;
      this.app = appConfig.data;
    },

    async updateUserConfig(updates: Partial<UserConfig>) {
      // Update user config via backend API
      await api.put("/api/config/user", updates);
      await this.loadConfig(); // Reload to confirm changes
    },
  },
});
```

### State Synchronization

For UI state that should persist across sessions:

- **Backend Storage**: Store in `state.hjson` on the server
- **API Access**: Provide safe endpoints for UI state
- **Frontend Sync**: Automatically sync UI changes to backend
- **Conflict Resolution**: Server state takes precedence on startup

## Development Integration (SPECIFICATION)

### Environment Variable Overrides

The application **MAY** support environment variable overrides for development:

```bash
# EXAMPLE: Override API keys for testing
LUMI_OPENAI_API_KEY=test-key-123
LUMI_GEMINI_API_KEY=test-key-456

# EXAMPLE: Override configuration paths for testing
LUMI_CONFIG_DIR=/tmp/lumi-test

# EXAMPLE: Override user preferences for testing
LUMI_THEME=light
LUMI_LANGUAGE=es
```

### Configuration in Development

During development, the application **SHOULD** support:

- **REQUIRED**: Fallback to default values when config files don't exist
- **RECOMMENDED**: Environment variable overrides for testing
- **OPTIONAL**: In-memory configuration for unit tests
- **OPTIONAL**: Mock configuration for integration tests

## Implementation Priority

### Phase 1 - Core Requirements (MUST IMPLEMENT)

1. **Directory Structure**: Create `~/.lumi/` with `cache/` and `tmp/` subdirectories
2. **Configuration File**: Support `config.hjson` with `apiKeys`, `user`, and `paths` sections
3. **State File**: Support `state.hjson` for application state persistence
4. **File Permissions**: Secure file permissions (600 for files, 700 for directories)
5. **Default Values**: Hardcoded fallbacks for all required configuration
6. **Client-Server Boundary**: Safe config API endpoints for frontend access
7. **API Key Security**: Never expose API keys to frontend

### Phase 2 - Recommended Features (SHOULD IMPLEMENT)

1. **Configuration Validation**: Basic type checking and format validation
2. **Frontend Config Store**: Pinia store for client-side configuration management
3. **State Synchronization**: UI state persistence across app restarts
4. **Environment Overrides**: Support for development environment variables
5. **Cache Management**: Basic cache cleanup and size monitoring
6. **Error Handling**: Graceful handling of missing or corrupted config files

### Phase 3 - Optional Enhancements (MAY IMPLEMENT)

1. **Advanced Validation**: Schema-based validation with detailed error messages
2. **Configuration Migration**: Version-based configuration upgrades
3. **Enhanced Security**: System keychain integration for API keys
4. **Real-time Sync**: Live configuration updates between frontend and backend
5. **Configuration UI**: Settings panel for managing configuration
6. **Diagnostic Tools**: Configuration validation and troubleshooting utilities

## Implementation Notes for Local Webapp

- **Primary Purpose**: Securely store and manage API keys for OpenAI, Gemini, and other services
- **Local Only**: No user authentication or multi-user concerns
- **Client-Server Split**: Backend handles sensitive config, frontend handles UI preferences
- **Security Boundary**: API keys never leave the backend server
- **Simple State**: UI state and preferences can be simpler without user sessions
- **Development Focus**: Easy to develop and debug locally
- **Graceful Degradation**: Application should work with minimal configuration

## Configuration Flow

1. **App Startup**: Backend loads config from `~/.lumi/config.hjson`
2. **Frontend Init**: Frontend requests safe config via API
3. **User Changes**: Frontend updates safe config via API endpoints
4. **Backend Updates**: Backend validates and saves config changes
5. **State Persistence**: UI state saved to `state.hjson` on backend
6. **API Calls**: Backend uses stored API keys for external service calls
