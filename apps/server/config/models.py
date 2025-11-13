"""Configuration models and types for Lumi UI backend."""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator


class ApiKeysConfig(BaseModel):
    """API keys configuration."""
    openai: Optional[str] = None
    gemini: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional API keys


class UserConfig(BaseModel):
    """User preferences configuration."""
    theme: Literal["light", "dark", "auto"] = "dark"
    language: str = Field(default="en", min_length=2, max_length=5)
    timezone: str = "UTC"
    date_format: str = "YYYY-MM-DD"
    time_format: Literal["12h", "24h"] = "24h"
    
    @validator("language")
    def validate_language(cls, v):
        """Validate language code format."""
        if not v.isalpha():
            raise ValueError("Language code must contain only letters")
        return v.lower()


class PathsConfig(BaseModel):
    """Directory paths configuration."""
    cache: str = "cache"
    tmp: str = "tmp"
    logs: str = "logs"


class AppConfig(BaseModel):
    """Application behavior settings."""
    auto_save: bool = True
    auto_save_interval: int = Field(default=30000, ge=1000)  # milliseconds
    max_recent_items: int = Field(default=10, ge=1, le=100)


class ApiConfig(BaseModel):
    """API and connectivity settings."""
    base_url: str = "http://localhost:8000"
    timeout: int = Field(default=30000, ge=1000)  # milliseconds
    retry_attempts: int = Field(default=3, ge=0, le=10)
    enable_cache: bool = True


class SecurityConfig(BaseModel):
    """Security settings."""
    enable_logging: bool = True
    log_level: Literal["debug", "info", "warn", "error"] = "info"


class LumiConfig(BaseModel):
    """Main configuration model."""
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    user: UserConfig = Field(default_factory=UserConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    class Config:
        extra = "allow"  # Allow additional configuration sections


# State models for application state persistence

class SessionState(BaseModel):
    """Application session state."""
    last_active: Optional[str] = None
    app_version: str = "1.0.0"


class RecentState(BaseModel):
    """Recent items and history."""
    conversations: list = Field(default_factory=list)
    searches: list = Field(default_factory=list)
    actions: list = Field(default_factory=list)


class PanelSizes(BaseModel):
    """Resizable panel sizes."""
    sidebar: int = 250
    main: int = 800
    details: int = 300


class WindowState(BaseModel):
    """Window position and size."""
    width: int = 1200
    height: int = 800
    maximized: bool = False


class UIState(BaseModel):
    """UI state persistence."""
    sidebar_collapsed: bool = False
    active_tab: str = "home"
    selected_theme: str = "dark"
    panel_sizes: PanelSizes = Field(default_factory=PanelSizes)
    window_state: WindowState = Field(default_factory=WindowState)


class UserData(BaseModel):
    """User data and preferences."""
    bookmarks: list = Field(default_factory=list)
    favorites: list = Field(default_factory=list)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class MetricsState(BaseModel):
    """Application metrics."""
    startup_count: int = 0
    total_usage_time: int = 0  # milliseconds
    last_version: str = "1.0.0"
    crash_count: int = 0


class LumiState(BaseModel):
    """Main application state model."""
    session: SessionState = Field(default_factory=SessionState)
    recent: RecentState = Field(default_factory=RecentState)
    ui: UIState = Field(default_factory=UIState)
    user_data: UserData = Field(default_factory=UserData)
    metrics: MetricsState = Field(default_factory=MetricsState)
    
    class Config:
        extra = "allow"  # Allow additional state sections


# API request/response models for safe configuration access

class UserConfigUpdate(BaseModel):
    """Model for updating user configuration from frontend."""
    theme: Optional[Literal["light", "dark", "auto"]] = None
    language: Optional[str] = None
    timezone: Optional[str] = None
    date_format: Optional[str] = None
    time_format: Optional[Literal["12h", "24h"]] = None
    
    @validator("language")
    def validate_language(cls, v):
        """Validate language code format."""
        if v is not None and not v.isalpha():
            raise ValueError("Language code must contain only letters")
        return v.lower() if v else v


class SafeUserConfig(BaseModel):
    """Safe user configuration for frontend access."""
    theme: str
    language: str
    timezone: str
    date_format: str
    time_format: str


class SafeAppConfig(BaseModel):
    """Safe application configuration for frontend access."""
    version: str
    auto_save: bool
    auto_save_interval: int
    max_recent_items: int


class UIStateUpdate(BaseModel):
    """Model for updating UI state from frontend."""
    sidebar_collapsed: Optional[bool] = None
    active_tab: Optional[str] = None
    selected_theme: Optional[str] = None
    panel_sizes: Optional[Dict[str, int]] = None
    window_state: Optional[Dict[str, Any]] = None
