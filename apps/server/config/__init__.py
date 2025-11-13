"""Configuration management package for Lumi UI backend."""

import logging
from typing import Optional

from .models import LumiConfig, LumiState
from .loader import ConfigurationLoader
from .filesystem import ConfigFileSystem
from .validator import ConfigurationValidator
from .state_manager import StateManager
from .api import router as config_router

logger = logging.getLogger(__name__)

# Global configuration instances
_config_loader: Optional[ConfigurationLoader] = None
_state_manager: Optional[StateManager] = None


def initialize_config(config_dir: Optional[str] = None) -> bool:
    """Initialize the configuration system.
    
    Args:
        config_dir: Override default config directory (for testing)
        
    Returns:
        True if initialization successful, False otherwise
    """
    global _config_loader, _state_manager
    
    try:
        # Initialize configuration loader
        _config_loader = ConfigurationLoader(config_dir)
        
        # Ensure directory structure exists
        if not _config_loader.fs.ensure_directory_structure():
            logger.error("Failed to create directory structure")
            return False
        
        # Load configuration and state
        config = _config_loader.load_configuration()
        state = _config_loader.load_state()
        
        # Initialize state manager
        _state_manager = StateManager(_config_loader)
        
        # Increment startup count
        _state_manager.increment_startup_count()
        
        # Enable logging directory if configured
        if config.security.enable_logging:
            _config_loader.fs.ensure_logs_directory()
        
        logger.info("Configuration system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize configuration system: {e}")
        return False


def get_config_loader() -> ConfigurationLoader:
    """Get the global configuration loader instance.
    
    Returns:
        Configuration loader instance
        
    Raises:
        RuntimeError: If configuration system not initialized
    """
    if _config_loader is None:
        raise RuntimeError("Configuration system not initialized. Call initialize_config() first.")
    return _config_loader


def get_state_manager() -> StateManager:
    """Get the global state manager instance.
    
    Returns:
        State manager instance
        
    Raises:
        RuntimeError: If configuration system not initialized
    """
    if _state_manager is None:
        raise RuntimeError("Configuration system not initialized. Call initialize_config() first.")
    return _state_manager


def get_config() -> LumiConfig:
    """Get the current configuration.
    
    Returns:
        Current configuration object
    """
    return get_config_loader().config


def get_state() -> LumiState:
    """Get the current application state.
    
    Returns:
        Current state object
    """
    return get_config_loader().state


def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service.
    
    Args:
        service: Service name (e.g., 'openai', 'gemini')
        
    Returns:
        API key or None if not found
    """
    return get_config_loader().get_api_key(service)


def update_user_config(updates: dict) -> bool:
    """Update user configuration.
    
    Args:
        updates: Configuration updates to apply
        
    Returns:
        True if successful, False otherwise
    """
    return get_config_loader().update_user_config(updates)


def save_state() -> bool:
    """Save current application state.
    
    Returns:
        True if successful, False otherwise
    """
    return get_config_loader().save_state(get_state())


def cleanup_temp_files(max_age_hours: int = 24) -> bool:
    """Clean up old temporary files.
    
    Args:
        max_age_hours: Maximum age of files to keep in hours
        
    Returns:
        True if successful, False otherwise
    """
    try:
        return get_config_loader().fs.cleanup_tmp_directory(max_age_hours)
    except Exception as e:
        logger.error(f"Failed to cleanup temp files: {e}")
        return False


def get_config_health() -> dict:
    """Get configuration system health status.
    
    Returns:
        Health status dictionary
    """
    try:
        loader = get_config_loader()
        fs = loader.fs
        
        return {
            "status": "healthy",
            "config_loaded": loader._config is not None,
            "state_loaded": loader._state is not None,
            "config_file_exists": fs.config_file.exists(),
            "state_file_exists": fs.state_file.exists(),
            "cache_dir_exists": fs.cache_dir.exists(),
            "tmp_dir_exists": fs.tmp_dir.exists(),
            "disk_space_ok": fs.check_disk_space(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Export main components
__all__ = [
    "initialize_config",
    "get_config_loader",
    "get_state_manager", 
    "get_config",
    "get_state",
    "get_api_key",
    "update_user_config",
    "save_state",
    "cleanup_temp_files",
    "get_config_health",
    "config_router",
    "LumiConfig",
    "LumiState",
    "ConfigurationLoader",
    "ConfigFileSystem",
    "ConfigurationValidator",
    "StateManager",
]
