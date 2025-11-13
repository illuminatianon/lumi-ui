"""Configuration loading and management for Lumi UI."""

import os
import hjson
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .models import LumiConfig, LumiState
from .filesystem import ConfigFileSystem

logger = logging.getLogger(__name__)


class ConfigurationLoader:
    """Handles loading and saving of configuration and state."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            config_dir: Override default config directory (for testing)
        """
        self.fs = ConfigFileSystem(config_dir)
        self._config: Optional[LumiConfig] = None
        self._state: Optional[LumiState] = None
    
    def load_configuration(self) -> LumiConfig:
        """Load configuration with proper precedence order.
        
        Returns:
            Loaded configuration object
        """
        try:
            # 1. Start with default configuration
            config_data = self._get_default_config()
            
            # 2. Load user configuration if exists
            user_config = self._load_user_config()
            if user_config:
                config_data = self._merge_configs(config_data, user_config)
            
            # 3. Apply environment variable overrides
            env_overrides = self._get_environment_overrides()
            if env_overrides:
                config_data = self._merge_configs(config_data, env_overrides)
            
            # 4. Validate and create configuration object
            self._config = LumiConfig(**config_data)
            
            logger.info("Configuration loaded successfully")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Fall back to default configuration
            self._config = LumiConfig()
            return self._config
    
    def save_configuration(self, config: LumiConfig) -> bool:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory structure exists
            if not self.fs.ensure_directory_structure():
                return False
            
            # Create backup of existing config
            if self.fs.config_file.exists():
                self.fs.create_backup(self.fs.config_file)
            
            # Convert to dict and save as HJSON
            config_dict = config.dict()
            hjson_content = hjson.dumps(config_dict, indent=2, sort_keys=True)
            
            self.fs.config_file.write_text(hjson_content, encoding='utf-8')
            
            # Set secure permissions
            self.fs.ensure_config_file_permissions()
            
            self._config = config
            logger.info(f"Configuration saved to {self.fs.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_state(self) -> LumiState:
        """Load application state.
        
        Returns:
            Loaded state object
        """
        try:
            if self.fs.state_file.exists():
                content = self.fs.state_file.read_text(encoding='utf-8')
                state_data = hjson.loads(content)
                self._state = LumiState(**state_data)
                logger.info("Application state loaded successfully")
            else:
                self._state = LumiState()
                logger.info("Created new application state")
            
            return self._state
            
        except Exception as e:
            logger.error(f"Failed to load application state: {e}")
            # Fall back to default state
            self._state = LumiState()
            return self._state
    
    def save_state(self, state: LumiState) -> bool:
        """Save application state to file.
        
        Args:
            state: State to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory structure exists
            if not self.fs.ensure_directory_structure():
                return False
            
            # Update last active timestamp
            state.session.last_active = datetime.now().isoformat()
            
            # Convert to dict and save as HJSON
            state_dict = state.dict()
            hjson_content = hjson.dumps(state_dict, indent=2, sort_keys=True)
            
            self.fs.state_file.write_text(hjson_content, encoding='utf-8')
            
            # Set secure permissions
            self.fs.ensure_config_file_permissions()
            
            self._state = state
            logger.info(f"Application state saved to {self.fs.state_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save application state: {e}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "api_keys": {
                "openai": None,
                "gemini": None
            },
            "user": {
                "theme": "dark",
                "language": "en",
                "timezone": "UTC",
                "date_format": "YYYY-MM-DD",
                "time_format": "24h"
            },
            "paths": {
                "cache": "cache",
                "tmp": "tmp",
                "logs": "logs"
            },
            "app": {
                "auto_save": True,
                "auto_save_interval": 30000,
                "max_recent_items": 10
            },
            "api": {
                "base_url": "http://localhost:8000",
                "timeout": 30000,
                "retry_attempts": 3,
                "enable_cache": True
            },
            "security": {
                "enable_logging": True,
                "log_level": "info"
            }
        }
    
    def _load_user_config(self) -> Optional[Dict[str, Any]]:
        """Load user configuration from file.
        
        Returns:
            User configuration dictionary or None if not found
        """
        try:
            if not self.fs.config_file.exists():
                return None
            
            content = self.fs.config_file.read_text(encoding='utf-8')
            return hjson.loads(content)
            
        except Exception as e:
            logger.warning(f"Failed to load user configuration: {e}")
            return None

    def _get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables.

        Returns:
            Environment overrides dictionary
        """
        overrides = {}

        # API key overrides
        if os.getenv("LUMI_OPENAI_API_KEY"):
            overrides.setdefault("api_keys", {})["openai"] = os.getenv("LUMI_OPENAI_API_KEY")

        if os.getenv("LUMI_GEMINI_API_KEY"):
            overrides.setdefault("api_keys", {})["gemini"] = os.getenv("LUMI_GEMINI_API_KEY")

        # User preference overrides
        if os.getenv("LUMI_THEME"):
            overrides.setdefault("user", {})["theme"] = os.getenv("LUMI_THEME")

        if os.getenv("LUMI_LANGUAGE"):
            overrides.setdefault("user", {})["language"] = os.getenv("LUMI_LANGUAGE")

        # Path overrides
        if os.getenv("LUMI_CONFIG_DIR"):
            # This would require reinitializing the filesystem manager
            logger.warning("LUMI_CONFIG_DIR override not supported after initialization")

        return overrides

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def update_user_config(self, updates: Dict[str, Any]) -> bool:
        """Update user configuration with new values.

        Args:
            updates: Configuration updates to apply

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._config:
                self.load_configuration()

            # Apply updates to current config
            current_dict = self._config.dict()
            updated_dict = self._merge_configs(current_dict, {"user": updates})

            # Validate updated configuration
            updated_config = LumiConfig(**updated_dict)

            # Save updated configuration
            return self.save_configuration(updated_config)

        except Exception as e:
            logger.error(f"Failed to update user configuration: {e}")
            return False

    def get_safe_config(self) -> Dict[str, Any]:
        """Get safe configuration values for frontend access.

        Returns:
            Safe configuration dictionary (no API keys)
        """
        if not self._config:
            self.load_configuration()

        return {
            "user": {
                "theme": self._config.user.theme,
                "language": self._config.user.language,
                "timezone": self._config.user.timezone,
                "date_format": self._config.user.date_format,
                "time_format": self._config.user.time_format,
            },
            "app": {
                "version": "1.0.0",
                "auto_save": self._config.app.auto_save,
                "auto_save_interval": self._config.app.auto_save_interval,
                "max_recent_items": self._config.app.max_recent_items,
            }
        }

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a specific service.

        Args:
            service: Service name (e.g., 'openai', 'gemini')

        Returns:
            API key or None if not found
        """
        if not self._config:
            self.load_configuration()

        return getattr(self._config.api_keys, service, None)

    @property
    def config(self) -> LumiConfig:
        """Get current configuration."""
        if not self._config:
            self.load_configuration()
        return self._config

    @property
    def state(self) -> LumiState:
        """Get current state."""
        if not self._state:
            self.load_state()
        return self._state
