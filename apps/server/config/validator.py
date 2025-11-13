"""Configuration validation and error handling for Lumi UI."""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from pydantic import ValidationError

from .models import LumiConfig, LumiState

logger = logging.getLogger(__name__)


class ConfigurationValidator:
    """Validates configuration values and provides error handling."""
    
    # API key patterns for validation
    API_KEY_PATTERNS = {
        "openai": re.compile(r"^sk-[a-zA-Z0-9]{48}$"),
        "gemini": re.compile(r"^[a-zA-Z0-9_-]{39}$"),
    }
    
    # Supported language codes (ISO 639-1)
    SUPPORTED_LANGUAGES = {
        "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"
    }
    
    # Supported themes
    SUPPORTED_THEMES = {"light", "dark", "auto"}
    
    # Supported time formats
    SUPPORTED_TIME_FORMATS = {"12h", "24h"}
    
    # Supported log levels
    SUPPORTED_LOG_LEVELS = {"debug", "info", "warn", "error"}
    
    @classmethod
    def validate_config(cls, config_data: Dict[str, Any]) -> Tuple[bool, List[str], Optional[LumiConfig]]:
        """Validate configuration data.
        
        Args:
            config_data: Configuration data to validate
            
        Returns:
            Tuple of (is_valid, error_messages, validated_config)
        """
        errors = []
        
        try:
            # First, try to create the config object (Pydantic validation)
            config = LumiConfig(**config_data)
            
            # Additional custom validation
            custom_errors = cls._validate_custom_rules(config)
            errors.extend(custom_errors)
            
            # If we have custom errors, return them but still provide the config
            # (it might be usable with warnings)
            is_valid = len(errors) == 0
            return is_valid, errors, config
            
        except ValidationError as e:
            # Pydantic validation errors
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                message = f"Field '{field}': {error['msg']}"
                errors.append(message)
            
            return False, errors, None
        except Exception as e:
            errors.append(f"Unexpected validation error: {str(e)}")
            return False, errors, None
    
    @classmethod
    def validate_state(cls, state_data: Dict[str, Any]) -> Tuple[bool, List[str], Optional[LumiState]]:
        """Validate state data.
        
        Args:
            state_data: State data to validate
            
        Returns:
            Tuple of (is_valid, error_messages, validated_state)
        """
        errors = []
        
        try:
            # Try to create the state object (Pydantic validation)
            state = LumiState(**state_data)
            
            # Additional custom validation for state
            custom_errors = cls._validate_state_custom_rules(state)
            errors.extend(custom_errors)
            
            is_valid = len(errors) == 0
            return is_valid, errors, state
            
        except ValidationError as e:
            # Pydantic validation errors
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                message = f"State field '{field}': {error['msg']}"
                errors.append(message)
            
            return False, errors, None
        except Exception as e:
            errors.append(f"Unexpected state validation error: {str(e)}")
            return False, errors, None
    
    @classmethod
    def _validate_custom_rules(cls, config: LumiConfig) -> List[str]:
        """Apply custom validation rules to configuration.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Validate API keys format
        if config.api_keys.openai:
            if not cls.API_KEY_PATTERNS["openai"].match(config.api_keys.openai):
                errors.append("OpenAI API key format is invalid")
        
        if config.api_keys.gemini:
            if not cls.API_KEY_PATTERNS["gemini"].match(config.api_keys.gemini):
                errors.append("Gemini API key format is invalid")
        
        # Validate user preferences
        if config.user.theme not in cls.SUPPORTED_THEMES:
            errors.append(f"Unsupported theme: {config.user.theme}")
        
        if config.user.language not in cls.SUPPORTED_LANGUAGES:
            errors.append(f"Unsupported language: {config.user.language}")
        
        if config.user.time_format not in cls.SUPPORTED_TIME_FORMATS:
            errors.append(f"Unsupported time format: {config.user.time_format}")
        
        # Validate security settings
        if config.security.log_level not in cls.SUPPORTED_LOG_LEVELS:
            errors.append(f"Unsupported log level: {config.security.log_level}")
        
        # Validate numeric ranges
        if config.app.auto_save_interval < 1000:
            errors.append("Auto-save interval must be at least 1000ms")
        
        if config.api.timeout < 1000:
            errors.append("API timeout must be at least 1000ms")
        
        if config.api.retry_attempts < 0 or config.api.retry_attempts > 10:
            errors.append("API retry attempts must be between 0 and 10")
        
        return errors
    
    @classmethod
    def _validate_state_custom_rules(cls, state: LumiState) -> List[str]:
        """Apply custom validation rules to state.
        
        Args:
            state: State object to validate
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Validate metrics are non-negative
        if state.metrics.startup_count < 0:
            errors.append("Startup count cannot be negative")
        
        if state.metrics.total_usage_time < 0:
            errors.append("Total usage time cannot be negative")
        
        if state.metrics.crash_count < 0:
            errors.append("Crash count cannot be negative")
        
        # Validate UI state ranges
        if state.ui.panel_sizes.sidebar < 100 or state.ui.panel_sizes.sidebar > 1000:
            errors.append("Sidebar width must be between 100 and 1000 pixels")
        
        if state.ui.window_state.width < 800 or state.ui.window_state.width > 5000:
            errors.append("Window width must be between 800 and 5000 pixels")
        
        if state.ui.window_state.height < 600 or state.ui.window_state.height > 3000:
            errors.append("Window height must be between 600 and 3000 pixels")
        
        return errors
    
    @classmethod
    def sanitize_config_for_logging(cls, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from config for logging.
        
        Args:
            config_data: Configuration data to sanitize
            
        Returns:
            Sanitized configuration data
        """
        sanitized = config_data.copy()
        
        # Remove API keys
        if "api_keys" in sanitized:
            api_keys = sanitized["api_keys"].copy()
            for key in api_keys:
                if api_keys[key]:
                    api_keys[key] = "***REDACTED***"
            sanitized["api_keys"] = api_keys
        
        return sanitized
    
    @classmethod
    def get_validation_summary(cls, config: LumiConfig) -> Dict[str, Any]:
        """Get a summary of configuration validation status.
        
        Args:
            config: Configuration to summarize
            
        Returns:
            Validation summary dictionary
        """
        summary = {
            "api_keys_configured": 0,
            "api_keys_total": 2,
            "user_preferences_valid": True,
            "paths_configured": True,
            "warnings": []
        }
        
        # Count configured API keys
        if config.api_keys.openai:
            summary["api_keys_configured"] += 1
        if config.api_keys.gemini:
            summary["api_keys_configured"] += 1
        
        # Check for common issues
        if not config.api_keys.openai and not config.api_keys.gemini:
            summary["warnings"].append("No API keys configured")
        
        if config.user.language not in cls.SUPPORTED_LANGUAGES:
            summary["user_preferences_valid"] = False
            summary["warnings"].append(f"Unsupported language: {config.user.language}")
        
        return summary
