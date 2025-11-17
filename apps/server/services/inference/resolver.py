"""Model resolver for registry-based model naming."""

import logging
from typing import Dict, Any, Tuple, Optional

from .base import Provider
from .registry import ProviderRegistry
from .models import UnifiedRequest

logger = logging.getLogger(__name__)


class ModelResolver:
    """Resolves registry-style model names to providers and models."""
    
    def __init__(self, provider_registry: ProviderRegistry):
        """Initialize the model resolver.
        
        Args:
            provider_registry: Provider registry instance
        """
        self.provider_registry = provider_registry
    
    def parse_model_name(self, model_name: str) -> Tuple[str, str]:
        """Parse 'provider/model' format.
        
        Args:
            model_name: Model name in format 'provider/model'
            
        Returns:
            Tuple of (provider_name, model_name)
            
        Raises:
            ValueError: If model name format is invalid
        """
        if "/" not in model_name:
            raise ValueError(f"Model name must be in 'provider/model' format, got: {model_name}")
        
        parts = model_name.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model name format: {model_name}")
        
        provider_name, model_name = parts
        
        if not provider_name or not model_name:
            raise ValueError(f"Provider and model names cannot be empty: {model_name}")
        
        return provider_name, model_name
    
    async def resolve_request(self, config: Dict[str, Any]) -> Tuple[Provider, str, Dict[str, Any]]:
        """Resolve config to provider, model, and validated params.
        
        Args:
            config: Request configuration containing model and parameters
            
        Returns:
            Tuple of (provider_instance, model_name, validated_params)
            
        Raises:
            ValueError: If model or provider not found, or parameters invalid
        """
        model_full_name = config.get("model")
        if not model_full_name:
            raise ValueError("Model name is required")
        
        # Parse provider/model
        provider_name, model_name = self.parse_model_name(model_full_name)
        
        # Get provider
        provider = self.provider_registry.get_provider(provider_name)
        if not provider:
            available_providers = self.provider_registry.get_available_providers()
            raise ValueError(f"Provider '{provider_name}' not found. Available: {available_providers}")
        
        # Validate model exists
        supported_models = provider.get_supported_models()
        if model_name not in supported_models:
            available = list(supported_models.keys())
            raise ValueError(f"Model '{model_name}' not supported by {provider_name}. Available: {available}")
        
        # Validate and map parameters
        params = {k: v for k, v in config.items() if k != "model"}
        validated_params = provider.validate_model_params(model_name, params)
        
        return provider, model_name, validated_params
    
    def build_unified_request(self, config: Dict[str, Any]) -> UnifiedRequest:
        """Build a UnifiedRequest from config.
        
        Args:
            config: Request configuration
            
        Returns:
            UnifiedRequest instance
        """
        return UnifiedRequest(
            prompt=config.get("prompt", ""),
            system_message=config.get("system_message"),
            attachments=config.get("attachments", []),
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens"),
            top_p=config.get("top_p"),
            frequency_penalty=config.get("frequency_penalty"),
            presence_penalty=config.get("presence_penalty"),
            stop_sequences=config.get("stop_sequences"),
            reasoning_effort=config.get("reasoning_effort"),
            response_format=config.get("response_format", "text"),
            stream=config.get("stream", False),
            model=config.get("model", "auto")
        )
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models in registry format.
        
        Returns:
            Dictionary with available models and metadata
        """
        models_by_provider = self.provider_registry.list_available_models()
        
        # Flatten to a single list with metadata
        all_models = []
        for provider_name, models in models_by_provider.items():
            for model_registry_name in models:
                _, model_name = self.parse_model_name(model_registry_name)
                provider_class = self.provider_registry._providers[provider_name]
                supported_models = provider_class.get_supported_models()
                model_config = supported_models.get(model_name)
                
                if model_config:
                    all_models.append({
                        "registry_name": model_registry_name,
                        "provider": provider_name,
                        "model": model_name,
                        "display_name": model_config.display_name,
                        "capabilities": model_config.capabilities.model_dump(),
                        "supported_parameters": list(model_config.supported_parameters)
                    })
        
        return {
            "models": all_models,
            "providers": models_by_provider,
            "total_models": len(all_models)
        }
