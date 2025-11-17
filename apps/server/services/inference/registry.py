"""Model and provider registry system."""

import logging
from typing import Dict, Optional, Type, List, Any
from datetime import datetime, timedelta

from .models import ModelConfig, ParameterMapping, ModelCapabilities, ProviderConfig
from .base import Provider

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for model configurations with static definitions and runtime caching."""

    def __init__(self):
        """Initialize the model registry."""
        self._static_registry = self._build_static_registry()
        self._runtime_cache: Dict[str, Dict[str, ModelConfig]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(hours=1)  # Cache for 1 hour

    def _build_static_registry(self) -> Dict[str, Dict[str, ModelConfig]]:
        """Build the static model registry with known model configurations."""
        return {
            "openai": {
                "gpt-4o": ModelConfig(
                    name="gpt-4o",
                    display_name="GPT-4o",
                    provider="openai",
                    capabilities=ModelCapabilities(
                        text_generation=True,
                        vision=True,
                        function_calling=True,
                        streaming=True,
                        json_mode=True,
                        max_context_length=128000,
                    ),
                    parameter_mapping=ParameterMapping(
                        max_tokens_param="max_tokens",
                        temperature_param="temperature",
                        top_p_param="top_p",
                    ),
                    supported_parameters={
                        "max_tokens",
                        "temperature",
                        "top_p",
                        "frequency_penalty",
                        "presence_penalty",
                        "stop",
                        "stream",
                    },
                    cost_per_1k_tokens=0.005,
                    context_window=128000,
                ),
                "gpt-5": ModelConfig(
                    name="gpt-5",
                    display_name="GPT-5",
                    provider="openai",
                    capabilities=ModelCapabilities(
                        text_generation=True,
                        vision=True,
                        reasoning=True,
                        function_calling=True,
                        streaming=True,
                        json_mode=True,
                        max_context_length=200000,
                    ),
                    parameter_mapping=ParameterMapping(
                        max_tokens_param="max_completion_tokens",
                        temperature_param=None,  # Not supported
                        top_p_param=None,
                        custom_params={"reasoning_effort": "medium"},
                    ),
                    supported_parameters={
                        "max_completion_tokens",
                        "reasoning_effort",
                        "stream",
                    },
                    cost_per_1k_tokens=0.01,
                    context_window=200000,
                ),
                "o1-preview": ModelConfig(
                    name="o1-preview",
                    display_name="o1-preview",
                    provider="openai",
                    capabilities=ModelCapabilities(
                        text_generation=True, reasoning=True, max_context_length=128000
                    ),
                    parameter_mapping=ParameterMapping(
                        max_tokens_param="max_completion_tokens",
                        temperature_param=None,
                        top_p_param=None,
                    ),
                    supported_parameters={"max_completion_tokens"},
                    cost_per_1k_tokens=0.015,
                    context_window=128000,
                ),
                "dall-e-3": ModelConfig(
                    name="dall-e-3",
                    display_name="DALL-E 3",
                    provider="openai",
                    capabilities=ModelCapabilities(
                        image_generation=True, text_generation=False
                    ),
                    parameter_mapping=ParameterMapping(),
                    supported_parameters={"size", "quality", "style", "n"},
                    cost_per_1k_tokens=0.04,  # Per image cost
                    context_window=4000,
                ),
            },
            "google": {
                "gemini-2.5-flash-image": ModelConfig(
                    name="gemini-2.5-flash-image",
                    display_name="Gemini 2.5 Flash Image",
                    provider="google",
                    capabilities=ModelCapabilities(
                        text_generation=True,
                        image_editing=True,
                        vision=True,
                        function_calling=False,
                        streaming=False,
                        max_context_length=1000000,
                    ),
                    parameter_mapping=ParameterMapping(
                        max_tokens_param="max_output_tokens",
                        temperature_param="temperature",
                        top_p_param="top_p",
                    ),
                    supported_parameters={
                        "max_output_tokens",
                        "temperature",
                        "top_p",
                        "stop_sequences",
                    },
                    cost_per_1k_tokens=0.00075,
                    context_window=1000000,
                ),
                "gemini-1.5-pro": ModelConfig(
                    name="gemini-1.5-pro",
                    display_name="Gemini 1.5 Pro",
                    provider="google",
                    capabilities=ModelCapabilities(
                        text_generation=True,
                        vision=True,
                        function_calling=True,
                        streaming=True,
                        max_context_length=2000000,
                    ),
                    parameter_mapping=ParameterMapping(
                        max_tokens_param="max_output_tokens",
                        temperature_param="temperature",
                        top_p_param="top_p",
                    ),
                    supported_parameters={
                        "max_output_tokens",
                        "temperature",
                        "top_p",
                        "stop_sequences",
                    },
                    cost_per_1k_tokens=0.0035,
                    context_window=2000000,
                ),
                "gemini-1.5-flash": ModelConfig(
                    name="gemini-1.5-flash",
                    display_name="Gemini 1.5 Flash",
                    provider="google",
                    capabilities=ModelCapabilities(
                        text_generation=True,
                        vision=True,
                        function_calling=True,
                        streaming=True,
                        max_context_length=1000000,
                    ),
                    parameter_mapping=ParameterMapping(
                        max_tokens_param="max_output_tokens",
                        temperature_param="temperature",
                        top_p_param="top_p",
                    ),
                    supported_parameters={
                        "max_output_tokens",
                        "temperature",
                        "top_p",
                        "stop_sequences",
                    },
                    cost_per_1k_tokens=0.00075,
                    context_window=1000000,
                ),
            },
        }

    async def get_model_config(
        self, provider: str, model_name: str, provider_instance: Optional[Provider] = None
    ) -> Optional[ModelConfig]:
        """Get model configuration with fallback strategy.

        Args:
            provider: Provider name
            model_name: Model name
            provider_instance: Optional provider instance for dynamic discovery

        Returns:
            Model configuration or None if not found
        """
        # 1. Check static registry first
        if (
            provider in self._static_registry
            and model_name in self._static_registry[provider]
        ):
            return self._static_registry[provider][model_name]

        # 2. Check runtime cache
        if (
            provider in self._runtime_cache
            and model_name in self._runtime_cache[provider]
        ):
            # Check if cache is still valid
            if provider in self._cache_timestamps:
                cache_age = datetime.now() - self._cache_timestamps[provider]
                if cache_age < self._cache_ttl:
                    return self._runtime_cache[provider][model_name]

        # 3. Try dynamic discovery if provider is provided
        if provider_instance:
            try:
                discovered = await provider_instance.discover_models()
                if model_name in discovered:
                    # Cache the discovered models
                    if provider not in self._runtime_cache:
                        self._runtime_cache[provider] = {}
                    self._runtime_cache[provider].update(discovered)
                    self._cache_timestamps[provider] = datetime.now()
                    return discovered[model_name]
            except Exception as e:
                logger.warning(f"Model discovery failed for {provider}: {e}")

        # 4. Return None if not found
        logger.warning(f"Model configuration not found: {provider}/{model_name}")
        return None

    def get_available_models(
        self, provider: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Get available models by provider.

        Args:
            provider: Optional provider filter

        Returns:
            Dictionary mapping provider names to lists of model names
        """
        if provider:
            return {provider: list(self._static_registry.get(provider, {}).keys())}

        return {
            prov: list(models.keys()) for prov, models in self._static_registry.items()
        }


class ProviderRegistry:
    """Registry for providers."""

    def __init__(self):
        """Initialize the provider registry."""
        self._providers: Dict[str, Type[Provider]] = {}
        self._instances: Dict[str, Provider] = {}

    def register_provider(self, provider_class: Type[Provider], config: Optional[Dict[str, Any]] = None):
        """Register a provider class with auto-discovery of provider name.

        Args:
            provider_class: Provider class
            config: Optional configuration for the provider
        """
        provider_name = provider_class.get_provider_name()
        self._providers[provider_name] = provider_class

        # Create instance if config provided
        if config:
            try:
                instance = provider_class(config)
                self._instances[provider_name] = instance
                logger.info(f"Registered and initialized provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")
                logger.info(f"Registered provider class: {provider_name}")
        else:
            logger.info(f"Registered provider class: {provider_name}")

    def register_provider_legacy(self, name: str, provider_class: Type[Provider]):
        """Legacy method for registering providers by name."""
        self._providers[name] = provider_class
        logger.info(f"Registered provider: {name}")

    def get_provider(
        self, name: str, config: Optional[Dict] = None
    ) -> Optional[Provider]:
        """Get a provider instance.

        Args:
            name: Provider name
            config: Optional configuration for the provider

        Returns:
            Provider instance or None if not found
        """
        if name not in self._providers:
            return None

        # Return cached instance if available and no new config provided
        if name in self._instances and config is None:
            return self._instances[name]

        # Create new instance
        if config is None:
            config = {}

        try:
            instance = self._providers[name](config)
            self._instances[name] = instance
            return instance
        except Exception as e:
            logger.error(f"Failed to create provider {name}: {e}")
            return None

    def get_available_providers(self) -> List[str]:
        """Get list of registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models in registry format (provider/model)."""
        result = {}
        for provider_name, provider_class in self._providers.items():
            try:
                supported_models = provider_class.get_supported_models()
                result[provider_name] = [f"{provider_name}/{model}" for model in supported_models.keys()]
            except Exception as e:
                logger.error(f"Failed to get models for provider {provider_name}: {e}")
                result[provider_name] = []
        return result

    def get_model_config_by_registry_name(self, registry_name: str) -> Optional[ModelConfig]:
        """Get model configuration by registry name (provider/model).

        Args:
            registry_name: Model name in format 'provider/model'

        Returns:
            Model configuration or None if not found
        """
        if "/" not in registry_name:
            logger.warning(f"Invalid registry name format: {registry_name}. Expected 'provider/model'")
            return None

        provider_name, model_name = registry_name.split("/", 1)

        if provider_name not in self._providers:
            logger.warning(f"Provider '{provider_name}' not found")
            return None

        try:
            provider_class = self._providers[provider_name]
            supported_models = provider_class.get_supported_models()
            return supported_models.get(model_name)
        except Exception as e:
            logger.error(f"Failed to get model config for {registry_name}: {e}")
            return None

    def is_provider_available(self, name: str) -> bool:
        """Check if a provider is registered and available.

        Args:
            name: Provider name

        Returns:
            True if provider is available, False otherwise
        """
        if name not in self._shims:
            return False

        shim = self.get_shim(name)
        return shim is not None and shim.is_available()
