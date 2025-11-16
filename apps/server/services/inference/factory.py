"""Factory for creating unified inference service instances."""

import logging
from typing import Optional

from config import get_config
from .service import UnifiedInferenceService
from .models import InferenceConfig, ProviderConfig, ModelConfig, ModelCapabilities, ParameterMapping

logger = logging.getLogger(__name__)

# Global service instance
_unified_service: Optional[UnifiedInferenceService] = None


def get_unified_inference_service() -> Optional[UnifiedInferenceService]:
    """Get the global unified inference service instance.
    
    Returns:
        Unified inference service instance or None if not enabled/configured
    """
    global _unified_service
    
    if _unified_service is None:
        _unified_service = create_unified_inference_service()
    
    return _unified_service


def create_unified_inference_service() -> Optional[UnifiedInferenceService]:
    """Create a new unified inference service instance from configuration.
    
    Returns:
        Unified inference service instance or None if not enabled/configured
    """
    try:
        config = get_config()
        
        # Check if unified inference is enabled
        if not config.inference.enabled:
            logger.info("Unified inference is disabled in configuration")
            return None
        
        # Build inference configuration from app config
        inference_config = _build_inference_config(config)
        
        # Create and return service
        service = UnifiedInferenceService(inference_config)
        logger.info("Unified inference service created successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to create unified inference service: {e}")
        return None


def _build_inference_config(app_config) -> InferenceConfig:
    """Build InferenceConfig from application configuration."""
    providers = {}
    
    # Configure OpenAI provider
    if app_config.inference.openai.enabled and app_config.api_keys.openai:
        providers["openai"] = ProviderConfig(
            enabled=True,
            api_key=app_config.api_keys.openai,
            base_url=app_config.inference.openai.base_url,
            models=_get_openai_models(),
            default_model=app_config.inference.openai.default_model,
            rate_limits={"default": app_config.inference.openai.rate_limit_rpm} if app_config.inference.openai.rate_limit_rpm else {}
        )
    
    # Configure Google provider
    if app_config.inference.google.enabled and app_config.api_keys.gemini:
        providers["google"] = ProviderConfig(
            enabled=True,
            api_key=app_config.api_keys.gemini,
            base_url=app_config.inference.google.base_url,
            models=_get_google_models(),
            default_model=app_config.inference.google.default_model,
            rate_limits={"default": app_config.inference.google.rate_limit_rpm} if app_config.inference.google.rate_limit_rpm else {}
        )
    
    return InferenceConfig(
        providers=providers,
        default_provider=app_config.inference.default_provider,
        fallback_providers=app_config.inference.fallback_providers,
        provider_selection_strategy=app_config.inference.provider_selection_strategy
    )


def _get_openai_models() -> dict:
    """Get OpenAI model configurations."""
    return {
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
                max_context_length=128000
            ),
            parameter_mapping=ParameterMapping(
                max_tokens_param="max_tokens",
                temperature_param="temperature",
                top_p_param="top_p"
            ),
            supported_parameters={"max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop", "stream"},
            cost_per_1k_tokens=0.005,
            context_window=128000
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
                max_context_length=200000
            ),
            parameter_mapping=ParameterMapping(
                max_tokens_param="max_completion_tokens",
                temperature_param=None,
                top_p_param=None,
                custom_params={"reasoning_effort": "medium"}
            ),
            supported_parameters={"max_completion_tokens", "reasoning_effort", "stream"},
            cost_per_1k_tokens=0.01,
            context_window=200000
        ),
        "dall-e-3": ModelConfig(
            name="dall-e-3",
            display_name="DALL-E 3",
            provider="openai",
            capabilities=ModelCapabilities(
                image_generation=True,
                text_generation=False
            ),
            parameter_mapping=ParameterMapping(),
            supported_parameters={"size", "quality", "style", "n"},
            cost_per_1k_tokens=0.04,
            context_window=4000
        )
    }


def _get_google_models() -> dict:
    """Get Google model configurations."""
    return {
        "gemini-1.5-pro": ModelConfig(
            name="gemini-1.5-pro",
            display_name="Gemini 1.5 Pro",
            provider="google",
            capabilities=ModelCapabilities(
                text_generation=True,
                vision=True,
                function_calling=True,
                streaming=True,
                max_context_length=2000000
            ),
            parameter_mapping=ParameterMapping(
                max_tokens_param="max_output_tokens",
                temperature_param="temperature",
                top_p_param="top_p"
            ),
            supported_parameters={"max_output_tokens", "temperature", "top_p", "stop_sequences"},
            cost_per_1k_tokens=0.0035,
            context_window=2000000
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
                max_context_length=1000000
            ),
            parameter_mapping=ParameterMapping(
                max_tokens_param="max_output_tokens",
                temperature_param="temperature",
                top_p_param="top_p"
            ),
            supported_parameters={"max_output_tokens", "temperature", "top_p", "stop_sequences"},
            cost_per_1k_tokens=0.00075,
            context_window=1000000
        )
    }
