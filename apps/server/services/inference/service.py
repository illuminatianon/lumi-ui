"""Unified inference service implementation."""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from .models import (
    UnifiedRequest,
    UnifiedResponse,
    ModelConfig,
    InferenceConfig,
    RequestType,
    Attachment,
    AttachmentType
)
from .base import Provider
from .registry import ModelRegistry, ProviderRegistry
from .resolver import ModelResolver
from .providers import OpenAIProvider, GoogleProvider

logger = logging.getLogger(__name__)


class UnifiedInferenceService:
    """Main unified inference service that orchestrates provider selection and request processing."""
    
    def __init__(self, config: InferenceConfig):
        """Initialize the unified inference service.
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.model_registry = ModelRegistry()
        self.provider_registry = ProviderRegistry()
        self.model_resolver = ModelResolver(self.provider_registry)
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize and register providers."""
        # Auto-register provider classes
        self.provider_registry.register_provider(OpenAIProvider)
        self.provider_registry.register_provider(GoogleProvider)

        # Initialize provider instances with configuration
        for provider_name, provider_config in self.config.providers.items():
            if provider_config.enabled:
                provider = self.provider_registry.get_provider(provider_name, provider_config.model_dump())
                if provider and provider.is_available():
                    logger.info(f"Provider {provider_name} initialized successfully")
                else:
                    logger.warning(f"Provider {provider_name} failed to initialize")
    
    async def process_request(self, request: UnifiedRequest) -> UnifiedResponse:
        """Single entry point for all inference requests.
        
        Args:
            request: The unified request to process
            
        Returns:
            Unified response containing the result
        """
        # 1. Analyze request to determine intent
        request_type = self._analyze_request(request)
        
        # 2. Select appropriate model based on request type and attachments
        model_config = await self._select_model(request, request_type)
        
        # 3. Get provider
        provider = self.provider_registry.get_provider(model_config.provider)
        if not provider:
            raise ValueError(f"Provider {model_config.provider} not available")
        
        # 4. Validate request compatibility
        if not provider.validate_request(request, model_config):
            raise ValueError(f"Request not compatible with model {model_config.name}")

        # 5. Process request
        try:
            return await provider.process_request(model_config.name, request)
        except Exception as e:
            # Try fallback providers if configured
            if self.config.fallback_providers:
                return await self._try_fallback_providers(request, request_type, e)
            raise
    
    def _analyze_request(self, request: UnifiedRequest) -> RequestType:
        """Analyze request to determine what the user wants."""
        has_images = any(att.attachment_type == AttachmentType.IMAGE for att in request.attachments)
        has_documents = any(att.attachment_type in [AttachmentType.PDF, AttachmentType.TEXT, AttachmentType.DOCUMENT]
                           for att in request.attachments)

        if request.response_format == "image":
            if has_images:
                return RequestType.IMAGE_EDIT
            else:
                return RequestType.IMAGE_GENERATION
        elif has_images:
            return RequestType.VISION_ANALYSIS
        elif has_documents:
            return RequestType.DOCUMENT_ANALYSIS
        else:
            return RequestType.TEXT_GENERATION
    
    async def _select_model(self, request: UnifiedRequest, request_type: RequestType) -> ModelConfig:
        """Select best model for the request type."""
        if request.model != "auto":
            # User specified a model in provider/model format
            if "/" in request.model:
                provider_name, model_name = request.model.split("/", 1)
                if provider_name in self.config.providers:
                    provider = self.provider_registry.get_provider(provider_name)
                    if provider:
                        # Get model config directly from the provider
                        supported_models = provider.get_supported_models()
                        if model_name in supported_models:
                            return supported_models[model_name]
                        else:
                            available = list(supported_models.keys())
                            raise ValueError(f"Model {model_name} not found in provider {provider_name}. Available: {available}")
                    else:
                        raise ValueError(f"Provider {provider_name} not available")
                else:
                    raise ValueError(f"Provider {provider_name} not found")
            else:
                # Legacy support: try to find model in any provider (for backward compatibility)
                for provider_name in self.config.providers.keys():
                    provider = self.provider_registry.get_provider(provider_name)
                    if provider:
                        supported_models = provider.get_supported_models()
                        if request.model in supported_models:
                            return supported_models[request.model]
                raise ValueError(f"Model {request.model} not found")
        
        # Auto-select based on request type and capabilities
        if request_type == RequestType.VISION_ANALYSIS:
            return await self._get_best_vision_model()
        elif request_type == RequestType.IMAGE_GENERATION:
            return await self._get_best_image_gen_model()
        elif request_type == RequestType.IMAGE_EDIT:
            return await self._get_best_image_edit_model()
        else:
            return await self._get_best_text_model()
    
    async def _get_best_vision_model(self) -> ModelConfig:
        """Get the best available vision model."""
        candidates = []
        
        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue
                
            for model_name, model_config in provider_config.models.items():
                if model_config.capabilities.vision:
                    candidates.append(model_config)
        
        if not candidates:
            raise ValueError("No vision models available")
        
        # Select based on strategy (for now, just pick the first available)
        return candidates[0]
    
    async def _get_best_image_gen_model(self) -> ModelConfig:
        """Get the best available image generation model."""
        candidates = []
        
        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue
                
            for model_name, model_config in provider_config.models.items():
                if model_config.capabilities.image_generation:
                    candidates.append(model_config)
        
        if not candidates:
            raise ValueError("No image generation models available")
        
        return candidates[0]
    
    async def _get_best_image_edit_model(self) -> ModelConfig:
        """Get the best available image editing model."""
        # For now, fallback to image generation
        return await self._get_best_image_gen_model()
    
    async def _get_best_text_model(self) -> ModelConfig:
        """Get the best available text model."""
        candidates = []
        
        for provider_name, provider_config in self.config.providers.items():
            if not provider_config.enabled:
                continue
                
            for model_name, model_config in provider_config.models.items():
                if model_config.capabilities.text_generation:
                    candidates.append(model_config)
        
        if not candidates:
            raise ValueError("No text generation models available")
        
        # Select based on strategy (for now, just pick the first available)
        return candidates[0]

    async def _try_fallback_providers(self, request: UnifiedRequest, request_type: RequestType, original_error: Exception) -> UnifiedResponse:
        """Try fallback providers when primary provider fails."""
        for provider_name in self.config.fallback_providers:
            try:
                model_config = await self._select_model_for_provider(request, request_type, provider_name)
                if model_config:
                    provider = self.provider_registry.get_provider(provider_name)
                    if provider and provider.validate_request(request, model_config):
                        logger.info(f"Trying fallback provider: {provider_name}")
                        return await provider.process_request(model_config.name, request)
            except Exception as e:
                logger.warning(f"Fallback provider {provider_name} failed: {e}")
                continue

        # If all fallbacks failed, raise the original error
        raise original_error

    async def _select_model_for_provider(self, request: UnifiedRequest, request_type: RequestType, provider_name: str) -> Optional[ModelConfig]:
        """Select a model from a specific provider for the request type."""
        provider_config = self.config.providers.get(provider_name)
        if not provider_config or not provider_config.enabled:
            return None

        for model_name, model_config in provider_config.models.items():
            if self._model_supports_request_type(model_config, request_type):
                return model_config

        return None

    def _model_supports_request_type(self, model_config: ModelConfig, request_type: RequestType) -> bool:
        """Check if a model supports the given request type."""
        if request_type == RequestType.VISION_ANALYSIS:
            return model_config.capabilities.vision
        elif request_type == RequestType.IMAGE_GENERATION:
            return model_config.capabilities.image_generation
        elif request_type == RequestType.IMAGE_EDIT:
            return model_config.capabilities.image_editing
        elif request_type == RequestType.TEXT_GENERATION:
            return model_config.capabilities.text_generation
        else:
            return False

    async def process_registry_request(self, config: Dict[str, Any]) -> UnifiedResponse:
        """Process request using registry-style model naming (provider/model).

        Args:
            config: Request configuration with registry-style model name

        Returns:
            Unified response from the provider

        Raises:
            ValueError: If model or provider not found
        """
        # Resolve provider and model
        provider, model_name, validated_params = await self.model_resolver.resolve_request(config)

        # Build unified request
        request = self.model_resolver.build_unified_request(config)

        # Process with provider
        try:
            response = await provider.process_request(model_name, request)
            logger.info(f"Registry request processed successfully: {config.get('model')}")
            return response
        except Exception as e:
            logger.error(f"Registry request failed for {config.get('model')}: {e}")
            raise

    def list_available_models(self) -> Dict[str, Any]:
        """List all available models in registry format.

        Returns:
            Dictionary with available models and metadata
        """
        return self.model_resolver.list_available_models()

    # Convenience methods for common use cases
    async def chat(self, message: str, attachments: List[Attachment] = None, **kwargs) -> str:
        """Simple chat interface - returns just the text response.

        Args:
            message: The message to send
            attachments: Optional list of attachments
            **kwargs: Additional parameters

        Returns:
            Text response from the model
        """
        request = UnifiedRequest(
            prompt=message,
            attachments=attachments or [],
            response_format="text",
            **kwargs
        )
        response = await self.process_request(request)
        return response.content or ""

    async def analyze_image(self, image: Attachment, question: str, **kwargs) -> str:
        """Analyze an image with a question.

        Args:
            image: Image attachment to analyze
            question: Question about the image
            **kwargs: Additional parameters

        Returns:
            Analysis result as text
        """
        request = UnifiedRequest(
            prompt=question,
            attachments=[image],
            response_format="text",
            **kwargs
        )
        response = await self.process_request(request)
        return response.content or ""

    async def generate_image(self, prompt: str, reference_image: Attachment = None, **kwargs) -> List[str]:
        """Generate images, optionally with a reference.

        Args:
            prompt: Description of the image to generate
            reference_image: Optional reference image
            **kwargs: Additional parameters

        Returns:
            List of generated image URLs or base64 strings
        """
        request = UnifiedRequest(
            prompt=prompt,
            attachments=[reference_image] if reference_image else [],
            response_format="image",
            **kwargs
        )
        response = await self.process_request(request)
        return response.images

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models by provider.

        Returns:
            Dictionary mapping provider names to lists of model names
        """
        return self.model_registry.get_available_models()

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all providers.

        Returns:
            Dictionary mapping provider names to availability status
        """
        status = {}
        for provider_name in self.provider_registry.get_available_providers():
            status[provider_name] = self.provider_registry.is_provider_available(provider_name)
        return status
