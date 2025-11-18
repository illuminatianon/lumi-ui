"""Base provider interface."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from .models import (
    UnifiedRequest,
    UnifiedResponse,
    ModelConfig,
    Attachment,
    AttachmentType,
    RequestType
)

logger = logging.getLogger(__name__)


class Provider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider.

        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.provider_name = self.get_provider_name()

    @classmethod
    @abstractmethod
    def get_provider_name(cls) -> str:
        """Return the provider name for registry.

        Returns:
            Provider name (e.g., 'openai', 'google', 'anthropic')
        """
        pass

    @classmethod
    @abstractmethod
    def get_supported_models(cls) -> Dict[str, ModelConfig]:
        """Return all models supported by this provider.

        Returns:
            Dictionary mapping model names to their configurations
        """
        pass

    @abstractmethod
    async def process_request(self, model_name: str, request: UnifiedRequest) -> UnifiedResponse:
        """Process a unified request for a specific model.

        Args:
            model_name: Name of the model to use (without provider prefix)
            request: The unified request to process

        Returns:
            Unified response containing the result
        """
        pass

    def validate_model_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and map parameters for the specific model.

        Args:
            model_name: Name of the model
            params: Parameters to validate and map

        Returns:
            Validated and mapped parameters
        """
        supported_models = self.get_supported_models()
        if model_name not in supported_models:
            available = list(supported_models.keys())
            raise ValueError(f"Model '{model_name}' not supported by {self.provider_name}. Available: {available}")

        model_config = supported_models[model_name]
        return self.map_parameters(params, model_config)

    @abstractmethod
    def map_parameters(self, unified_params: Dict[str, Any], model_config: ModelConfig) -> Dict[str, Any]:
        """Map unified parameters to model-specific parameters.

        Args:
            unified_params: Parameters in unified format
            model_config: Configuration for the target model

        Returns:
            Parameters mapped to provider-specific format
        """
        pass

    # Legacy methods for backward compatibility during transition
    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Legacy method - use get_supported_models() instead."""
        return self.get_supported_models()

    async def process_request_legacy(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Legacy method - use process_request(model_name, request) instead."""
        return await self.process_request(model_config.name, request)
    
    @abstractmethod
    def prepare_attachments(self, attachments: List[Attachment], model_config: ModelConfig) -> Any:
        """Convert attachments to provider-specific format.
        
        Args:
            attachments: List of unified attachments
            model_config: Configuration for the target model
            
        Returns:
            Attachments in provider-specific format
        """
        pass
    
    def determine_request_type(self, request: UnifiedRequest, model_config: ModelConfig) -> RequestType:
        """Determine what type of request this is based on attachments and response format.
        
        Args:
            request: The unified request
            model_config: Configuration for the target model
            
        Returns:
            The type of request
        """
        has_images = any(att.attachment_type == AttachmentType.IMAGE for att in request.attachments)
        has_documents = any(att.attachment_type in [AttachmentType.PDF, AttachmentType.TEXT, AttachmentType.DOCUMENT]
                           for att in request.attachments)

        if request.response_format == "image":
            if has_images:
                return RequestType.IMAGE_EDIT  # Image-to-image
            else:
                return RequestType.IMAGE_GENERATION  # Text-to-image
        elif has_images:
            return RequestType.VISION_ANALYSIS  # Image analysis/vision
        elif has_documents:
            return RequestType.DOCUMENT_ANALYSIS  # Document analysis
        else:
            return RequestType.TEXT_GENERATION  # Pure text generation
    
    def validate_request(self, request: UnifiedRequest, model_config: ModelConfig) -> bool:
        """Validate that the request is compatible with the model.

        Args:
            request: The unified request to validate
            model_config: Configuration for the target model

        Returns:
            True if request is valid for this model, False otherwise
        """
        request_type = self.determine_request_type(request, model_config)

        # Debug logging
        logger.debug(f"Validating request for {model_config.name}")
        logger.debug(f"Request type: {request_type}")
        logger.debug(f"Model capabilities: {model_config.capabilities}")

        # Check if model supports the required capabilities
        if request_type == RequestType.VISION_ANALYSIS and not model_config.capabilities.vision:
            logger.warning(f"Model {model_config.name} does not support vision analysis")
            return False
        elif request_type == RequestType.IMAGE_GENERATION and not model_config.capabilities.image_generation:
            logger.warning(f"Model {model_config.name} does not support image generation. Capabilities: {model_config.capabilities}")
            return False
        elif request_type == RequestType.IMAGE_EDIT:
            # Check image editing capability
            if not model_config.capabilities.image_generation:
                logger.warning(f"Model {model_config.name} does not support image generation (required for image editing)")
                return False
            if not model_config.capabilities.image_editing:
                logger.warning(f"Model {model_config.name} does not support image editing")
                return False

        logger.debug(f"Request validation passed for {model_config.name}")
        return True
    
    def get_supported_models(self) -> List[str]:
        """Get list of model names supported by this provider.
        
        Returns:
            List of supported model names
        """
        return list(self.config.get('models', {}).keys())
    
    def is_available(self) -> bool:
        """Check if this provider is available and properly configured.

        Returns:
            True if provider is available, False otherwise
        """
        return self.config.get('api_key') is not None
