"""Base provider shim interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List

from .models import (
    UnifiedRequest,
    UnifiedResponse,
    ModelConfig,
    Attachment,
    AttachmentType,
    RequestType
)


class ProviderShim(ABC):
    """Abstract base class for provider shims."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider shim.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.provider_name = self.__class__.__name__.lower().replace('shim', '')
    
    @abstractmethod
    async def process_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Process a unified request and return a unified response.
        
        Args:
            request: The unified request to process
            model_config: Configuration for the specific model to use
            
        Returns:
            Unified response containing the result
        """
        pass
    
    @abstractmethod
    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Discover available models and their capabilities from the provider.
        
        Returns:
            Dictionary mapping model names to their configurations
        """
        pass
    
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
        
        # Check if model supports the required capabilities
        if request_type == RequestType.VISION_ANALYSIS and not model_config.capabilities.vision:
            return False
        elif request_type == RequestType.IMAGE_GENERATION and not model_config.capabilities.image_generation:
            return False
        elif request_type == RequestType.IMAGE_EDIT and not model_config.capabilities.image_editing:
            return False
        
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
        return (
            self.config.get('enabled', False) and
            self.config.get('api_key') is not None
        )
