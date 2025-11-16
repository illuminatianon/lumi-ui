"""Unified inference infrastructure package."""

from .models import (
    UnifiedRequest,
    UnifiedResponse,
    Attachment,
    AttachmentType,
    TokenUsage,
    ModelConfig,
    ParameterMapping,
    ProviderConfig,
    InferenceConfig,
    ModelCapabilities,
    RequestType
)

from .base import ProviderShim
from .service import UnifiedInferenceService
from .registry import ModelRegistry, ProviderRegistry

__all__ = [
    # Models
    "UnifiedRequest",
    "UnifiedResponse", 
    "Attachment",
    "AttachmentType",
    "TokenUsage",
    "ModelConfig",
    "ParameterMapping",
    "ProviderConfig",
    "InferenceConfig",
    "ModelCapabilities",
    "RequestType",
    
    # Base classes
    "ProviderShim",
    
    # Services
    "UnifiedInferenceService",
    
    # Registries
    "ModelRegistry",
    "ProviderRegistry",
]
