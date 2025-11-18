"""Unified inference models and types."""

import base64
import mimetypes
from enum import Enum
from typing import Optional, List, Dict, Any, Literal, Set
from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class AttachmentType(str, Enum):
    """Types of attachments supported by the inference system."""
    IMAGE = "image"
    TEXT = "text"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class RequestType(str, Enum):
    """Types of inference requests."""
    TEXT_GENERATION = "text"
    VISION_ANALYSIS = "vision"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    DOCUMENT_ANALYSIS = "document"


class MessageRole(str, Enum):
    """Roles for chat messages in multi-turn conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single message in a multi-turn conversation."""
    role: MessageRole
    content: str

    class Config:
        use_enum_values = True


class Attachment(BaseModel):
    """Unified attachment that works across all providers."""
    content: bytes
    mime_type: str
    filename: Optional[str] = None
    attachment_type: AttachmentType
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_file(cls, path: str) -> 'Attachment':
        """Create attachment from file path."""
        file_path = Path(path)
        content = file_path.read_bytes()
        mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"

        return cls(
            content=content,
            mime_type=mime_type,
            filename=file_path.name,
            attachment_type=cls._detect_type(mime_type),
            metadata={"file_size": len(content), "source": "file"}
        )

    @classmethod
    def from_base64(cls, data: str, mime_type: str, filename: Optional[str] = None) -> 'Attachment':
        """Create attachment from base64 data."""
        content = base64.b64decode(data)
        return cls(
            content=content,
            mime_type=mime_type,
            filename=filename,
            attachment_type=cls._detect_type(mime_type),
            metadata={"source": "base64"}
        )

    def to_base64(self) -> str:
        """Convert to base64 string."""
        return base64.b64encode(self.content).decode('utf-8')

    def to_text(self) -> str:
        """Extract text content."""
        if self.attachment_type == AttachmentType.TEXT:
            return self.content.decode('utf-8')
        else:
            raise ValueError(f"Cannot extract text from {self.attachment_type}")

    @classmethod
    def _detect_type(cls, mime_type: str) -> AttachmentType:
        """Detect attachment type from MIME type."""
        if mime_type.startswith("image/"):
            return AttachmentType.IMAGE
        elif mime_type.startswith("text/"):
            return AttachmentType.TEXT
        elif mime_type == "application/pdf":
            return AttachmentType.PDF
        elif mime_type.startswith("audio/"):
            return AttachmentType.AUDIO
        elif mime_type.startswith("video/"):
            return AttachmentType.VIDEO
        else:
            return AttachmentType.UNKNOWN


class TokenUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, usage) -> 'TokenUsage':
        """Create from OpenAI usage object."""
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )


class UnifiedRequest(BaseModel):
    """Unified request that works for text, vision, and generation tasks.

    Supports both single-turn (prompt + system_message) and multi-turn (messages) formats.
    Either prompt or messages must be provided, but not both.
    """
    # Single-turn format (backward compatible)
    prompt: Optional[str] = None
    system_message: Optional[str] = None

    # Multi-turn format (new)
    messages: Optional[List[Message]] = None

    # Common fields
    attachments: List[Attachment] = Field(default_factory=list)

    # Standard parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None

    # Reasoning parameters (for thinking models)
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"

    # Response format control
    response_format: Literal["text", "image", "json"] = "text"
    stream: bool = False

    # Model selection
    model: str = "auto"

    # Provider-specific parameters
    extras: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_prompt_or_messages(self):
        """Ensure either prompt or messages is provided, but not both."""
        has_prompt = self.prompt is not None
        has_messages = self.messages is not None and len(self.messages) > 0

        if not has_prompt and not has_messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")

        if has_prompt and has_messages:
            raise ValueError("Cannot provide both 'prompt' and 'messages'. Use one or the other.")

        return self

    def is_multi_turn(self) -> bool:
        """Check if this is a multi-turn conversation request."""
        return self.messages is not None and len(self.messages) > 0

    class Config:
        arbitrary_types_allowed = True


class UnifiedResponse(BaseModel):
    """Unified response that can contain text, images, or both."""
    content: Optional[str] = None  # Text content
    images: List[str] = Field(default_factory=list)  # Base64 encoded images or URLs

    # Metadata
    model_used: str
    provider: str
    usage: TokenUsage
    finish_reason: str
    attachments_processed: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParameterMapping(BaseModel):
    """Maps unified parameter names to model-specific parameter names."""
    max_tokens_param: str = "max_tokens"  # Could be "max_completion_tokens" for GPT-5
    temperature_param: Optional[str] = "temperature"  # None if not supported
    top_p_param: Optional[str] = "top_p"
    # Model-specific parameters
    custom_params: Dict[str, Any] = Field(default_factory=dict)


class ModelCapabilities(BaseModel):
    """Model capabilities and features."""
    text_generation: bool = True
    vision: bool = False
    image_generation: bool = False
    image_editing: bool = False
    function_calling: bool = False
    streaming: bool = False
    json_mode: bool = False
    reasoning: bool = False
    max_context_length: Optional[int] = None


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str
    display_name: str
    provider: str
    capabilities: ModelCapabilities
    parameter_mapping: ParameterMapping
    supported_parameters: Set[str]
    cost_per_1k_tokens: Optional[float] = None
    context_window: Optional[int] = None


class ProviderConfig(BaseModel):
    """Configuration for a provider."""
    api_key: str
    base_url: Optional[str] = None
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    default_model: str
    rate_limits: Dict[str, int] = Field(default_factory=dict)  # requests per minute by model
    custom_headers: Dict[str, str] = Field(default_factory=dict)


class InferenceConfig(BaseModel):
    """Main inference configuration."""
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    default_provider: str = "auto"
    fallback_providers: List[str] = Field(default_factory=list)
    provider_selection_strategy: str = "cost_optimized"  # or "performance", "reliability"
