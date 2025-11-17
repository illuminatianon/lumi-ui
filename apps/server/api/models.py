"""API models for AI endpoints."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TextGenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The user prompt")
    model: str = Field(..., description="Model to use in 'provider/model' format (e.g., 'openai/gpt-4o', 'google/gemini-1.5-pro')")
    system_message: Optional[str] = Field(None, description="Optional system message")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(None, description="Top-p for generation")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")
    attachments: Optional[List[str]] = Field(None, description="List of attachment file paths or base64 data")

    @field_validator('model')
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """Validate model name is in provider/model format."""
        if "/" not in v:
            raise ValueError("Model name must be in 'provider/model' format (e.g., 'openai/gpt-4o')")
        parts = v.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Invalid model format. Use 'provider/model' (e.g., 'openai/gpt-4o')")
        return v


class TextGenerationResponse(BaseModel):
    """Response model for text generation."""
    text: str
    model_used: str
    success: bool
    error: Optional[str] = None


class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis."""
    image_data: str = Field(..., description="Base64 encoded image data")
    prompt: str = Field(..., description="Question or instruction about the image")
    model: str = Field(..., description="Model to use in 'provider/model' format (e.g., 'openai/gpt-4o', 'google/gemini-1.5-pro')")

    @field_validator('model')
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """Validate model name is in provider/model format."""
        if "/" not in v:
            raise ValueError("Model name must be in 'provider/model' format (e.g., 'openai/gpt-4o')")
        parts = v.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Invalid model format. Use 'provider/model' (e.g., 'openai/gpt-4o')")
        return v


class ImageAnalysisResponse(BaseModel):
    """Response model for image analysis."""
    text: str
    model_used: str
    success: bool
    error: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., description="Description of the image to generate")
    model: str = Field(..., description="Model to use in 'provider/model' format (e.g., 'openai/dall-e-3', 'google/gemini-2.5-flash-image')")

    @field_validator('model')
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """Validate model name is in provider/model format."""
        if "/" not in v:
            raise ValueError("Model name must be in 'provider/model' format (e.g., 'openai/dall-e-3')")
        parts = v.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Invalid model format. Use 'provider/model' (e.g., 'openai/dall-e-3')")
        return v

    # Size and aspect ratio parameters
    size: Optional[str] = Field(None, description="Image size (e.g., 1024x1024, 1536x1024)")
    aspect_ratio: Optional[str] = Field(None, description="Aspect ratio (e.g., 1:1, 16:9, 9:16)")

    # Quality and style parameters
    quality: Optional[str] = Field("standard", description="Image quality (standard, hd)")
    style: Optional[str] = Field(None, description="Image style (vivid, natural)")

    # Advanced parameters
    response_modalities: Optional[List[str]] = Field(None, description="Response types (Text, Image)")
    reference_images: Optional[List[str]] = Field(None, description="Base64 encoded reference images for editing/style transfer")

    # Legacy parameter for backward compatibility
    n: int = Field(default=1, ge=1, le=1, description="Number of images (most models support only 1)")


class GeneratedImage(BaseModel):
    """Model for a generated image."""
    url: str
    revised_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ImageGenerationResponse(BaseModel):
    """Response model for image generation."""
    images: List[GeneratedImage]
    model_used: str
    success: bool
    error: Optional[str] = None

    # Additional metadata
    generation_metadata: Optional[Dict[str, Any]] = Field(None, description="Model-specific generation metadata")
    text_content: Optional[str] = Field(None, description="Text content generated along with images (for models that support it)")


class ModelStatusResponse(BaseModel):
    """Response model for model availability status."""
    openai_chat: bool
    openai_dalle: bool
    openai_gpt_image: bool
    google_chat: bool
    google_image: bool

    # Available image generation models
    available_image_models: List[str] = Field(default_factory=list)

    # Model capabilities
    model_capabilities: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Detailed capabilities for each model")


class RegistryRequest(BaseModel):
    """Generic request model for registry-style model requests."""
    model: str = Field(..., description="Model to use in 'provider/model' format")
    prompt: str = Field(..., description="The prompt or instruction")
    system_message: Optional[str] = Field(None, description="Optional system message")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(None, description="Top-p for generation")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")
    attachments: Optional[List[str]] = Field(None, description="List of attachment file paths or base64 data")

    # Image generation specific parameters
    size: Optional[str] = Field(None, description="Image size (e.g., 1024x1024)")
    aspect_ratio: Optional[str] = Field(None, description="Aspect ratio (e.g., 1:1, 16:9)")
    quality: Optional[str] = Field(None, description="Image quality (standard, hd)")
    style: Optional[str] = Field(None, description="Image style (vivid, natural)")

    # GPT-5 specific parameters
    reasoning_effort: Optional[str] = Field(None, description="Reasoning effort for GPT-5 (low, medium, high)")

    @field_validator('model')
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """Validate model name is in provider/model format."""
        if "/" not in v:
            raise ValueError("Model name must be in 'provider/model' format (e.g., 'openai/gpt-4o')")
        parts = v.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("Invalid model format. Use 'provider/model' (e.g., 'openai/gpt-4o')")
        return v


class RegistryResponse(BaseModel):
    """Generic response model for registry-style requests."""
    content: str = Field(..., description="Generated content (text or image data)")
    model_used: str = Field(..., description="Full model name that was used")
    provider: str = Field(..., description="Provider that handled the request")
    success: bool = Field(..., description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if request failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the response")


class ModelListResponse(BaseModel):
    """Response model for listing available models."""
    models: List[Dict[str, Any]] = Field(..., description="List of available models with metadata")
    providers: Dict[str, List[str]] = Field(..., description="Models grouped by provider")
    total_models: int = Field(..., description="Total number of available models")
