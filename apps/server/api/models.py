"""API models for AI endpoints."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class TextGenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="The user prompt")
    model: str = Field(default="auto", description="Model to use (openai, google, auto)")
    system_message: Optional[str] = Field(None, description="Optional system message")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature for generation")


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
    model: str = Field(default="auto", description="Model to use (openai, google, auto)")


class ImageAnalysisResponse(BaseModel):
    """Response model for image analysis."""
    text: str
    model_used: str
    success: bool
    error: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""
    prompt: str = Field(..., description="Description of the image to generate")
    model: str = Field(default="auto", description="Model to use (openai, google, auto)")
    size: str = Field(default="1024x1024", description="Image size (for DALL-E)")
    quality: str = Field(default="standard", description="Image quality (for DALL-E: standard, hd)")
    n: int = Field(default=1, ge=1, le=1, description="Number of images (DALL-E 3 supports only 1)")


class GeneratedImage(BaseModel):
    """Model for a generated image."""
    url: str
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """Response model for image generation."""
    images: List[GeneratedImage]
    model_used: str
    success: bool
    error: Optional[str] = None


class ModelStatusResponse(BaseModel):
    """Response model for model availability status."""
    openai_chat: bool
    openai_dalle: bool
    google_chat: bool
    google_image: bool
