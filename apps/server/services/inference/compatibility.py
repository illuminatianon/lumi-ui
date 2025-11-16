"""Compatibility layer for unified inference service."""

import logging
from typing import Dict, Any, Optional, List
import base64

from .factory import get_unified_inference_service
from .models import UnifiedRequest, Attachment, AttachmentType
from ..langchain_service import LangChainService

logger = logging.getLogger(__name__)


class InferenceCompatibilityService:
    """Compatibility wrapper that can use either LangChain or Unified inference."""
    
    def __init__(self):
        """Initialize the compatibility service."""
        self.unified_service = get_unified_inference_service()
        self.langchain_service = LangChainService()
        
        # Determine which service to use
        self.use_unified = self.unified_service is not None
        
        if self.use_unified:
            logger.info("Using unified inference service")
        else:
            logger.info("Using LangChain service (unified inference disabled)")
    
    async def generate_text(
        self, 
        prompt: str, 
        model: str = "auto",
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text using the appropriate service.
        
        Args:
            prompt: The user prompt
            model: Model to use
            system_message: Optional system message
            temperature: Optional temperature override
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with text, model_used, success, etc.
        """
        if self.use_unified:
            return await self._generate_text_unified(prompt, model, system_message, temperature, **kwargs)
        else:
            return await self.langchain_service.generate_text(prompt, model, system_message, temperature)
    
    async def process_image_with_text(
        self,
        image_data: str,
        prompt: str,
        model: str = "auto",
        image_format: str = "base64",
        **kwargs
    ) -> Dict[str, Any]:
        """Process image with text using the appropriate service.
        
        Args:
            image_data: Base64 encoded image data
            prompt: Text prompt about the image
            model: Model to use
            image_format: Format of image data (always base64 for now)
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with text, model_used, success, etc.
        """
        if self.use_unified:
            return await self._process_image_unified(image_data, prompt, model, **kwargs)
        else:
            return await self.langchain_service.process_image_with_text(image_data, prompt, model, image_format)
    
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using the appropriate service.
        
        Args:
            prompt: Description of the image to generate
            size: Image size
            quality: Image quality
            n: Number of images
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with images, model_used, success, etc.
        """
        if self.use_unified:
            return await self._generate_image_unified(prompt, size, quality, n, **kwargs)
        else:
            return await self.langchain_service.generate_image(prompt, size, quality, n)
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get status of available AI models."""
        if self.use_unified:
            # Convert unified format to LangChain format for compatibility
            models = self.unified_service.get_available_models()
            provider_status = self.unified_service.get_provider_status()
            
            return {
                "openai_chat": "openai" in provider_status and provider_status["openai"],
                "openai_dalle": "openai" in provider_status and provider_status["openai"],
                "google_chat": "google" in provider_status and provider_status["google"],
                "google_image": False  # Google doesn't have image generation yet
            }
        else:
            return self.langchain_service.get_available_models()
    
    async def _generate_text_unified(self, prompt: str, model: str, system_message: Optional[str], temperature: Optional[float], **kwargs) -> Dict[str, Any]:
        """Generate text using unified service."""
        try:
            request = UnifiedRequest(
                prompt=prompt,
                system_message=system_message,
                model=model,
                temperature=temperature,
                **kwargs
            )
            
            response = await self.unified_service.process_request(request)
            
            return {
                "text": response.content,
                "model_used": response.model_used,
                "success": True,
                "provider": response.provider,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Unified text generation failed: {e}")
            return {
                "text": "",
                "model_used": model,
                "success": False,
                "error": str(e)
            }
    
    async def _process_image_unified(self, image_data: str, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """Process image using unified service."""
        try:
            # Create attachment from base64 data
            attachment = Attachment.from_base64(image_data, "image/jpeg")  # Assume JPEG for now
            
            request = UnifiedRequest(
                prompt=prompt,
                attachments=[attachment],
                model=model,
                **kwargs
            )
            
            response = await self.unified_service.process_request(request)
            
            return {
                "text": response.content,
                "model_used": response.model_used,
                "success": True,
                "provider": response.provider,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            logger.error(f"Unified image processing failed: {e}")
            return {
                "text": "",
                "model_used": model,
                "success": False,
                "error": str(e)
            }
    
    async def _generate_image_unified(self, prompt: str, size: str, quality: str, n: int, **kwargs) -> Dict[str, Any]:
        """Generate image using unified service."""
        try:
            request = UnifiedRequest(
                prompt=prompt,
                response_format="image",
                extras={"size": size, "quality": quality, "n": n},
                **kwargs
            )
            
            response = await self.unified_service.process_request(request)
            
            # Convert to LangChain format
            images = [{"url": url} for url in response.images]
            
            return {
                "images": images,
                "model_used": response.model_used,
                "success": True,
                "provider": response.provider
            }
        except Exception as e:
            logger.error(f"Unified image generation failed: {e}")
            return {
                "images": [],
                "model_used": "dall-e-3",
                "success": False,
                "error": str(e)
            }
