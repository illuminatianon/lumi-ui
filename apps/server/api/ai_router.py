"""AI API endpoints using LangChain integrations."""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import base64

from services.langchain_service import LangChainService
from .models import (
    TextGenerationRequest, TextGenerationResponse,
    ImageAnalysisRequest, ImageAnalysisResponse,
    ImageGenerationRequest, ImageGenerationResponse,
    ModelStatusResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai"])

# Global service instance (lazy initialization)
langchain_service = None


def get_langchain_service() -> LangChainService:
    """Get or create the LangChain service instance."""
    global langchain_service
    if langchain_service is None:
        langchain_service = LangChainService()
    return langchain_service


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get the status of available AI models."""
    try:
        service = get_langchain_service()
        status = service.get_available_models()
        return ModelStatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model status")


@router.post("/generate-text", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    """Generate text using AI models."""
    try:
        service = get_langchain_service()
        result = await service.generate_text(
            prompt=request.prompt,
            model=request.model,
            system_message=request.system_message,
            temperature=request.temperature
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Text generation failed"))
        
        return TextGenerationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text generation endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze image with text prompt using vision models."""
    try:
        service = get_langchain_service()
        result = await service.process_image_with_text(
            image_data=request.image_data,
            prompt=request.prompt,
            model=request.model,
            image_format="base64"
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Image analysis failed"))
        
        return ImageAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analyze-image-upload", response_model=ImageAnalysisResponse)
async def analyze_image_upload(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form(default="auto")
):
    """Analyze uploaded image with text prompt."""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and encode image
        image_bytes = await file.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        service = get_langchain_service()
        result = await service.process_image_with_text(
            image_data=image_b64,
            prompt=prompt,
            model=model,
            image_format="base64"
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Image analysis failed"))
        
        return ImageAnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image upload analysis endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate-image", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """Generate image using DALL-E."""
    try:
        service = get_langchain_service()
        result = await service.generate_image(
            prompt=request.prompt,
            size=request.size,
            quality=request.quality,
            n=request.n
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Image generation failed"))
        
        return ImageGenerationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/refresh-models")
async def refresh_models():
    """Refresh AI model initialization (useful when API keys are updated)."""
    try:
        service = get_langchain_service()
        service.refresh_models()
        return {"message": "Models refreshed successfully"}
    except Exception as e:
        logger.error(f"Failed to refresh models: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh models")
