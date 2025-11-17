from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Load environment variables
load_dotenv()

# Import configuration system
from config import initialize_config, config_router, get_config_health

# Import AI models and services
from api.models import RegistryRequest, RegistryResponse, ModelListResponse
from services.inference.factory import get_unified_inference_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize configuration system
config_initialized = initialize_config()
if not config_initialized:
    logger.error("Failed to initialize configuration system")

app = FastAPI(
    title="Lumi API",
    description="Backend API for Lumi UI application",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(config_router)

# Initialize AI inference service
try:
    inference_service = get_unified_inference_service()
    if inference_service:
        logger.info("AI inference service initialized successfully")
    else:
        logger.warning("AI inference service not available (disabled or misconfigured)")
except Exception as e:
    logger.error(f"Failed to initialize AI inference service: {e}")
    inference_service = None


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    message: str
    version: str


class MessageResponse(BaseModel):
    message: str
    timestamp: str


# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint returning API health status."""
    return HealthResponse(
        status="healthy", message="Lumi API is running", version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    config_health = get_config_health()
    overall_status = "healthy" if config_health.get("status") == "healthy" else "degraded"

    return HealthResponse(
        status=overall_status,
        message=f"API is operational (config: {config_health.get('status', 'unknown')})",
        version="1.0.0"
    )


@app.get("/api/hello", response_model=MessageResponse)
async def hello():
    """Simple hello endpoint."""
    from datetime import datetime

    return MessageResponse(
        message="Hello from Lumi API!", timestamp=datetime.now().isoformat()
    )


# AI Endpoints
@app.post("/api/ai/chat", response_model=RegistryResponse)
async def chat_completion(request: RegistryRequest):
    """Process chat completion with registry-style model naming."""
    if not inference_service:
        return RegistryResponse(
            content="",
            model_used="",
            provider="",
            success=False,
            error="AI inference service not available"
        )

    try:
        # Convert request to config dict
        config = request.model_dump()

        # Process with registry resolver
        response = await inference_service.process_registry_request(config)

        return RegistryResponse(
            content=response.content or "",
            model_used=config["model"],
            provider=config["model"].split("/")[0],
            success=True,
            metadata={
                "token_usage": response.token_usage.model_dump() if response.token_usage else None,
                "finish_reason": response.finish_reason
            }
        )
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return RegistryResponse(
            content="",
            model_used=request.model,
            provider=request.model.split("/")[0] if "/" in request.model else "",
            success=False,
            error=str(e)
        )


@app.get("/api/ai/models", response_model=ModelListResponse)
async def list_models():
    """List all available models in registry format."""
    if not inference_service:
        return ModelListResponse(
            models=[],
            providers={},
            total_models=0
        )

    try:
        models_data = inference_service.list_available_models()
        return ModelListResponse(**models_data)
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return ModelListResponse(
            models=[],
            providers={},
            total_models=0
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
