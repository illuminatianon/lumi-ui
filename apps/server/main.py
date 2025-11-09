from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

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
        status="healthy",
        message="Lumi API is running",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="API is operational",
        version="1.0.0"
    )

@app.get("/api/hello", response_model=MessageResponse)
async def hello():
    """Simple hello endpoint."""
    from datetime import datetime
    return MessageResponse(
        message="Hello from Lumi API!",
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
