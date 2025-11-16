"""Configuration models for the unified inference infrastructure."""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class ProviderType(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


class Capability(str, Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"
    JSON_MODE = "json_mode"
    CODE_GENERATION = "code_generation"
    EMBEDDINGS = "embeddings"
    REASONING = "reasoning"  # Model has explicit reasoning/thinking capabilities


class SelectionStrategy(str, Enum):
    """Provider selection strategies."""
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    ROUND_ROBIN = "round_robin"
    MANUAL = "manual"
    AUTO = "auto"


class ReasoningCapabilities(BaseModel):
    """Reasoning/thinking specific capabilities."""
    supports_reasoning: bool = False
    reasoning_effort_levels: List[str] = []  # e.g., ["low", "medium", "high"]
    default_reasoning_effort: Optional[str] = None
    exposes_reasoning_tokens: bool = False  # Whether reasoning is visible in response
    reasoning_token_cost: Optional[float] = None  # Cost per reasoning token if different


class ModelCapabilities(BaseModel):
    """Capabilities supported by a specific model."""
    text_generation: bool = True
    vision: bool = False
    image_generation: bool = False
    function_calling: bool = False
    streaming: bool = False
    json_mode: bool = False
    code_generation: bool = False
    embeddings: bool = False
    max_context_length: Optional[int] = None
    supports_system_messages: bool = True
    supports_multimodal: bool = False

    # Reasoning capabilities
    reasoning: ReasoningCapabilities = Field(default_factory=ReasoningCapabilities)


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str = Field(..., description="Model identifier used by the provider")
    display_name: str = Field(..., description="Human-readable model name")
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    cost_per_1k_tokens: Optional[float] = Field(None, description="Cost per 1000 tokens")
    max_tokens_per_request: Optional[int] = Field(None, description="Maximum tokens per request")
    context_window: Optional[int] = Field(None, description="Maximum context window size")
    default_temperature: float = Field(0.7, ge=0.0, le=2.0)
    default_max_tokens: Optional[int] = Field(None, ge=1)
    
    # Provider-specific parameters
    provider_specific: Dict[str, Any] = Field(default_factory=dict)


class RateLimits(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: Optional[int] = Field(None, ge=1)
    tokens_per_minute: Optional[int] = Field(None, ge=1)
    concurrent_requests: Optional[int] = Field(None, ge=1)


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""
    enabled: bool = True
    provider_type: ProviderType
    api_key: str = Field(..., description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Custom base URL for the provider")
    organization_id: Optional[str] = Field(None, description="Organization ID (for providers that support it)")
    
    # Models configuration
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    default_model: str = Field(..., description="Default model to use for this provider")
    
    # Rate limiting and reliability
    rate_limits: RateLimits = Field(default_factory=RateLimits)
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")
    retry_attempts: int = Field(3, ge=0, description="Number of retry attempts")
    retry_delay: float = Field(1.0, ge=0.1, description="Base retry delay in seconds")
    
    # Custom headers and provider-specific settings
    custom_headers: Dict[str, str] = Field(default_factory=dict)
    provider_specific: Dict[str, Any] = Field(default_factory=dict)


class GlobalDefaults(BaseModel):
    """Global default parameters for inference requests."""
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    
    # Request handling defaults
    timeout: int = Field(30, ge=1)
    retry_attempts: int = Field(3, ge=0)
    retry_delay: float = Field(1.0, ge=0.1)
    
    # Streaming and response format
    stream: bool = False
    response_format: Optional[str] = Field(None, description="Default response format (json, text)")


class LoadBalancing(BaseModel):
    """Load balancing configuration."""
    enabled: bool = False
    strategy: SelectionStrategy = SelectionStrategy.ROUND_ROBIN
    weights: Dict[str, float] = Field(default_factory=dict, description="Provider weights for weighted selection")
    health_check_interval: int = Field(60, ge=10, description="Health check interval in seconds")


class Monitoring(BaseModel):
    """Monitoring and observability configuration."""
    enabled: bool = True
    log_requests: bool = True
    log_responses: bool = False  # May contain sensitive data
    track_usage: bool = True
    track_costs: bool = True
    metrics_endpoint: Optional[str] = None


class InferenceConfig(BaseModel):
    """Main configuration for the unified inference infrastructure."""
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)
    default_provider: str = Field("auto", description="Default provider to use")
    fallback_providers: List[str] = Field(default_factory=list, description="Fallback providers in order")
    
    # Selection and routing
    provider_selection_strategy: SelectionStrategy = SelectionStrategy.AUTO
    load_balancing: LoadBalancing = Field(default_factory=LoadBalancing)
    
    # Global settings
    global_defaults: GlobalDefaults = Field(default_factory=GlobalDefaults)
    monitoring: Monitoring = Field(default_factory=Monitoring)
    
    # Feature flags
    enable_caching: bool = False
    enable_request_validation: bool = True
    enable_response_validation: bool = True
    
    class Config:
        extra = "allow"  # Allow additional configuration sections


# Example configuration for common providers
EXAMPLE_OPENAI_CONFIG = ModelConfig(
    name="gpt-4o",
    display_name="GPT-4 Omni",
    capabilities=ModelCapabilities(
        text_generation=True,
        vision=True,
        function_calling=True,
        streaming=True,
        json_mode=True,
        max_context_length=128000,
        supports_multimodal=True,
        reasoning=ReasoningCapabilities(
            supports_reasoning=False
        )
    ),
    cost_per_1k_tokens=0.03,
    max_tokens_per_request=4096,
    context_window=128000
)

EXAMPLE_GPT5_CONFIG = ModelConfig(
    name="gpt-5",
    display_name="GPT-5",
    capabilities=ModelCapabilities(
        text_generation=True,
        vision=True,
        function_calling=True,
        streaming=True,
        json_mode=True,
        max_context_length=200000,
        supports_multimodal=True,
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            reasoning_effort_levels=["low", "medium", "high"],
            default_reasoning_effort="medium",
            exposes_reasoning_tokens=True,
            reasoning_token_cost=0.01  # Different cost for reasoning tokens
        )
    ),
    cost_per_1k_tokens=0.05,
    max_tokens_per_request=8192,
    context_window=200000,
    provider_specific={
        "max_tokens_param": "max_completion_tokens",
        "supports_temperature": False
    }
)

EXAMPLE_O1_CONFIG = ModelConfig(
    name="o1-preview",
    display_name="OpenAI o1 Preview",
    capabilities=ModelCapabilities(
        text_generation=True,
        vision=False,
        function_calling=False,
        streaming=False,
        json_mode=False,
        max_context_length=128000,
        supports_multimodal=False,
        reasoning=ReasoningCapabilities(
            supports_reasoning=True,
            reasoning_effort_levels=[],  # o1 doesn't expose effort control
            exposes_reasoning_tokens=False,  # Reasoning is hidden
            reasoning_token_cost=0.015
        )
    ),
    cost_per_1k_tokens=0.06,
    max_tokens_per_request=32768,
    context_window=128000,
    provider_specific={
        "max_tokens_param": "max_completion_tokens",
        "supports_temperature": False
    }
)

EXAMPLE_GEMINI_CONFIG = ModelConfig(
    name="gemini-1.5-pro",
    display_name="Gemini 1.5 Pro",
    capabilities=ModelCapabilities(
        text_generation=True,
        vision=True,
        function_calling=True,
        streaming=True,
        max_context_length=1000000,
        supports_multimodal=True,
        reasoning=ReasoningCapabilities(
            supports_reasoning=False  # Gemini doesn't have explicit reasoning mode
        )
    ),
    cost_per_1k_tokens=0.0125,
    max_tokens_per_request=8192,
    context_window=1000000
)
