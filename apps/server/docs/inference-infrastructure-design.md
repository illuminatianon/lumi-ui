# Inference Infrastructure Design Document

## Overview

This document outlines the design for a new unified inference infrastructure that will replace the current LangChain-based implementation. The goal is to create a provider-agnostic interface that smooths over differences in supported parameters, API formats, and capabilities across different AI providers.

## Current State Analysis

### Existing Implementation
- **LangChainService**: Handles OpenAI and Google Gemini integration
- **Provider-specific logic**: Scattered throughout the service with conditional handling
- **Configuration**: Basic provider selection with limited parameter support
- **API Surface**: REST endpoints for text generation, image analysis, and image generation

### Current Limitations
1. **Tight coupling** to LangChain abstractions
2. **Inconsistent parameter handling** across providers
3. **Limited extensibility** for new providers
4. **Provider-specific code paths** scattered throughout the service
5. **Configuration complexity** when adding new providers or models

## Critical Challenge: Model-Specific Parameters

### The Problem
Different models within the same provider can have completely different parameter sets:

**OpenAI Examples:**
- **GPT-4o**: Uses `max_tokens`, supports `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`
- **GPT-5**: Uses `max_completion_tokens` (not `max_tokens`), supports `reasoning_effort` (low/medium/high), may not support `temperature`
- **o1 models**: Use `max_completion_tokens`, no `temperature` support, have reasoning-specific parameters

**Parameter Mapping Challenges:**
```python
# This won't work for GPT-5:
openai_params = {
    "model": "gpt-5",
    "max_tokens": 1000,  # ❌ Should be max_completion_tokens
    "temperature": 0.7,  # ❌ Not supported
}

# GPT-5 requires:
openai_params = {
    "model": "gpt-5",
    "max_completion_tokens": 1000,  # ✅ Correct parameter
    "reasoning_effort": "medium",   # ✅ GPT-5 specific
    # No temperature parameter
}
```

### Current State Issues
1. **Hardcoded Parameters**: Current LangChain service assumes all OpenAI models use same parameters
2. **No Model Introspection**: No way to discover what parameters a model actually supports
3. **Breaking Changes**: New models break existing code when they change parameter names
4. **Provider Inconsistency**: Each provider handles this differently

## Design Goals

### Primary Objectives
1. **Unified Interface**: Single API surface regardless of underlying provider
2. **Provider Agnostic**: Easy addition of new providers without changing core logic
3. **Dynamic Parameter Mapping**: Automatically map parameters based on specific model capabilities
4. **Model-Aware Configuration**: Support different parameter sets for different models
5. **Graceful Degradation**: Ignore unsupported parameters rather than failing
6. **Backward Compatibility**: Maintain existing API contracts during migration

### Secondary Objectives
1. **Performance**: Minimize overhead from abstraction layers
2. **Observability**: Rich logging and metrics for debugging and monitoring
3. **Testability**: Clear separation of concerns for easier testing
4. **Configuration**: Flexible, hierarchical configuration system
5. **Future-Proof**: Handle new models and parameters without code changes

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                     │
├─────────────────────────────────────────────────────────────┤
│                 Inference Service                           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Request Processor                          ││
│  │  • Parameter validation & normalization                ││
│  │  • Provider selection logic                            ││
│  │  • Response formatting                                 ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                Provider Shim Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐│
│  │   OpenAI    │ │   Google    │ │  Anthropic  │ │  ...   ││
│  │    Shim     │ │    Shim     │ │    Shim     │ │  Shim  ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘│
├─────────────────────────────────────────────────────────────┤
│                Provider SDKs/APIs                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐│
│  │   OpenAI    │ │   Google    │ │  Anthropic  │ │  ...   ││
│  │     SDK     │ │     SDK     │ │     SDK     │ │   SDK  ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Model-Aware Configuration Schema
```python
class ParameterMapping(BaseModel):
    """Maps unified parameter names to model-specific parameter names."""
    max_tokens_param: str = "max_tokens"  # Could be "max_completion_tokens" for GPT-5
    temperature_param: Optional[str] = "temperature"  # None if not supported
    top_p_param: Optional[str] = "top_p"
    # Model-specific parameters
    custom_params: Dict[str, Any] = {}  # e.g., {"reasoning_effort": "medium"}

class ModelConfig(BaseModel):
    name: str
    capabilities: List[Capability]
    parameter_mapping: ParameterMapping
    supported_parameters: Set[str]  # Parameters this model actually supports
    cost_per_token: Optional[float]
    context_window: int

class ProviderConfig:
    enabled: bool
    api_key: str
    models: Dict[str, ModelConfig]
    default_model: str
    # Model discovery settings
    auto_discover_models: bool = True
    model_cache_ttl: int = 3600  # Cache model info for 1 hour
```

### 2. Unified Provider Shim Interface
```python
class ProviderShim(ABC):
    @abstractmethod
    async def process_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Single method that handles all request types based on attachments and response_format."""
        pass

    @abstractmethod
    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Discover available models and their capabilities from the provider."""
        pass

    @abstractmethod
    def map_parameters(self, unified_params: Dict[str, Any], model_config: ModelConfig) -> Dict[str, Any]:
        """Map unified parameters to model-specific parameters."""
        pass

    @abstractmethod
    def prepare_attachments(self, attachments: List[Attachment], model_config: ModelConfig) -> Any:
        """Convert attachments to provider-specific format."""
        pass

    def determine_request_type(self, request: UnifiedRequest, model_config: ModelConfig) -> str:
        """Determine what type of request this is based on attachments and response format."""
        has_images = any(att.attachment_type == AttachmentType.IMAGE for att in request.attachments)

        if request.response_format == "image":
            if has_images:
                return "image_edit"  # Image-to-image
            else:
                return "image_generation"  # Text-to-image
        elif has_images:
            return "vision"  # Image analysis/vision
        else:
            return "text"  # Pure text generation
```

### 3. Unified Request/Response Models with Attachments

```python
class AttachmentType(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    UNKNOWN = "unknown"

class Attachment(BaseModel):
    """Unified attachment that works across all providers."""
    content: bytes
    mime_type: str
    filename: Optional[str] = None
    attachment_type: AttachmentType
    metadata: Dict[str, Any] = {}

    @classmethod
    def from_file(cls, path: str) -> 'Attachment':
        """Create attachment from file path."""
        import mimetypes
        from pathlib import Path

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
    def from_url(cls, url: str) -> 'Attachment':
        """Create attachment from URL."""
        # Implementation would download and process
        pass

    @classmethod
    def from_base64(cls, data: str, mime_type: str, filename: Optional[str] = None) -> 'Attachment':
        """Create attachment from base64 data."""
        import base64
        content = base64.b64decode(data)
        return cls(
            content=content,
            mime_type=mime_type,
            filename=filename,
            attachment_type=cls._detect_type(mime_type)
        )

    def to_base64(self) -> str:
        """Convert to base64 string."""
        import base64
        return base64.b64encode(self.content).decode('utf-8')

    def to_pil_image(self) -> 'PIL.Image':
        """Convert to PIL Image (for image attachments)."""
        if self.attachment_type != AttachmentType.IMAGE:
            raise ValueError("Not an image attachment")
        from PIL import Image
        from io import BytesIO
        return Image.open(BytesIO(self.content))

    def to_text(self) -> str:
        """Extract text content."""
        if self.attachment_type == AttachmentType.TEXT:
            return self.content.decode('utf-8')
        elif self.attachment_type == AttachmentType.PDF:
            # Use PDF extraction library
            return self._extract_pdf_text()
        else:
            raise ValueError(f"Cannot extract text from {self.attachment_type}")

class UnifiedRequest(BaseModel):
    """Unified request that works for text, vision, and generation tasks."""
    prompt: str
    system_message: Optional[str] = None
    attachments: List[Attachment] = []

    # Standard parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop_sequences: Optional[List[str]] = None

    # Reasoning parameters (for thinking models)
    reasoning_effort: Optional[str] = None  # "low", "medium", "high" - model will use default if not specified

    # Response format control
    response_format: Literal["text", "image", "json"] = "text"
    stream: bool = False

    # Model selection
    model: str = "auto"

    # Provider-specific parameters
    extras: Dict[str, Any] = {}

class UnifiedResponse(BaseModel):
    """Unified response that can contain text, images, or both."""
    content: Optional[str] = None  # Text content
    images: List[str] = []  # Base64 encoded images or URLs

    # Metadata
    model_used: str
    provider: str
    usage: TokenUsage
    finish_reason: str
    attachments_processed: List[Dict[str, Any]] = []  # Info about how attachments were used
    metadata: Dict[str, Any] = {}
```

## Implementation Plan

### Phase 1: Foundation (Week 1-2)
1. Create base interfaces and abstract classes
2. Implement unified configuration schema
3. Create request/response models
4. Set up basic provider registry

### Phase 2: Provider Shims (Week 3-4)
1. Implement OpenAI shim (migrate from current implementation)
2. Implement Google/Gemini shim
3. Add Anthropic shim for future expansion
4. Create provider capability detection

### Phase 3: Core Service (Week 5-6)
1. Implement unified inference service
2. Add provider selection logic
3. Implement parameter normalization
4. Add response formatting

### Phase 4: Migration & Testing (Week 7-8)
1. Create migration layer for backward compatibility
2. Comprehensive testing suite
3. Performance benchmarking
4. Documentation updates

## Migration Strategy

### Backward Compatibility
- Maintain existing API endpoints during transition
- Create adapter layer that translates old requests to new format
- Gradual migration of internal consumers

### Configuration Migration
- Automatic migration of existing LangChain config to new format
- Validation and warnings for deprecated settings
- Clear migration guide for users

### Testing Strategy
- Unit tests for each provider shim
- Integration tests for the unified service
- End-to-end tests for API compatibility
- Performance regression tests

## Benefits

1. **Easier Provider Addition**: New providers require only implementing the shim interface
2. **Consistent Behavior**: Same parameters work across all providers (when supported)
3. **Better Error Handling**: Graceful degradation for unsupported features
4. **Improved Testing**: Clear interfaces make mocking and testing easier
5. **Future-Proof**: Architecture supports new capabilities and providers

## Smart Attachment Processing and Request Routing

### Unified Service with Intelligent Routing

```python
class UnifiedInferenceService:
    async def process_request(self, request: UnifiedRequest) -> UnifiedResponse:
        """Single entry point for all inference requests."""

        # 1. Analyze request to determine intent
        request_type = self._analyze_request(request)

        # 2. Select appropriate model based on request type and attachments
        model_config = await self._select_model(request, request_type)

        # 3. Get provider shim
        shim = self.provider_registry.get_shim(model_config.provider)

        # 4. Process request
        return await shim.process_request(request, model_config)

    def _analyze_request(self, request: UnifiedRequest) -> RequestType:
        """Analyze request to determine what the user wants."""
        has_images = any(att.attachment_type == AttachmentType.IMAGE for att in request.attachments)
        has_documents = any(att.attachment_type in [AttachmentType.PDF, AttachmentType.TEXT]
                           for att in request.attachments)

        if request.response_format == "image":
            if has_images:
                return RequestType.IMAGE_EDIT
            else:
                return RequestType.IMAGE_GENERATION
        elif has_images:
            return RequestType.VISION_ANALYSIS
        elif has_documents:
            return RequestType.DOCUMENT_ANALYSIS
        else:
            return RequestType.TEXT_GENERATION

    async def _select_model(self, request: UnifiedRequest, request_type: RequestType) -> ModelConfig:
        """Select best model for the request type."""
        if request.model != "auto":
            return await self.model_manager.get_model_config(request.model)

        # Auto-select based on request type and capabilities
        if request_type == RequestType.VISION_ANALYSIS:
            return self._get_best_vision_model()
        elif request_type == RequestType.IMAGE_GENERATION:
            return self._get_best_image_gen_model()
        elif request_type == RequestType.IMAGE_EDIT:
            return self._get_best_image_edit_model()
        else:
            return self._get_best_text_model()

# Convenience methods for common use cases
class UnifiedInferenceService:
    async def chat(self, message: str, attachments: List[Attachment] = None, **kwargs) -> str:
        """Simple chat interface - returns just the text response."""
        request = UnifiedRequest(
            prompt=message,
            attachments=attachments or [],
            response_format="text",
            **kwargs
        )
        response = await self.process_request(request)
        return response.content

    async def analyze_image(self, image: Attachment, question: str, **kwargs) -> str:
        """Analyze an image with a question."""
        request = UnifiedRequest(
            prompt=question,
            attachments=[image],
            response_format="text",
            **kwargs
        )
        response = await self.process_request(request)
        return response.content

    async def generate_image(self, prompt: str, reference_image: Attachment = None, **kwargs) -> List[str]:
        """Generate images, optionally with a reference."""
        request = UnifiedRequest(
            prompt=prompt,
            attachments=[reference_image] if reference_image else [],
            response_format="image",
            **kwargs
        )
        response = await self.process_request(request)
        return response.images
```

### Provider-Specific Attachment Handling

```python
class OpenAIShim(ProviderShim):
    async def process_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        request_type = self.determine_request_type(request, model_config)

        if request_type == "vision":
            return await self._handle_vision_request(request, model_config)
        elif request_type == "image_generation":
            return await self._handle_image_generation(request, model_config)
        elif request_type == "image_edit":
            return await self._handle_image_edit(request, model_config)
        else:
            return await self._handle_text_request(request, model_config)

    async def _handle_vision_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle vision requests with image attachments."""
        # Build messages with images in OpenAI format
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})

        # Build user message with text and images
        content = [{"type": "text", "text": request.prompt}]

        for attachment in request.attachments:
            if attachment.attachment_type == AttachmentType.IMAGE:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{attachment.mime_type};base64,{attachment.to_base64()}"
                    }
                })

        messages.append({"role": "user", "content": content})

        # Map parameters and make request
        params = self.map_parameters(request.dict(), model_config)
        params.update({
            "model": model_config.name,
            "messages": messages
        })

        response = await self.client.chat.completions.create(**params)

        return UnifiedResponse(
            content=response.choices[0].message.content,
            model_used=response.model,
            provider="openai",
            usage=TokenUsage.from_openai(response.usage),
            finish_reason=response.choices[0].finish_reason,
            attachments_processed=[
                {"type": att.attachment_type, "filename": att.filename}
                for att in request.attachments
            ]
        )

    async def _handle_image_generation(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle DALL-E image generation."""
        # Check if there are reference images for style/composition
        reference_prompt = request.prompt
        if request.attachments:
            # Could analyze reference images and modify prompt
            reference_prompt = await self._enhance_prompt_with_references(request.prompt, request.attachments)

        response = await self.client.images.generate(
            model="dall-e-3",
            prompt=reference_prompt,
            size=request.extras.get("size", "1024x1024"),
            quality=request.extras.get("quality", "standard"),
            n=1
        )

        return UnifiedResponse(
            images=[img.url for img in response.data],
            model_used="dall-e-3",
            provider="openai",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),  # DALL-E doesn't report tokens
            finish_reason="completed",
            attachments_processed=[
                {"type": att.attachment_type, "used_as": "reference"}
                for att in request.attachments
            ]
        )
```

## Solutions for Model-Specific Parameters

### Approach 1: Static Model Registry (Recommended)
Maintain a curated registry of model configurations that gets updated as new models are released.

```python
# Model registry with known parameter mappings
MODEL_REGISTRY = {
    "openai": {
        "gpt-4o": ModelConfig(
            name="gpt-4o",
            parameter_mapping=ParameterMapping(
                max_tokens_param="max_tokens",
                temperature_param="temperature",
                top_p_param="top_p"
            ),
            supported_parameters={"max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty"},
            custom_params={}
        ),
        "gpt-5": ModelConfig(
            name="gpt-5",
            parameter_mapping=ParameterMapping(
                max_tokens_param="max_completion_tokens",
                temperature_param=None,  # Not supported
                top_p_param=None,
                custom_params={"reasoning_effort": "medium"}
            ),
            supported_parameters={"max_completion_tokens", "reasoning_effort"},
        ),
        "o1-preview": ModelConfig(
            name="o1-preview",
            parameter_mapping=ParameterMapping(
                max_tokens_param="max_completion_tokens",
                temperature_param=None,
                top_p_param=None
            ),
            supported_parameters={"max_completion_tokens"},
        )
    }
}
```

### Approach 2: Dynamic Model Discovery (Future Enhancement)
Query provider APIs to discover model capabilities at runtime.

```python
class OpenAIModelDiscovery:
    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Discover OpenAI models and attempt to infer their parameters."""
        client = AsyncOpenAI(api_key=self.api_key)

        # Get list of available models
        models = await client.models.list()

        discovered = {}
        for model in models.data:
            # Try to infer parameter support based on model family
            config = self._infer_model_config(model.id)
            discovered[model.id] = config

        return discovered

    def _infer_model_config(self, model_id: str) -> ModelConfig:
        """Infer model configuration based on model ID patterns."""
        if model_id.startswith("gpt-5"):
            return self._get_gpt5_config(model_id)
        elif model_id.startswith("o1"):
            return self._get_o1_config(model_id)
        elif model_id.startswith("gpt-4"):
            return self._get_gpt4_config(model_id)
        else:
            return self._get_default_config(model_id)
```

### Approach 3: Hybrid Strategy (Recommended Implementation)
Combine static registry with runtime validation and fallback discovery.

```python
class ModelManager:
    def __init__(self, provider_shim: ProviderShim):
        self.provider_shim = provider_shim
        self.static_registry = MODEL_REGISTRY
        self.runtime_cache = {}

    async def get_model_config(self, model_id: str) -> ModelConfig:
        """Get model configuration with fallback strategy."""

        # 1. Check static registry first
        if model_id in self.static_registry.get(self.provider_shim.provider_name, {}):
            return self.static_registry[self.provider_shim.provider_name][model_id]

        # 2. Check runtime cache
        if model_id in self.runtime_cache:
            return self.runtime_cache[model_id]

        # 3. Try dynamic discovery
        try:
            discovered = await self.provider_shim.discover_models()
            if model_id in discovered:
                self.runtime_cache[model_id] = discovered[model_id]
                return discovered[model_id]
        except Exception as e:
            logger.warning(f"Model discovery failed: {e}")

        # 4. Fallback to conservative defaults
        return self._get_conservative_defaults(model_id)
```

## Detailed Component Specifications

### Provider Shim Implementation Details

#### Model-Aware OpenAI Shim
```python
class OpenAIShim(ProviderShim):
    def __init__(self, config: ProviderConfig):
        self.client = AsyncOpenAI(api_key=config.api_key)
        self.config = config
        self.model_manager = ModelManager(self)

    async def generate_text(self, request: TextRequest, model_config: ModelConfig) -> TextResponse:
        # Map unified parameters to model-specific OpenAI format
        openai_params = self.map_parameters(request.dict(), model_config)

        # Add required parameters
        openai_params["model"] = model_config.name
        openai_params["messages"] = self._build_messages(request)

        # Add model-specific custom parameters
        openai_params.update(model_config.parameter_mapping.custom_params)

        # Remove None values and unsupported parameters
        openai_params = {
            k: v for k, v in openai_params.items()
            if v is not None and k in model_config.supported_parameters
        }

        response = await self.client.chat.completions.create(**openai_params)

        return TextResponse(
            content=response.choices[0].message.content,
            model_used=response.model,
            provider="openai",
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ),
            finish_reason=response.choices[0].finish_reason,
            metadata={"response_id": response.id}
        )

    def map_parameters(self, unified_params: Dict[str, Any], model_config: ModelConfig) -> Dict[str, Any]:
        """Map unified parameters to OpenAI model-specific parameters."""
        mapped = {}
        mapping = model_config.parameter_mapping

        # Map standard parameters
        if unified_params.get("max_tokens") and mapping.max_tokens_param:
            mapped[mapping.max_tokens_param] = unified_params["max_tokens"]

        if unified_params.get("temperature") and mapping.temperature_param:
            mapped[mapping.temperature_param] = unified_params["temperature"]

        if unified_params.get("top_p") and mapping.top_p_param:
            mapped[mapping.top_p_param] = unified_params["top_p"]

        # Handle frequency/presence penalty for models that support them
        if "frequency_penalty" in model_config.supported_parameters:
            mapped["frequency_penalty"] = unified_params.get("frequency_penalty")

        if "presence_penalty" in model_config.supported_parameters:
            mapped["presence_penalty"] = unified_params.get("presence_penalty")

        # Add stop sequences if supported
        if "stop" in model_config.supported_parameters:
            mapped["stop"] = unified_params.get("stop_sequences")

        # Add streaming if supported
        if "stream" in model_config.supported_parameters:
            mapped["stream"] = unified_params.get("stream", False)

        # Handle reasoning parameters for thinking models
        if model_config.capabilities.reasoning.supports_reasoning:
            reasoning_effort = unified_params.get("reasoning_effort")
            if reasoning_effort:
                if reasoning_effort in model_config.capabilities.reasoning.reasoning_effort_levels:
                    mapped["reasoning_effort"] = reasoning_effort
                else:
                    # Use default if invalid effort level provided
                    mapped["reasoning_effort"] = model_config.capabilities.reasoning.default_reasoning_effort
            elif model_config.capabilities.reasoning.default_reasoning_effort:
                # Use model default if no effort specified
                mapped["reasoning_effort"] = model_config.capabilities.reasoning.default_reasoning_effort

        return mapped

    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Discover available OpenAI models."""
        models = await self.client.models.list()
        discovered = {}

        for model in models.data:
            # Use static registry if available, otherwise infer
            if model.id in MODEL_REGISTRY.get("openai", {}):
                discovered[model.id] = MODEL_REGISTRY["openai"][model.id]
            else:
                discovered[model.id] = self._infer_model_config(model.id)

        return discovered
```

#### Google Gemini Shim
```python
class GeminiShim(ProviderShim):
    def __init__(self, config: ProviderConfig):
        self.client = genai.GenerativeModel(config.default_model)
        self.config = config
        genai.configure(api_key=config.api_key)

    async def generate_text(self, request: TextRequest) -> TextResponse:
        # Map unified parameters to Gemini format
        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            top_p=request.top_p,
            stop_sequences=request.stop_sequences,
        )

        # Gemini doesn't support frequency/presence penalty - log warning
        if request.frequency_penalty or request.presence_penalty:
            logger.warning("Gemini doesn't support frequency/presence penalty")

        # Build prompt from system message and user prompt
        prompt = self._build_prompt(request)

        response = await self.client.generate_content_async(
            prompt,
            generation_config=generation_config
        )

        return TextResponse(
            content=response.text,
            model_used=self.config.default_model,
            provider="google",
            usage=TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count,
                completion_tokens=response.usage_metadata.candidates_token_count,
                total_tokens=response.usage_metadata.total_token_count
            ),
            finish_reason=response.candidates[0].finish_reason.name,
            metadata={"safety_ratings": response.candidates[0].safety_ratings}
        )
```

### Configuration Schema Details

#### Complete Configuration Structure
```python
class GlobalDefaults(BaseModel):
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0

class ModelCapabilities(BaseModel):
    text_generation: bool = True
    vision: bool = False
    image_generation: bool = False
    function_calling: bool = False
    streaming: bool = False
    json_mode: bool = False
    max_context_length: Optional[int] = None

class ModelConfig(BaseModel):
    name: str
    display_name: str
    capabilities: ModelCapabilities
    cost_per_1k_tokens: Optional[float] = None
    max_tokens_per_request: Optional[int] = None
    context_window: Optional[int] = None

class ProviderConfig(BaseModel):
    enabled: bool = True
    api_key: str
    base_url: Optional[str] = None
    models: Dict[str, ModelConfig]
    default_model: str
    rate_limits: Dict[str, int] = {}  # requests per minute by model
    custom_headers: Dict[str, str] = {}

class InferenceConfig(BaseModel):
    providers: Dict[str, ProviderConfig]
    default_provider: str = "auto"
    fallback_providers: List[str] = []
    global_defaults: GlobalDefaults = GlobalDefaults()
    provider_selection_strategy: str = "cost_optimized"  # or "performance", "reliability"
```

### Provider Selection Logic

#### Selection Strategies
```python
class ProviderSelector:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.strategies = {
            "cost_optimized": self._cost_optimized_selection,
            "performance": self._performance_optimized_selection,
            "reliability": self._reliability_optimized_selection,
            "round_robin": self._round_robin_selection,
        }

    async def select_provider(self, request: TextRequest) -> str:
        strategy = self.config.provider_selection_strategy
        return await self.strategies[strategy](request)

    async def _cost_optimized_selection(self, request: TextRequest) -> str:
        # Select provider with lowest cost for the requested capability
        available_providers = self._get_available_providers(request)

        costs = []
        for provider_name in available_providers:
            provider = self.config.providers[provider_name]
            model = provider.models.get(provider.default_model)
            if model and model.cost_per_1k_tokens:
                costs.append((provider_name, model.cost_per_1k_tokens))

        if costs:
            return min(costs, key=lambda x: x[1])[0]

        return available_providers[0] if available_providers else None
```

## Error Handling and Resilience

### Retry Logic
```python
class RetryHandler:
    def __init__(self, config: GlobalDefaults):
        self.max_attempts = config.retry_attempts
        self.base_delay = config.retry_delay

    async def execute_with_retry(self, func, *args, **kwargs):
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except (RateLimitError, TimeoutError, ConnectionError) as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                    continue
                break
            except Exception as e:
                # Don't retry on non-transient errors
                raise e

        raise last_exception
```

### Fallback Mechanism
```python
class FallbackHandler:
    def __init__(self, config: InferenceConfig):
        self.fallback_providers = config.fallback_providers

    async def execute_with_fallback(self, request: TextRequest, primary_provider: str):
        providers_to_try = [primary_provider] + self.fallback_providers

        for provider_name in providers_to_try:
            try:
                provider = self._get_provider_shim(provider_name)
                return await provider.generate_text(request)
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                if provider_name == providers_to_try[-1]:
                    raise e
                continue
```

## Addressing the Parameter Problem

### Immediate Solutions

1. **Model Registry Updates**: Maintain a living registry of model parameter mappings
2. **Parameter Validation**: Validate parameters against model capabilities before sending requests
3. **Graceful Fallbacks**: When a parameter isn't supported, log a warning but continue
4. **Version Detection**: Detect model versions and apply appropriate parameter mappings

### Example Implementation Strategy

```python
class UnifiedInferenceService:
    async def generate_text(self, prompt: str, model: str = "auto", **kwargs) -> TextResponse:
        # 1. Resolve model selection
        provider_name, model_id = self._resolve_model(model)

        # 2. Get model configuration
        model_config = await self.model_manager.get_model_config(model_id)

        # 3. Create unified request
        request = TextRequest(
            prompt=prompt,
            model=model_id,
            **kwargs
        )

        # 4. Get provider shim
        shim = self.provider_registry.get_shim(provider_name)

        # 5. Generate with model-aware shim
        return await shim.generate_text(request, model_config)
```

### Migration Strategy from Current LangChain Implementation

#### Phase 1: Add Model Registry
1. Create static model registry with known OpenAI and Google models
2. Add parameter mapping for current models (gpt-4o, gemini-1.5-pro)
3. Update existing LangChain service to use registry for parameter validation

#### Phase 2: Implement Model-Aware Shims
1. Create new provider shim interfaces
2. Implement OpenAI shim with proper parameter mapping
3. Add support for GPT-5 and reasoning models
4. Implement Google shim with Gemini-specific handling

#### Phase 3: Unified Service Layer
1. Create unified inference service that uses shims
2. Add provider selection logic
3. Implement fallback mechanisms
4. Add comprehensive logging and monitoring

#### Phase 4: Deprecate LangChain Service
1. Create compatibility layer for existing API endpoints
2. Gradually migrate internal consumers to new service
3. Update documentation and examples
4. Remove old LangChain service

### Benefits of This Approach

1. **Handles Current Issues**: Solves the GPT-5 parameter problem immediately
2. **Future-Proof**: New models can be added by updating the registry
3. **Backward Compatible**: Existing code continues to work
4. **Provider Agnostic**: Same approach works for all providers
5. **Maintainable**: Clear separation between model knowledge and business logic

## Usage Examples: The Power of Unified Attachments

### Simple Cases
```python
service = UnifiedInferenceService()

# Pure text
response = await service.chat("What is machine learning?")

# Image analysis
screenshot = Attachment.from_file("bug_screenshot.png")
response = await service.chat("What's wrong in this screenshot?", attachments=[screenshot])

# Document analysis
pdf = Attachment.from_file("research_paper.pdf")
response = await service.chat("Summarize the key findings", attachments=[pdf])

# Reasoning/thinking models
response = await service.chat(
    "Solve this complex math problem step by step: ...",
    model="gpt-5",
    reasoning_effort="high"  # Use maximum reasoning for complex problems
)

# Auto-reasoning (model picks appropriate effort level)
response = await service.chat(
    "What's the best approach to this coding problem?",
    model="auto"  # Will pick GPT-5 or o1 if available for reasoning tasks
)
```

### Complex Multimodal Cases
```python
# Code review with screenshot
code_file = Attachment.from_file("main.py")
error_screenshot = Attachment.from_file("error.png")
response = await service.chat(
    "Fix the bug shown in the screenshot based on this code",
    attachments=[code_file, error_screenshot]
)

# Image generation with reference
reference_image = Attachment.from_file("style_reference.jpg")
response = await service.process_request(UnifiedRequest(
    prompt="Create a landscape in this artistic style",
    attachments=[reference_image],
    response_format="image"
))

# Mixed analysis and generation
chart_image = Attachment.from_file("sales_chart.png")
response = await service.process_request(UnifiedRequest(
    prompt="Analyze this chart and create an improved version",
    attachments=[chart_image],
    response_format="image"  # Want a new chart back
))
```

### Provider Abstraction in Action
```python
# Same code works across providers automatically
configs = [
    {"model": "gpt-4o"},           # OpenAI vision
    {"model": "gemini-1.5-pro"},   # Google vision
    {"model": "claude-3-opus"},    # Anthropic vision
]

image = Attachment.from_file("diagram.png")
for config in configs:
    response = await service.chat(
        "Explain this diagram",
        attachments=[image],
        **config
    )
    print(f"{config['model']}: {response}")
```

### Automatic Model Selection
```python
# System automatically picks the best model for each task
requests = [
    # Picks best vision model
    UnifiedRequest(prompt="What's in this image?", attachments=[image]),

    # Picks best image generation model
    UnifiedRequest(prompt="Draw a cat", response_format="image"),

    # Picks best text model
    UnifiedRequest(prompt="Write a poem"),

    # Picks best document analysis model
    UnifiedRequest(prompt="Summarize this", attachments=[pdf])
]

for req in requests:
    response = await service.process_request(req)
    print(f"Used: {response.model_used}")
```

## Initial Implementation Scope

### Working Providers (Phase 1-3)
- **OpenAI**: Full implementation with GPT-4o, GPT-5, o1 models, DALL-E, vision support
- **Google**: Full implementation with Gemini models, vision support (no image generation yet)

### Provider Stubs (Future Implementation)
```python
class AnthropicShim(ProviderShim):
    """Stub implementation for Anthropic Claude models."""

    async def process_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        raise NotImplementedError(
            "Anthropic provider not yet implemented. "
            "Currently supported providers: OpenAI, Google. "
            "Please use model='auto' or specify an OpenAI/Google model."
        )

    async def discover_models(self) -> Dict[str, ModelConfig]:
        return {}  # Return empty for now

class AzureOpenAIShim(ProviderShim):
    """Stub implementation for Azure OpenAI."""

    async def process_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        raise NotImplementedError(
            "Azure OpenAI provider not yet implemented. "
            "Use the regular OpenAI provider for now."
        )

class OllamaShim(ProviderShim):
    """Stub implementation for local Ollama models."""

    async def process_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        raise NotImplementedError(
            "Ollama provider not yet implemented. "
            "Local model support coming in future release."
        )
```

### Configuration for Stubs
```python
# These providers can be configured but will fail gracefully
PROVIDER_REGISTRY = {
    "openai": OpenAIShim,      # ✅ Working
    "google": GoogleShim,      # ✅ Working
    "anthropic": AnthropicShim,  # ❌ Stub only
    "azure": AzureOpenAIShim,    # ❌ Stub only
    "ollama": OllamaShim,        # ❌ Stub only
}

# Default config marks stubs as disabled
DEFAULT_CONFIG = {
    "providers": {
        "openai": {"enabled": True, ...},
        "google": {"enabled": True, ...},
        "anthropic": {"enabled": False, ...},  # Disabled by default
        "azure": {"enabled": False, ...},
        "ollama": {"enabled": False, ...}
    }
}
```

## Key Benefits Achieved

1. **Single Interface**: One method handles text, vision, generation, and mixed requests
2. **Attachment Flexibility**: Drop any file type into any request
3. **Provider Abstraction**: Same code works with OpenAI and Google (more coming)
4. **Smart Routing**: System picks the best model for each task automatically
5. **Format Conversion**: Automatic conversion between provider-specific formats
6. **Future Proof**: New providers can be added without changing existing code
7. **Graceful Degradation**: Unsupported providers fail with helpful error messages

## Next Steps

1. **Review and approve this design document**
2. **Start with Phase 1**: Create model registry and update current service
3. **Prototype attachment system**: Test the unified attachment handling
4. **Implement OpenAI shim**: Full multimodal support with GPT-5
5. **Plan full implementation**: Detailed specs for the unified service
6. **Set up development environment and tooling**
