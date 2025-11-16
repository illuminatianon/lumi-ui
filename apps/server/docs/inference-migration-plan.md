# Inference Infrastructure Migration Plan

## Overview

This document outlines the practical steps to migrate from the current LangChain-based inference service to the new unified inference infrastructure, with immediate focus on solving the model-specific parameter problem (e.g., GPT-5's `max_completion_tokens` vs `max_tokens`).

## Current State Assessment

### Immediate Problems to Solve
1. **GPT-5 Parameter Incompatibility**: Current service uses `max_tokens` but GPT-5 requires `max_completion_tokens`
2. **Reasoning Model Support**: No support for `reasoning_effort` parameter
3. **Hardcoded Parameter Mapping**: All OpenAI models assumed to have same parameters
4. **No Model Discovery**: No way to detect what parameters a model supports

### Current Architecture Issues
- Single `ChatOpenAI` instance for all OpenAI models
- Parameters set at initialization time, not per-request
- No model-specific configuration
- LangChain abstractions hide provider-specific differences

## Migration Strategy

### Phase 1: Quick Fix for Current Issues (Week 1)
**Goal**: Get GPT-5 and reasoning models working with minimal changes

#### 1.1 Add Model-Specific Parameter Mapping
```python
# Add to existing LangChainService
MODEL_PARAMETER_MAPPING = {
    "gpt-4o": {
        "max_tokens_param": "max_tokens",
        "supports_temperature": True,
        "supports_reasoning": False
    },
    "gpt-5": {
        "max_tokens_param": "max_completion_tokens", 
        "supports_temperature": False,
        "supports_reasoning": True,
        "default_reasoning_effort": "medium"
    },
    "o1-preview": {
        "max_tokens_param": "max_completion_tokens",
        "supports_temperature": False,
        "supports_reasoning": False
    }
}
```

#### 1.2 Update generate_text Method
```python
async def generate_text(self, prompt: str, model: str = "auto", **kwargs):
    # Determine actual model to use
    actual_model = self._resolve_model_name(model)
    
    # Get model-specific parameter mapping
    mapping = MODEL_PARAMETER_MAPPING.get(actual_model, {})
    
    # Build parameters based on model capabilities
    params = self._build_model_specific_params(kwargs, mapping, actual_model)
    
    # Use direct OpenAI client instead of LangChain for model-specific calls
    if actual_model.startswith("gpt-5") or actual_model.startswith("o1"):
        return await self._call_openai_direct(prompt, actual_model, params)
    else:
        # Use existing LangChain path for compatible models
        return await self._call_langchain(prompt, actual_model, params)
```

#### 1.3 Direct OpenAI Client Integration
```python
async def _call_openai_direct(self, prompt: str, model: str, params: dict):
    """Direct OpenAI API call for models with special parameter requirements."""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(api_key=self._config.api_keys.openai)
    
    messages = [{"role": "user", "content": prompt}]
    if params.get("system_message"):
        messages.insert(0, {"role": "system", "content": params["system_message"]})
    
    openai_params = {
        "model": model,
        "messages": messages,
        **params  # Already mapped to model-specific format
    }
    
    response = await client.chat.completions.create(**openai_params)
    
    return {
        "text": response.choices[0].message.content,
        "model_used": response.model,
        "success": True
    }
```

### Phase 2: Unified Service Foundation (Week 2-3)
**Goal**: Create the new unified service alongside the existing one

#### 2.1 Create Base Interfaces
- `ProviderShim` abstract base class
- `ModelConfig` and `ParameterMapping` classes
- `UnifiedInferenceService` main service class

#### 2.2 Implement OpenAI Shim
- Full OpenAI provider shim with model-aware parameter mapping
- Support for all current OpenAI models (GPT-4o, GPT-5, o1, etc.)
- Model discovery and capability detection

#### 2.3 Create Model Registry
- Static registry with known model configurations
- Runtime caching for discovered models
- Fallback mechanisms for unknown models

### Phase 3: Google/Gemini Implementation (Week 4-5)
**Goal**: Complete Google/Gemini shim alongside OpenAI (only working providers initially)

#### 3.1 Google/Gemini Shim (Full Implementation)
- Implement Gemini provider shim with full multimodal support
- Handle Google-specific parameter differences (no reasoning support yet)
- Support for Gemini vision models
- Note: Google doesn't have image generation in Gemini yet

#### 3.2 Provider Stubs (Non-functional)
- Create stub implementations for future providers:
  - `AnthropicShim` (raises NotImplementedError)
  - `AzureOpenAIShim` (raises NotImplementedError)
  - `OllamaShim` (raises NotImplementedError)
- Stubs allow configuration but fail gracefully with helpful error messages
- Registry includes these providers but marks them as `enabled: false` by default

### Phase 4: Migration and Deprecation (Week 6-7)
**Goal**: Migrate existing consumers and deprecate old service (OpenAI + Google only)

#### 4.1 Compatibility Layer
```python
class LangChainCompatibilityService:
    """Compatibility wrapper that translates old API calls to new service."""
    
    def __init__(self, unified_service: UnifiedInferenceService):
        self.unified_service = unified_service
    
    async def generate_text(self, prompt: str, model: str = "auto", **kwargs):
        # Translate old parameters to new format
        return await self.unified_service.generate_text(prompt, model, **kwargs)
```

#### 4.2 Gradual Migration
1. Update API endpoints to use new service internally
2. Migrate example code and documentation
3. Update tests to use new service
4. Remove old LangChain service

## Implementation Priority

### Immediate (This Week)
1. **Fix GPT-5 support**: Add model-specific parameter mapping to existing service
2. **Test reasoning models**: Verify o1-preview and GPT-5 work correctly
3. **Update configuration**: Add model-specific settings to config

### Short Term (Next 2 Weeks)  
1. **Create unified service**: Implement new architecture alongside existing
2. **OpenAI shim**: Full implementation with all model support
3. **Model registry**: Static registry with runtime discovery

### Medium Term (Next Month)
1. **Provider expansion**: Add Google/Gemini support
2. **Migration**: Move existing consumers to new service
3. **Documentation**: Update all docs and examples

## Risk Mitigation

### Backward Compatibility
- Keep existing API endpoints unchanged during migration
- Use feature flags to switch between old and new implementations
- Comprehensive testing before deprecating old service

### Rollback Plan
- Maintain old LangChain service until new service is proven stable
- Feature flags allow instant rollback if issues arise
- Separate deployment of new service components

### Testing Strategy
- Unit tests for each provider shim
- Integration tests for model-specific parameter handling
- End-to-end tests for API compatibility
- Performance regression tests

## Success Metrics

1. **GPT-5 Support**: Successfully generate text with GPT-5 using correct parameters
2. **Parameter Accuracy**: All model-specific parameters mapped correctly
3. **API Compatibility**: Existing API consumers work without changes
4. **Performance**: No significant performance regression
5. **Maintainability**: Easy to add new models and providers
