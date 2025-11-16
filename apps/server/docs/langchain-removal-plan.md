# Complete SDK Removal and Pure REST API Migration Plan

## Executive Summary

This document outlines the plan to remove ALL AI SDK dependencies (LangChain, OpenAI SDK, Google SDK) from the Lumi UI backend and migrate to pure REST API calls using httpx. This will eliminate bloated dependencies while preserving the excellent unified inference abstractions, giving us full control over API interactions.

## Current State Analysis

### LangChain Usage Patterns

**Dependencies (pyproject.toml):**
- `langchain>=1.0.0` - Core framework
- `langchain-openai>=1.0.0` - OpenAI integration
- `langchain-google-genai>=2.0.0,<3.0.0` - Google integration

**Current LangChain Usage:**
1. **services/langchain_service.py** - Legacy service using LangChain wrappers
2. **services/inference/compatibility.py** - Compatibility layer between LangChain and unified inference
3. **tests/test_langchain_service.py** - LangChain-specific tests
4. **examples/langchain_examples.py** - LangChain usage examples

### Current SDK Usage (To Be Removed)

**Heavy Dependencies to Remove:**
- **OpenAI SDK**: `openai>=1.0.0` - Uses `AsyncOpenAI` client (bloated, unnecessary)
- **Google SDK**: `google.generativeai>=0.3.0,<0.8.0` - Overly complicated, bloated
- **LangChain**: All LangChain packages (already identified for removal)

**Target Architecture:**
- **Pure REST**: Use `httpx` for all HTTP calls
- **Lightweight**: No vendor SDKs, just HTTP requests and JSON parsing
- **Full Control**: Direct access to all API features without SDK limitations

### Salvageable Components

**Keep (Modify for REST):**
- `services/inference/models.py` - Data models and abstractions ✓
- `services/inference/base.py` - Provider shim interface ✓
- `services/inference/registry.py` - Model and provider registries ✓
- `services/inference/service.py` - Main unified service ✓

**Rewrite for Pure REST:**
- `services/inference/providers/openai_shim.py` - Replace AsyncOpenAI with httpx
- `services/inference/providers/google_shim.py` - Replace google.generativeai with httpx

**Remove Completely:**
- `api/ai_router.py` - Auto-generated, never used
- `services/langchain_service.py` - Legacy LangChain service
- `services/inference/compatibility.py` - Compatibility layer

## Migration Strategy

### Phase 1: Complete SDK Removal Strategy

**Remove ALL AI SDKs:**
- `langchain>=1.0.0` and related packages
- `openai>=1.0.0` - Replace with pure REST
- `google-generativeai>=0.3.0,<0.8.0` - Replace with pure REST

**Use Pure HTTP:**
- `httpx>=0.25.0` - Already in dev dependencies, promote to main
- Direct JSON request/response handling
- Custom error handling and retry logic

### Phase 2: Provider Shim Rewrite

**OpenAI Shim (`openai_shim.py`):**
- Replace `AsyncOpenAI` with `httpx.AsyncClient`
- Implement direct calls to:
  - `POST /v1/chat/completions` - Text and vision
  - `POST /v1/images/generations` - DALL-E
  - `POST /v1/images/edits` - Image editing
- Custom authentication header handling
- Direct JSON parsing and error handling

**Google Shim (`google_shim.py`):**
- Replace `google.generativeai` with `httpx.AsyncClient`
- Implement direct calls to:
  - `POST /v1beta/models/{model}:generateContent` - Text and vision
  - Custom multipart handling for images
- Direct API key authentication
- Custom response parsing

### Phase 3: Remove Legacy Components

**Complete Removal:**
- `services/langchain_service.py`
- `services/inference/compatibility.py`
- `api/ai_router.py` - Auto-generated, never used
- `tests/test_langchain_service.py`
- `examples/langchain_examples.py`
- `docs/langchain-integration.md`
- `LANGCHAIN_BOOTSTRAP.md`

### Phase 4: Dependency Cleanup

**Final Dependencies:**
- `httpx>=0.25.0` - Pure HTTP client
- `pydantic>=2.5.0` - Data validation (keep)
- `pillow>=12.0.0` - Image processing (keep)

**Remove Everything Else:**
- All LangChain packages
- OpenAI SDK
- Google SDK

## Implementation Plan

### Step 1: Rewrite Provider Shims for Pure REST
- Rewrite `openai_shim.py` to use httpx instead of AsyncOpenAI
- Rewrite `google_shim.py` to use httpx instead of google.generativeai
- Implement custom JSON handling and error management
- Add proper authentication and request formatting

### Step 2: Remove All Legacy Components
- Delete `services/langchain_service.py`
- Delete `services/inference/compatibility.py`
- Delete `api/ai_router.py` (auto-generated, never used)
- Remove all SDK imports and references

### Step 3: Clean Up Dependencies Completely
- Remove ALL AI SDK packages from `pyproject.toml`
- Promote `httpx` from dev to main dependencies
- Update setup scripts

### Step 4: Testing and Validation
- Test pure REST implementation thoroughly
- Verify all inference functionality works
- Performance testing vs old SDK approach

## Benefits of Complete SDK Removal

### Immediate Benefits
1. **Massive Dependency Reduction** - Remove 5+ heavy AI packages
2. **Full API Control** - No SDK limitations on any features
3. **Dramatically Reduced Complexity** - Pure HTTP, no abstraction conflicts
4. **Much Faster Startup** - Minimal imports, no SDK initialization
5. **Smaller Bundle Size** - Eliminate megabytes of unused SDK code

### Long-term Benefits
1. **Complete API Access** - Every API feature available immediately
2. **Superior Error Handling** - Direct HTTP status codes and error messages
3. **Maximum Performance** - Zero SDK overhead, pure HTTP speed
4. **Ultimate Maintainability** - Only maintain HTTP requests, not SDK updates
5. **Future-Proof** - Works with any API changes, no SDK update dependencies

## Risk Assessment

### Low Risk
- **Unified inference system is already complete** - No functionality loss
- **Provider shims already use direct APIs** - No implementation changes needed
- **API endpoints already use unified inference** - No breaking changes

### Mitigation Strategies
- **Comprehensive testing** - Verify all functionality works post-migration
- **Gradual rollout** - Remove components incrementally
- **Rollback plan** - Keep LangChain dependencies in git history

## Success Criteria

1. **All AI functionality preserved** - Text generation, image analysis, image generation
2. **No breaking API changes** - Existing endpoints continue to work
3. **Improved performance** - Faster response times and lower memory usage
4. **Cleaner codebase** - Reduced complexity and dependencies
5. **Better image generation** - Full DALL-E feature access

## Timeline

- **Phase 1**: Analysis and planning (Complete)
- **Phase 2**: Remove LangChain service layer (1-2 hours)
- **Phase 3**: Dependency cleanup (30 minutes)
- **Phase 4**: Documentation and testing (1-2 hours)

**Total Estimated Time: 3-5 hours**

## Detailed TODO List

### Phase 1: Rewrite Provider Shims for Pure REST
- [ ] **OpenAI Shim Rewrite** (`services/inference/providers/openai_shim.py`):
  - [ ] Replace `from openai import AsyncOpenAI` with `import httpx`
  - [ ] Implement `POST /v1/chat/completions` for text/vision
  - [ ] Implement `POST /v1/images/generations` for DALL-E
  - [ ] Implement `POST /v1/images/edits` for image editing
  - [ ] Add custom authentication headers
  - [ ] Add proper JSON request/response handling
  - [ ] Add custom error handling and retry logic

- [ ] **Google Shim Rewrite** (`services/inference/providers/google_shim.py`):
  - [ ] Replace `import google.generativeai as genai` with `import httpx`
  - [ ] Implement `POST /v1beta/models/{model}:generateContent`
  - [ ] Add custom multipart handling for vision requests
  - [ ] Add API key authentication
  - [ ] Add custom JSON parsing for Google's response format
  - [ ] Handle Google's specific error responses

### Phase 2: Complete Legacy Removal
- [ ] Delete `services/langchain_service.py`
- [ ] Delete `services/inference/compatibility.py`
- [ ] Delete `api/ai_router.py` (auto-generated, never used)
- [ ] Delete `tests/test_langchain_service.py`
- [ ] Delete `examples/langchain_examples.py`
- [ ] Delete `docs/langchain-integration.md`
- [ ] Delete `LANGCHAIN_BOOTSTRAP.md`
- [ ] Update `services/__init__.py` to remove all legacy exports

### Phase 3: Complete Dependency Cleanup
- [ ] **Remove ALL AI SDKs from `pyproject.toml`**:
  - [ ] Remove `langchain>=1.0.0`
  - [ ] Remove `langchain-openai>=1.0.0`
  - [ ] Remove `langchain-google-genai>=2.0.0,<3.0.0`
  - [ ] Remove `openai>=1.0.0`
  - [ ] Remove `google-generativeai>=0.3.0,<0.8.0`
- [ ] **Add pure HTTP dependency**:
  - [ ] Move `httpx>=0.25.0` from dev to main dependencies
- [ ] Update `scripts/setup.sh` to remove SDK installations
- [ ] Update main `README.md` to remove SDK references

### Phase 4: Configuration Cleanup
- [ ] Remove LangChain-specific configuration from `config/models.py`
- [ ] Update `config/loader.py` to remove LangChain defaults
- [ ] Clean up any SDK-related environment variable handling
- [ ] Update configuration documentation

### Phase 5: Testing and Validation
- [ ] **Create comprehensive REST API tests**:
  - [ ] Test OpenAI text generation via pure REST
  - [ ] Test OpenAI vision analysis via pure REST
  - [ ] Test OpenAI image generation via pure REST
  - [ ] Test Google text generation via pure REST
  - [ ] Test Google vision analysis via pure REST
- [ ] **Performance testing**:
  - [ ] Measure startup time improvement
  - [ ] Measure memory usage reduction
  - [ ] Measure request latency comparison
- [ ] **Integration testing**:
  - [ ] Test unified inference service end-to-end
  - [ ] Verify all model capabilities work correctly
  - [ ] Test error handling and retry logic

### Phase 6: Documentation and Examples
- [ ] Create new examples showing pure REST usage
- [ ] Document the REST API implementation approach
- [ ] Update setup and installation instructions
- [ ] Document performance improvements and benefits

## API Implementation Details

### OpenAI REST API Endpoints
```
Base URL: https://api.openai.com
Authentication: Bearer {api_key}

Text/Vision: POST /v1/chat/completions
Image Gen:   POST /v1/images/generations
Image Edit:  POST /v1/images/edits
```

### Google Gemini REST API Endpoints
```
Base URL: https://generativelanguage.googleapis.com
Authentication: ?key={api_key}

Text/Vision: POST /v1beta/models/{model}:generateContent
```

## Next Steps

1. **Start with Phase 1** - Rewrite provider shims for pure REST (core functionality)
2. **Continue with Phase 2** - Remove all legacy components completely
3. **Execute Phase 3** - Clean up all SDK dependencies
4. **Complete Phase 4** - Update configuration
5. **Finish with Phases 5-6** - Test thoroughly and document

**Estimated Time: 4-6 hours for complete SDK removal and pure REST implementation**
