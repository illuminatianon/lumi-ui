"""Google/Gemini provider shim implementation using pure REST API calls."""

import logging
import json
import base64
from typing import Dict, Any, List, Optional
import httpx

from ..base import ProviderShim
from ..models import (
    UnifiedRequest,
    UnifiedResponse,
    ModelConfig,
    Attachment,
    AttachmentType,
    TokenUsage,
    RequestType,
    ModelCapabilities,
    ParameterMapping
)

logger = logging.getLogger(__name__)


class GoogleShim(ProviderShim):
    """Google/Gemini provider shim using pure REST API calls."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Google shim.

        Args:
            config: Provider configuration including api_key
        """
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = 'https://generativelanguage.googleapis.com'
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    def is_available(self) -> bool:
        """Check if the provider is available."""
        return bool(self.api_key)

    async def process_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Process a unified request using Google Gemini API.
        
        Args:
            request: The unified request to process
            model_config: Configuration for the specific model to use
            
        Returns:
            Unified response containing the result
        """
        request_type = self.determine_request_type(request, model_config)
        
        if request_type == RequestType.VISION_ANALYSIS:
            return await self._handle_vision_request(request, model_config)
        elif request_type == RequestType.IMAGE_GENERATION:
            # Google doesn't have image generation in Gemini yet
            raise NotImplementedError("Google Gemini doesn't support image generation yet")
        elif request_type == RequestType.IMAGE_EDIT:
            raise NotImplementedError("Google Gemini doesn't support image editing yet")
        else:
            return await self._handle_text_request(request, model_config)
    
    async def _handle_text_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle pure text generation requests via REST API."""
        # Build content for Gemini API
        content = self._build_prompt(request)

        # Map parameters
        generation_config = self._build_generation_config(request, model_config)

        payload = {
            "contents": [{"parts": [{"text": content}]}],
            "generationConfig": generation_config
        }

        try:
            url = f'/v1beta/models/{model_config.name}:generateContent'
            params = {'key': self.api_key}

            response = await self.client.post(url, json=payload, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract response content
            content = data['candidates'][0]['content']['parts'][0]['text']

            # Extract usage metadata if available
            usage_metadata = data.get('usageMetadata', {})

            return UnifiedResponse(
                content=content,
                model_used=model_config.name,
                provider="google",
                usage=TokenUsage(
                    prompt_tokens=usage_metadata.get('promptTokenCount', 0),
                    completion_tokens=usage_metadata.get('candidatesTokenCount', 0),
                    total_tokens=usage_metadata.get('totalTokenCount', 0)
                ),
                finish_reason=data['candidates'][0].get('finishReason', 'STOP'),
                metadata={
                    "safety_ratings": data['candidates'][0].get('safetyRatings', [])
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Google text request failed: {e}")
            raise
    
    async def _handle_vision_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle vision requests with image attachments via REST API."""
        # Build content parts for Gemini API
        parts = [{"text": request.prompt}]

        # Add images as inline data
        for attachment in request.attachments:
            if attachment.attachment_type == AttachmentType.IMAGE:
                parts.append({
                    "inline_data": {
                        "mime_type": attachment.mime_type,
                        "data": attachment.to_base64()
                    }
                })

        # Map parameters
        generation_config = self._build_generation_config(request, model_config)

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config
        }

        try:
            url = f'/v1beta/models/{model_config.name}:generateContent'
            params = {'key': self.api_key}

            response = await self.client.post(url, json=payload, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract response content
            content = data['candidates'][0]['content']['parts'][0]['text']

            # Extract usage metadata if available
            usage_metadata = data.get('usageMetadata', {})

            return UnifiedResponse(
                content=content,
                model_used=model_config.name,
                provider="google",
                usage=TokenUsage(
                    prompt_tokens=usage_metadata.get('promptTokenCount', 0),
                    completion_tokens=usage_metadata.get('candidatesTokenCount', 0),
                    total_tokens=usage_metadata.get('totalTokenCount', 0)
                ),
                finish_reason=data['candidates'][0].get('finishReason', 'STOP'),
                attachments_processed=[
                    {"type": att.attachment_type, "filename": att.filename}
                    for att in request.attachments
                ],
                metadata={
                    "safety_ratings": data['candidates'][0].get('safetyRatings', [])
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Google vision request failed: {e}")
            raise
    
    def _build_generation_config(self, request: UnifiedRequest, model_config: ModelConfig) -> Dict[str, Any]:
        """Build Gemini generation configuration from unified request."""
        params = self.map_parameters(request.model_dump(), model_config)

        config_params = {}
        if "temperature" in params and params["temperature"] is not None:
            config_params["temperature"] = params["temperature"]
        if "max_output_tokens" in params and params["max_output_tokens"] is not None:
            config_params["maxOutputTokens"] = params["max_output_tokens"]
        if "top_p" in params and params["top_p"] is not None:
            config_params["topP"] = params["top_p"]
        if "stop_sequences" in params and params["stop_sequences"] is not None:
            config_params["stopSequences"] = params["stop_sequences"]

        return config_params

    def _build_prompt(self, request: UnifiedRequest) -> str:
        """Build prompt from system message and user prompt."""
        if request.system_message:
            return f"System: {request.system_message}\n\nUser: {request.prompt}"
        return request.prompt

    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Discover available Google models via REST API."""
        try:
            url = '/v1beta/models'
            params = {'key': self.api_key}

            response = await self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            discovered = {}
            for model in data.get('models', []):
                # Only include generative models
                if 'generateContent' in model.get('supportedGenerationMethods', []):
                    model_id = model['name'].split('/')[-1]  # Extract model name from full path
                    discovered[model_id] = self._infer_model_config(model_id, model)

            return discovered
        except httpx.HTTPStatusError as e:
            logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
            return {}
        except Exception as e:
            logger.error(f"Failed to discover Google models: {e}")
            return {}

    def map_parameters(self, unified_params: Dict[str, Any], model_config: ModelConfig) -> Dict[str, Any]:
        """Map unified parameters to Google-specific parameters."""
        mapped = {}
        mapping = model_config.parameter_mapping

        # Map standard parameters
        if unified_params.get("max_tokens") and mapping.max_tokens_param:
            mapped[mapping.max_tokens_param] = unified_params["max_tokens"]

        if unified_params.get("temperature") and mapping.temperature_param:
            mapped[mapping.temperature_param] = unified_params["temperature"]

        if unified_params.get("top_p") and mapping.top_p_param:
            mapped[mapping.top_p_param] = unified_params["top_p"]

        # Add stop sequences if supported
        if "stop_sequences" in model_config.supported_parameters:
            mapped["stop_sequences"] = unified_params.get("stop_sequences")

        # Log warning for unsupported parameters
        if unified_params.get("frequency_penalty") or unified_params.get("presence_penalty"):
            logger.warning("Google Gemini doesn't support frequency/presence penalty")

        if unified_params.get("reasoning_effort"):
            logger.warning("Google Gemini doesn't support reasoning effort parameter")

        return mapped

    def prepare_attachments(self, attachments: List[Attachment], model_config: ModelConfig) -> Any:
        """Convert attachments to Google-specific format."""
        # Google handles attachments as PIL Images in content, so return as-is
        return attachments

    def _infer_model_config(self, model_id: str, model_info=None) -> ModelConfig:
        """Infer model configuration based on model ID and info."""
        # Default configuration
        capabilities = ModelCapabilities(text_generation=True)
        parameter_mapping = ParameterMapping(
            max_tokens_param="max_output_tokens",
            temperature_param="temperature",
            top_p_param="top_p"
        )
        supported_parameters = {"max_output_tokens", "temperature", "top_p", "stop_sequences"}

        # Adjust based on model name patterns
        if "vision" in model_id.lower() or "pro" in model_id.lower():
            capabilities.vision = True
            capabilities.function_calling = True

        context_window = 32768  # Conservative default
        if "1.5" in model_id:
            if "pro" in model_id.lower():
                context_window = 2000000
            else:
                context_window = 1000000

        return ModelConfig(
            name=model_id,
            display_name=model_id,
            provider="google",
            capabilities=capabilities,
            parameter_mapping=parameter_mapping,
            supported_parameters=supported_parameters,
            context_window=context_window
        )
