"""OpenAI provider shim implementation using pure REST API calls."""

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
    RequestType
)

logger = logging.getLogger(__name__)


class OpenAIShim(ProviderShim):
    """OpenAI provider shim using pure REST API calls."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI shim.

        Args:
            config: Provider configuration including api_key
        """
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url', 'https://api.openai.com')
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
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
        """Process a unified request using OpenAI API.
        
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
            return await self._handle_image_generation(request, model_config)
        elif request_type == RequestType.IMAGE_EDIT:
            return await self._handle_image_edit(request, model_config)
        else:
            return await self._handle_text_request(request, model_config)
    
    async def _handle_text_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle pure text generation requests via REST API."""
        # Build messages
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.append({"role": "user", "content": request.prompt})

        # Map parameters
        params = self.map_parameters(request.model_dump(), model_config)
        payload = {
            "model": model_config.name,
            "messages": messages,
            **{k: v for k, v in params.items() if v is not None and k in model_config.supported_parameters}
        }

        try:
            response = await self.client.post('/v1/chat/completions', json=payload)
            response.raise_for_status()
            data = response.json()

            return UnifiedResponse(
                content=data['choices'][0]['message']['content'],
                model_used=data['model'],
                provider="openai",
                usage=TokenUsage(
                    prompt_tokens=data['usage']['prompt_tokens'],
                    completion_tokens=data['usage']['completion_tokens'],
                    total_tokens=data['usage']['total_tokens']
                ),
                finish_reason=data['choices'][0]['finish_reason'],
                metadata={"response_id": data['id']}
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenAI text request failed: {e}")
            raise
    
    async def _handle_vision_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle vision requests with image attachments via REST API."""
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

        # Map parameters
        params = self.map_parameters(request.model_dump(), model_config)
        payload = {
            "model": model_config.name,
            "messages": messages,
            **{k: v for k, v in params.items() if v is not None and k in model_config.supported_parameters}
        }

        try:
            response = await self.client.post('/v1/chat/completions', json=payload)
            response.raise_for_status()
            data = response.json()

            return UnifiedResponse(
                content=data['choices'][0]['message']['content'],
                model_used=data['model'],
                provider="openai",
                usage=TokenUsage(
                    prompt_tokens=data['usage']['prompt_tokens'],
                    completion_tokens=data['usage']['completion_tokens'],
                    total_tokens=data['usage']['total_tokens']
                ),
                finish_reason=data['choices'][0]['finish_reason'],
                attachments_processed=[
                    {"type": att.attachment_type, "filename": att.filename}
                    for att in request.attachments
                ],
                metadata={"response_id": data['id']}
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenAI vision request failed: {e}")
            raise
    
    async def _handle_image_generation(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle DALL-E image generation via REST API."""
        payload = {
            "model": model_config.name,
            "prompt": request.prompt,
            "size": request.extras.get("size", "1024x1024"),
            "quality": request.extras.get("quality", "standard"),
            "n": 1
        }

        try:
            response = await self.client.post('/v1/images/generations', json=payload)
            response.raise_for_status()
            data = response.json()

            return UnifiedResponse(
                images=[img['url'] for img in data['data']],
                model_used=model_config.name,
                provider="openai",
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="completed",
                attachments_processed=[
                    {"type": att.attachment_type, "used_as": "reference"}
                    for att in request.attachments
                ]
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenAI image generation failed: {e}")
            raise
    
    async def _handle_image_edit(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle image editing requests."""
        # For now, treat as image generation with reference
        # TODO: Implement proper image editing when OpenAI supports it
        return await self._handle_image_generation(request, model_config)

    async def discover_models(self) -> Dict[str, ModelConfig]:
        """Discover available OpenAI models via REST API."""
        try:
            response = await self.client.get('/v1/models')
            response.raise_for_status()
            data = response.json()

            discovered = {}
            for model in data['data']:
                # Use static registry if available, otherwise infer
                if model['id'] in ["gpt-4o", "gpt-5", "o1-preview", "dall-e-3"]:
                    # These are handled by static registry
                    continue
                else:
                    # Infer configuration for unknown models
                    discovered[model['id']] = self._infer_model_config(model['id'])

            return discovered
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            return {}
        except Exception as e:
            logger.error(f"Failed to discover OpenAI models: {e}")
            return {}

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
        if model_config.capabilities.reasoning:
            reasoning_effort = unified_params.get("reasoning_effort")
            if reasoning_effort and "reasoning_effort" in model_config.supported_parameters:
                mapped["reasoning_effort"] = reasoning_effort
            elif mapping.custom_params.get("reasoning_effort"):
                # Use model default if no effort specified
                mapped["reasoning_effort"] = mapping.custom_params["reasoning_effort"]

        return mapped

    def prepare_attachments(self, attachments: List[Attachment], model_config: ModelConfig) -> Any:
        """Convert attachments to OpenAI-specific format."""
        # OpenAI handles attachments inline in messages, so return as-is
        return attachments

    def _infer_model_config(self, model_id: str) -> ModelConfig:
        """Infer model configuration based on model ID patterns."""
        from ..models import ModelCapabilities, ParameterMapping

        # Default configuration for unknown models
        capabilities = ModelCapabilities(text_generation=True)
        parameter_mapping = ParameterMapping()
        supported_parameters = {"max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop", "stream"}

        # Adjust based on model name patterns
        if "gpt-4" in model_id.lower():
            capabilities.vision = True
            capabilities.function_calling = True
            capabilities.json_mode = True
        elif "dall-e" in model_id.lower():
            capabilities = ModelCapabilities(image_generation=True, text_generation=False)
            supported_parameters = {"size", "quality", "style", "n"}

        return ModelConfig(
            name=model_id,
            display_name=model_id,
            provider="openai",
            capabilities=capabilities,
            parameter_mapping=parameter_mapping,
            supported_parameters=supported_parameters,
            context_window=8192  # Conservative default
        )
