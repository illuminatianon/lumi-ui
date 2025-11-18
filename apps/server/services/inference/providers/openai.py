"""OpenAI provider implementation using pure REST API calls."""

import logging
import json
import base64
from typing import Dict, Any, List, Optional
import httpx

from ..base import Provider
from ..models import (
    UnifiedRequest,
    UnifiedResponse,
    ModelConfig,
    Attachment,
    AttachmentType,
    TokenUsage,
    RequestType,
    ParameterMapping,
    ModelCapabilities
)

logger = logging.getLogger(__name__)


class OpenAIProvider(Provider):
    """OpenAI provider using pure REST API calls."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider.

        Args:
            config: Provider configuration including api_key
        """
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url') or 'https://api.openai.com'
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            },
            timeout=60.0
        )

    @classmethod
    def get_provider_name(cls) -> str:
        """Return the provider name for registry."""
        return "openai"

    @classmethod
    def get_supported_models(cls) -> Dict[str, ModelConfig]:
        """Return all models supported by OpenAI."""
        return {
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                display_name="GPT-4o",
                provider="openai",
                capabilities=ModelCapabilities(
                    text_generation=True,
                    vision=True,
                    function_calling=True,
                    streaming=True,
                    json_mode=True,
                    max_context_length=128000,
                    supports_multimodal=True
                ),
                parameter_mapping=ParameterMapping(
                    max_tokens_param="max_tokens",
                    temperature_param="temperature",
                    top_p_param="top_p"
                ),
                supported_parameters={
                    "max_tokens", "temperature", "top_p",
                    "frequency_penalty", "presence_penalty", "stop", "stream"
                },
                cost_per_1k_tokens=0.005,
                context_window=128000
            ),
            "gpt-5": ModelConfig(
                name="gpt-5",
                display_name="GPT-5",
                provider="openai",
                capabilities=ModelCapabilities(
                    text_generation=True,
                    vision=True,
                    function_calling=True,
                    streaming=True,
                    json_mode=True,
                    max_context_length=200000,
                    supports_multimodal=True
                ),
                parameter_mapping=ParameterMapping(
                    max_tokens_param="max_completion_tokens",
                    temperature_param=None,  # GPT-5 ignores temperature
                    top_p_param=None,
                    custom_params={"reasoning_effort": "medium"}
                ),
                supported_parameters={
                    "max_completion_tokens", "reasoning_effort", "stream"
                },
                cost_per_1k_tokens=0.01,
                context_window=200000
            ),
            "dall-e-3": ModelConfig(
                name="dall-e-3",
                display_name="DALL-E 3",
                provider="openai",
                capabilities=ModelCapabilities(
                    image_generation=True,
                    text_generation=False
                ),
                parameter_mapping=ParameterMapping(),
                supported_parameters={"size", "quality", "style", "n"},
                cost_per_1k_tokens=0.04,
                context_window=4000
            )
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    def is_available(self) -> bool:
        """Check if the provider is available."""
        return bool(self.api_key)

    async def process_request(self, model_name: str, request: UnifiedRequest) -> UnifiedResponse:
        """Process a unified request for a specific model.

        Args:
            model_name: Name of the model to use
            request: The unified request to process

        Returns:
            Unified response containing the result
        """
        # Get model config
        supported_models = self.get_supported_models()
        if model_name not in supported_models:
            available = list(supported_models.keys())
            raise ValueError(f"Model '{model_name}' not supported. Available: {available}")

        model_config = supported_models[model_name]
        request_type = self.determine_request_type(request, model_config)

        if request_type == RequestType.VISION_ANALYSIS:
            return await self._handle_vision_request(request, model_config)
        elif request_type == RequestType.IMAGE_GENERATION:
            return await self._handle_image_generation(request, model_config)
        elif request_type == RequestType.IMAGE_EDIT:
            return await self._handle_image_edit(request, model_config)
        else:
            return await self._handle_text_request(request, model_config)
    
    def _build_messages(self, request: UnifiedRequest) -> List[Dict[str, Any]]:
        """Build messages array from either single-turn or multi-turn format."""
        if request.is_multi_turn():
            # Use messages directly - convert to OpenAI format
            return [{"role": msg.role, "content": msg.content} for msg in request.messages]
        else:
            # Build from prompt and system_message (backward compatibility)
            messages = []
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            messages.append({"role": "user", "content": request.prompt})
            return messages

    def _build_messages_with_attachments(self, request: UnifiedRequest) -> List[Dict[str, Any]]:
        """Build messages array with attachment support for vision requests."""
        if request.is_multi_turn():
            # For multi-turn, attachments are assumed to be part of the last user message
            # This is a simplification - in the future we could support attachments per message
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            # If there are attachments, modify the last user message to include them
            if request.attachments:
                # Find the last user message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]["role"] == "user":
                        # Convert text content to multimodal format
                        content = [{"type": "text", "text": messages[i]["content"]}]

                        # Add images
                        for attachment in request.attachments:
                            if attachment.attachment_type == AttachmentType.IMAGE:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{attachment.mime_type};base64,{attachment.to_base64()}"
                                    }
                                })

                        messages[i]["content"] = content
                        break

            return messages
        else:
            # Single-turn with attachments (existing logic)
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
            return messages

    async def _handle_text_request(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Handle pure text generation requests via REST API."""
        # Build messages
        messages = self._build_messages(request)

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

            # Extract content from response
            message = data['choices'][0]['message']
            content = message.get('content', '')

            # Debug logging for empty content (useful for reasoning models)
            if not content:
                usage = data.get('usage', {})
                reasoning_tokens = usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0)
                if reasoning_tokens > 0:
                    logger.warning(f"Empty content from {model_config.name} - used {reasoning_tokens} reasoning tokens, may need higher max_tokens")
                else:
                    logger.warning(f"Empty content from {model_config.name}. Full response: {data}")

            return UnifiedResponse(
                content=content,
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
        messages = self._build_messages_with_attachments(request)

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
        """Handle OpenAI image generation (DALL-E 3, GPT Image 1) via REST API."""

        # Build base payload
        payload = {
            "model": model_config.name,
            "prompt": request.prompt,
            "n": 1
        }

        # Handle model-specific parameters
        if model_config.name == "gpt-image-1":
            # GPT Image 1 specific parameters
            if "aspect_ratio" in request.extras:
                aspect_ratio = request.extras["aspect_ratio"]
                # Map aspect ratio to size for GPT Image 1
                size_mapping = {
                    "1:1": "1024x1024",
                    "3:2": "1536x1024",
                    "2:3": "1024x1536"
                }
                payload["size"] = size_mapping.get(aspect_ratio, "1024x1024")
            else:
                payload["size"] = request.extras.get("size", "1024x1024")

            # Quality parameter
            if "quality" in request.extras:
                payload["quality"] = request.extras["quality"]

            # Style parameter
            if "style" in request.extras:
                payload["style"] = request.extras["style"]

        else:
            # DALL-E 3 and other models
            payload["size"] = request.extras.get("size", "1024x1024")
            payload["quality"] = request.extras.get("quality", "standard")

            # Style parameter for DALL-E 3
            if "style" in request.extras:
                payload["style"] = request.extras["style"]

        try:
            response = await self.client.post('/v1/images/generations', json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract images and any revised prompts
            images = []
            for img_data in data['data']:
                images.append(img_data['url'])

            # Extract revised prompt if available (DALL-E 3 feature)
            revised_prompt = None
            if data['data'] and 'revised_prompt' in data['data'][0]:
                revised_prompt = data['data'][0]['revised_prompt']

            return UnifiedResponse(
                images=images,
                model_used=model_config.name,
                provider="openai",
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="completed",
                attachments_processed=[
                    {"type": att.attachment_type, "used_as": "reference"}
                    for att in request.attachments
                ],
                metadata={
                    "revised_prompt": revised_prompt,
                    "original_size": payload.get("size"),
                    "quality": payload.get("quality"),
                    "style": payload.get("style")
                }
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

    # Legacy compatibility methods
    async def process_request_legacy(self, request: UnifiedRequest, model_config: ModelConfig) -> UnifiedResponse:
        """Legacy method for backward compatibility."""
        return await self.process_request(model_config.name, request)

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
